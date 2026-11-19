"""
ANPR System Master Launcher
===========================
Launches the Production Backend & Responsive Web UI (Desktop & Mobile).
Runs the disclosed startup license check, dynamic hardware/ONNX Runtime
resolution, and serves the FastAPI application cleanly in both console and
Nuitka console-disabled Windows standalone builds.

Usage:
  python run.py                     # Runs unified production server (Web UI + APIs)
  python run.py --source synthetic  # Runs with simulated entrance gate video feed
  python run.py --device cpu        # Forces CPU mode
  python run.py --check-license     # Force-enables startup license validation
"""

import os
import sys
import argparse
from pathlib import Path

BASE_DIR = (
    Path(sys.executable).resolve().parent
    if ("__compiled__" in globals() or getattr(sys, "frozen", False))
    else Path(__file__).resolve().parent
)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.core.device import resolve_device
from app.core.license_check import enforce_license_or_exit, LicenseMonitor
from app.database.connection import init_db


def _ensure_stdio_for_windowed_mode() -> bool:
    """
    When built with Nuitka `--windows-console-mode=disable`, sys.stdout and
    sys.stderr are None. Redirect them to `logs/runtime_console.log` so print()
    and uvicorn never raise AttributeError on missing console streams.
    Returns True if running in console-disabled/windowed mode.
    """
    windowed = sys.stdout is None or sys.stderr is None
    if windowed:
        try:
            log_file = settings.LOGS_DIR / "runtime_console.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            stream = open(log_file, "a", encoding="utf-8", buffering=1)
        except Exception:
            stream = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
    return windowed


def print_banner(device_str: str, diag: dict, license_status: str):
    onnx_providers = diag.get("onnx_providers", [])
    print("=" * 72)
    print("    AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) & GATE ACCESS SYSTEM    ")
    print("                      PRODUCTION RBAC SUITE                           ")
    print("=" * 72)
    print(f"  ● Inference Backend:    {settings.INFERENCE_BACKEND.upper()}")
    print(f"  ● Compute Device:       {device_str.upper()} ({diag.get('name', 'CPU')})")
    if onnx_providers:
        print(f"  ● ONNX Providers:       {', '.join(onnx_providers)}")
    if "total_memory_mb" in diag:
        print(f"  ● GPU VRAM:             {diag.get('total_memory_mb'):,.0f} MB")
    print(f"  ● License Status:       {license_status}")
    print(f"  ● Video Source:         {settings.CAMERA_SOURCE}")
    print(f"  ● Event Strategy:       {settings.EVENT_STRATEGY}")
    print(f"  ● Recognition Cooldown: {settings.RECOGNITION_COOLDOWN}s")
    print(f"  ● SQLite Database:      {settings.DATABASE_URL}")
    print(f"  ● Mobile Push (ntfy):   {settings.NTFY_SERVER}/{settings.NTFY_TOPIC}")
    print("-" * 72)
    print(f"  ★ Production Web App:   http://localhost:{settings.API_PORT}/ (Mobile & Desktop)")
    print(f"  ★ REST API Docs:        http://localhost:{settings.API_PORT}/docs")
    print(f"  ★ Default Admin Login:  admin / admin123 (Role: ADMIN)")
    print(f"  ★ Default Guard Login:  guard / guard123 (Role: OPERATOR)")
    print(f"  ★ Default Audit Login:  auditor / auditor123 (Role: AUDITOR)")
    print("=" * 72)
    print("Press CTRL+C to gracefully stop all services.")
    print("=" * 72 + "\n")


def main():
    windowed = _ensure_stdio_for_windowed_mode()

    parser = argparse.ArgumentParser(description="ANPR Production Master Launcher")
    parser.add_argument("--source", type=str, help="Override camera source ('0', video path, or 'synthetic')")
    parser.add_argument("--device", type=str, help="Override compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port to run server on")
    parser.add_argument("--host", type=str, default=settings.API_HOST, help="Host binding")
    parser.add_argument("--check-license", action="store_true", help="Force startup HTTPS license validation")

    args = parser.parse_args()

    if args.source:
        settings.CAMERA_SOURCE = args.source
    if args.device:
        settings.DEVICE = args.device
    if args.check_license:
        settings.LICENSE_CHECK_ENABLED = True

    # 1. Disclosed startup license check (exits cleanly with clear notice if inactive)
    lic_result = enforce_license_or_exit(force=args.check_license)
    os.environ["_ANPR_LICENSE_VERIFIED_PID"] = str(os.getpid())

    # 2. Initialize SQLite database and seed default RBAC users
    init_db()

    # 3. Dynamic hardware / ONNX Runtime provider resolution
    resolved_dev, diag = resolve_device(settings.DEVICE)
    print_banner(resolved_dev, diag, lic_result.status)

    # 4. Start optional periodic license re-check monitor (only active when LICENSE_CHECK_ENABLED=true)
    monitor = LicenseMonitor()
    monitor.start()

    # 5. Direct ASGI import ensures Nuitka statically bundles app.main and all routers
    import uvicorn
    from app.main import app as fastapi_app

    try:
        uvicorn_kwargs = {
            "host": args.host,
            "port": args.port,
            "log_level": "info",
            "reload": False,
        }
        if windowed:
            # Avoid uvicorn's default TTY formatter crashing when console is disabled
            uvicorn_kwargs["log_config"] = None
        uvicorn.run(fastapi_app, **uvicorn_kwargs)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
