"""
ANPR System Master Launcher
===========================
Launches the Production Backend & Responsive Web UI (Desktop & Mobile).
Provides dynamic hardware resolution, ASCII telemetry banners, and graceful shutdown.

Usage:
  python run.py                     # Runs unified production server (Web UI + APIs)
  python run.py --source synthetic  # Runs with simulated entrance gate video feed
  python run.py --device cuda       # Forces CUDA hardware acceleration
  python run.py --device cpu        # Forces CPU mode
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.core.device import resolve_device, get_device_diagnostics
from app.database.connection import init_db


def print_banner(device_str: str, diag: dict):
    print("=" * 72)
    print("    AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) & GATE ACCESS SYSTEM    ")
    print("                      PRODUCTION RBAC SUITE                           ")
    print("=" * 72)
    print(f"  ● Compute Device:       {device_str.upper()} ({diag.get('name', 'CPU')})")
    if "total_memory_mb" in diag:
        print(f"  ● GPU VRAM:             {diag.get('total_memory_mb'):,.0f} MB")
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
    parser = argparse.ArgumentParser(description="ANPR Production Master Launcher")
    parser.add_argument("--source", type=str, help="Override camera source ('0', video path, or 'synthetic')")
    parser.add_argument("--device", type=str, help="Override compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port to run server on")
    parser.add_argument("--host", type=str, default=settings.API_HOST, help="Host binding")

    args = parser.parse_args()

    if args.source:
        settings.CAMERA_SOURCE = args.source
    if args.device:
        settings.DEVICE = args.device

    # Initialize SQLite database and seed default RBAC users
    init_db()

    # Dynamic hardware acceleration resolution
    resolved_dev, diag = resolve_device(settings.DEVICE)
    print_banner(resolved_dev, diag)

    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info", reload=False)


if __name__ == "__main__":
    main()
