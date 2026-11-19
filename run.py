"""
ANPR Gate Control — Native Windows Desktop Launcher (PyQt6)
===========================================================
Startup sequence:
  1. Register bundled ONNX Runtime / DirectML DLL directories (Nuitka builds).
  2. Redirect stdio safely when running with `--windows-console-mode=disable`.
  3. Run disclosed startup HTTPS license check (`license_check.py`) BEFORE any
     window opens; show clear visible MessageBox and exit on failure.
  4. Initialize SQLite database & default RBAC accounts (admin / guard / auditor).
  5. Show native PyQt6 RBAC Login Dialog before opening the Main Window.
  6. Launch native PyQt6 Main Window with live ONNX Runtime DirectML/CPU feed,
     camera switching, and background QThreads (no browser, no console).

Optional flags:
  python run.py --source synthetic  # Start with simulated entrance gate feed
  python run.py --check-license     # Force startup HTTPS license validation
  python run.py --api-only          # Run internal FastAPI engine on 127.0.0.1 only
"""

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = (
    Path(sys.executable).resolve().parent
    if ("__compiled__" in globals() or getattr(sys, "frozen", False))
    else Path(__file__).resolve().parent
)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure bundled ONNX Runtime / DirectML native DLLs are discoverable before any imports
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    for _dll_dir in (BASE_DIR, BASE_DIR / "onnxruntime" / "capi"):
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir))
            except OSError:
                pass

from app.core.config import settings
from app.core.license_check import LicenseMonitor, enforce_license_or_exit
from app.core.logging_config import setup_logging
from app.database.connection import init_db


def _ensure_stdio_for_windowed_mode() -> bool:
    """
    When built with Nuitka `--windows-console-mode=disable`, sys.stdout and
    sys.stderr are None. Redirect them to `logs/runtime_console.log` so logging
    and third-party libraries never raise AttributeError on missing streams.
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


def _launch_native_gui(license_status: str) -> int:
    """Launch the native PyQt6 RBAC Login Dialog followed by the Main Window."""
    from PyQt6.QtWidgets import QApplication
    from app.gui.backend_bridge import BackendBridge
    from app.gui.dialogs import LoginDialog
    from app.gui.main_window import MainWindow
    from app.gui.styles import DARK_STYLESHEET

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName(settings.APP_NAME)
    qt_app.setStyle("Fusion")
    qt_app.setStyleSheet(DARK_STYLESHEET)

    bridge = BackendBridge()

    # Show native RBAC Login Dialog before opening the Main Window
    login_dlg = LoginDialog(bridge)
    if login_dlg.exec() != LoginDialog.DialogCode.Accepted or login_dlg.session is None:
        return 0

    window = MainWindow(
        bridge=bridge,
        session=login_dlg.session,
        auto_start_engine=True,
        license_status=license_status,
    )

    # Periodic disclosed license monitor hooked safely to Qt main thread
    monitor = LicenseMonitor(
        on_revoked=lambda res: window.license_revoked_signal.emit(res.user_message)
    )
    monitor.start()

    window.show()
    try:
        return qt_app.exec()
    finally:
        monitor.stop()
        bridge.stop_pipeline()


def _launch_internal_api(port: int, windowed: bool) -> None:
    """Optional loopback-only (127.0.0.1) FastAPI engine mode for local tooling."""
    import uvicorn
    from app.main import app as fastapi_app

    monitor = LicenseMonitor()
    monitor.start()
    try:
        uvicorn_kwargs = {
            "host": "127.0.0.1",  # Strictly loopback-only; never exposed beyond localhost
            "port": port,
            "log_level": "info",
            "reload": False,
        }
        if windowed:
            uvicorn_kwargs["log_config"] = None
        uvicorn.run(fastapi_app, **uvicorn_kwargs)
    finally:
        monitor.stop()


def main():
    windowed = _ensure_stdio_for_windowed_mode()
    setup_logging()

    parser = argparse.ArgumentParser(description="ANPR Native Windows Desktop Application")
    parser.add_argument("--source", type=str, help="Override camera source ('0', video path, or 'synthetic')")
    parser.add_argument("--device", type=str, help="Override compute device ('auto', 'cpu')")
    parser.add_argument("--check-license", action="store_true", help="Force startup HTTPS license validation")
    parser.add_argument("--api-only", action="store_true", help="Run internal FastAPI engine strictly on 127.0.0.1")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Loopback port for --api-only mode")
    args = parser.parse_args()

    if args.source:
        settings.CAMERA_SOURCE = args.source
    if args.device:
        settings.DEVICE = args.device
    if args.check_license:
        settings.LICENSE_CHECK_ENABLED = True

    # 1. Disclosed startup license check BEFORE any window opens
    lic_result = enforce_license_or_exit(force=args.check_license)
    os.environ["_ANPR_LICENSE_VERIFIED_PID"] = str(os.getpid())

    # 2. Initialize SQLite database and seed default RBAC accounts
    init_db()

    # 3. Launch Native PyQt6 GUI (default) or internal loopback-only API engine
    if args.api_only:
        _launch_internal_api(args.port, windowed)
    else:
        exit_code = _launch_native_gui(lic_result.status)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
