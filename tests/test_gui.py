"""
Native PyQt6 Desktop GUI & In-Process Backend Bridge Test Suite
===============================================================
Verifies:
  - Native LoginDialog RBAC authentication (admin, guard, auditor, invalid)
  - MainWindow RBAC visibility gating across all 7 tabs & action buttons
  - Camera device enumeration and live camera switching
  - Manual Gate Entry, Vehicle Registry, Event History, CSV export, and RBAC guards
  - Startup license gate runs before opening any Qt window
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Run Qt cleanly in offscreen mode during automated pytest runs
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PyQt6.QtWidgets import QApplication

from app.database.connection import init_db
from app.gui.backend_bridge import BackendBridge, PermissionDeniedError
from app.gui.dialogs import LoginDialog
from app.gui.main_window import MainWindow
from app.vision.camera_manager import CameraManager


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    init_db()
    return app


def test_login_dialog_rbac_roles_and_invalid_password(qt_app):
    bridge = BackendBridge()
    dlg = LoginDialog(bridge)

    # Invalid password stays rejected with visible error
    dlg.username_edit.setText("admin")
    dlg.password_edit.setText("wrongpass")
    dlg._handle_login()
    assert dlg.session is None
    assert "incorrect" in dlg.error_label.text().lower()

    # Valid admin, guard, auditor logins
    for user, pw, expected_role in [
        ("admin", "admin123", "ADMIN"),
        ("guard", "guard123", "OPERATOR"),
        ("auditor", "auditor123", "AUDITOR"),
    ]:
        dlg2 = LoginDialog(bridge)
        dlg2.username_edit.setText(user)
        dlg2.password_edit.setText(pw)
        dlg2._handle_login()
        assert dlg2.session is not None
        assert dlg2.session.role == expected_role
        dlg2.close()
    dlg.close()


def test_main_window_rbac_tab_and_control_gating(qt_app):
    bridge = BackendBridge()

    # 1. ADMIN sees all 7 tabs and admin/operator controls
    admin_sess = bridge.authenticate("admin", "admin123")
    win = MainWindow(bridge, admin_sess, auto_start_engine=False, license_status="active")
    assert not win._nav_buttons["users"].isHidden()
    assert not win._nav_buttons["system"].isHidden()
    assert not win.btn_manual_entry.isHidden()
    assert not win.btn_quick_in.isHidden()
    assert not win.btn_purge_vehicles.isHidden()
    assert not win.btn_purge_events.isHidden()
    assert not win.btn_purge_clips.isHidden()
    assert win.camera_combo.isEnabled()

    # 2. Switch to OPERATOR (guard) -> hides admin tabs & purges, keeps manual entry & camera
    win.session = bridge.authenticate("guard", "guard123")
    win._apply_rbac_ui()
    assert win._nav_buttons["users"].isHidden()
    assert win._nav_buttons["system"].isHidden()
    assert not win.btn_manual_entry.isHidden()
    assert not win.btn_quick_in.isHidden()
    assert win.btn_purge_vehicles.isHidden()
    assert win.camera_combo.isEnabled()

    # 3. Switch to AUDITOR -> read-only, hides manual entry & disables camera switching
    win.session = bridge.authenticate("auditor", "auditor123")
    win._apply_rbac_ui()
    assert win._nav_buttons["users"].isHidden()
    assert win._nav_buttons["system"].isHidden()
    assert win.btn_manual_entry.isHidden()
    assert win.btn_quick_in.isHidden()
    assert not win.camera_combo.isEnabled()
    win.close()


def test_camera_enumeration_and_live_switching(qt_app):
    cm = CameraManager(source="synthetic")
    cm.start()
    try:
        from app.services.camera_service import CameraService
        svc = CameraService(camera_manager=cm)
        devices = svc.list_cameras(max_index=2)
        assert any(d["source"] == "synthetic" for d in devices)

        # Switch live to synthetic
        ok = svc.switch_source("synthetic")
        assert ok is True
        assert cm.source == "synthetic"
        assert cm.is_connected is True
    finally:
        cm.stop()


def test_backend_bridge_rbac_enforcement_and_csv_export(qt_app, tmp_path):
    bridge = BackendBridge()
    admin = bridge.authenticate("admin", "admin123")
    guard = bridge.authenticate("guard", "guard123")
    auditor = bridge.authenticate("auditor", "auditor123")

    # Auditor cannot record manual entry or list users
    with pytest.raises(PermissionDeniedError):
        bridge.manual_gate_entry(auditor, "KA01GUI999", "IN")
    with pytest.raises(PermissionDeniedError):
        bridge.list_users(auditor)

    # Guard cannot purge data or list users
    with pytest.raises(PermissionDeniedError):
        bridge.purge_system_data(guard, "events")
    with pytest.raises(PermissionDeniedError):
        bridge.list_users(guard)

    # Admin CAN list users, record manual entry, and export CSVs
    users = bridge.list_users(admin)
    assert len(users) >= 3

    res = bridge.manual_gate_entry(admin, "KA01GUI101", "IN", "PyQt6 test")
    assert res["status"] == "success"

    veh_csv = tmp_path / "vehicles.csv"
    ev_csv = tmp_path / "events.csv"
    v_count = bridge.export_vehicles_csv(str(veh_csv))
    e_count = bridge.export_events_csv(str(ev_csv))
    assert v_count >= 1 and veh_csv.exists()
    assert e_count >= 1 and ev_csv.exists()

    # Clean up test vehicle
    bridge.delete_vehicle(admin, "KA01GUI101")


def test_run_py_gates_startup_with_license_check_before_gui():
    """Verify run.py calls enforce_license_or_exit BEFORE launching Qt GUI."""
    import run as launcher

    with patch.object(launcher, "enforce_license_or_exit", side_effect=SystemExit(1)) as mock_lic, \
         patch.object(launcher, "_launch_native_gui") as mock_gui, \
         patch("sys.argv", ["run.py", "--check-license"]):
        with pytest.raises(SystemExit) as exc:
            launcher.main()
        assert exc.value.code == 1
        mock_lic.assert_called_once()
        mock_gui.assert_not_called()
