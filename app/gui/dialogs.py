"""
Native PyQt6 Dialogs for the ANPR Desktop Suite
===============================================
Provides native equivalents for every modal in the original web UI plus
a native OpenCV-backed MP4 clip player and vehicle history inspector:
  - LoginDialog          (RBAC Authentication + Demo Role quick-switch)
  - ManualGateDialog     (Manual IN/OUT gate trigger)
  - CreateUserDialog     (Admin RBAC user creation)
  - ImagePreviewDialog   (High-res plate/vehicle snapshot inspection)
  - ClipPlayerDialog     (Native MP4 clip playback + scrub + Save As)
  - VehicleHistoryDialog (Per-vehicle chronological movement history)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.gui.backend_bridge import AuthenticatedSession, BackendBridge
from app.gui.workers import bgr_to_qimage


class LoginDialog(QDialog):
    """Native RBAC Login dialog shown prior to opening the main window."""

    def __init__(self, bridge: BackendBridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.session: Optional[AuthenticatedSession] = None

        self.setWindowTitle("ANPR Gate Security — Sign In")
        self.setFixedSize(430, 420)
        self.setModal(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 24)
        root.setSpacing(14)

        # Header
        title = QLabel("ANPR Gate Security")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #ffffff;")
        subtitle = QLabel("Role-Based Access Control Login")
        subtitle.setObjectName("Muted")
        root.addWidget(title)
        root.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1f2937; max-height: 1px;")
        root.addWidget(sep)

        # Credentials form
        form = QVBoxLayout()
        form.setSpacing(8)

        lbl_u = QLabel("Username")
        lbl_u.setStyleSheet("font-size: 12px; color: #cbd5e1; font-weight: 600;")
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("admin / guard / auditor")
        self.username_edit.setMinimumHeight(36)

        lbl_p = QLabel("Password")
        lbl_p.setStyleSheet("font-size: 12px; color: #cbd5e1; font-weight: 600;")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setMinimumHeight(36)
        self.password_edit.returnPressed.connect(self._handle_login)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #f87171; font-size: 12px;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()

        self.btn_sign_in = QPushButton("Sign In")
        self.btn_sign_in.setObjectName("PrimaryBtn")
        self.btn_sign_in.setMinimumHeight(40)
        self.btn_sign_in.clicked.connect(self._handle_login)

        form.addWidget(lbl_u)
        form.addWidget(self.username_edit)
        form.addWidget(lbl_p)
        form.addWidget(self.password_edit)
        form.addWidget(self.error_label)
        form.addSpacing(4)
        form.addWidget(self.btn_sign_in)
        root.addLayout(form)

        # Quick-switch demo role buttons (matches web UI)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #1f2937; max-height: 1px;")
        root.addWidget(sep2)

        quick_lbl = QLabel("Quick Switch (Demo Roles):")
        quick_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        quick_lbl.setObjectName("Muted")
        root.addWidget(quick_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        b_admin = QPushButton("Admin")
        b_admin.setStyleSheet(
            "background-color: #3b0764; color: #e9d5ff; border: 1px solid #7e22ce; padding: 7px;"
        )
        b_admin.clicked.connect(lambda: self._quick_login("admin", "admin123"))

        b_guard = QPushButton("Guard")
        b_guard.setStyleSheet(
            "background-color: #172554; color: #bfdbfe; border: 1px solid #1d4ed8; padding: 7px;"
        )
        b_guard.clicked.connect(lambda: self._quick_login("guard", "guard123"))

        b_auditor = QPushButton("Auditor")
        b_auditor.setStyleSheet(
            "background-color: #064e3b; color: #a7f3d0; border: 1px solid #047857; padding: 7px;"
        )
        b_auditor.clicked.connect(lambda: self._quick_login("auditor", "auditor123"))

        btn_row.addWidget(b_admin)
        btn_row.addWidget(b_guard)
        btn_row.addWidget(b_auditor)
        root.addLayout(btn_row)

        self.username_edit.setFocus()

    def _quick_login(self, username: str, password: str) -> None:
        self.username_edit.setText(username)
        self.password_edit.setText(password)
        self._handle_login()

    def _handle_login(self) -> None:
        self.error_label.hide()
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            self.error_label.setText("Please enter both username and password.")
            self.error_label.show()
            return
        try:
            self.session = self.bridge.authenticate(username, password)
            self.accept()
        except Exception as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()


class ManualGateDialog(QDialog):
    """Manual Gate Entry modal (ADMIN & OPERATOR roles)."""

    def __init__(self, default_type: str = "IN", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Gate Entry")
        self.setFixedSize(400, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        hdr = QLabel("Record Manual Gate Movement")
        hdr.setObjectName("SectionTitle")
        layout.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(10)

        self.plate_edit = QLineEdit()
        self.plate_edit.setPlaceholderText("e.g. KA01AB1234")
        self.plate_edit.setFont(QFont("Consolas", 11))

        self.type_combo = QComboBox()
        self.type_combo.addItem("IN (Vehicle Entering)", "IN")
        self.type_combo.addItem("OUT (Vehicle Exiting)", "OUT")
        idx = self.type_combo.findData(default_type.upper())
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("e.g. Guest visitor, buzzer opened")

        form.addRow("Plate Number:", self.plate_edit)
        form.addRow("Movement Type:", self.type_combo)
        form.addRow("Officer Notes:", self.notes_edit)
        layout.addLayout(form)

        layout.addStretch()

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Record Event")
        btn_ok.setObjectName("PrimaryBtn")
        btn_ok.clicked.connect(self._validate_and_accept)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        layout.addLayout(btns)

    def _validate_and_accept(self) -> None:
        if not self.plate_edit.text().strip():
            QMessageBox.warning(self, "Missing Plate", "Please enter a vehicle registration plate number.")
            return
        self.accept()

    def values(self):
        return (
            self.plate_edit.text().strip().upper(),
            str(self.type_combo.currentData()),
            self.notes_edit.text().strip() or "Gate Operator Manual Trigger",
        )


class CreateUserDialog(QDialog):
    """Create New RBAC User modal (ADMIN only)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New RBAC User")
        self.setFixedSize(420, 320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        hdr = QLabel("Create RBAC User Account")
        hdr.setObjectName("SectionTitle")
        layout.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(10)

        self.full_name_edit = QLineEdit()
        self.username_edit = QLineEdit()
        self.email_edit = QLineEdit()
        self.role_combo = QComboBox()
        self.role_combo.addItem("OPERATOR (Gate Guard)", "OPERATOR")
        self.role_combo.addItem("AUDITOR (Read-only)", "AUDITOR")
        self.role_combo.addItem("ADMIN (Full Security Chief)", "ADMIN")
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)

        form.addRow("Full Name:", self.full_name_edit)
        form.addRow("Username:", self.username_edit)
        form.addRow("Email:", self.email_edit)
        form.addRow("Role:", self.role_combo)
        form.addRow("Password:", self.password_edit)
        layout.addLayout(form)

        layout.addStretch()

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_create = QPushButton("Create Account")
        btn_create.setObjectName("PurpleBtn")
        btn_create.clicked.connect(self._validate_and_accept)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_create)
        layout.addLayout(btns)

    def _validate_and_accept(self) -> None:
        if not all([
            self.full_name_edit.text().strip(),
            self.username_edit.text().strip(),
            self.email_edit.text().strip(),
            self.password_edit.text(),
        ]):
            QMessageBox.warning(self, "Incomplete Form", "Please fill in all fields.")
            return
        self.accept()

    def values(self):
        return {
            "full_name": self.full_name_edit.text().strip(),
            "username": self.username_edit.text().strip().lower(),
            "email": self.email_edit.text().strip().lower(),
            "role": str(self.role_combo.currentData()),
            "password": self.password_edit.text(),
        }


class ImagePreviewDialog(QDialog):
    """High-resolution preview dialog for captured plate crops and vehicle snapshots."""

    def __init__(self, image_path: Path, title: str = "Event Snapshot", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(title)
        self.resize(820, 580)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        top = QHBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setObjectName("SectionTitle")
        btn_save = QPushButton("Save Image Copy...")
        btn_save.clicked.connect(self._save_copy)
        top.addWidget(lbl_title)
        top.addStretch()
        top.addWidget(btn_save)
        layout.addLayout(top)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background-color: #000000; border-radius: 10px;")
        self._pixmap = QPixmap(str(image_path))
        layout.addWidget(self.img_label, 1)
        self._update_scaled()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._pixmap.isNull():
            self.img_label.setText(f"Could not load image:\n{self.image_path}")
            return
        scaled = self._pixmap.scaled(
            max(200, self.img_label.width() - 8),
            max(160, self.img_label.height() - 8),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.img_label.setPixmap(scaled)

    def _save_copy(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", self.image_path.name, "Images (*.jpg *.png);;All Files (*)"
        )
        if dest:
            shutil.copy2(self.image_path, dest)
            QMessageBox.information(self, "Saved", f"Saved image to:\n{dest}")


class ClipPlayerDialog(QDialog):
    """Native OpenCV-powered MP4 clip player with Play/Pause, Scrubber, and Save MP4."""

    def __init__(self, clip_path: Path, title: str = "Recorded Clip", parent=None):
        super().__init__(parent)
        self.clip_path = clip_path
        self.setWindowTitle(title)
        self.resize(860, 600)

        self._cap = cv2.VideoCapture(str(clip_path))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 20.0)
        if self._fps < 5.0 or self._fps > 60.0:
            self._fps = 20.0
        self._total_frames = max(1, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self._paused = False
        self._current_pixmap: Optional[QPixmap] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setObjectName("SectionTitle")
        btn_save = QPushButton("Save MP4 Copy...")
        btn_save.setObjectName("PrimaryBtn")
        btn_save.clicked.connect(self._save_copy)
        top.addWidget(lbl)
        top.addStretch()
        top.addWidget(btn_save)
        layout.addLayout(top)

        self.video_label = QLabel("Loading clip...")
        self.video_label.setObjectName("VideoSurface")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, 1)

        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("Pause")
        self.btn_play.clicked.connect(self._toggle_pause)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self._restart)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, self._total_frames - 1))
        self.slider.sliderMoved.connect(self._seek)

        self.lbl_frame = QLabel(f"0 / {self._total_frames}")
        self.lbl_frame.setStyleSheet("font-family: Consolas; font-size: 12px; color: #9ca3af;")

        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(btn_restart)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.lbl_frame)
        layout.addLayout(ctrl)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._timer.start(int(1000 / self._fps))
        self._next_frame()

    def _next_frame(self) -> None:
        if self._paused or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return
        pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(min(pos, self.slider.maximum()))
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"{pos} / {self._total_frames}")
        self._current_pixmap = QPixmap.fromImage(bgr_to_qimage(frame))
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        scaled = self._current_pixmap.scaled(
            max(200, self.video_label.width() - 6),
            max(140, self.video_label.height() - 6),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_pixmap()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.btn_play.setText("Play" if self._paused else "Pause")

    def _restart(self) -> None:
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._paused = False
            self.btn_play.setText("Pause")
            self._next_frame()

    def _seek(self, frame_idx: int) -> None:
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            was_paused = self._paused
            self._paused = False
            self._next_frame()
            self._paused = was_paused

    def _save_copy(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Video Clip", self.clip_path.name, "MP4 Video (*.mp4);;All Files (*)"
        )
        if dest:
            shutil.copy2(self.clip_path, dest)
            QMessageBox.information(self, "Saved", f"Saved video clip to:\n{dest}")

    def closeEvent(self, event) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


class VehicleHistoryDialog(QDialog):
    """Displays complete chronological IN/OUT history for one vehicle."""

    def __init__(self, plate: str, events: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Movement History — {plate}")
        self.resize(700, 440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hdr = QLabel(f"Chronological Gate History: {plate} ({len(events)} events)")
        hdr.setObjectName("SectionTitle")
        layout.addWidget(hdr)

        table = QTableWidget(len(events), 4)
        table.setHorizontalHeaderLabels(["Type", "Timestamp", "Confidence", "Raw OCR / Notes"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)

        for r, ev in enumerate(events):
            ts = ev["event_timestamp"]
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            table.setItem(r, 0, QTableWidgetItem(ev["event_type"]))
            table.setItem(r, 1, QTableWidgetItem(ts_str))
            table.setItem(r, 2, QTableWidgetItem(f"{ev['ocr_confidence'] * 100:.1f}%"))
            table.setItem(r, 3, QTableWidgetItem(ev["raw_ocr_text"] or "-"))

        layout.addWidget(table, 1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
