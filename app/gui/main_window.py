"""
Native PyQt6 Main Desktop Window for ANPR & Vehicle Gate Control
================================================================
Replaces the browser-based frontend with a fast, self-contained native
Windows desktop interface while preserving 100% of existing features,
RBAC roles (ADMIN / OPERATOR / AUDITOR), and ONNX Runtime DirectML/CPU
detection via in-process backend calls and background QThreads.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.config import settings
from app.gui.backend_bridge import AuthenticatedSession, BackendBridge, PermissionDeniedError
from app.gui.dialogs import (
    ClipPlayerDialog,
    CreateUserDialog,
    ImagePreviewDialog,
    LoginDialog,
    ManualGateDialog,
    VehicleHistoryDialog,
)
from app.gui.workers import CameraProbeWorker, EngineInitWorker, VideoFeedWorker

logger = logging.getLogger("anpr.gui.main_window")


def _fmt_dt(val) -> str:
    if val is None:
        return "-"
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)[:19].replace("T", " ")


def _badge_label(text: str, bg: str, fg: str, border: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(
        f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
        "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px; "
        "font-family: Consolas, monospace;"
    )
    return lbl


class MainWindow(QMainWindow):
    """Main native desktop window for the ANPR Gate Access System."""

    license_revoked_signal = pyqtSignal(str)

    TAB_SPECS = [
        ("live", "Live Gate Feed", "Automated Number Plate Recognition System", False),
        ("analytics", "Dashboard", "Gate Inflow & Outflow Analytics", False),
        ("vehicles", "Vehicle Registry", "Vehicle Registry & Status Lookup", False),
        ("events", "Event History", "Historical Movement Logs & Snapshots", False),
        ("clips", "Recorded Clips", "Stored Video Clips Archive (captures/events/)", False),
        ("users", "RBAC Users", "RBAC Access Control Directory (ADMIN)", True),
        ("system", "System & Settings", "Hardware Diagnostics, Settings & Cleanup (ADMIN)", True),
    ]

    def __init__(
        self,
        bridge: BackendBridge,
        session: AuthenticatedSession,
        auto_start_engine: bool = True,
        license_status: str = "active",
        parent=None,
    ):
        super().__init__(parent)
        self.bridge = bridge
        self.session = session
        self.license_status = license_status

        self._active_tab = "live"
        self._last_pixmap: Optional[QPixmap] = None
        self._camera_devices: List[Dict[str, str]] = []
        self._is_fullscreen_video = False
        self._engine_ready = False
        self._log_mode = "anpr"

        self.setWindowTitle("ANPR Gate Operations & Vehicle Access Control")
        self.resize(1380, 860)
        self.setMinimumSize(1024, 680)

        self.license_revoked_signal.connect(self._on_license_revoked)

        self._build_ui()
        self._apply_rbac_ui()

        # Debounce timer for vehicle search
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(300)
        self._search_timer.timeout.connect(self.load_vehicles)

        # Periodic UI telemetry / KPI refresh (every 2.5s)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2500)
        self._poll_timer.timeout.connect(self.refresh_live_telemetry)
        self._poll_timer.start()

        # Background workers (never freeze GUI)
        self._video_worker = VideoFeedWorker(self.bridge, parent=self)
        self._video_worker.frame_ready.connect(self._on_video_frame)
        self._video_worker.start()

        self._probe_worker: Optional[CameraProbeWorker] = None
        self._engine_worker: Optional[EngineInitWorker] = None

        if auto_start_engine:
            self._start_engine_async()
        else:
            self._engine_ready = True
            self.rescan_cameras()
            self.refresh_all_data()

    # ============================================================== UI BUILD
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 1. Left Navigation Sidebar
        self.sidebar = self._build_sidebar()
        root.addWidget(self.sidebar)

        # 2. Right Content Area (Top Bar + Stacked Tabs)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        self.topbar = self._build_topbar()
        right.addWidget(self.topbar)

        self.stack = QStackedWidget()
        self._tab_indices: Dict[str, int] = {}

        tab_builders = {
            "live": self._build_live_tab,
            "analytics": self._build_analytics_tab,
            "vehicles": self._build_vehicles_tab,
            "events": self._build_events_tab,
            "clips": self._build_clips_tab,
            "users": self._build_users_tab,
            "system": self._build_system_tab,
        }
        for key, _, _, _ in self.TAB_SPECS:
            widget = tab_builders[key]()
            idx = self.stack.addWidget(widget)
            self._tab_indices[key] = idx

        right.addWidget(self.stack, 1)
        root.addLayout(right, 1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_license_status = QLabel(f"License: {self.license_status.upper()}")
        self.lbl_license_status.setStyleSheet("color: #9ca3af; padding-right: 10px; font-family: Consolas;")
        self.status_bar.addPermanentWidget(self.lbl_license_status)
        self.show_toast("Initializing ONNX Runtime detectors...")

    def _build_sidebar(self) -> QFrame:
        sb = QFrame()
        sb.setObjectName("Sidebar")
        sb.setFixedWidth(235)
        layout = QVBoxLayout(sb)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(8)

        # Brand
        brand_title = QLabel("ANPR CONTROL")
        brand_title.setStyleSheet("font-size: 15px; font-weight: 800; color: #ffffff; letter-spacing: 1px;")
        brand_sub = QLabel("Native Windows Gate Suite")
        brand_sub.setStyleSheet("font-size: 11px; color: #64748b;")
        layout.addWidget(brand_title)
        layout.addWidget(brand_sub)
        layout.addSpacing(12)

        # Navigation buttons
        self._nav_buttons: Dict[str, QPushButton] = {}
        for key, label, _, _ in self.TAB_SPECS:
            btn = QPushButton(label)
            btn.setObjectName("NavBtn")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, k=key: self.switch_tab(k))
            layout.addWidget(btn)
            self._nav_buttons[key] = btn

        layout.addStretch(1)

        # User Profile + Switch User/Logout
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #1e293b; max-height: 1px;")
        layout.addWidget(sep)

        user_row = QHBoxLayout()
        user_info = QVBoxLayout()
        user_info.setSpacing(3)
        self.lbl_user_name = QLabel(self.session.full_name)
        self.lbl_user_name.setStyleSheet("font-size: 12px; font-weight: 700; color: #ffffff;")
        self.lbl_role_badge = _badge_label(self.session.role, "#172554", "#93c5fd", "#1d4ed8")
        self.lbl_role_badge.setFixedWidth(96)
        user_info.addWidget(self.lbl_user_name)
        user_info.addWidget(self.lbl_role_badge)

        btn_logout = QPushButton("Logout")
        btn_logout.setToolTip("Sign out and switch RBAC user")
        btn_logout.setFixedHeight(30)
        btn_logout.clicked.connect(self.logout_and_switch_user)

        user_row.addLayout(user_info, 1)
        user_row.addWidget(btn_logout)
        layout.addLayout(user_row)

        # Hardware status chip
        self.lbl_hw_chip = QLabel("● Detecting Hardware...")
        self.lbl_hw_chip.setStyleSheet(
            "background-color: #030712; border: 1px solid #1f2937; border-radius: 6px; "
            "padding: 6px 8px; color: #34d399; font-size: 11px; font-family: Consolas;"
        )
        self.lbl_hw_chip.setWordWrap(True)
        layout.addWidget(self.lbl_hw_chip)

        return sb

    def _build_topbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setMinimumHeight(64)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 10, 18, 10)
        layout.setSpacing(10)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        self.lbl_page_title = QLabel("Live Gate Feed")
        self.lbl_page_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #ffffff;")
        self.lbl_page_subtitle = QLabel("Automated Number Plate Recognition System")
        self.lbl_page_subtitle.setStyleSheet("font-size: 11px; color: #94a3b8;")
        title_col.addWidget(self.lbl_page_title)
        title_col.addWidget(self.lbl_page_subtitle)
        layout.addLayout(title_col)

        layout.addStretch(1)

        # Camera Selection Dropdown (live switching)
        lbl_cam = QLabel("Camera:")
        lbl_cam.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(210)
        self.camera_combo.setToolTip("Select camera device or video source live")
        self.camera_combo.activated.connect(self._on_camera_selected)

        self.btn_rescan_cam = QPushButton("Rescan")
        self.btn_rescan_cam.setToolTip("Probe available local cameras via OpenCV")
        self.btn_rescan_cam.clicked.connect(self.rescan_cameras)

        # Pipeline Start/Stop Toggle
        self.btn_pipeline_toggle = QPushButton("Stop Pipeline")
        self.btn_pipeline_toggle.setObjectName("WarningBtn")
        self.btn_pipeline_toggle.setToolTip("Start or pause live ANPR detection & capture")
        self.btn_pipeline_toggle.clicked.connect(self.toggle_pipeline)

        # Quick IN / OUT + Manual Entry (ADMIN & OPERATOR only)
        self.btn_quick_in = QPushButton("Quick IN")
        self.btn_quick_in.setObjectName("SuccessBtn")
        self.btn_quick_in.clicked.connect(lambda: self.quick_gate_action("IN"))

        self.btn_quick_out = QPushButton("Quick OUT")
        self.btn_quick_out.setObjectName("WarningBtn")
        self.btn_quick_out.clicked.connect(lambda: self.quick_gate_action("OUT"))

        self.btn_manual_entry = QPushButton("Manual Entry")
        self.btn_manual_entry.setObjectName("PrimaryBtn")
        self.btn_manual_entry.clicked.connect(self.open_manual_gate_dialog)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_all_data)

        layout.addWidget(lbl_cam)
        layout.addWidget(self.camera_combo)
        layout.addWidget(self.btn_rescan_cam)
        layout.addWidget(self.btn_pipeline_toggle)
        layout.addWidget(self.btn_quick_in)
        layout.addWidget(self.btn_quick_out)
        layout.addWidget(self.btn_manual_entry)
        layout.addWidget(self.btn_refresh)
        return bar

    # ------------------------------------------------------ Tab 1: Live Feed
    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(16)

        # Left 2/3: Live Video Panel
        video_col = QVBoxLayout()
        video_col.setSpacing(10)

        # Status badge strip above video
        badge_row = QHBoxLayout()
        badge_row.setSpacing(8)
        self.badge_live = _badge_label("● STARTING", "#7f1d1d", "#fecaca", "#ef4444")
        self.badge_fps = _badge_label("-- FPS", "#064e3b", "#34d399", "#059669")
        self.badge_rec = _badge_label("MONITORING", "#1f2937", "#d1d5db", "#374151")

        self.btn_reconnect_live = QPushButton("Reconnect Camera")
        self.btn_reconnect_live.clicked.connect(self.reconnect_camera)
        self.btn_fullscreen = QPushButton("Fullscreen")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen_video)

        badge_row.addWidget(self.badge_live)
        badge_row.addWidget(self.badge_fps)
        badge_row.addWidget(self.badge_rec)
        badge_row.addStretch(1)
        badge_row.addWidget(self.btn_reconnect_live)
        badge_row.addWidget(self.btn_fullscreen)
        video_col.addLayout(badge_row)

        self.video_label = QLabel("Starting ONNX Runtime DirectML / CPU pipeline...\nWaiting for camera frames...")
        self.video_label.setObjectName("VideoSurface")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(560, 360)
        video_col.addWidget(self.video_label, 1)

        layout.addLayout(video_col, 2)

        # Right 1/3: Telemetry Card + Recent Gate Events
        side_col = QVBoxLayout()
        side_col.setSpacing(14)

        # Pipeline Telemetry Card
        tel_card = QFrame()
        tel_card.setObjectName("Card")
        tel_layout = QVBoxLayout(tel_card)
        tel_layout.setContentsMargins(14, 14, 14, 14)
        tel_layout.setSpacing(10)

        tel_hdr = QLabel("PIPELINE TELEMETRY")
        tel_hdr.setStyleSheet("font-size: 11px; font-weight: 700; color: #94a3b8; letter-spacing: 1px;")
        tel_layout.addWidget(tel_hdr)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.lbl_tel_infer = self._make_metric_tile(grid, 0, 0, "YOLO Inference", "-- ms", "#ffffff")
        self.lbl_tel_ocr = self._make_metric_tile(grid, 0, 1, "Clip OCR Time", "-- ms", "#ffffff")
        self.lbl_tel_clips = self._make_metric_tile(grid, 1, 0, "Clips Recorded", "0 / 0", "#fbbf24")
        self.lbl_tel_device = self._make_metric_tile(grid, 1, 1, "Compute Device", "--", "#34d399")
        tel_layout.addLayout(grid)

        clip_box = QFrame()
        clip_box.setObjectName("SubCard")
        cb_layout = QHBoxLayout(clip_box)
        cb_layout.setContentsMargins(10, 8, 10, 8)
        cb_left = QVBoxLayout()
        cb_left.setSpacing(2)
        cb_l1 = QLabel("Last Clip Result")
        cb_l1.setStyleSheet("font-size: 10px; color: #9ca3af;")
        self.lbl_tel_clip_result = QLabel("IDLE")
        self.lbl_tel_clip_result.setStyleSheet("font-family: Consolas; font-size: 12px; color: #e5e7eb;")
        cb_left.addWidget(cb_l1)
        cb_left.addWidget(self.lbl_tel_clip_result)
        self.lbl_tel_vram = QLabel("Host RAM")
        self.lbl_tel_vram.setStyleSheet("font-family: Consolas; font-size: 11px; color: #60a5fa;")
        cb_layout.addLayout(cb_left, 1)
        cb_layout.addWidget(self.lbl_tel_vram)
        tel_layout.addWidget(clip_box)

        side_col.addWidget(tel_card)

        # Recent Gate Events Card
        ev_card = QFrame()
        ev_card.setObjectName("Card")
        ev_layout = QVBoxLayout(ev_card)
        ev_layout.setContentsMargins(14, 14, 14, 14)
        ev_layout.setSpacing(10)

        ev_hdr_row = QHBoxLayout()
        ev_hdr = QLabel("RECENT GATE EVENTS")
        ev_hdr.setStyleSheet("font-size: 11px; font-weight: 700; color: #94a3b8; letter-spacing: 1px;")
        self.lbl_live_event_count = _badge_label("0 today", "#1f2937", "#d1d5db", "#374151")
        ev_hdr_row.addWidget(ev_hdr)
        ev_hdr_row.addStretch(1)
        ev_hdr_row.addWidget(self.lbl_live_event_count)
        ev_layout.addLayout(ev_hdr_row)

        self.recent_events_scroll = QScrollArea()
        self.recent_events_scroll.setWidgetResizable(True)
        self.recent_events_container = QWidget()
        self.recent_events_layout = QVBoxLayout(self.recent_events_container)
        self.recent_events_layout.setContentsMargins(0, 0, 4, 0)
        self.recent_events_layout.setSpacing(8)
        self.recent_events_layout.addStretch(1)
        self.recent_events_scroll.setWidget(self.recent_events_container)
        ev_layout.addWidget(self.recent_events_scroll, 1)

        side_col.addWidget(ev_card, 1)
        layout.addLayout(side_col, 1)
        return page

    def _make_metric_tile(
        self, grid: QGridLayout, row: int, col: int, title: str, value: str, color: str
    ) -> QLabel:
        box = QFrame()
        box.setObjectName("SubCard")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("font-size: 10px; color: #9ca3af;")
        val = QLabel(value)
        val.setStyleSheet(f"font-family: Consolas; font-size: 14px; font-weight: 700; color: {color};")
        v.addWidget(t)
        v.addWidget(val)
        grid.addWidget(box, row, col)
        return val

    # ------------------------------------------------------ Tab 2: Dashboard
    def _build_analytics_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(16)

        # 4 KPI Cards
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(14)
        self.lbl_kpi_total = self._make_kpi_card(kpi_row, "KpiBlue", "REGISTERED VEHICLES", "Total in database", "#ffffff")
        self.lbl_kpi_inside = self._make_kpi_card(kpi_row, "KpiEmerald", "VEHICLES INSIDE", "Active on premises", "#34d399")
        self.lbl_kpi_entries = self._make_kpi_card(kpi_row, "KpiIndigo", "TODAY'S INFLOW", "Entries recorded", "#ffffff")
        self.lbl_kpi_exits = self._make_kpi_card(kpi_row, "KpiAmber", "TODAY'S OUTFLOW", "Exits recorded", "#fbbf24")
        layout.addLayout(kpi_row)

        # Today's Real-time Activity Table
        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)

        hdr = QLabel("Today's Real-time Movement Activity")
        hdr.setObjectName("SectionTitle")
        cl.addWidget(hdr)

        self.tbl_dashboard = QTableWidget(0, 5)
        self.tbl_dashboard.setHorizontalHeaderLabels(["Type", "Plate Number", "Time", "Confidence", "Plate Crop"])
        self.tbl_dashboard.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_dashboard.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_dashboard.setAlternatingRowColors(True)
        self.tbl_dashboard.verticalHeader().setDefaultSectionSize(42)
        cl.addWidget(self.tbl_dashboard, 1)

        layout.addWidget(card, 1)
        return page

    def _make_kpi_card(self, parent_layout: QHBoxLayout, obj_name: str, title: str, subtitle: str, color: str) -> QLabel:
        card = QFrame()
        card.setObjectName(obj_name)
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(4)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 11px; font-weight: 600; color: #9ca3af;")
        lbl_v = QLabel("--")
        lbl_v.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {color};")
        lbl_s = QLabel(subtitle)
        lbl_s.setStyleSheet("font-size: 11px; color: #6b7280;")
        v.addWidget(lbl_t)
        v.addWidget(lbl_v)
        v.addWidget(lbl_s)
        parent_layout.addWidget(card, 1)
        return lbl_v

    # ----------------------------------------------- Tab 3: Vehicle Registry
    def _build_vehicles_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.vehicle_search_edit = QLineEdit()
        self.vehicle_search_edit.setPlaceholderText("Search Plate (e.g. KA01AB1234)...")
        self.vehicle_search_edit.textChanged.connect(lambda: self._search_timer.start())

        self.vehicle_status_combo = QComboBox()
        self.vehicle_status_combo.addItem("All Statuses", "")
        self.vehicle_status_combo.addItem("Currently IN", "IN")
        self.vehicle_status_combo.addItem("Currently OUT", "OUT")
        self.vehicle_status_combo.currentIndexChanged.connect(self.load_vehicles)

        btn_export_veh = QPushButton("Export CSV")
        btn_export_veh.clicked.connect(self.export_vehicles_csv)

        self.btn_purge_vehicles = QPushButton("Delete All")
        self.btn_purge_vehicles.setObjectName("DangerBtn")
        self.btn_purge_vehicles.clicked.connect(lambda: self.purge_system_data("vehicles"))

        bar.addWidget(self.vehicle_search_edit, 1)
        bar.addWidget(self.vehicle_status_combo)
        bar.addWidget(btn_export_veh)
        bar.addWidget(self.btn_purge_vehicles)
        layout.addLayout(bar)

        self.tbl_vehicles = QTableWidget(0, 6)
        self.tbl_vehicles.setHorizontalHeaderLabels(
            ["Registration", "Current Status", "Entries", "Exits", "Last Seen", "Actions"]
        )
        self.tbl_vehicles.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_vehicles.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_vehicles.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_vehicles.setAlternatingRowColors(True)
        self.tbl_vehicles.verticalHeader().setDefaultSectionSize(44)
        layout.addWidget(self.tbl_vehicles, 1)
        return page

    # -------------------------------------------------- Tab 4: Event History
    def _build_events_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        bar = QHBoxLayout()
        bar.setSpacing(10)
        self.event_plate_edit = QLineEdit()
        self.event_plate_edit.setPlaceholderText("Filter Plate...")
        self.event_plate_edit.returnPressed.connect(self.load_events)

        btn_filter = QPushButton("Filter")
        btn_filter.setObjectName("PrimaryBtn")
        btn_filter.clicked.connect(self.load_events)

        self.event_type_combo = QComboBox()
        self.event_type_combo.addItem("All Events", "")
        self.event_type_combo.addItem("Entries Only (IN)", "IN")
        self.event_type_combo.addItem("Exits Only (OUT)", "OUT")
        self.event_type_combo.currentIndexChanged.connect(self.load_events)

        btn_export_ev = QPushButton("Export CSV")
        btn_export_ev.clicked.connect(self.export_events_csv)

        self.btn_purge_events = QPushButton("Clear Events")
        self.btn_purge_events.setObjectName("DangerBtn")
        self.btn_purge_events.clicked.connect(lambda: self.purge_system_data("events"))

        bar.addWidget(self.event_plate_edit, 1)
        bar.addWidget(btn_filter)
        bar.addWidget(self.event_type_combo)
        bar.addWidget(btn_export_ev)
        bar.addWidget(self.btn_purge_events)
        layout.addLayout(bar)

        self.tbl_events = QTableWidget(0, 6)
        self.tbl_events.setHorizontalHeaderLabels(
            ["Type", "Plate Number", "Timestamp", "Confidence", "Raw OCR", "Media & Actions"]
        )
        self.tbl_events.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_events.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_events.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_events.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_events.setAlternatingRowColors(True)
        self.tbl_events.verticalHeader().setDefaultSectionSize(46)
        layout.addWidget(self.tbl_events, 1)
        return page

    # ------------------------------------------------- Tab 5: Recorded Clips
    def _build_clips_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        hdr = QHBoxLayout()
        t_col = QVBoxLayout()
        t1 = QLabel("Stored Video Clips")
        t1.setObjectName("SectionTitle")
        t2 = QLabel("All plate-triggered recordings saved in captures/events/")
        t2.setObjectName("Muted")
        t_col.addWidget(t1)
        t_col.addWidget(t2)

        btn_ref = QPushButton("Refresh")
        btn_ref.clicked.connect(self.load_clips)
        self.btn_purge_clips = QPushButton("Delete All Clips")
        self.btn_purge_clips.setObjectName("DangerBtn")
        self.btn_purge_clips.clicked.connect(lambda: self.purge_system_data("captures"))

        hdr.addLayout(t_col)
        hdr.addStretch(1)
        hdr.addWidget(btn_ref)
        hdr.addWidget(self.btn_purge_clips)
        layout.addLayout(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.clips_grid_widget = QWidget()
        self.clips_grid = QGridLayout(self.clips_grid_widget)
        self.clips_grid.setContentsMargins(0, 0, 6, 0)
        self.clips_grid.setSpacing(14)
        scroll.setWidget(self.clips_grid_widget)
        layout.addWidget(scroll, 1)
        return page

    # ----------------------------------------------------- Tab 6: RBAC Users
    def _build_users_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        hdr = QHBoxLayout()
        t_col = QVBoxLayout()
        t1 = QLabel("RBAC User Directory")
        t1.setObjectName("SectionTitle")
        t2 = QLabel("Manage security officers, operators, and compliance auditors")
        t2.setObjectName("Muted")
        t_col.addWidget(t1)
        t_col.addWidget(t2)

        btn_add = QPushButton("Add User")
        btn_add.setObjectName("PurpleBtn")
        btn_add.clicked.connect(self.open_create_user_dialog)

        hdr.addLayout(t_col)
        hdr.addStretch(1)
        hdr.addWidget(btn_add)
        layout.addLayout(hdr)

        self.tbl_users = QTableWidget(0, 5)
        self.tbl_users.setHorizontalHeaderLabels(["Full Name", "Username", "Email", "Role", "Status"])
        self.tbl_users.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_users.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_users.setAlternatingRowColors(True)
        self.tbl_users.verticalHeader().setDefaultSectionSize(40)
        layout.addWidget(self.tbl_users, 1)
        return page

    # -------------------------------------------- Tab 7: System & Settings
    def _build_system_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(14)

        # Row 1: Hardware + Notifications cards
        row1 = QHBoxLayout()
        row1.setSpacing(14)

        hw_card = QFrame()
        hw_card.setObjectName("Card")
        hw_l = QVBoxLayout(hw_card)
        hw_l.setContentsMargins(16, 14, 16, 14)
        hw_l.setSpacing(8)
        hw_hdr = QLabel("Hardware Acceleration & Camera")
        hw_hdr.setObjectName("SectionTitle")
        self.lbl_sys_hw = QLabel("Loading hardware telemetry...")
        self.lbl_sys_hw.setWordWrap(True)
        self.lbl_sys_hw.setStyleSheet("font-size: 12px; color: #d1d5db; line-height: 1.5;")
        btn_reconn = QPushButton("Reconnect Camera")
        btn_reconn.clicked.connect(self.reconnect_camera)
        hw_l.addWidget(hw_hdr)
        hw_l.addWidget(self.lbl_sys_hw, 1)
        hw_l.addWidget(btn_reconn, alignment=Qt.AlignmentFlag.AlignLeft)

        notif_card = QFrame()
        notif_card.setObjectName("Card")
        nf_l = QVBoxLayout(notif_card)
        nf_l.setContentsMargins(16, 14, 16, 14)
        nf_l.setSpacing(8)
        nf_hdr = QLabel("Mobile Push (ntfy.sh)")
        nf_hdr.setObjectName("SectionTitle")
        self.lbl_sys_notif = QLabel("Loading notification telemetry...")
        self.lbl_sys_notif.setWordWrap(True)
        self.lbl_sys_notif.setStyleSheet("font-size: 12px; color: #d1d5db;")
        nf_l.addWidget(nf_hdr)
        nf_l.addWidget(self.lbl_sys_notif, 1)

        row1.addWidget(hw_card, 1)
        row1.addWidget(notif_card, 1)
        layout.addLayout(row1)

        # Row 2: Live Pipeline & Notification Settings
        set_card = QFrame()
        set_card.setObjectName("Card")
        sl = QVBoxLayout(set_card)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(10)
        set_hdr = QLabel("Live Pipeline & Notification Settings")
        set_hdr.setObjectName("SectionTitle")
        sl.addWidget(set_hdr)

        form_grid = QGridLayout()
        form_grid.setSpacing(10)

        self.spin_det_conf = QDoubleSpinBox()
        self.spin_det_conf.setRange(0.05, 0.95)
        self.spin_det_conf.setSingleStep(0.05)
        self.spin_det_conf.setValue(float(settings.DETECTION_CONFIDENCE))

        self.spin_cooldown = QSpinBox()
        self.spin_cooldown.setRange(0, 3600)
        self.spin_cooldown.setValue(int(settings.RECOGNITION_COOLDOWN))
        self.spin_cooldown.setSuffix(" s")

        self.combo_strategy = QComboBox()
        self.combo_strategy.addItem("Toggle (IN <-> OUT)", "toggle")
        self.combo_strategy.addItem("Virtual Line Crossing", "line_crossing")
        s_idx = self.combo_strategy.findData(settings.EVENT_STRATEGY)
        if s_idx >= 0:
            self.combo_strategy.setCurrentIndex(s_idx)

        self.edit_ntfy_topic = QLineEdit(settings.NTFY_TOPIC)
        self.chk_clip_mode = QCheckBox("Event-Triggered Video Clip Recording")
        self.chk_clip_mode.setChecked(bool(settings.CLIP_RECORDING_ENABLED))
        self.chk_notif_enabled = QCheckBox("Enable ntfy.sh Push Notifications")
        self.chk_notif_enabled.setChecked(bool(settings.NOTIFICATIONS_ENABLED))

        form_grid.addWidget(QLabel("Plate Detection Confidence:"), 0, 0)
        form_grid.addWidget(self.spin_det_conf, 0, 1)
        form_grid.addWidget(QLabel("Recognition Cooldown:"), 0, 2)
        form_grid.addWidget(self.spin_cooldown, 0, 3)
        form_grid.addWidget(QLabel("Event Strategy:"), 1, 0)
        form_grid.addWidget(self.combo_strategy, 1, 1)
        form_grid.addWidget(QLabel("ntfy.sh Topic:"), 1, 2)
        form_grid.addWidget(self.edit_ntfy_topic, 1, 3)
        form_grid.addWidget(self.chk_clip_mode, 2, 0, 1, 2)
        form_grid.addWidget(self.chk_notif_enabled, 2, 2, 1, 2)
        sl.addLayout(form_grid)

        btn_apply_settings = QPushButton("Apply Settings")
        btn_apply_settings.setObjectName("PrimaryBtn")
        btn_apply_settings.clicked.connect(self.apply_runtime_settings)
        sl.addWidget(btn_apply_settings, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(set_card)

        # Row 3: Data Management & Cleanup (ADMIN only)
        purge_card = QFrame()
        purge_card.setObjectName("CardDanger")
        pl = QVBoxLayout(purge_card)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.setSpacing(10)

        p_hdr_row = QHBoxLayout()
        p_hdr = QLabel("Data Management & Cleanup")
        p_hdr.setObjectName("SectionTitle")
        p_hdr_row.addWidget(p_hdr)
        p_hdr_row.addStretch(1)
        p_hdr_row.addWidget(_badge_label("ADMIN ONLY", "#450a0a", "#fca5a5", "#991b1b"))
        pl.addLayout(p_hdr_row)

        p_desc = QLabel(
            "Delete stored vehicle logs, movement events, recorded video clips, and captured snapshots. "
            "RBAC user accounts and camera definitions are always preserved."
        )
        p_desc.setObjectName("Muted")
        p_desc.setWordWrap(True)
        pl.addWidget(p_desc)

        p_btns = QHBoxLayout()
        p_btns.setSpacing(10)
        b_ev = QPushButton("Delete All Events & Media")
        b_ev.clicked.connect(lambda: self.purge_system_data("events"))
        b_cap = QPushButton("Clear Captured Files Only")
        b_cap.clicked.connect(lambda: self.purge_system_data("captures"))
        b_log = QPushButton("Clear Application Logs")
        b_log.clicked.connect(lambda: self.purge_system_data("logs"))
        b_all = QPushButton("Reset All Data (Full Purge)")
        b_all.setObjectName("DangerBtn")
        b_all.clicked.connect(lambda: self.purge_system_data("all"))
        for b in (b_ev, b_cap, b_log, b_all):
            p_btns.addWidget(b)
        pl.addLayout(p_btns)
        layout.addWidget(purge_card)

        # Row 4: Live Application & License Audit Logs
        log_card = QFrame()
        log_card.setObjectName("Card")
        ll = QVBoxLayout(log_card)
        ll.setContentsMargins(16, 14, 16, 14)
        ll.setSpacing(8)

        lhdr = QHBoxLayout()
        ltitle = QLabel("Live Application & License Audit Logs")
        ltitle.setObjectName("SectionTitle")
        btn_app_log = QPushButton("anpr.log")
        btn_app_log.clicked.connect(lambda: self._switch_log_view("anpr"))
        btn_lic_log = QPushButton("license.log")
        btn_lic_log.clicked.connect(lambda: self._switch_log_view("license"))
        btn_ref_log = QPushButton("Refresh Logs")
        btn_ref_log.clicked.connect(self.load_logs)
        lhdr.addWidget(ltitle)
        lhdr.addStretch(1)
        lhdr.addWidget(btn_app_log)
        lhdr.addWidget(btn_lic_log)
        lhdr.addWidget(btn_ref_log)
        ll.addLayout(lhdr)

        self.txt_logs = QPlainTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setMinimumHeight(210)
        ll.addWidget(self.txt_logs)
        layout.addWidget(log_card)

        scroll.setWidget(container)
        return scroll

    # ========================================================== ENGINE & CAM
    def _start_engine_async(self) -> None:
        self._engine_worker = EngineInitWorker(self.bridge, auto_start=True, parent=self)
        self._engine_worker.ready.connect(self._on_engine_ready)
        self._engine_worker.failed.connect(self._on_engine_failed)
        self._engine_worker.start()

    def _on_engine_ready(self) -> None:
        self._engine_ready = True
        self._update_pipeline_button()
        self.rescan_cameras()
        self.refresh_all_data()
        self.show_toast("ANPR ONNX Runtime detection pipeline active.")

    def _on_engine_failed(self, err: str) -> None:
        self.video_label.setText(f"Engine initialization error:\n{err}")
        self.show_toast(f"Engine error: {err}", is_error=True)
        QMessageBox.critical(self, "ANPR Engine Initialization Failed", err)

    def rescan_cameras(self) -> None:
        if self._probe_worker is not None and self._probe_worker.isRunning():
            return
        self.btn_rescan_cam.setEnabled(False)
        self.btn_rescan_cam.setText("Scanning...")
        self._probe_worker = CameraProbeWorker(self.bridge, max_index=6, parent=self)
        self._probe_worker.devices_ready.connect(self._on_cameras_enumerated)
        self._probe_worker.start()

    def _on_cameras_enumerated(self, devices: List[Dict[str, str]]) -> None:
        self.btn_rescan_cam.setEnabled(True)
        self.btn_rescan_cam.setText("Rescan")
        self._camera_devices = devices

        current_src = (
            str(self.bridge.camera_service.camera_manager.source)
            if self.bridge.camera_service is not None
            else str(settings.CAMERA_SOURCE)
        )

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        found_current = False
        for d in devices:
            self.camera_combo.addItem(d["label"], d["source"])
            if str(d["source"]) == current_src:
                found_current = True

        if not found_current and current_src:
            self.camera_combo.addItem(f"Custom Source ({Path(current_src).name})", current_src)

        self.camera_combo.addItem("Open Video File...", "__OPEN_FILE__")

        idx = self.camera_combo.findData(current_src)
        if idx >= 0:
            self.camera_combo.setCurrentIndex(idx)
        self.camera_combo.blockSignals(False)

    def _on_camera_selected(self, index: int) -> None:
        source = self.camera_combo.itemData(index)
        if not source:
            return
        if source == "__OPEN_FILE__":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
            )
            if not path:
                self._on_cameras_enumerated(self._camera_devices)
                return
            source = path

        ok = self.bridge.switch_camera(str(source))
        self.show_toast(
            f"Switched camera source to '{source}'" if ok else f"Switching to '{source}' (reconnecting...)",
            is_error=not ok,
        )
        self.rescan_cameras()
        self.refresh_live_telemetry()

    def toggle_pipeline(self) -> None:
        if self.bridge.is_pipeline_running():
            self.bridge.stop_pipeline()
            self.video_label.setText("Pipeline Paused.\nClick 'Start Pipeline' to resume live ANPR detection.")
            self.show_toast("ANPR pipeline paused.")
        else:
            self.bridge.start_pipeline()
            self.show_toast("ANPR pipeline started.")
        self._update_pipeline_button()
        self.refresh_live_telemetry()

    def _update_pipeline_button(self) -> None:
        running = self.bridge.is_pipeline_running()
        if running:
            self.btn_pipeline_toggle.setText("Stop Pipeline")
            self.btn_pipeline_toggle.setObjectName("WarningBtn")
            self.badge_live.setText("● LIVE")
            self.badge_live.setStyleSheet(
                "background-color: #991b1b; color: #ffffff; border: 1px solid #ef4444; "
                "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
            )
        else:
            self.btn_pipeline_toggle.setText("Start Pipeline")
            self.btn_pipeline_toggle.setObjectName("SuccessBtn")
            self.badge_live.setText("● PAUSED")
            self.badge_live.setStyleSheet(
                "background-color: #1f2937; color: #9ca3af; border: 1px solid #374151; "
                "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
            )
        self.btn_pipeline_toggle.style().unpolish(self.btn_pipeline_toggle)
        self.btn_pipeline_toggle.style().polish(self.btn_pipeline_toggle)

    def _on_video_frame(self, qimg: QImage) -> None:
        self._last_pixmap = QPixmap.fromImage(qimg)
        self._render_video_pixmap()

    def _render_video_pixmap(self) -> None:
        if self._last_pixmap is None or self._last_pixmap.isNull():
            return
        target = self.video_label
        scaled = self._last_pixmap.scaled(
            max(200, target.width() - 4),
            max(140, target.height() - 4),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        target.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_video_pixmap()

    def toggle_fullscreen_video(self) -> None:
        self._is_fullscreen_video = not self._is_fullscreen_video
        if self._is_fullscreen_video:
            self.sidebar.hide()
            self.showFullScreen()
            self.btn_fullscreen.setText("Exit Fullscreen")
        else:
            self.showNormal()
            self.sidebar.show()
            self.btn_fullscreen.setText("Fullscreen")

    # ========================================================== RBAC & NAV
    def _apply_rbac_ui(self) -> None:
        role = self.session.role.upper()
        is_admin = role == "ADMIN"
        can_operate = role in ("ADMIN", "OPERATOR")

        self.lbl_user_name.setText(self.session.full_name)
        self.lbl_role_badge.setText(role)
        if is_admin:
            self.lbl_role_badge.setStyleSheet(
                "background-color: #3b0764; color: #e9d5ff; border: 1px solid #7e22ce; "
                "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
            )
        elif role == "OPERATOR":
            self.lbl_role_badge.setStyleSheet(
                "background-color: #172554; color: #93c5fd; border: 1px solid #1d4ed8; "
                "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
            )
        else:
            self.lbl_role_badge.setStyleSheet(
                "background-color: #064e3b; color: #6ee7b7; border: 1px solid #047857; "
                "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
            )

        # Gate admin-only navigation tabs
        for key, _, _, admin_only in self.TAB_SPECS:
            self._nav_buttons[key].setVisible((not admin_only) or is_admin)

        # Gate operator controls (hidden for AUDITOR)
        self.btn_manual_entry.setVisible(can_operate)
        self.btn_quick_in.setVisible(can_operate)
        self.btn_quick_out.setVisible(can_operate)
        self.btn_pipeline_toggle.setVisible(can_operate)
        self.camera_combo.setEnabled(can_operate)
        self.btn_rescan_cam.setVisible(can_operate)
        self.btn_reconnect_live.setVisible(can_operate)

        # Gate admin-only destructive purge buttons
        self.btn_purge_vehicles.setVisible(is_admin)
        self.btn_purge_events.setVisible(is_admin)
        self.btn_purge_clips.setVisible(is_admin)

        # If current tab is now forbidden, jump back to Live Gate Feed
        if not is_admin and self._active_tab in ("users", "system"):
            self.switch_tab("live")
        else:
            self.switch_tab(self._active_tab)

    def switch_tab(self, tab_key: str) -> None:
        if tab_key in ("users", "system") and self.session.role != "ADMIN":
            return
        self._active_tab = tab_key
        self.stack.setCurrentIndex(self._tab_indices[tab_key])

        for key, label, subtitle, _ in self.TAB_SPECS:
            btn = self._nav_buttons[key]
            btn.setProperty("active", key == tab_key)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            if key == tab_key:
                self.lbl_page_title.setText(label)
                self.lbl_page_subtitle.setText(subtitle)

        if tab_key == "vehicles":
            self.load_vehicles()
        elif tab_key == "events":
            self.load_events()
        elif tab_key == "clips":
            self.load_clips()
        elif tab_key == "users":
            self.load_users()
        elif tab_key == "system":
            self.load_system_details()
        else:
            self.refresh_live_telemetry()

    def logout_and_switch_user(self) -> None:
        dlg = LoginDialog(self.bridge, parent=self)
        if dlg.exec() == LoginDialog.DialogCode.Accepted and dlg.session is not None:
            self.session = dlg.session
            self._apply_rbac_ui()
            self.refresh_all_data()
            self.show_toast(f"Signed in as {self.session.full_name} ({self.session.role})")

    # ======================================================= DATA REFRESHERS
    def refresh_all_data(self) -> None:
        self.refresh_live_telemetry()
        if self._active_tab == "vehicles":
            self.load_vehicles()
        elif self._active_tab == "events":
            self.load_events()
        elif self._active_tab == "clips":
            self.load_clips()
        elif self._active_tab == "users":
            self.load_users()
        elif self._active_tab == "system":
            self.load_system_details()

    def refresh_live_telemetry(self) -> None:
        try:
            summary = self.bridge.get_dashboard_summary()
            self.lbl_kpi_total.setText(str(summary["total_registered_vehicles"]))
            self.lbl_kpi_inside.setText(str(summary["vehicles_currently_inside"]))
            self.lbl_kpi_entries.setText(str(summary["today_entries"]))
            self.lbl_kpi_exits.setText(str(summary["today_exits"]))
            total_today = int(summary["today_entries"]) + int(summary["today_exits"])
            self.lbl_live_event_count.setText(f"{total_today} today")

            recent = summary.get("recent_events") or []
            self._render_recent_events_list(recent)
            self._render_dashboard_table(recent)
        except Exception as exc:
            logger.debug(f"Dashboard refresh error: {exc}")

        try:
            st = self.bridge.get_system_status()
            cam = st.get("camera", {})
            met = st.get("pipeline_metrics", {})
            diag = st.get("hardware_diagnostics", {})
            clip = met.get("clip_recorder", {})

            fps = cam.get("fps_actual") or met.get("processing_fps") or 0
            self.badge_fps.setText(f"{fps} FPS")
            self.lbl_tel_infer.setText(f"{float(met.get('inference_time_ms', 0.0)):.1f} ms")
            self.lbl_tel_ocr.setText(f"{float(met.get('ocr_time_ms', 0.0)):.0f} ms")
            self.lbl_tel_clips.setText(f"{clip.get('clips_confirmed', 0)} / {clip.get('clips_recorded', 0)}")
            active_dev = str(diag.get("active_device", "cpu")).upper()
            self.lbl_tel_device.setText(active_dev)
            self.lbl_tel_clip_result.setText(str(clip.get("last_clip_result") or "IDLE"))

            if diag.get("total_vram_mb"):
                self.lbl_tel_vram.setText(
                    f"{round(diag['allocated_vram_mb'])} / {round(diag['total_vram_mb'])} MB"
                )
            else:
                self.lbl_tel_vram.setText("Host RAM")

            self.lbl_hw_chip.setText(f"● {active_dev} • {diag.get('device_name', 'CPU')}")

            if clip.get("is_recording"):
                self.badge_rec.setText(f"● REC CLIP {clip.get('recording_elapsed_s', 0)}s")
                self.badge_rec.setStyleSheet(
                    "background-color: #450a0a; color: #fca5a5; border: 1px solid #ef4444; "
                    "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
                )
            elif clip.get("is_processing_clip"):
                self.badge_rec.setText("PROCESSING CLIP...")
                self.badge_rec.setStyleSheet(
                    "background-color: #451a03; color: #fcd34d; border: 1px solid #f59e0b; "
                    "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
                )
            else:
                self.badge_rec.setText(f"CLIP MODE • {clip.get('clips_pending', 0)} QUEUED")
                self.badge_rec.setStyleSheet(
                    "background-color: #1f2937; color: #d1d5db; border: 1px solid #374151; "
                    "border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: 11px;"
                )

            if self._engine_ready:
                self._update_pipeline_button()
        except Exception as exc:
            logger.debug(f"Telemetry refresh error: {exc}")

    def _render_recent_events_list(self, events: list) -> None:
        while self.recent_events_layout.count():
            item = self.recent_events_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not events:
            empty = QLabel("No movements recorded yet today.")
            empty.setObjectName("Muted")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.recent_events_layout.addWidget(empty)
            self.recent_events_layout.addStretch(1)
            return

        for ev in events[:8]:
            ev_type = ev.event_type if hasattr(ev, "event_type") else ev["event_type"]
            plate = ev.plate_number if hasattr(ev, "plate_number") else ev["plate_number"]
            ts = ev.event_timestamp if hasattr(ev, "event_timestamp") else ev["event_timestamp"]
            conf = float((ev.ocr_confidence if hasattr(ev, "ocr_confidence") else ev["ocr_confidence"]) or 0.0)
            p_path = (ev.plate_image_path if hasattr(ev, "plate_image_path") else ev.get("plate_image_path")) or ""

            row_card = QFrame()
            row_card.setObjectName("SubCard")
            rl = QHBoxLayout(row_card)
            rl.setContentsMargins(8, 6, 8, 6)
            rl.setSpacing(8)

            badge = (
                _badge_label("IN", "#064e3b", "#34d399", "#059669")
                if ev_type == "IN"
                else _badge_label("OUT", "#451a03", "#fbbf24", "#d97706")
            )
            badge.setFixedWidth(44)
            rl.addWidget(badge)

            info = QVBoxLayout()
            info.setSpacing(1)
            lp = QLabel(plate)
            lp.setStyleSheet("font-family: Consolas; font-weight: 700; color: #ffffff; font-size: 12px;")
            lt = QLabel(f"{_fmt_dt(ts)[-8:]} • {conf * 100:.0f}% conf")
            lt.setStyleSheet("font-size: 10px; color: #9ca3af;")
            info.addWidget(lp)
            info.addWidget(lt)
            rl.addLayout(info, 1)

            resolved = self.bridge.resolve_capture_path(p_path)
            if resolved is not None:
                btn_img = QPushButton("Plate")
                btn_img.setFixedHeight(26)
                btn_img.clicked.connect(
                    lambda checked=False, p=resolved, pl=plate: ImagePreviewDialog(p, f"Plate Crop: {pl}", self).exec()
                )
                rl.addWidget(btn_img)

            self.recent_events_layout.addWidget(row_card)

        self.recent_events_layout.addStretch(1)

    def _render_dashboard_table(self, events: list) -> None:
        rows = events[:15]
        self.tbl_dashboard.setRowCount(len(rows))
        for r, ev in enumerate(rows):
            ev_type = ev.event_type if hasattr(ev, "event_type") else ev["event_type"]
            plate = ev.plate_number if hasattr(ev, "plate_number") else ev["plate_number"]
            ts = ev.event_timestamp if hasattr(ev, "event_timestamp") else ev["event_timestamp"]
            conf = float((ev.ocr_confidence if hasattr(ev, "ocr_confidence") else ev["ocr_confidence"]) or 0.0)
            p_path = (ev.plate_image_path if hasattr(ev, "plate_image_path") else ev.get("plate_image_path")) or ""

            badge = (
                _badge_label("IN", "#064e3b", "#34d399", "#059669")
                if ev_type == "IN"
                else _badge_label("OUT", "#451a03", "#fbbf24", "#d97706")
            )
            self.tbl_dashboard.setCellWidget(r, 0, badge)
            item_p = QTableWidgetItem(plate)
            item_p.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            self.tbl_dashboard.setItem(r, 1, item_p)
            self.tbl_dashboard.setItem(r, 2, QTableWidgetItem(_fmt_dt(ts)))
            self.tbl_dashboard.setItem(r, 3, QTableWidgetItem(f"{conf * 100:.1f}%"))

            resolved = self.bridge.resolve_capture_path(p_path)
            if resolved is not None:
                btn = QPushButton("View Plate")
                btn.clicked.connect(
                    lambda checked=False, p=resolved, pl=plate: ImagePreviewDialog(p, f"Plate Crop: {pl}", self).exec()
                )
                self.tbl_dashboard.setCellWidget(r, 4, btn)
            else:
                self.tbl_dashboard.setItem(r, 4, QTableWidgetItem("-"))

    # ------------------------------------------------------- Vehicle Registry
    def load_vehicles(self) -> None:
        data = self.bridge.list_vehicles(
            search=self.vehicle_search_edit.text(),
            status_filter=str(self.vehicle_status_combo.currentData() or ""),
            limit=100,
        )
        items = data["items"]
        is_admin = self.session.role == "ADMIN"
        self.tbl_vehicles.setRowCount(len(items))

        for r, v in enumerate(items):
            plate = v["plate_number"]
            is_in = v["current_status"] == "IN"

            p_item = QTableWidgetItem(plate)
            p_item.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            self.tbl_vehicles.setItem(r, 0, p_item)

            badge = (
                _badge_label("INSIDE", "#064e3b", "#34d399", "#059669")
                if is_in
                else _badge_label("OUTSIDE", "#1f2937", "#9ca3af", "#374151")
            )
            self.tbl_vehicles.setCellWidget(r, 1, badge)
            self.tbl_vehicles.setItem(r, 2, QTableWidgetItem(str(v["total_entries"])))
            self.tbl_vehicles.setItem(r, 3, QTableWidgetItem(str(v["total_exits"])))
            self.tbl_vehicles.setItem(r, 4, QTableWidgetItem(_fmt_dt(v["last_seen_at"])))

            act_w = QWidget()
            al = QHBoxLayout(act_w)
            al.setContentsMargins(4, 2, 4, 2)
            al.setSpacing(6)

            btn_hist = QPushButton("History")
            btn_hist.clicked.connect(lambda checked=False, pl=plate: self._show_vehicle_history(pl))
            al.addWidget(btn_hist)

            if is_admin:
                next_st = "OUT" if is_in else "IN"
                btn_tog = QPushButton(f"Toggle {next_st}")
                btn_tog.setObjectName("PrimaryBtn")
                btn_tog.clicked.connect(lambda checked=False, pl=plate, ns=next_st: self.override_vehicle_status(pl, ns))
                btn_del = QPushButton("Delete")
                btn_del.setObjectName("DangerBtn")
                btn_del.clicked.connect(lambda checked=False, pl=plate: self.delete_vehicle(pl))
                al.addWidget(btn_tog)
                al.addWidget(btn_del)

            self.tbl_vehicles.setCellWidget(r, 5, act_w)

    def _show_vehicle_history(self, plate: str) -> None:
        events = self.bridge.get_vehicle_history(plate)
        VehicleHistoryDialog(plate, events, self).exec()

    def override_vehicle_status(self, plate: str, new_status: str) -> None:
        ans = QMessageBox.question(
            self,
            "Confirm Status Override",
            f"Administratively override vehicle {plate} status to {new_status}?",
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            self.bridge.override_vehicle_status(self.session, plate, new_status)
            self.show_toast(f"Vehicle {plate} status updated to {new_status}")
            self.load_vehicles()
            self.refresh_live_telemetry()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)

    def delete_vehicle(self, plate: str) -> None:
        ans = QMessageBox.question(
            self,
            "Delete Vehicle",
            f"Delete vehicle {plate} along with ALL its movement history and saved images/clips?",
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            res = self.bridge.delete_vehicle(self.session, plate)
            self.show_toast(
                f"Deleted {plate} ({res['deleted_events']} events, {res['deleted_files']} files)"
            )
            self.load_vehicles()
            self.refresh_live_telemetry()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)

    # ---------------------------------------------------------- Event History
    def load_events(self) -> None:
        data = self.bridge.list_events(
            plate=self.event_plate_edit.text(),
            event_type=str(self.event_type_combo.currentData() or ""),
            limit=80,
        )
        items = data["items"]
        is_admin = self.session.role == "ADMIN"
        self.tbl_events.setRowCount(len(items))

        for r, ev in enumerate(items):
            ev_type = ev["event_type"]
            plate = ev["plate_number"]
            badge = (
                _badge_label("IN", "#064e3b", "#34d399", "#059669")
                if ev_type == "IN"
                else _badge_label("OUT", "#451a03", "#fbbf24", "#d97706")
            )
            self.tbl_events.setCellWidget(r, 0, badge)
            p_item = QTableWidgetItem(plate)
            p_item.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            self.tbl_events.setItem(r, 1, p_item)
            self.tbl_events.setItem(r, 2, QTableWidgetItem(_fmt_dt(ev["event_timestamp"])))
            self.tbl_events.setItem(r, 3, QTableWidgetItem(f"{ev['ocr_confidence'] * 100:.1f}%"))
            self.tbl_events.setItem(r, 4, QTableWidgetItem(ev["raw_ocr_text"] or "-"))

            act_w = QWidget()
            al = QHBoxLayout(act_w)
            al.setContentsMargins(4, 2, 4, 2)
            al.setSpacing(6)

            if ev["clip_filename"]:
                clip_path = settings.EVENTS_DIR / ev["clip_filename"]
                if clip_path.exists():
                    btn_clip = QPushButton("▶ Clip")
                    btn_clip.setObjectName("PrimaryBtn")
                    btn_clip.clicked.connect(
                        lambda checked=False, cp=clip_path, t=f"Recorded Clip: {plate} ({ev_type})":
                        ClipPlayerDialog(cp, t, self).exec()
                    )
                    al.addWidget(btn_clip)

            plate_img = self.bridge.resolve_capture_path(ev["plate_image_path"])
            if plate_img is not None:
                btn_p = QPushButton("Plate")
                btn_p.clicked.connect(
                    lambda checked=False, p=plate_img, pl=plate:
                    ImagePreviewDialog(p, f"Plate Crop: {pl}", self).exec()
                )
                al.addWidget(btn_p)

            veh_img = self.bridge.resolve_capture_path(ev["vehicle_image_path"])
            if veh_img is not None:
                btn_v = QPushButton("Vehicle")
                btn_v.clicked.connect(
                    lambda checked=False, p=veh_img, pl=plate:
                    ImagePreviewDialog(p, f"Vehicle Context: {pl}", self).exec()
                )
                al.addWidget(btn_v)

            if is_admin:
                btn_del = QPushButton("Delete")
                btn_del.setObjectName("DangerBtn")
                btn_del.clicked.connect(lambda checked=False, eid=ev["id"], pl=plate: self.delete_event(eid, pl))
                al.addWidget(btn_del)

            self.tbl_events.setCellWidget(r, 5, act_w)

    def delete_event(self, event_id: int, plate: str) -> None:
        ans = QMessageBox.question(
            self,
            "Delete Event",
            f"Delete event #{event_id} for {plate} and its captured media?",
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            res = self.bridge.delete_event(self.session, event_id)
            self.show_toast(f"Deleted event #{event_id} ({res['deleted_files']} files removed)")
            self.load_events()
            self.refresh_live_telemetry()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)

    # --------------------------------------------------------- Recorded Clips
    def load_clips(self) -> None:
        while self.clips_grid.count():
            item = self.clips_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        data = self.bridge.list_clips(limit=60)
        items = data["items"]
        if not items:
            empty = QLabel("No video clips stored yet. Clips are automatically recorded whenever a plate is detected.")
            empty.setObjectName("Muted")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.clips_grid.addWidget(empty, 0, 0, 1, 3)
            return

        is_admin = self.session.role == "ADMIN"
        cols = 3
        for idx, c in enumerate(items):
            card = QFrame()
            card.setObjectName("Card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 10, 10, 10)
            cl.setSpacing(8)

            # Thumbnail preview
            thumb_lbl = QLabel()
            thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_lbl.setFixedHeight(150)
            thumb_lbl.setStyleSheet("background-color: #000000; border-radius: 8px; color: #6b7280;")
            if c["thumbnail_path"] and Path(c["thumbnail_path"]).exists():
                pm = QPixmap(c["thumbnail_path"])
                if not pm.isNull():
                    thumb_lbl.setPixmap(
                        pm.scaled(300, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    )
            else:
                thumb_lbl.setText("MP4 VIDEO")
            cl.addWidget(thumb_lbl)

            top_meta = QHBoxLayout()
            label_text = c["plate_number"] or "Unconfirmed Clip"
            badge = (
                _badge_label(label_text, "#064e3b", "#34d399", "#059669")
                if c["plate_number"]
                else _badge_label(label_text, "#1f2937", "#d1d5db", "#374151")
            )
            size_lbl = QLabel(f"{c['size_kb']} KB")
            size_lbl.setStyleSheet("font-family: Consolas; font-size: 11px; color: #9ca3af;")
            top_meta.addWidget(badge)
            top_meta.addStretch(1)
            top_meta.addWidget(size_lbl)
            cl.addLayout(top_meta)

            fn_lbl = QLabel(c["filename"])
            fn_lbl.setStyleSheet("font-family: Consolas; font-size: 11px; color: #e5e7eb;")
            dt_lbl = QLabel(_fmt_dt(c["created_at"]))
            dt_lbl.setStyleSheet("font-size: 10px; color: #9ca3af;")
            cl.addWidget(fn_lbl)
            cl.addWidget(dt_lbl)

            btns = QHBoxLayout()
            btns.setSpacing(6)
            clip_path = Path(c["file_path"])
            btn_play = QPushButton("▶ Play")
            btn_play.setObjectName("PrimaryBtn")
            btn_play.clicked.connect(
                lambda checked=False, cp=clip_path, t=f"Clip: {label_text}": ClipPlayerDialog(cp, t, self).exec()
            )
            btn_save = QPushButton("Save MP4")
            btn_save.clicked.connect(lambda checked=False, cp=clip_path: self._save_clip_copy(cp))
            btns.addWidget(btn_play)
            btns.addWidget(btn_save)

            if is_admin:
                btn_del = QPushButton("Delete")
                btn_del.setObjectName("DangerBtn")
                btn_del.clicked.connect(lambda checked=False, fn=c["filename"]: self.delete_clip(fn))
                btns.addWidget(btn_del)

            cl.addLayout(btns)
            self.clips_grid.addWidget(card, idx // cols, idx % cols)

    def _save_clip_copy(self, clip_path: Path) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Video Clip", clip_path.name, "MP4 Video (*.mp4);;All Files (*)"
        )
        if dest:
            shutil.copy2(clip_path, dest)
            self.show_toast(f"Exported clip to {Path(dest).name}")

    def delete_clip(self, filename: str) -> None:
        ans = QMessageBox.question(self, "Delete Clip", f"Delete stored video clip {filename}?")
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            self.bridge.delete_clip(self.session, filename)
            self.show_toast(f"Deleted clip {filename}")
            self.load_clips()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)

    # ------------------------------------------------------------ RBAC Users
    def load_users(self) -> None:
        try:
            users = self.bridge.list_users(self.session)
        except PermissionDeniedError as exc:
            self.show_toast(str(exc), is_error=True)
            return
        self.tbl_users.setRowCount(len(users))
        for r, u in enumerate(users):
            self.tbl_users.setItem(r, 0, QTableWidgetItem(u["full_name"]))
            self.tbl_users.setItem(r, 1, QTableWidgetItem(u["username"]))
            self.tbl_users.setItem(r, 2, QTableWidgetItem(u["email"]))
            self.tbl_users.setCellWidget(r, 3, _badge_label(u["role"], "#1f2937", "#93c5fd", "#374151"))
            status_lbl = (
                _badge_label("Active", "#064e3b", "#34d399", "#059669")
                if u["is_active"]
                else _badge_label("Inactive", "#450a0a", "#f87171", "#991b1b")
            )
            self.tbl_users.setCellWidget(r, 4, status_lbl)

    def open_create_user_dialog(self) -> None:
        dlg = CreateUserDialog(parent=self)
        if dlg.exec() == CreateUserDialog.DialogCode.Accepted:
            vals = dlg.values()
            try:
                res = self.bridge.create_user(self.session, **vals)
                self.show_toast(f"User '{res['username']}' ({res['role']}) created successfully.")
                self.load_users()
            except Exception as exc:
                self.show_toast(str(exc), is_error=True)
                QMessageBox.warning(self, "Create User Failed", str(exc))

    # ---------------------------------------------------- System & Settings
    def load_system_details(self) -> None:
        st = self.bridge.get_system_status()
        diag = st.get("hardware_diagnostics", {})
        notif = st.get("notifications", {})
        cam = st.get("camera", {})

        providers = ", ".join(diag.get("onnx_providers", [])) or "N/A"
        vram_line = (
            f"VRAM: {round(diag['allocated_vram_mb'])} MB / {round(diag['total_vram_mb'])} MB ({diag.get('utilization_pct', 0)}%)"
            if diag.get("total_vram_mb")
            else f"CPU Threads: {diag.get('cpu_count', 'N/A')}"
        )
        self.lbl_sys_hw.setText(
            f"<b>Active Device:</b> {diag.get('active_device', 'cpu').upper()}<br>"
            f"<b>Hardware Name:</b> {diag.get('device_name', 'CPU')}<br>"
            f"<b>ONNX Providers:</b> {providers}<br>"
            f"<b>{vram_line}</b><br>"
            f"<b>Camera Stream:</b> {cam.get('source')} ({cam.get('configured_resolution')})<br>"
            f"<b>Capture FPS:</b> {cam.get('fps_actual', 0)} FPS"
        )

        self.lbl_sys_notif.setText(
            f"<b>Server:</b> {notif.get('server')}<br>"
            f"<b>Topic:</b> {notif.get('topic')}<br>"
            f"<b>Photo Attachments:</b> {'Enabled' if notif.get('attach_photo') else 'Disabled'} "
            f"({notif.get('total_photos_sent', 0)} photos sent)<br>"
            f"<b>Notifications Sent:</b> {notif.get('total_dispatched', 0)}<br>"
            f"<b>Queue Pending:</b> {notif.get('queue_pending', 0)}<br>"
            f"<b>Last Status:</b> {notif.get('last_status', 'idle')}"
        )
        self.load_logs()

    def _switch_log_view(self, mode: str) -> None:
        self._log_mode = mode
        self.load_logs()

    def load_logs(self) -> None:
        lines = (
            self.bridge.get_license_logs(100)
            if self._log_mode == "license"
            else self.bridge.get_logs(120)
        )
        self.txt_logs.setPlainText("\n".join(lines))
        sb = self.txt_logs.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_runtime_settings(self) -> None:
        try:
            self.bridge.update_runtime_settings(
                self.session,
                detection_confidence=self.spin_det_conf.value(),
                recognition_cooldown=self.spin_cooldown.value(),
                event_strategy=str(self.combo_strategy.currentData()),
                clip_recording_enabled=self.chk_clip_mode.isChecked(),
                notifications_enabled=self.chk_notif_enabled.isChecked(),
                ntfy_topic=self.edit_ntfy_topic.text(),
            )
            self.show_toast("Runtime settings applied successfully.")
            self.load_system_details()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)

    # ------------------------------------------ Manual Entry / Purge / Export
    def open_manual_gate_dialog(self, default_type: str = "IN") -> None:
        dlg = ManualGateDialog(default_type=default_type, parent=self)
        if dlg.exec() == ManualGateDialog.DialogCode.Accepted:
            plate, ev_type, notes = dlg.values()
            try:
                self.bridge.manual_gate_entry(self.session, plate, ev_type, notes)
                self.show_toast(f"Manual event recorded: {plate} ({ev_type})")
                self.refresh_all_data()
            except Exception as exc:
                self.show_toast(str(exc), is_error=True)
                QMessageBox.warning(self, "Manual Entry Rejected", str(exc))

    def quick_gate_action(self, ev_type: str) -> None:
        plate, ok = QInputDialog.getText(
            self,
            f"Quick Gate {ev_type}",
            f"Enter Vehicle Registration Number for Gate {ev_type}:",
        )
        if not ok or not plate.strip():
            return
        try:
            self.bridge.manual_gate_entry(
                self.session, plate.strip().upper(), ev_type, f"Quick Gate {ev_type} Action"
            )
            self.show_toast(f"Gate {ev_type} recorded for {plate.strip().upper()}")
            self.refresh_all_data()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)
            QMessageBox.warning(self, "Action Rejected", str(exc))

    def reconnect_camera(self) -> None:
        ok = self.bridge.reconnect_camera()
        self.show_toast(
            "Camera reconnected successfully." if ok else "Camera reconnect failed.",
            is_error=not ok,
        )
        self.refresh_live_telemetry()

    def purge_system_data(self, scope: str) -> None:
        labels = {
            "events": "ALL movement events and captured images/clips",
            "vehicles": "ALL registered vehicles, events, and captured media",
            "captures": "ALL captured image and video clip files on disk",
            "logs": "the application log file",
            "all": "ALL vehicles, events, system logs, and captured media (Full Reset)",
        }
        ans = QMessageBox.warning(
            self,
            "Confirm Permanent Deletion",
            f"Permanently delete {labels.get(scope, scope)}?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            res = self.bridge.purge_system_data(self.session, scope)
            self.show_toast(
                f"Purged ({scope}): {res['deleted_vehicles']} vehicles, "
                f"{res['deleted_events']} events, {res['deleted_files']} files"
            )
            self.refresh_all_data()
            self.load_vehicles()
            self.load_events()
            self.load_clips()
            if self.session.role == "ADMIN":
                self.load_logs()
        except Exception as exc:
            self.show_toast(str(exc), is_error=True)

    def export_vehicles_csv(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(self, "Export Vehicles CSV", "vehicles.csv", "CSV Files (*.csv)")
        if not dest:
            return
        try:
            count = self.bridge.export_vehicles_csv(
                dest,
                search=self.vehicle_search_edit.text(),
                status_filter=str(self.vehicle_status_combo.currentData() or ""),
            )
            self.show_toast(f"Exported {count} vehicles to {Path(dest).name}")
        except Exception as exc:
            self.show_toast(f"CSV export failed: {exc}", is_error=True)

    def export_events_csv(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(self, "Export Events CSV", "events.csv", "CSV Files (*.csv)")
        if not dest:
            return
        try:
            count = self.bridge.export_events_csv(
                dest,
                plate=self.event_plate_edit.text(),
                event_type=str(self.event_type_combo.currentData() or ""),
            )
            self.show_toast(f"Exported {count} events to {Path(dest).name}")
        except Exception as exc:
            self.show_toast(f"CSV export failed: {exc}", is_error=True)

    # ======================================================= STATUS & EXIT
    def show_toast(self, message: str, is_error: bool = False, timeout_ms: int = 3500) -> None:
        prefix = "[ERROR] " if is_error else ""
        self.status_bar.setStyleSheet(
            "QStatusBar { background-color: #0f172a; border-top: 1px solid #1e293b; color: "
            + ("#f87171" if is_error else "#34d399")
            + "; }"
        )
        self.status_bar.showMessage(f"{prefix}{message}", timeout_ms)

    def _on_license_revoked(self, message: str) -> None:
        QMessageBox.critical(self, f"{settings.APP_NAME} — License Inactive", message)
        self.close()

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        if self._video_worker is not None and self._video_worker.isRunning():
            self._video_worker.stop()
        if self._probe_worker is not None and self._probe_worker.isRunning():
            self._probe_worker.wait(1500)
        if self._engine_worker is not None and self._engine_worker.isRunning():
            self._engine_worker.wait(2000)
        self.bridge.stop_pipeline()
        super().closeEvent(event)
