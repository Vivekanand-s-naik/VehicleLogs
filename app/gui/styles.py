"""
Modern Dark Security-Control QSS Stylesheet for PyQt6 Desktop GUI.
Matches the sleek slate/blue/emerald/amber visual language of the ANPR Control Suite.
"""

DARK_STYLESHEET = """
QWidget {
    background-color: #0b1120;
    color: #e5e7eb;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QMainWindow, QDialog {
    background-color: #0b1120;
}

QFrame#Sidebar {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

QFrame#TopBar {
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
}

QFrame#Card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
}

QFrame#CardDanger {
    background-color: #111827;
    border: 1px solid #7f1d1d;
    border-radius: 12px;
}

QFrame#KpiBlue {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
}

QFrame#KpiEmerald {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #10b981;
    border-radius: 10px;
}

QFrame#KpiIndigo {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
}

QFrame#KpiAmber {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
}

QFrame#SubCard {
    background-color: #0b1120;
    border: 1px solid #1f2937;
    border-radius: 8px;
}

QPushButton#NavBtn {
    background-color: transparent;
    color: #cbd5e1;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    font-weight: 500;
}

QPushButton#NavBtn:hover {
    background-color: #1e293b;
    color: #ffffff;
}

QPushButton#NavBtn[active="true"] {
    background-color: rgba(37, 99, 235, 0.22);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.45);
    font-weight: 600;
}

QPushButton {
    background-color: #1f2937;
    color: #f3f4f6;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #374151;
}

QPushButton:disabled {
    background-color: #111827;
    color: #6b7280;
    border-color: #1f2937;
}

QPushButton#PrimaryBtn {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
    font-weight: 600;
}

QPushButton#PrimaryBtn:hover {
    background-color: #1d4ed8;
}

QPushButton#SuccessBtn {
    background-color: #059669;
    color: #ffffff;
    border: 1px solid #10b981;
    font-weight: 600;
}

QPushButton#SuccessBtn:hover {
    background-color: #047857;
}

QPushButton#WarningBtn {
    background-color: #d97706;
    color: #ffffff;
    border: 1px solid #f59e0b;
    font-weight: 600;
}

QPushButton#WarningBtn:hover {
    background-color: #b45309;
}

QPushButton#DangerBtn {
    background-color: #7f1d1d;
    color: #fecaca;
    border: 1px solid #b91c1c;
    font-weight: 600;
}

QPushButton#DangerBtn:hover {
    background-color: #991b1b;
    color: #ffffff;
}

QPushButton#PurpleBtn {
    background-color: #7c3aed;
    color: #ffffff;
    border: 1px solid #8b5cf6;
    font-weight: 600;
}

QPushButton#PurpleBtn:hover {
    background-color: #6d28d9;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #030712;
    color: #f9fafb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 6px 10px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #3b82f6;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #111827;
    color: #f3f4f6;
    border: 1px solid #374151;
    selection-background-color: #2563eb;
}

QTableWidget {
    background-color: #0f172a;
    alternate-background-color: #111827;
    color: #e5e7eb;
    gridline-color: #1f2937;
    border: 1px solid #1f2937;
    border-radius: 10px;
    selection-background-color: rgba(37, 99, 235, 0.35);
}

QTableWidget::item {
    padding: 6px 8px;
}

QHeaderView::section {
    background-color: #111827;
    color: #9ca3af;
    border: none;
    border-bottom: 1px solid #1f2937;
    padding: 8px 10px;
    font-weight: 600;
    font-size: 12px;
}

QPlainTextEdit {
    background-color: #030712;
    color: #d1d5db;
    border: 1px solid #1f2937;
    border-radius: 10px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    padding: 8px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background: #0f172a;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QStatusBar {
    background-color: #0f172a;
    color: #9ca3af;
    border-top: 1px solid #1e293b;
    font-size: 12px;
}

QCheckBox {
    spacing: 8px;
    color: #e5e7eb;
}

QLabel#Muted {
    color: #9ca3af;
}

QLabel#SectionTitle {
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
}

QLabel#VideoSurface {
    background-color: #000000;
    border: 1px solid #1f2937;
    border-radius: 12px;
    color: #6b7280;
}
"""
