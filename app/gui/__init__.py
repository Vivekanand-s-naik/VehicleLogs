"""
Native PyQt6 Desktop Interface for the ANPR & Gate Access System.
Replaces the browser-based web UI while using the existing FastAPI/SQLAlchemy/RBAC
and ONNX Runtime detection pipeline cleanly in-process (zero exposed network ports).
"""

from app.gui.main_window import MainWindow
from app.gui.dialogs import LoginDialog

__all__ = ["MainWindow", "LoginDialog"]
