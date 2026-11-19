"""
Background QThread Workers for Non-Blocking Native GUI Responsiveness
=====================================================================
Keeps ONNX model loading, live video frame conversion, and camera index
enumeration completely off the PyQt6 main GUI thread.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from app.gui.backend_bridge import BackendBridge

logger = logging.getLogger("anpr.gui.workers")


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR ndarray into a deep-copied QImage safe to emit across threads."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class EngineInitWorker(QThread):
    """Loads ONNX Runtime DirectML/CPU detectors + OCR engines and starts the pipeline off-thread."""

    ready = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, bridge: BackendBridge, auto_start: bool = True, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.auto_start = auto_start

    def run(self) -> None:
        try:
            self.bridge.init_engine()
            if self.auto_start:
                self.bridge.start_pipeline()
            self.ready.emit()
        except Exception as exc:
            logger.exception(f"Background ANPR engine initialization failed: {exc}")
            self.failed.emit(str(exc))


class VideoFeedWorker(QThread):
    """
    Polls EventService's latest annotated BGR frame (with vehicle/plate bboxes & OCR overlays)
    at up to ~30 FPS, converts to QImage off-thread, and emits `frame_ready`.
    """

    frame_ready = pyqtSignal(QImage)

    def __init__(self, bridge: BackendBridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        last_id = -1
        while self._running:
            svc = self.bridge.event_service
            if svc is None or not svc.is_running:
                time.sleep(0.08)
                continue

            frame_id, frame = svc.get_latest_annotated_frame_with_id()
            if frame is None or frame_id == last_id:
                time.sleep(0.02)
                continue

            last_id = frame_id
            try:
                qimg = bgr_to_qimage(frame)
                self.frame_ready.emit(qimg)
            except Exception as exc:
                logger.debug(f"Frame conversion skipped: {exc}")
                time.sleep(0.03)


class CameraProbeWorker(QThread):
    """Enumerates local camera indices via cv2.VideoCapture off the GUI thread."""

    devices_ready = pyqtSignal(list)

    def __init__(self, bridge: BackendBridge, max_index: int = 6, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.max_index = max_index

    def run(self) -> None:
        try:
            devices: List[Dict[str, str]] = self.bridge.list_cameras(max_index=self.max_index)
            self.devices_ready.emit(devices)
        except Exception as exc:
            logger.warning(f"Camera enumeration error: {exc}")
            self.devices_ready.emit([
                {"source": "synthetic", "label": "Synthetic Test Feed (Gate 1)", "kind": "synthetic"}
            ])
