"""
Camera Service
==============
High-level management of video capture streams, health diagnostics,
and reconnection operations.
"""

import logging
from typing import Dict, Any, Optional
from app.core.config import settings
from app.vision.camera_manager import CameraManager

logger = logging.getLogger("anpr.camera_service")


class CameraService:
    _instance: Optional["CameraService"] = None

    def __init__(self, camera_manager: Optional[CameraManager] = None):
        self.camera_manager = camera_manager or CameraManager()

    @classmethod
    def get_instance(cls, camera_manager: Optional[CameraManager] = None) -> "CameraService":
        if cls._instance is None:
            cls._instance = cls(camera_manager=camera_manager)
        elif camera_manager is not None and cls._instance.camera_manager is not camera_manager:
            cls._instance.camera_manager = camera_manager
        return cls._instance

    def start(self):
        self.camera_manager.start()

    def stop(self):
        self.camera_manager.stop()

    def reconnect(self) -> bool:
        return self.camera_manager.reconnect()

    def get_status(self) -> Dict[str, Any]:
        return {
            "source": self.camera_manager.source,
            "is_connected": self.camera_manager.is_connected,
            "fps_actual": round(self.camera_manager.fps_actual, 1),
            "total_frames_captured": self.camera_manager.total_frames_captured,
            "reconnect_attempts": self.camera_manager.reconnect_attempts,
            "configured_resolution": f"{self.camera_manager.width}x{self.camera_manager.height}",
            "configured_fps": self.camera_manager.fps
        }
