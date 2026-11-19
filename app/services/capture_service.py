"""
Image Capture & Archiving Service
=================================
Saves vehicle and cropped license plate images with collision-resistant naming:
captures/vehicles/{plate}_{event_type}_{timestamp}.jpg
captures/plates/{plate}_{event_type}_{timestamp}.jpg
Provides image retrieval and retention policy cleanup.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings

logger = logging.getLogger("anpr.capture")


class CaptureService:
    def __init__(self):
        self._ensure_directories()

    def _ensure_directories(self):
        """Create capture directories if missing."""
        settings.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        settings.VEHICLES_DIR.mkdir(parents=True, exist_ok=True)
        settings.PLATES_DIR.mkdir(parents=True, exist_ok=True)
        settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    def save_event_images(
        self,
        plate_number: str,
        event_type: str,
        vehicle_image: Optional[np.ndarray],
        plate_image: Optional[np.ndarray]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Save vehicle and plate crop images to disk.
        Returns relative paths (relative to BASE_DIR) suitable for database persistence.
        """
        self._ensure_directories()
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_plate = "".join(c for c in plate_number if c.isalnum())
        unique_suffix = f"{int(time.time() * 1000) % 10000:04d}"

        base_filename = f"{safe_plate}_{event_type.upper()}_{timestamp_str}_{unique_suffix}.jpg"

        rel_vehicle_path = None
        rel_plate_path = None

        # 1. Save vehicle image
        if vehicle_image is not None and vehicle_image.size > 0:
            veh_dest = settings.VEHICLES_DIR / base_filename
            try:
                cv2.imwrite(str(veh_dest), vehicle_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                rel_vehicle_path = str(veh_dest.relative_to(settings.CAPTURES_DIR.parent)).replace("\\", "/")
            except Exception as e:
                logger.error(f"Failed to save vehicle image {veh_dest}: {e}")

        # 2. Save cropped plate image
        if plate_image is not None and plate_image.size > 0:
            plate_dest = settings.PLATES_DIR / base_filename
            try:
                cv2.imwrite(str(plate_dest), plate_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                rel_plate_path = str(plate_dest.relative_to(settings.CAPTURES_DIR.parent)).replace("\\", "/")
            except Exception as e:
                logger.error(f"Failed to save plate image {plate_dest}: {e}")

        return rel_vehicle_path, rel_plate_path

    def purge_expired_images(self, days: Optional[int] = None) -> int:
        """
        Delete images older than IMAGE_RETENTION_DAYS.
        Returns the number of deleted files.
        """
        retention_days = days or settings.IMAGE_RETENTION_DAYS
        cutoff = time.time() - (retention_days * 86400)
        deleted_count = 0

        for folder in (settings.VEHICLES_DIR, settings.PLATES_DIR, settings.EVENTS_DIR):
            if not folder.exists():
                continue
            for img_file in folder.glob("*.jpg"):
                try:
                    if img_file.stat().st_mtime < cutoff:
                        img_file.unlink()
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove old capture {img_file}: {e}")

        logger.info(f"Purged {deleted_count} captures older than {retention_days} days.")
        return deleted_count
