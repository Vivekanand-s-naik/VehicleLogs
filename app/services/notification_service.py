"""
Mobile Push Notification Service (ntfy.sh) with Clip Photo Attachments
======================================================================
Dispatches real-time vehicle entry/exit notifications to mobile devices via ntfy.sh.
When a clip/vehicle photo is available, uploads the compressed JPEG snapshot
directly via ntfy.sh's HTTP PUT attachment API (`Filename` + `Message` headers)
so Android/iOS notifications display the vehicle & zoomed plate photo inline.
Falls back seamlessly to text-only HTTP POST if no photo is available or if an
attachment upload is rejected.
"""

import time
import queue
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import cv2
import numpy as np
import requests
from app.core.config import settings

logger = logging.getLogger("anpr.notifications")


def build_notification_photo(
    vehicle_image: Optional[np.ndarray],
    plate_image: Optional[np.ndarray],
    plate_number: str,
    event_type: str,
    confidence: float,
) -> Optional[bytes]:
    """
    Create a crisp, compact JPEG notification card from the clip's best frame:
    - Resizes vehicle/clip frame to max width 960px.
    - Overlays the zoomed cropped number plate inset in the bottom-right corner.
    - Adds a clean caption bar showing plate, movement type, and confidence.
    Returns JPEG bytes (~50-120 KB), or None if no image is available.
    """
    base = None
    if vehicle_image is not None and vehicle_image.size > 0:
        base = vehicle_image.copy()
    elif plate_image is not None and plate_image.size > 0:
        base = plate_image.copy()
    else:
        return None

    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    h, w = base.shape[:2]
    max_w = 960
    if w > max_w:
        scale = max_w / float(w)
        base = cv2.resize(base, (max_w, max(64, int(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = base.shape[:2]
    elif w < 320:
        scale = 320.0 / max(1, w)
        base = cv2.resize(base, (320, max(64, int(h * scale))), interpolation=cv2.INTER_CUBIC)
        h, w = base.shape[:2]

    # Overlay zoomed plate crop inset in bottom-right corner when both images exist
    if (
        vehicle_image is not None and vehicle_image.size > 0
        and plate_image is not None and plate_image.size > 0
    ):
        p_img = plate_image.copy()
        if p_img.ndim == 2:
            p_img = cv2.cvtColor(p_img, cv2.COLOR_GRAY2BGR)
        ph, pw = p_img.shape[:2]
        inset_w = min(int(w * 0.38), 280)
        inset_h = max(32, min(int(h * 0.28), int(inset_w * ph / max(1, pw))))
        inset_w = max(48, int(inset_h * pw / max(1, ph)))
        if inset_w <= w - 20 and inset_h <= h - 48:
            resized_plate = cv2.resize(p_img, (inset_w, inset_h), interpolation=cv2.INTER_CUBIC)
            pad = 10
            y2 = h - 38
            y1 = max(pad, y2 - inset_h)
            x2 = w - pad
            x1 = max(pad, x2 - inset_w)
            resized_plate = cv2.resize(resized_plate, (x2 - x1, y2 - y1))
            border_color = (80, 230, 120) if event_type.upper() == "IN" else (60, 180, 255)
            cv2.rectangle(base, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), border_color, 2)
            base[y1:y2, x1:x2] = resized_plate

    # Add crisp bottom caption bar
    bar_h = 30
    overlay = base.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (16, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, base, 0.22, 0, base)
    accent = (80, 230, 120) if event_type.upper() == "IN" else (60, 190, 255)
    caption = f"{plate_number}  |  {event_type.upper()}  |  {confidence * 100:.0f}%"
    cv2.putText(base, caption, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.58, accent, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    return buf.tobytes() if ok else None


class NotificationItem:
    def __init__(
        self,
        plate_number: str,
        event_type: str,
        timestamp_str: str,
        confidence: float,
        state_name: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        clip_path: Optional[str] = None,
    ):
        self.plate_number = plate_number
        self.event_type = event_type
        self.timestamp_str = timestamp_str
        self.confidence = confidence
        self.state_name = state_name
        self.image_bytes = image_bytes
        self.image_path = image_path
        self.clip_path = clip_path
        self.created_at = time.time()
        self.attempts = 0


class NotificationService:
    _instance: Optional["NotificationService"] = None

    def __init__(self):
        self.server = settings.NTFY_SERVER.rstrip("/")
        self.topic = settings.NTFY_TOPIC
        self.token = settings.NTFY_TOKEN
        self.enabled = settings.NOTIFICATIONS_ENABLED
        self.attach_photo = settings.NOTIFICATION_ATTACH_PHOTO
        self.timeout = settings.NTFY_TIMEOUT
        self.max_retries = settings.NTFY_MAX_RETRIES

        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Telemetry
        self.total_dispatched: int = 0
        self.total_photos_sent: int = 0
        self.total_failed: int = 0
        self.last_status: str = "IDLE"

        self._start_worker()

    @classmethod
    def get_instance(cls) -> "NotificationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _start_worker(self):
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._process_queue,
            name="NotificationWorkerThread",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(
            f"Notification service worker active. Target: {self.server}/{self.topic} "
            f"(photo_attachments={self.attach_photo})"
        )

    def stop(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def dispatch_event(
        self,
        plate_number: str,
        event_type: str,
        timestamp_str: str,
        confidence: float,
        state_name: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[str] = None,
        clip_path: Optional[str] = None,
    ):
        """Non-blocking enqueue of vehicle movement notification (with optional clip photo)."""
        if not self.enabled:
            logger.debug("Notifications are disabled in settings.")
            return

        item = NotificationItem(
            plate_number=plate_number,
            event_type=event_type,
            timestamp_str=timestamp_str,
            confidence=confidence,
            state_name=state_name,
            image_bytes=image_bytes,
            image_path=image_path,
            clip_path=clip_path,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("Notification queue full! Dropping notification.")

    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                item: NotificationItem = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            success = self._send_http(item)
            if success:
                self.total_dispatched += 1
                self.last_status = f"Sent: {item.plate_number} ({item.event_type})"
            else:
                item.attempts += 1
                if item.attempts < self.max_retries:
                    time.sleep(1.0 * (2 ** (item.attempts - 1)))
                    self._queue.put(item)
                else:
                    self.total_failed += 1
                    self.last_status = f"Failed after {self.max_retries} attempts: {item.plate_number}"
                    logger.error(f"Permanently failed to deliver notification for {item.plate_number}")

            self._queue.task_done()

    @staticmethod
    def _load_photo_bytes(item: NotificationItem) -> Optional[bytes]:
        if item.image_bytes:
            return item.image_bytes
        if item.image_path:
            try:
                p = Path(item.image_path)
                if not p.is_absolute():
                    p = settings.CAPTURES_DIR.parent / p
                if p.exists() and p.is_file():
                    return p.read_bytes()
            except Exception as e:
                logger.debug(f"Could not read notification image {item.image_path}: {e}")
        return None

    @staticmethod
    def _clean_header(val: str) -> str:
        """Sanitize header values so newlines/control chars never break HTTP requests."""
        return " ".join(str(val).replace("\r", " ").replace("\n", " | ").split())

    def _send_http(self, item: NotificationItem) -> bool:
        """
        Send notification to ntfy.sh:
        - With clip photo: HTTP PUT with binary JPEG body + `Filename` & `Message` headers.
        - Without photo (or fallback): HTTP POST with plain UTF-8 body.
        """
        url = f"{self.server}/{self.topic}"
        is_entry = item.event_type.upper() == "IN"

        title = f"Vehicle Entry: {item.plate_number}" if is_entry else f"Vehicle Exit: {item.plate_number}"
        tag = "car,white_check_mark" if is_entry else "car,arrow_right"
        clip_name = Path(item.clip_path).name if item.clip_path else None

        lines = [
            f"Registration: {item.plate_number}",
            f"Status: {item.event_type}",
            f"Time: {item.timestamp_str}",
            f"Confidence: {item.confidence * 100:.1f}%",
        ]
        if item.state_name:
            lines.append(f"State: {item.state_name}")
        if clip_name:
            lines.append(f"Clip Saved: {clip_name}")

        headers = {
            "Title": self._clean_header(title),
            "Priority": "default",
            "Tags": tag,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        # 1. Try photo attachment via HTTP PUT when available & enabled
        photo_bytes = self._load_photo_bytes(item) if self.attach_photo else None
        if photo_bytes:
            safe_plate = "".join(c for c in item.plate_number if c.isalnum()) or "plate"
            put_headers = dict(headers)
            put_headers["Filename"] = f"{safe_plate}_{item.event_type.upper()}.jpg"
            put_headers["Message"] = self._clean_header(" | ".join(lines))
            try:
                resp = requests.put(url, data=photo_bytes, headers=put_headers, timeout=self.timeout)
                if resp.status_code == 200:
                    self.total_photos_sent += 1
                    logger.info(
                        f"Photo notification sent to {url} for {item.plate_number} "
                        f"({item.event_type}, {len(photo_bytes):,} bytes)."
                    )
                    return True
                logger.warning(
                    f"ntfy.sh PUT photo upload returned {resp.status_code}: {resp.text[:160]}. "
                    "Falling back to text notification."
                )
            except Exception as e:
                logger.warning(f"Network error sending photo notification to ntfy.sh: {e}")
                return False

        # 2. Text-only notification via HTTP POST
        body = "\n".join(lines)
        try:
            resp = requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                logger.info(f"Notification sent to {url} for vehicle {item.plate_number} ({item.event_type}).")
                return True
            logger.warning(f"ntfy.sh responded with status {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            logger.warning(f"Network error sending notification to ntfy.sh: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "attach_photo": self.attach_photo,
            "server": self.server,
            "topic": self.topic,
            "queue_pending": self._queue.qsize(),
            "total_dispatched": self.total_dispatched,
            "total_photos_sent": self.total_photos_sent,
            "total_failed": self.total_failed,
            "last_status": self.last_status,
        }
