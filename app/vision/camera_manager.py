"""
Camera Manager: Threaded Video Capture with Exponential Backoff
===============================================================
Acquires webcam/video stream on a dedicated thread, maintains a latest-frame
buffer (zero lag), automatically detects disconnects, and reconnects with
exponential backoff. Supports synthetic frame generation for headless tests.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.core.exceptions import CameraError, CameraDisconnectedError

logger = logging.getLogger("anpr.camera")


def enumerate_video_devices(
    max_index: int = 6,
    active_source: Optional[str] = None,
    active_resolution: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Enumerate available local camera devices via cv2.VideoCapture index probing
    without disrupting a currently open active camera index.
    Always includes the built-in Synthetic Test Feed as a fallback option.
    """
    devices: List[Dict[str, str]] = []
    active_idx: Optional[int] = None
    if active_source is not None:
        try:
            active_idx = int(str(active_source).strip())
        except (ValueError, TypeError):
            active_idx = None

    use_dshow = hasattr(cv2, "CAP_DSHOW")
    for idx in range(max_index):
        if active_idx is not None and idx == active_idx:
            res_suffix = f" ({active_resolution})" if active_resolution else " (Active)"
            devices.append({
                "source": str(idx),
                "label": f"Camera {idx}{res_suffix}",
                "kind": "webcam",
            })
            continue

        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                res_str = f" ({w}x{h})" if (w > 0 and h > 0) else ""
                devices.append({
                    "source": str(idx),
                    "label": f"Camera {idx}{res_str}",
                    "kind": "webcam",
                })
        except Exception as exc:
            logger.debug(f"Camera probe failed for index {idx}: {exc}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    devices.append({
        "source": "synthetic",
        "label": "Synthetic Test Feed (Gate 1)",
        "kind": "synthetic",
    })
    return devices


class CameraManager:
    def __init__(
        self,
        source: Optional[str] = None,
        width: int = settings.CAMERA_WIDTH,
        height: int = settings.CAMERA_HEIGHT,
        fps: int = settings.CAMERA_FPS
    ):
        self.source = source if source is not None else settings.CAMERA_SOURCE
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._last_frame_time: float = 0.0
        self._is_connected: bool = False
        self._is_synthetic: bool = (str(self.source).lower() == "synthetic")

        # Telemetry
        self.fps_actual: float = 0.0
        self.total_frames_captured: int = 0
        self.consecutive_read_failures: int = 0
        self.reconnect_attempts: int = 0

        # Synthetic generator variables
        self._synth_tick = 0

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def start(self):
        """Start background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Camera thread is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
        self._thread.start()
        logger.info(f"Camera manager started with source='{self.source}'.")

    def stop(self):
        """Signal capture thread to stop and release resources."""
        logger.info("Stopping camera capture thread...")
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release_capture()
        self._is_connected = False
        logger.info("Camera manager stopped.")

    def _open_capture(self) -> bool:
        """Attempt to open VideoCapture source."""
        if self._is_synthetic:
            self._is_connected = True
            logger.info("Synthetic frame generator enabled for camera.")
            return True

        self._release_capture()
        src = self.source
        # Check if source is integer camera index
        try:
            src = int(src)
        except (ValueError, TypeError):
            pass

        logger.info(f"Opening camera source: {src} ...")
        # On Windows, try cv2.CAP_DSHOW for USB webcams if integer index
        if isinstance(src, int) and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            logger.warning(f"Could not open camera source: {src}")
            cap.release()
            return False

        # Set properties. MJPG lets most USB webcams deliver 720p at 30 FPS
        # (raw YUY2 is often limited to ~10 FPS at 720p over USB 2.0).
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(
            f"Camera negotiated {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
            f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} FPS"
        )

        self._cap = cap
        self._is_connected = True
        self.consecutive_read_failures = 0
        logger.info(f"Camera source {src} successfully connected.")
        return True

    def _release_capture(self):
        """Release underlying OpenCV VideoCapture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.debug(f"Error releasing VideoCapture: {e}")
            self._cap = None

    def _capture_loop(self):
        """Worker thread loop maintaining latest frame buffer with exponential backoff reconnects."""
        retry_delay = settings.CAMERA_RECONNECT_INITIAL_DELAY

        while not self._stop_event.is_set():
            if not self._is_connected:
                # Attempt connection with backoff
                if self.reconnect_attempts >= settings.CAMERA_RECONNECT_MAX_ATTEMPTS:
                    logger.error(f"Exceeded max reconnect attempts ({settings.CAMERA_RECONNECT_MAX_ATTEMPTS}).")
                    # If real camera fails, fallback to synthetic mode so system continues operating
                    logger.warning("Switching automatically to synthetic fallback video feed.")
                    self._is_synthetic = True
                    self._is_connected = True
                    self.reconnect_attempts = 0
                    continue

                self.reconnect_attempts += 1
                logger.info(f"Reconnection attempt {self.reconnect_attempts} in {retry_delay:.1f}s...")
                time.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * settings.CAMERA_RECONNECT_BACKOFF_FACTOR,
                    settings.CAMERA_RECONNECT_MAX_DELAY
                )

                if self._open_capture():
                    retry_delay = settings.CAMERA_RECONNECT_INITIAL_DELAY
                    self.reconnect_attempts = 0
                continue

            # Connected: read next frame
            if self._is_synthetic:
                frame = self._generate_synthetic_frame()
                ret = True
                time.sleep(1.0 / max(1, self.fps))
            else:
                ret, frame = self._cap.read()

            if not ret or frame is None:
                self.consecutive_read_failures += 1
                logger.warning(f"Failed to read frame (failure count: {self.consecutive_read_failures})")
                if self.consecutive_read_failures > 5:
                    logger.error("Camera disconnected or stream ended.")
                    self._is_connected = False
                    self._release_capture()
                time.sleep(0.05)
                continue

            self.consecutive_read_failures = 0
            now = time.time()
            if self._last_frame_time > 0:
                dt = now - self._last_frame_time
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps_actual = (self.fps_actual * 0.9) + (current_fps * 0.1)

            self._last_frame_time = now
            self.total_frames_captured += 1

            # Store in single latest-frame buffer
            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1

        self._release_capture()

    def get_latest_frame_with_id(self) -> Tuple[int, Optional[np.ndarray]]:
        """Return (frame_id, frame copy). frame_id increases with every captured frame."""
        with self._lock:
            if self._latest_frame is None:
                return 0, None
            return self._frame_id, self._latest_frame.copy()

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieve a copy of the latest frame.
        Non-blocking: returns (True, frame) if fresh frame is ready,
        or (False, None) if camera is disconnected.
        """
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def reconnect(self) -> bool:
        """Trigger an immediate reconnect attempt."""
        logger.info("Manual camera reconnect requested.")
        with self._lock:
            self._is_connected = False
            self._release_capture()
            self.reconnect_attempts = 0
        return self._open_capture()

    def switch_source(self, new_source: str) -> bool:
        """
        Live-switch the active video source (camera index, video file path, or 'synthetic')
        without restarting the application.
        """
        new_src = str(new_source).strip()
        logger.info(f"Switching camera source from '{self.source}' -> '{new_src}'...")
        was_running = self._thread is not None and self._thread.is_alive()
        if was_running:
            self.stop()

        self.source = new_src
        settings.CAMERA_SOURCE = new_src
        self._is_synthetic = (new_src.lower() == "synthetic")
        self.reconnect_attempts = 0
        self.consecutive_read_failures = 0
        with self._lock:
            self._latest_frame = None

        opened = self._open_capture()
        if was_running:
            self.start()
        return opened

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generate realistic simulated entrance gate frame with a moving vehicle
        and an Indian license plate ("KA01AB1234") for testing & headless execution.
        """
        self._synth_tick += 1
        h, w = self.height, self.width
        # Dark asphalt background
        frame = np.full((h, w, 3), (35, 35, 35), dtype=np.uint8)

        # Entrance gate road markings
        cv2.line(frame, (w // 4, 0), (w // 4, h), (200, 200, 200), 4)
        cv2.line(frame, (3 * w // 4, 0), (3 * w // 4, h), (200, 200, 200), 4)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 215, 255), 2)  # Center lane yellow

        # Moving car body (approaching gate downwards)
        cycle_len = 120
        phase = (self._synth_tick % cycle_len) / cycle_len
        car_y = int((phase * (h + 200)) - 100)
        car_x = w // 2 - 120
        car_w = 240
        car_h = 160

        if -100 <= car_y <= h:
            # Car body (Metallic Blue)
            cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (180, 70, 30), -1)
            cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (255, 255, 255), 2)
            # Windshield
            cv2.rectangle(frame, (car_x + 20, car_y + 15), (car_x + car_w - 20, car_y + 60), (90, 90, 90), -1)

            # Indian License Plate: White plate with black border
            plate_w = 140
            plate_h = 36
            plate_x = car_x + (car_w - plate_w) // 2
            plate_y = car_y + car_h - plate_h - 10

            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 2)
            # IND blue strip on left
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 18, plate_y + plate_h), (180, 60, 20), -1)
            cv2.putText(frame, "IND", (plate_x + 1, plate_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Registration text: "KA01AB1234"
            cv2.putText(
                frame, "KA01AB1234", (plate_x + 22, plate_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA
            )

        # Header info
        cv2.putText(frame, "FEED: SYNTHETIC TEST STREAM (GATE 1)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
