"""
Application Configuration using Pydantic Settings
=================================================
Validates environment variables and provides structured system settings.
"""

import sys
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


def _resolve_base_dir() -> Path:
    """Resolve workspace root for both source runs and Nuitka standalone builds."""
    if "__compiled__" in globals() or getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


BASE_DIR = _resolve_base_dir()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project Information
    APP_NAME: str = "Production ANPR & Vehicle State Manager"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Device & Hardware Acceleration
    # Options: "auto" (dynamic probe for CUDA -> MPS -> CPU), "cuda", "cuda:0", "cpu"
    DEVICE: str = Field(default="auto", description="Compute device for YOLO and EasyOCR ('auto', 'cuda', 'cpu')")

    # Camera Configuration
    CAMERA_INDEX: int = Field(default=0, description="Default USB webcam index")
    CAMERA_SOURCE: str = Field(default="0", description="Camera index (e.g. '0') or video file path or 'synthetic'")
    CAMERA_WIDTH: int = Field(default=1280, description="Camera capture frame width")
    CAMERA_HEIGHT: int = Field(default=720, description="Camera capture frame height")
    CAMERA_FPS: int = Field(default=30, description="Target capture FPS")
    CAMERA_BUFFER_SIZE: int = Field(default=1, description="Buffer size for latest-frame capture")
    CAMERA_RECONNECT_MAX_ATTEMPTS: int = Field(default=10, description="Max reconnection attempts")
    CAMERA_RECONNECT_INITIAL_DELAY: float = Field(default=1.0, description="Initial retry delay in seconds")
    CAMERA_RECONNECT_BACKOFF_FACTOR: float = Field(default=1.8, description="Exponential backoff factor")
    CAMERA_RECONNECT_MAX_DELAY: float = Field(default=30.0, description="Max retry delay in seconds")

    # Vision & Models
    INFERENCE_BACKEND: str = Field(
        default="onnx",
        description="Inference runtime backend: 'onnx' (lightweight DirectML/CPU, no PyTorch), 'pytorch', or 'auto'"
    )
    YOLO_MODEL_PATH: str = Field(default="models/license_plate_detector.pt", description="Path to license plate PyTorch weights")
    VEHICLE_MODEL_PATH: str = Field(default="models/vehicle_detector.pt", description="Path to vehicle PyTorch weights")
    PLATE_ONNX_MODEL_PATH: str = Field(default="models/license_plate_detector.onnx", description="Path to license plate ONNX model")
    VEHICLE_ONNX_MODEL_PATH: str = Field(default="models/vehicle_detector.onnx", description="Path to vehicle ONNX model")
    DETECTION_CONFIDENCE: float = Field(default=0.25, description="Min confidence for plate detection (catches distant plates too)")
    VEHICLE_CONFIDENCE: float = Field(default=0.35, description="Min confidence for vehicle detection")
    IOU_THRESHOLD: float = Field(default=0.45, description="IoU threshold for NMS")
    FRAME_SKIP: int = Field(default=1, description="Process inference every Nth frame (1 = every frame)")
    PLATE_PADDING_RATIO: float = Field(default=0.08, description="Fractional padding around plate bbox crop")

    PLATE_DETECT_IMGSZ: int = Field(default=640, description="Live YOLO inference size for plate detection")
    CLIP_DETECT_IMGSZ: int = Field(default=960, description="High-res YOLO inference size used when processing recorded clips")
    VEHICLE_DETECT_INTERVAL: int = Field(default=3, description="Run vehicle detector every N processed frames")
    USE_HALF_PRECISION: bool = Field(default=True, description="Use FP16 YOLO inference on CUDA")

    # Event-Triggered Video Clip Recording & Deferred Processing
    CLIP_RECORDING_ENABLED: bool = Field(default=True, description="Record video clip on plate detection and process clip asynchronously")
    RECORD_PRE_ROLL_SECONDS: float = Field(default=0.8, description="Seconds of pre-roll frames included before plate trigger")
    RECORD_POST_ROLL_SECONDS: float = Field(default=1.5, description="Stop recording after plate is absent for this many seconds")
    RECORD_MAX_DURATION: float = Field(default=4.5, description="Max clip duration (seconds) before finalizing and processing")
    RECORD_MIN_FRAMES: int = Field(default=3, description="Minimum recorded frames required to process a clip")
    CLIP_TOP_K_CROPS: int = Field(default=12, description="Number of sharpest plate crops evaluated per recorded clip")
    CLIP_QUEUE_SIZE: int = Field(default=8, description="Max pending recorded video clips awaiting processing")
    KEEP_ALL_CLIPS: bool = Field(default=True, description="Keep all recorded video clips in captures/events/ instead of discarding unreadable/cooldown clips")

    # OCR & Validation
    OCR_CONFIDENCE: float = Field(default=0.58, description="Minimum OCR score threshold")
    OCR_EARLY_EXIT_CONFIDENCE: float = Field(default=0.80, description="Stop trying preprocessing variants above this confidence")
    OCR_DECODER: str = Field(default="greedy", description="EasyOCR decoder: 'greedy' (fast) or 'beamsearch'")
    OCR_MIN_PLATE_WIDTH: int = Field(default=14, description="Minimum plate crop width (px) — accepts distant plates")
    OCR_INTERVAL_PER_TRACK: float = Field(default=0.08, description="Minimum seconds between OCR attempts for one plate track")
    OCR_MAX_ATTEMPTS_PER_TRACK: int = Field(default=25, description="Give up OCR on a plate track after N attempts")
    OCR_QUEUE_SIZE: int = Field(default=4, description="Bounded OCR job queue size")
    RECOGNITION_WINDOW: int = Field(default=8, description="Number of multi-frame observations for voting")
    RECOGNITION_MIN_VOTES: int = Field(default=2, description="Agreeing reads required to confirm a plate")
    RECOGNITION_COOLDOWN: int = Field(default=60, description="Seconds before an identical vehicle can re-trigger state")
    PRESENCE_TIMEOUT: float = Field(default=8.0, description="Seconds a plate must be unseen before it can trigger a new event")

    # Event Strategy
    # 'toggle': flips state between IN and OUT on confirmed re-appearance after cooldown
    # 'line_crossing': uses tracked centroid crossing virtual line (top-down = IN, bottom-up = OUT)
    EVENT_STRATEGY: str = Field(default="toggle", description="Vehicle entry/exit detection strategy ('toggle', 'line_crossing')")
    VIRTUAL_LINE_Y: float = Field(default=0.55, description="Normalized Y position for virtual line crossing (0.0 to 1.0)")

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./vehicle_logs.db", description="SQLAlchemy database URL")
    DATABASE_TIMEOUT: float = Field(default=30.0, description="SQLite lock timeout in seconds")
    DATABASE_WAL_MODE: bool = Field(default=True, description="Enable SQLite Write-Ahead Logging")

    # Notifications (ntfy.sh)
    NTFY_SERVER: str = Field(default="https://ntfy.sh", description="ntfy server URL")
    NTFY_TOPIC: str = Field(default="anpr_vehicle_events_demo", description="ntfy topic name")
    NTFY_TOKEN: Optional[str] = Field(default=None, description="Optional bearer token for private ntfy topics")
    NOTIFICATIONS_ENABLED: bool = Field(default=True, description="Enable mobile push notifications")
    NOTIFICATION_ATTACH_PHOTO: bool = Field(default=True, description="Attach clip vehicle + plate photo to ntfy.sh notifications")
    NTFY_TIMEOUT: float = Field(default=10.0, description="HTTP timeout for ntfy requests (includes photo upload)")
    NTFY_MAX_RETRIES: int = Field(default=3, description="Max retries for notification delivery")

    # Captures & Archiving
    CAPTURES_DIR: Path = Field(default=BASE_DIR / "captures")
    VEHICLES_DIR: Path = Field(default=BASE_DIR / "captures" / "vehicles")
    PLATES_DIR: Path = Field(default=BASE_DIR / "captures" / "plates")
    EVENTS_DIR: Path = Field(default=BASE_DIR / "captures" / "events")
    IMAGE_RETENTION_DAYS: int = Field(default=90, description="Days to retain captured images")

    # Logging & Paths
    LOGS_DIR: Path = Field(default=BASE_DIR / "logs")
    LOG_FILE_PATH: Path = Field(default=BASE_DIR / "logs" / "anpr.log")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_MAX_BYTES: int = Field(default=10 * 1024 * 1024)  # 10 MB
    LOG_BACKUP_COUNT: int = Field(default=5)

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_KEY: Optional[str] = Field(default=None, description="Optional API key for administrative endpoints")

    # Disclosed Licensing & Remote Kill Switch
    LICENSE_CHECK_ENABLED: bool = Field(
        default=False,
        description="Enable startup & periodic HTTPS license check (enabled in packaged Windows builds)"
    )
    LICENSE_SERVER_URL: str = Field(
        default="https://my-license-server.example.com/api/v1/validate",
        description="HTTPS endpoint for startup/periodic license validation"
    )
    LICENSE_KEY: str = Field(default="", description="Customer license key assigned by licensor")
    LICENSE_SUPPORT_CONTACT: str = Field(
        default="support@yourcompany.example.com",
        description="Support contact displayed if license is inactive"
    )
    LICENSE_GRACE_HOURS: float = Field(default=72.0, description="Offline grace window (hours) after last active check")
    LICENSE_TIMEOUT_SECONDS: float = Field(default=6.0, description="HTTPS timeout for license server")
    LICENSE_MAX_RETRIES: int = Field(default=3, description="Retry attempts on transient network failures")
    LICENSE_RECHECK_INTERVAL_SECONDS: int = Field(default=3600, description="Periodic license check interval (seconds)")
    LICENSE_LOG_PATH: Path = Field(default=BASE_DIR / "logs" / "license.log")
    LICENSE_CACHE_PATH: Path = Field(default=BASE_DIR / "logs" / ".license_state.json")

    def resolve_model_path(self, relative_path: str) -> Path:
        """Resolve model path relative to BASE_DIR if not absolute."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return BASE_DIR / p


settings = Settings()
