"""
License Plate Detector (YOLOv8)
===============================
Runs dedicated YOLOv8 license plate detection model with dynamic device execution.
Filters invalid aspect ratios, applies NMS, and returns plate bounding boxes.
"""

import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from app.core.config import settings
from app.core.device import resolve_device
from app.core.exceptions import WeightsNotFoundError, ModelError

logger = logging.getLogger("anpr.detector.plate")


class PlateDetection:
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.tracking_id: Optional[int] = None

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]
    @property
    def width(self) -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)


class PlateDetector:
    _instance: Optional["PlateDetector"] = None
    _instance_lock = threading.Lock()

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        path = settings.resolve_model_path(weights_path or settings.YOLO_MODEL_PATH)
        if not path.exists():
            raise WeightsNotFoundError(f"License plate model weights not found at: {path}")

        resolved_dev, diag = resolve_device(device or settings.DEVICE)
        self.device = resolved_dev
        self.conf_threshold = settings.DETECTION_CONFIDENCE
        self.iou_threshold = settings.IOU_THRESHOLD
        self.imgsz = settings.PLATE_DETECT_IMGSZ
        self.half = bool(settings.USE_HALF_PRECISION and str(self.device).startswith("cuda"))
        self._lock = threading.Lock()

        logger.info(f"Loading License Plate Detector from '{path}' on '{self.device}'...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            for _ in range(2):
                self.model.predict(dummy, device=self.device, imgsz=self.imgsz, verbose=False)
            logger.info("License Plate Detector loaded and pre-warmed successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize License Plate Detector: {e}")
            raise ModelError(f"Plate model initialization failure: {e}")

    @classmethod
    def get_instance(cls) -> "PlateDetector":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def detect(
        self,
        frame: np.ndarray,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
    ) -> List[PlateDetection]:
        """
        Detect license plates in frame (including small/distant plates).
        Thread-safe for concurrent live + clip-processing access.
        """
        if frame is None or frame.size == 0:
            return []

        use_imgsz = imgsz or self.imgsz
        use_conf = conf if conf is not None else self.conf_threshold

        try:
            with self._lock:
                results = self.model.predict(
                    source=frame,
                    conf=use_conf,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=use_imgsz,
                    verbose=False
                )
        except Exception as e:
            logger.error(f"Error during plate detection inference: {e}")
            return []

        detections: List[PlateDetection] = []
        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        img_h, img_w = frame.shape[:2]

        for i in range(len(boxes)):
            det_conf = float(boxes.conf[i].cpu().item())
            cls_id = int(boxes.cls[i].cpu().item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

            x1 = max(0, min(img_w - 1, int(xyxy[0])))
            y1 = max(0, min(img_h - 1, int(xyxy[1])))
            x2 = max(0, min(img_w, int(xyxy[2])))
            y2 = max(0, min(img_h, int(xyxy[3])))

            w = x2 - x1
            h = y2 - y1

            # Allow small/distant plates (down to 14x8px) so clip recording triggers early
            if w < 14 or h < 8:
                continue

            aspect = w / max(1, h)
            if aspect < 0.80 or aspect > 8.0:
                continue

            detections.append(PlateDetection((x1, y1, x2, y2), det_conf, cls_id))

        return detections
