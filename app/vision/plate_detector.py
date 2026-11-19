"""
License Plate Detector (ONNX Runtime Primary / PyTorch Fallback)
================================================================
Runs the exported FP32 YOLOv8 license plate ONNX model via ONNX Runtime
(prioritizing DirectML GPU on Windows, falling back cleanly to CPU) with
zero PyTorch/Ultralytics runtime dependency when INFERENCE_BACKEND='onnx'.
Retains optional PyTorch/Ultralytics support for parity benchmarking & dev.
"""

import logging
import threading
from pathlib import Path
from typing import List, Tuple, Optional
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

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.conf_threshold = settings.DETECTION_CONFIDENCE
        self.iou_threshold = settings.IOU_THRESHOLD
        self.imgsz = settings.PLATE_DETECT_IMGSZ
        self._lock = threading.Lock()

        chosen_backend = (backend or settings.INFERENCE_BACKEND or "onnx").strip().lower()
        onnx_path = settings.resolve_model_path(
            weights_path if (weights_path and str(weights_path).lower().endswith(".onnx"))
            else settings.PLATE_ONNX_MODEL_PATH
        )
        pt_path = settings.resolve_model_path(
            weights_path if (weights_path and str(weights_path).lower().endswith(".pt"))
            else settings.YOLO_MODEL_PATH
        )

        use_onnx = (
            chosen_backend == "onnx"
            or (chosen_backend == "auto" and onnx_path.exists())
            or (weights_path is not None and str(weights_path).lower().endswith(".onnx"))
        )

        self.backend = "onnx" if use_onnx else "pytorch"
        self.onnx_detector = None
        self.model = None
        self.device = "onnx"

        if use_onnx:
            if not onnx_path.exists():
                raise WeightsNotFoundError(
                    f"License plate ONNX model not found at '{onnx_path}'. Run `python export_to_onnx.py`."
                )
            try:
                from app.vision.inference_onnx import ONNXYOLODetector
                self.onnx_detector = ONNXYOLODetector(
                    model_path=onnx_path,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    class_names={0: "license_plate"},
                )
                self.device = self.onnx_detector.active_providers[0]
                logger.info(f"PlateDetector using ONNX Runtime ({self.device}) -> {onnx_path.name}")
            except Exception as e:
                logger.error(f"Failed to initialize ONNX License Plate Detector: {e}")
                raise ModelError(f"Plate ONNX initialization failure: {e}")
        else:
            if not pt_path.exists():
                raise WeightsNotFoundError(f"License plate model weights not found at: {pt_path}")
            resolved_dev, _ = resolve_device(device or settings.DEVICE)
            self.device = resolved_dev
            logger.info(f"Loading PyTorch License Plate Detector from '{pt_path}' on '{self.device}'...")
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(pt_path))
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                for _ in range(2):
                    self.model.predict(dummy, device=self.device, imgsz=self.imgsz, verbose=False)
                logger.info("PyTorch License Plate Detector loaded and pre-warmed successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize PyTorch License Plate Detector: {e}")
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

        use_conf = conf if conf is not None else self.conf_threshold
        detections: List[PlateDetection] = []

        if self.onnx_detector is not None:
            try:
                raw_dets = self.onnx_detector.predict(
                    frame, conf=use_conf, iou=self.iou_threshold
                )
            except Exception as e:
                logger.error(f"Error during ONNX plate detection inference: {e}")
                return []

            for d in raw_dets:
                x1, y1, x2, y2 = d.bbox
                w, h = x2 - x1, y2 - y1
                if w < 14 or h < 8:
                    continue
                aspect = w / max(1, h)
                if aspect < 0.80 or aspect > 8.0:
                    continue
                detections.append(PlateDetection((x1, y1, x2, y2), d.confidence, d.class_id))
            return detections

        # PyTorch / Ultralytics fallback path
        use_imgsz = imgsz or self.imgsz
        try:
            with self._lock:
                results = self.model.predict(
                    source=frame,
                    conf=use_conf,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=use_imgsz,
                    verbose=False,
                )
        except Exception as e:
            logger.error(f"Error during PyTorch plate detection inference: {e}")
            return []

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
            w, h = x2 - x1, y2 - y1
            if w < 14 or h < 8:
                continue
            aspect = w / max(1, h)
            if aspect < 0.80 or aspect > 8.0:
                continue
            detections.append(PlateDetection((x1, y1, x2, y2), det_conf, cls_id))

        return detections
