"""
Vehicle Detector (ONNX Runtime Primary / PyTorch Fallback)
==========================================================
Detects cars, motorcycles, buses, and trucks using the exported FP32 YOLOv8n
ONNX model via ONNX Runtime (DirectML / CPU) without requiring PyTorch at runtime.
Associates detected license plate regions with their corresponding vehicles.
"""

import logging
import threading
from typing import List, Tuple, Optional
import numpy as np
from app.core.config import settings
from app.core.device import resolve_device
from app.core.exceptions import WeightsNotFoundError, ModelError

logger = logging.getLogger("anpr.detector.vehicle")


class VehicleDetection:
    # Target COCO class IDs: car (2), motorcycle (3), bus (5), truck (7)
    VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        class_id: int,
        tracking_id: Optional[int] = None,
    ):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = self.VEHICLE_CLASSES.get(class_id, "vehicle")
        self.tracking_id = tracking_id

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
    def centroid(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class VehicleDetector:
    _instance: Optional["VehicleDetector"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.conf_threshold = settings.VEHICLE_CONFIDENCE
        self.iou_threshold = settings.IOU_THRESHOLD
        self.target_classes = list(VehicleDetection.VEHICLE_CLASSES.keys())
        self._lock = threading.Lock()

        chosen_backend = (backend or settings.INFERENCE_BACKEND or "onnx").strip().lower()
        onnx_path = settings.resolve_model_path(
            weights_path if (weights_path and str(weights_path).lower().endswith(".onnx"))
            else settings.VEHICLE_ONNX_MODEL_PATH
        )
        pt_path = settings.resolve_model_path(
            weights_path if (weights_path and str(weights_path).lower().endswith(".pt"))
            else settings.VEHICLE_MODEL_PATH
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
                    f"Vehicle ONNX model not found at '{onnx_path}'. Run `python export_to_onnx.py`."
                )
            try:
                from app.vision.inference_onnx import ONNXYOLODetector
                self.onnx_detector = ONNXYOLODetector(
                    model_path=onnx_path,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                )
                self.device = self.onnx_detector.active_providers[0]
                logger.info(f"VehicleDetector using ONNX Runtime ({self.device}) -> {onnx_path.name}")
            except Exception as e:
                logger.error(f"Failed to initialize ONNX Vehicle Detector: {e}")
                raise ModelError(f"Vehicle ONNX initialization failure: {e}")
        else:
            if not pt_path.exists():
                raise WeightsNotFoundError(f"Vehicle detector weights not found at: {pt_path}")
            resolved_dev, _ = resolve_device(device or settings.DEVICE)
            self.device = resolved_dev
            logger.info(f"Loading PyTorch Vehicle Detector from '{pt_path}' on '{self.device}'...")
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(pt_path))
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                for _ in range(2):
                    self.model.predict(dummy, device=self.device, verbose=False)
                logger.info("PyTorch Vehicle Detector loaded and pre-warmed successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize PyTorch Vehicle Detector: {e}")
                raise ModelError(f"Vehicle detector initialization failure: {e}")

    @classmethod
    def get_instance(cls) -> "VehicleDetector":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Detect vehicles in frame (thread-safe)."""
        if frame is None or frame.size == 0:
            return []

        detections: List[VehicleDetection] = []
        if self.onnx_detector is not None:
            try:
                raw_dets = self.onnx_detector.predict(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=self.target_classes,
                )
            except Exception as e:
                logger.error(f"Error during ONNX vehicle detection inference: {e}")
                return []

            for d in raw_dets:
                x1, y1, x2, y2 = d.bbox
                if (x2 - x1) > 40 and (y2 - y1) > 40:
                    detections.append(VehicleDetection((x1, y1, x2, y2), d.confidence, d.class_id))
            return detections

        # PyTorch / Ultralytics fallback path
        try:
            with self._lock:
                results = self.model.predict(
                    source=frame,
                    classes=self.target_classes,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False,
                )
        except Exception as e:
            logger.error(f"Error during PyTorch vehicle detection inference: {e}")
            return []

        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        img_h, img_w = frame.shape[:2]
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].cpu().item())
            cls_id = int(boxes.cls[i].cpu().item())
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

            x1 = max(0, min(img_w - 1, int(xyxy[0])))
            y1 = max(0, min(img_h - 1, int(xyxy[1])))
            x2 = max(0, min(img_w, int(xyxy[2])))
            y2 = max(0, min(img_h, int(xyxy[3])))

            if (x2 - x1) > 40 and (y2 - y1) > 40:
                detections.append(VehicleDetection((x1, y1, x2, y2), conf, cls_id))

        return detections

    @staticmethod
    def associate_plate_to_vehicle(
        plate_bbox: Tuple[int, int, int, int],
        vehicles: List[VehicleDetection],
    ) -> Optional[VehicleDetection]:
        """Find the vehicle that encloses or is closest to the plate bounding box."""
        if not vehicles:
            return None

        px1, py1, px2, py2 = plate_bbox
        p_cx = (px1 + px2) // 2
        p_cy = (py1 + py2) // 2

        containing = [v for v in vehicles if v.contains_point(p_cx, p_cy)]
        if containing:
            return min(containing, key=lambda v: v.width * v.height)

        min_dist = float("inf")
        best_v = None
        for v in vehicles:
            vx, vy = v.centroid
            dist = ((vx - p_cx) ** 2 + (vy - p_cy) ** 2) ** 0.5
            if dist < max(v.width, v.height) and dist < min_dist:
                min_dist = dist
                best_v = v

        return best_v
