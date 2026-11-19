"""
Lightweight ONNX Runtime YOLO Inference Engine
==============================================
Replaces PyTorch and Ultralytics at inference time using ONLY:
  - onnxruntime (prioritizing DirectML on Windows GPUs, with automatic CPU fallback)
  - opencv-python
  - numpy

Implements exact Ultralytics YOLOv8 letterbox preprocessing, FP32 ONNX session
execution, class-aware NMS, and inverse coordinate scaling for numerical parity
with the original PyTorch models.
"""

import ast
import os
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("anpr.inference_onnx")

# Strict Windows deployment provider priority: DirectML GPU first, then CPU
PREFERRED_PROVIDERS: Tuple[str, ...] = ("DmlExecutionProvider", "CPUExecutionProvider")


def select_onnx_providers(
    preferred: Sequence[str] = PREFERRED_PROVIDERS,
) -> List[str]:
    """
    Resolve ONNX Runtime execution providers prioritizing
    ['DmlExecutionProvider', 'CPUExecutionProvider'] so standard Windows
    Intel/AMD/NVIDIA GPUs and CPUs run cleanly without CUDA.
    """
    available = set(ort.get_available_providers())
    selected = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in selected and "CPUExecutionProvider" in available:
        selected.append("CPUExecutionProvider")
    if not selected:
        selected = list(ort.get_available_providers()) or ["CPUExecutionProvider"]
    return selected


def build_session_options(intra_threads: Optional[int] = None) -> ort.SessionOptions:
    """Create optimized ONNX Runtime SessionOptions without thread spin-wait contention."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = intra_threads or min(4, max(1, (os.cpu_count() or 4) // 2))
    so.inter_op_num_threads = 1
    try:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    except Exception:
        pass
    return so


@dataclass
class ONNXDetection:
    bbox: Tuple[int, int, int, int]          # Clipped integer (x1, y1, x2, y2)
    bbox_float: Tuple[float, float, float, float]  # Exact sub-pixel (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str = ""


class ONNXYOLODetector:
    """
    Framework-free YOLOv8 detector powered exclusively by ONNX Runtime,
    OpenCV, and NumPy. Safe for concurrent calls across threads.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Optional[Sequence[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[Dict[int, str]] = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self._lock = threading.Lock()

        chosen_providers = select_onnx_providers(providers or PREFERRED_PROVIDERS)
        sess_options = build_session_options()

        logger.info(
            f"Loading ONNX model '{self.model_path.name}' with providers={chosen_providers}..."
        )
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=chosen_providers,
        )
        self.active_providers: List[str] = self.session.get_providers()

        inp = self.session.get_inputs()[0]
        self.input_name: str = inp.name
        self.output_name: str = self.session.get_outputs()[0].name

        # Read static input spatial dimensions (default 640x640 if dynamic)
        shape = inp.shape
        self.input_height: int = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 640
        self.input_width: int = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 640

        # Load class names from ONNX metadata (Ultralytics embeds `names`)
        self.names: Dict[int, str] = class_names or self._parse_metadata_names()

        # Pre-warm session so first live frame has low latency
        dummy = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        logger.info(
            f"ONNX model '{self.model_path.name}' ready "
            f"(input={self.input_width}x{self.input_height}, active_providers={self.active_providers})."
        )

    def _parse_metadata_names(self) -> Dict[int, str]:
        try:
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names")
            if raw:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return {int(k): str(v) for k, v in parsed.items()}
                if isinstance(parsed, list):
                    return {i: str(v) for i, v in enumerate(parsed)}
        except Exception as e:
            logger.debug(f"Could not parse ONNX class names metadata: {e}")
        return {}

    # ----------------------------------------------------------- preprocessing
    @staticmethod
    def letterbox(
        image: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Ultralytics-compatible aspect-preserving letterbox resize.
        Returns (padded_bgr, scale_ratio, (pad_left, pad_top)).
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = new_shape

        ratio = min(target_h / float(orig_h), target_w / float(orig_w))
        new_unpad_w = int(round(orig_w * ratio))
        new_unpad_h = int(round(orig_h * ratio))

        dw = (target_w - new_unpad_w) / 2.0
        dh = (target_h - new_unpad_h) / 2.0

        if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, ratio, (left, top)

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        padded, ratio, (pad_left, pad_top) = self.letterbox(
            frame, (self.input_height, self.input_width)
        )
        # BGR -> RGB, HWC -> CHW, normalize [0, 1] FP32, add batch dimension
        rgb = padded[:, :, ::-1]
        blob = np.transpose(rgb, (2, 0, 1))
        blob = np.ascontiguousarray(blob, dtype=np.float32) * (1.0 / 255.0)
        return blob[np.newaxis, ...], ratio, (pad_left, pad_top)

    # ---------------------------------------------------------- postprocessing
    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
        classes: Optional[Sequence[int]] = None,
    ) -> List[ONNXDetection]:
        """Decode YOLOv8 [1, 4+C, N] output tensor with class-aware NMS."""
        # Handle [1, 4+C, N] -> [N, 4+C]
        preds = raw_output[0] if raw_output.ndim == 3 else raw_output
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        if preds.size == 0 or preds.shape[1] < 5:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        if class_scores.shape[1] == 1:
            class_ids = np.zeros(len(preds), dtype=np.int32)
            scores = class_scores[:, 0]
        else:
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(len(preds)), class_ids]

        mask = scores >= conf_threshold
        if classes is not None:
            allowed = set(int(c) for c in classes)
            mask &= np.isin(class_ids, list(allowed))

        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Convert center xywh -> top-left xywh for OpenCV NMS + xyxy for scaling
        half_w = boxes_xywh[:, 2] * 0.5
        half_h = boxes_xywh[:, 3] * 0.5
        x1 = boxes_xywh[:, 0] - half_w
        y1 = boxes_xywh[:, 1] - half_h
        x2 = boxes_xywh[:, 0] + half_w
        y2 = boxes_xywh[:, 1] + half_h

        # Class-aware NMS (matches Ultralytics / torchvision.ops.batched_nms)
        max_wh = 7680.0
        offsets = class_ids.astype(np.float32) * max_wh
        nms_boxes = np.stack(
            [x1 + offsets, y1 + offsets, boxes_xywh[:, 2], boxes_xywh[:, 3]],
            axis=1,
        ).tolist()
        nms_scores = scores.astype(float).tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, iou_threshold)
        if len(indices) == 0:
            return []

        orig_h, orig_w = orig_shape
        pad_left, pad_top = pad
        inv_ratio = 1.0 / max(ratio, 1e-9)

        detections: List[ONNXDetection] = []
        for idx in np.array(indices).flatten():
            fx1 = float(np.clip((x1[idx] - pad_left) * inv_ratio, 0.0, orig_w))
            fy1 = float(np.clip((y1[idx] - pad_top) * inv_ratio, 0.0, orig_h))
            fx2 = float(np.clip((x2[idx] - pad_left) * inv_ratio, 0.0, orig_w))
            fy2 = float(np.clip((y2[idx] - pad_top) * inv_ratio, 0.0, orig_h))

            ix1 = max(0, min(orig_w - 1, int(round(fx1))))
            iy1 = max(0, min(orig_h - 1, int(round(fy1))))
            ix2 = max(0, min(orig_w, int(round(fx2))))
            iy2 = max(0, min(orig_h, int(round(fy2))))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            cid = int(class_ids[idx])
            detections.append(
                ONNXDetection(
                    bbox=(ix1, iy1, ix2, iy2),
                    bbox_float=(fx1, fy1, fx2, fy2),
                    confidence=float(scores[idx]),
                    class_id=cid,
                    class_name=self.names.get(cid, str(cid)),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # --------------------------------------------------------------- public API
    def predict(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Sequence[int]] = None,
    ) -> List[ONNXDetection]:
        """Run thread-safe ONNX Runtime inference on a BGR OpenCV frame."""
        if frame is None or frame.size == 0:
            return []

        use_conf = self.conf_threshold if conf is None else float(conf)
        use_iou = self.iou_threshold if iou is None else float(iou)

        blob, ratio, pad = self._preprocess(frame)
        with self._lock:
            outputs = self.session.run([self.output_name], {self.input_name: blob})

        return self._postprocess(
            outputs[0],
            orig_shape=frame.shape[:2],
            ratio=ratio,
            pad=pad,
            conf_threshold=use_conf,
            iou_threshold=use_iou,
            classes=classes,
        )
