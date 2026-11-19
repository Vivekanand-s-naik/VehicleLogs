"""
Top-level convenience facade for the ONNX Runtime inference engine.
Uses only onnxruntime, opencv-python, and numpy (zero PyTorch / Ultralytics imports).
"""

from app.vision.inference_onnx import (
    ONNXDetection,
    ONNXYOLODetector,
    PREFERRED_PROVIDERS,
    build_session_options,
    select_onnx_providers,
)

__all__ = [
    "ONNXDetection",
    "ONNXYOLODetector",
    "PREFERRED_PROVIDERS",
    "build_session_options",
    "select_onnx_providers",
]
