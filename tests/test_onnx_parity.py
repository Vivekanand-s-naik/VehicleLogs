"""
Tests for ONNX Model Export, Provider Selection, Zero-PyTorch Import Isolation,
and PyTorch (.pt) vs ONNX Runtime (.onnx) Numerical Parity.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from inference_onnx import (
    ONNXYOLODetector,
    PREFERRED_PROVIDERS,
    select_onnx_providers,
)
from app.vision.plate_detector import PlateDetector
from app.vision.vehicle_detector import VehicleDetector


def test_select_onnx_providers_prioritizes_directml():
    """Verify DirectML GPU is prioritized ahead of CPU when available on Windows."""
    assert PREFERRED_PROVIDERS == ("DmlExecutionProvider", "CPUExecutionProvider")

    with patch(
        "app.vision.inference_onnx.ort.get_available_providers",
        return_value=["CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"],
    ):
        selected = select_onnx_providers()
        assert selected == ["DmlExecutionProvider", "CPUExecutionProvider"]

    with patch(
        "app.vision.inference_onnx.ort.get_available_providers",
        return_value=["AzureExecutionProvider", "CPUExecutionProvider"],
    ):
        selected_cpu = select_onnx_providers()
        assert selected_cpu == ["CPUExecutionProvider"]


def test_inference_onnx_zero_training_framework_imports():
    """
    Prove that inference_onnx, PlateDetector, and VehicleDetector import cleanly
    when torch, torchvision, ultralytics, and easyocr are completely blocked.
    """
    code = r"""
import sys
for forbidden in ("torch", "torchvision", "ultralytics", "easyocr"):
    sys.modules[forbidden] = None

import inference_onnx
from app.vision.plate_detector import PlateDetector
from app.vision.vehicle_detector import VehicleDetector
from app.core.device import resolve_device

dev, diag = resolve_device("auto")
assert dev in ("dml", "cpu")
assert "onnx_providers" in diag

pd = PlateDetector(backend="onnx")
vd = VehicleDetector(backend="onnx")
assert pd.backend == "onnx" and pd.model is None
assert vd.backend == "onnx" and vd.model is None
print("ISOLATION_OK")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "ISOLATION_OK" in proc.stdout


def test_exported_onnx_graphs_valid_fp32_and_original_pt_preserved():
    """Verify both exported .onnx graphs pass onnx.checker and original .pt files remain intact."""
    import onnx

    for stem in ("license_plate_detector", "vehicle_detector"):
        pt_file = BASE_DIR / "models" / f"{stem}.pt"
        onnx_file = BASE_DIR / "models" / f"{stem}.onnx"

        assert pt_file.exists() and pt_file.stat().st_size > 1_000_000, f"Original {pt_file} missing!"
        assert onnx_file.exists() and onnx_file.stat().st_size > 1_000_000, f"Exported {onnx_file} missing!"

        model = onnx.load(str(onnx_file))
        onnx.checker.check_model(model, full_check=True)

        inp = model.graph.input[0]
        out = model.graph.output[0]
        assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        assert dims == [1, 3, 640, 640]


def test_numerical_parity_pytorch_vs_onnx():
    """
    Verify numerical parity between original PyTorch (.pt) models and standalone
    ONNX Runtime (.onnx) inference within standard FP32 floating-point tolerance.
    """
    from scripts.verify_onnx_parity import build_test_frames, compare_model

    bus_img, plated_img = build_test_frames()

    cases = [
        (
            "vehicle_detector",
            BASE_DIR / "models" / "vehicle_detector.pt",
            BASE_DIR / "models" / "vehicle_detector.onnx",
            bus_img,
            0.25,
        ),
        (
            "license_plate_detector",
            BASE_DIR / "models" / "license_plate_detector.pt",
            BASE_DIR / "models" / "license_plate_detector.onnx",
            plated_img,
            0.20,
        ),
    ]

    for name, pt_path, onnx_path, frame, conf in cases:
        rep = compare_model(name, pt_path, onnx_path, frame, conf=conf, iou=0.45)

        # Raw FP32 tensor parity
        assert rep["raw_box_max_diff"] < 1e-2, f"{name}: raw box diff too high ({rep['raw_box_max_diff']})"
        assert rep["raw_prob_max_diff"] < 1e-4, f"{name}: raw prob diff too high ({rep['raw_prob_max_diff']})"

        # End-to-end detections parity (boxes, class IDs, confidence scores)
        p = rep["pt_vs_onnx"]
        assert p["ref_count"] == p["cand_count"] > 0, f"{name}: detection count mismatch"
        assert p["class_mismatches"] == 0, f"{name}: class ID mismatch"
        assert p["min_iou"] >= 0.999, f"{name}: IoU {p['min_iou']} below 0.999"
        assert p["max_box_px_diff"] <= 0.1, f"{name}: box pixel diff {p['max_box_px_diff']} > 0.1"
        assert p["max_conf_diff"] < 1e-4, f"{name}: confidence diff {p['max_conf_diff']} >= 1e-4"

        # Full-resolution non-square letterbox + NMS parity
        f = rep["fullres_parity"]
        assert f["ref_count"] == f["cand_count"] > 0
        assert f["class_mismatches"] == 0
        assert f["min_iou"] >= 0.999
        assert f["max_conf_diff"] < 1e-4


def test_app_detectors_run_via_onnx_backend():
    """Smoke-test app PlateDetector and VehicleDetector end-to-end on real images via ONNX."""
    from scripts.verify_onnx_parity import build_test_frames

    bus_img, plated_img = build_test_frames()

    vd = VehicleDetector(backend="onnx")
    vehicles = vd.detect(bus_img)
    assert vd.backend == "onnx"
    assert any(v.class_name == "bus" for v in vehicles)

    pd = PlateDetector(backend="onnx")
    plates = pd.detect(plated_img, conf=0.20)
    assert pd.backend == "onnx"
    assert len(plates) >= 1
