"""
Numerical Parity Verification: Original PyTorch YOLOv8 (.pt) vs ONNX Runtime (.onnx)
====================================================================================
Compares BOTH exported models on real visual frames:
  1. vehicle_detector.pt        vs vehicle_detector.onnx       (real bus photo, 810x1080)
  2. license_plate_detector.pt  vs license_plate_detector.onnx (plated vehicle photo, 810x1080)

Runs three complementary parity checks per model:
  A. Raw FP32 Tensor Parity: identical [1, 3, 640, 640] tensor through PyTorch vs ONNX Runtime.
  B. End-to-End PyTorch (.pt) vs ONNXYOLODetector (.onnx) Parity on matched 640x640 geometry
     (controls for Ultralytics' default `rect=True` non-square padding on portrait inputs).
  C. Full-Resolution Non-Square (810x1080) Pre/Post-Processing Parity: Ultralytics ONNX wrapper
     vs our standalone OpenCV/NumPy `ONNXYOLODetector` (verifies letterbox, NMS, and coordinate
     un-scaling independently of PyTorch).
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.vision.inference_onnx import ONNXYOLODetector, ONNXDetection


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_test_frames() -> Tuple[np.ndarray, np.ndarray]:
    """Load real Ultralytics bus.jpg and create a realistic plated vehicle variant."""
    import ultralytics
    bus_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    if not bus_path.exists():
        raise FileNotFoundError(f"Sample image not found at {bus_path}")
    bus_img = cv2.imread(str(bus_path))

    # Render a clean Indian plate onto the front bumper of the bus
    plated = bus_img.copy()
    h, w = plated.shape[:2]
    pw, ph = 180, 44
    px, py = int(w * 0.28), int(h * 0.56)
    cv2.rectangle(plated, (px, py), (px + pw, py + ph), (255, 255, 255), -1)
    cv2.rectangle(plated, (px, py), (px + pw, py + ph), (20, 20, 20), 2)
    cv2.rectangle(plated, (px, py), (px + 18, py + ph), (170, 55, 20), -1)
    cv2.putText(plated, "KA01AB1234", (px + 22, py + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2, cv2.LINE_AA)
    return bus_img, plated


def _extract_ultralytics_dets(result) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
    dets = []
    if result.boxes is not None and len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            xyxy = tuple(float(v) for v in result.boxes.xyxy[i].cpu().numpy())
            c = float(result.boxes.conf[i].cpu().item())
            cid = int(result.boxes.cls[i].cpu().item())
            dets.append((xyxy, c, cid))
    dets.sort(key=lambda d: d[1], reverse=True)
    return dets


def _match_detections(
    ref_dets: List[Tuple[Tuple[float, float, float, float], float, int]],
    cand_dets: List[ONNXDetection],
) -> Dict[str, float | int]:
    used = set()
    ious, box_diffs, conf_diffs = [], [], []
    class_mismatches = 0

    for ref_box, ref_conf, ref_cid in ref_dets:
        best_j, best_iou = -1, -1.0
        for j, od in enumerate(cand_dets):
            if j in used:
                continue
            ov = box_iou(ref_box, od.bbox_float)
            if ov > best_iou:
                best_iou, best_j = ov, j
        if best_j >= 0:
            used.add(best_j)
            od = cand_dets[best_j]
            ious.append(best_iou)
            box_diffs.append(max(abs(a - b) for a, b in zip(ref_box, od.bbox_float)))
            conf_diffs.append(abs(ref_conf - od.confidence))
            if ref_cid != od.class_id:
                class_mismatches += 1

    return {
        "ref_count": len(ref_dets),
        "cand_count": len(cand_dets),
        "matched": len(ious),
        "min_iou": min(ious) if ious else 1.0,
        "max_box_px_diff": max(box_diffs) if box_diffs else 0.0,
        "max_conf_diff": max(conf_diffs) if conf_diffs else 0.0,
        "class_mismatches": class_mismatches,
    }


def compare_model(
    name: str,
    pt_path: Path,
    onnx_path: Path,
    frame: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict:
    import torch
    from ultralytics import YOLO

    pt_model = YOLO(str(pt_path))
    ul_onnx_model = YOLO(str(onnx_path), task="detect")
    onnx_det = ONNXYOLODetector(onnx_path, conf_threshold=conf, iou_threshold=iou)

    # A. Exact raw FP32 tensor parity on identical [1, 3, 640, 640] blob
    blob, _, _ = onnx_det._preprocess(frame)
    with torch.no_grad():
        pt_inner = pt_model.model.to("cpu").float().eval()
        pt_raw = pt_inner(torch.from_numpy(blob))
        pt_raw_np = (pt_raw[0] if isinstance(pt_raw, (tuple, list)) else pt_raw).cpu().numpy()
    onnx_raw_np = onnx_det.session.run([onnx_det.output_name], {onnx_det.input_name: blob})[0]
    raw_abs_diff = np.abs(pt_raw_np - onnx_raw_np)
    raw_box_max_diff = float(np.max(raw_abs_diff[:, :4, :]))
    raw_prob_max_diff = float(np.max(raw_abs_diff[:, 4:, :]))
    raw_mean_diff = float(np.mean(raw_abs_diff))

    # B. Matched-geometry 640x640 end-to-end parity: PyTorch (.pt) vs ONNXYOLODetector (.onnx)
    #    Uses the exact 640x640 letterboxed canvas so Ultralytics' default rect=True shortcut
    #    does not feed a different [640x480] aspect ratio to the .pt model.
    lb_canvas, _, _ = ONNXYOLODetector.letterbox(frame, (640, 640))
    pt_model.predict(lb_canvas, imgsz=640, conf=conf, iou=iou, device="cpu", verbose=False)
    onnx_det.predict(lb_canvas, conf=conf, iou=iou)

    t0 = time.perf_counter()
    pt_res_lb = pt_model.predict(lb_canvas, imgsz=640, conf=conf, iou=iou, device="cpu", verbose=False)[0]
    pt_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    onnx_res_lb = onnx_det.predict(lb_canvas, conf=conf, iou=iou)
    onnx_ms = (time.perf_counter() - t0) * 1000.0

    pt_dets_lb = _extract_ultralytics_dets(pt_res_lb)
    pt_vs_onnx = _match_detections(pt_dets_lb, onnx_res_lb)

    # C. Full-resolution non-square (810x1080) pre/post-processing parity:
    #    Ultralytics ONNX wrapper vs standalone OpenCV/NumPy ONNXYOLODetector
    ul_onnx_res = ul_onnx_model.predict(frame, imgsz=640, conf=conf, iou=iou, device="cpu", verbose=False)[0]
    ul_onnx_dets = _extract_ultralytics_dets(ul_onnx_res)
    onnx_res_full = onnx_det.predict(frame, conf=conf, iou=iou)
    fullres_parity = _match_detections(ul_onnx_dets, onnx_res_full)

    return {
        "model": name,
        "providers": onnx_det.active_providers,
        "raw_box_max_diff": raw_box_max_diff,
        "raw_prob_max_diff": raw_prob_max_diff,
        "raw_tensor_mean_diff": raw_mean_diff,
        "pt_vs_onnx": pt_vs_onnx,
        "fullres_parity": fullres_parity,
        "pt_cpu_ms": round(pt_ms, 2),
        "onnx_ms": round(onnx_ms, 2),
        "pt_dets": pt_dets_lb,
        "onnx_dets": onnx_res_lb,
    }


def main():
    bus_img, plated_img = build_test_frames()

    checks = [
        (
            "Vehicle Detector (YOLOv8n COCO)",
            BASE_DIR / "models" / "vehicle_detector.pt",
            BASE_DIR / "models" / "vehicle_detector.onnx",
            bus_img,
            0.25,
            0.45,
        ),
        (
            "License Plate Detector (YOLOv8 Custom)",
            BASE_DIR / "models" / "license_plate_detector.pt",
            BASE_DIR / "models" / "license_plate_detector.onnx",
            plated_img,
            0.20,
            0.45,
        ),
    ]

    all_ok = True
    print("=" * 80)
    print("NUMERICAL PARITY CHECK: PyTorch (.pt) vs Standalone ONNX Runtime (.onnx)")
    print("=" * 80)
    for name, pt_p, onnx_p, img, conf, iou in checks:
        rep = compare_model(name, pt_p, onnx_p, img, conf=conf, iou=iou)
        p = rep["pt_vs_onnx"]
        f = rep["fullres_parity"]

        passed = (
            rep["raw_box_max_diff"] < 1e-2      # < 0.01 px across all 8,400 anchors
            and rep["raw_prob_max_diff"] < 1e-4 # < 0.0001 probability diff across all anchors
            and rep["raw_tensor_mean_diff"] < 1e-4
            and p["ref_count"] == p["cand_count"] > 0
            and p["class_mismatches"] == 0
            and p["min_iou"] >= 0.999
            and p["max_box_px_diff"] <= 0.1
            and p["max_conf_diff"] < 1e-4
            and f["ref_count"] == f["cand_count"] > 0
            and f["class_mismatches"] == 0
            and f["min_iou"] >= 0.999
            and f["max_conf_diff"] < 1e-4
        )
        all_ok = all_ok and passed

        print(f"\nModel: {rep['model']}")
        print(f"  ONNX Runtime Providers:    {rep['providers']}")
        print(
            f"  [A] Raw FP32 Tensor Diff:  box_max={rep['raw_box_max_diff']:.2e} px, "
            f"prob_max={rep['raw_prob_max_diff']:.2e}, mean={rep['raw_tensor_mean_diff']:.2e}"
        )
        print(
            f"  [B] PyTorch vs ONNX (640): dets={p['ref_count']}=={p['cand_count']}, "
            f"cls_mismatch={p['class_mismatches']}, min_IoU={p['min_iou']:.6f}, "
            f"max_box_diff={p['max_box_px_diff']:.4f} px, max_conf_diff={p['max_conf_diff']:.2e}"
        )
        print(
            f"  [C] Full-Res 810x1080 NMS: dets={f['ref_count']}=={f['cand_count']}, "
            f"cls_mismatch={f['class_mismatches']}, min_IoU={f['min_iou']:.6f}, "
            f"max_box_diff={f['max_box_px_diff']:.4f} px, max_conf_diff={f['max_conf_diff']:.2e}"
        )
        print(f"  Latency (CPU 640x640):     PyTorch={rep['pt_cpu_ms']} ms | ONNX Runtime={rep['onnx_ms']} ms")
        print(f"  Verdict:                   {'PASS' if passed else 'FAIL'}")

        for idx, ((pb, pc, pcid), od) in enumerate(zip(rep["pt_dets"], rep["onnx_dets"])):
            print(
                f"      det[{idx}] cls={pcid}=={od.class_id} ({od.class_name}) | "
                f"conf={pc:.6f} vs {od.confidence:.6f} | "
                f"box={[round(v, 2) for v in pb]} vs {[round(v, 2) for v in od.bbox_float]}"
            )

    print("\n" + "=" * 80)
    print("OVERALL PARITY VERDICT:", "ALL MODELS PASSED" if all_ok else "FAILED")
    print("=" * 80)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
