"""
Standalone ONNX Model Exporter & Graph Verifier
================================================
Exports the project's Ultralytics YOLOv8 PyTorch (.pt) models to standard
FP32 ONNX (.onnx) files for lightweight Windows deployment via ONNX Runtime
(DirectML / CPU) without requiring PyTorch or Ultralytics at runtime.

Original .pt model files are never modified or deleted.

Models exported by default:
  1. models/license_plate_detector.pt -> models/license_plate_detector.onnx
  2. models/vehicle_detector.pt       -> models/vehicle_detector.onnx

Usage:
  python export_to_onnx.py
  python export_to_onnx.py --imgsz 640 --opset 17
  python export_to_onnx.py --model models/license_plate_detector.pt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS: List[Tuple[Path, str]] = [
    (BASE_DIR / "models" / "license_plate_detector.pt", "YOLOv8 License Plate Detector"),
    (BASE_DIR / "models" / "vehicle_detector.pt", "YOLOv8n Vehicle Detector (COCO)"),
]


def export_single_model(pt_path: Path, imgsz: int = 640, opset: int = 17, dynamic: bool = False) -> Path:
    """Export one Ultralytics YOLOv8 .pt checkpoint to standard FP32 .onnx and verify graph."""
    import onnx
    from ultralytics import YOLO

    if not pt_path.exists():
        raise FileNotFoundError(f"Source PyTorch weights not found: {pt_path}")

    print(f"\n{'=' * 70}")
    print(f"Exporting: {pt_path.name}  (imgsz={imgsz}, opset={opset}, FP32, dynamic={dynamic})")
    print(f"{'=' * 70}")

    model = YOLO(str(pt_path))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,  # Static 640x640 yields peak DirectML GPU performance on Windows
        simplify=False,   # Keep canonical FP32 graph without requiring onnxslim
    )

    onnx_path = Path(exported)
    if not onnx_path.exists():
        raise RuntimeError(f"Export reported '{exported}', but file does not exist.")

    # Verify the exported ONNX graph strictly using onnx.checker
    print(f"Verifying ONNX graph with onnx.checker.check_model({onnx_path.name})...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model, full_check=True)

    # Inspect input/output tensors and confirm FP32 precision
    graph = onnx_model.graph
    opsets = [(op.domain or "ai.onnx", op.version) for op in onnx_model.opset_import]

    for inp in graph.input:
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  [INPUT ] {inp.name}: shape={dims}, dtype={dtype}")
        if inp.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"Expected FP32 (FLOAT) input tensor, got {dtype}")

    for out in graph.output:
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  [OUTPUT] {out.name}: shape={dims}, dtype={dtype}")
        if out.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"Expected FP32 (FLOAT) output tensor, got {dtype}")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  [OK    ] Graph check PASSED | Opset={opsets} | Size={size_mb:.2f} MB | Path={onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch models to verified FP32 ONNX")
    parser.add_argument("--model", type=str, default=None, help="Export a single specific .pt file")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input resolution (default: 640)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic spatial axes (default: static)")
    args = parser.parse_args()

    targets = [(Path(args.model).resolve(), Path(args.model).name)] if args.model else DEFAULT_MODELS

    exported_files: List[Path] = []
    for pt_path, label in targets:
        print(f"\n>>> Target: {label}")
        out = export_single_model(pt_path, imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic)
        exported_files.append(out)

    print(f"\n{'=' * 70}")
    print(f"All {len(exported_files)} ONNX model(s) exported and verified cleanly:")
    for f in exported_files:
        print(f"  - {f}")
    print("Original .pt model files remain untouched.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] ONNX export failed: {exc}", file=sys.stderr)
        sys.exit(1)
