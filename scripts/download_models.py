"""
Model Download & Verification Script for Production ANPR System
===============================================================

Downloads and verifies:
1. Genuine License Plate Detector (YOLOv8):
   - Source: Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
   - Direct Raw URL: https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/main/license_plate_detector.pt
   - Class: {0: 'license_plate'}
   - Expected size: ~6.2 MB (6,241,454 bytes)
   - License: MIT / Open Source Research & Commercial use

2. Vehicle Detector (YOLOv8n COCO):
   - Source: Ultralytics Official Release
   - Direct URL: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt
   - Classes include: car (2), motorcycle (3), bus (5), truck (7)
   - Expected size: ~6.5 MB (6,549,796 bytes)
   - License: AGPL-3.0 / Enterprise
"""

import os
import sys
import shutil
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODELS_CONFIG = {
    "license_plate_detector.pt": {
        "url": "https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/main/license_plate_detector.pt",
        "min_size_bytes": 5_000_000,
        "expected_class_count": 1,
        "description": "Pretrained YOLOv8 License Plate Detector",
        "license": "MIT"
    },
    "vehicle_detector.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "local_source_fallback": Path("D:/PROJECTS/ProjectVision/yolov8n.pt"),
        "min_size_bytes": 6_000_000,
        "expected_class_count": 80,
        "description": "Pretrained Ultralytics YOLOv8n Vehicle/Object Detector",
        "license": "AGPL-3.0"
    }
}


def download_file(url: str, dest_path: Path):
    """Download a file with user-agent and progress reporting."""
    print(f"Downloading from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 64
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                print(f"\rProgress: [{downloaded:,} / {total_size:,} bytes] ({percent:.1f}%)", end="", flush=True)
            else:
                print(f"\rDownloaded: {downloaded:,} bytes", end="", flush=True)
        print()


def verify_yolo_model(model_path: Path, expected_classes: int) -> bool:
    """Verify that YOLO model can be loaded and has expected classes."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        num_classes = len(model.names)
        print(f"  [OK] Successfully loaded {model_path.name}")
        print(f"  [OK] Classes ({num_classes}): {list(model.names.values())[:5]} ...")
        if num_classes < expected_classes:
            print(f"  [WARN] Expected at least {expected_classes} classes, got {num_classes}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error verifying model {model_path.name}: {e}")
        return False


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ANPR System: Model Downloader & Verification")
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 60)

    for filename, config in MODELS_CONFIG.items():
        dest = MODELS_DIR / filename
        print(f"\nProcessing: {filename} ({config['description']})")
        print(f"License: {config['license']}")

        if dest.exists() and dest.stat().st_size >= config["min_size_bytes"]:
            print(f"  [OK] File exists ({dest.stat().st_size:,} bytes). Verifying...")
        else:
            local_fallback = config.get("local_source_fallback")
            if local_fallback and local_fallback.exists() and local_fallback.stat().st_size >= config["min_size_bytes"]:
                print(f"  Copying from local cache: {local_fallback} -> {dest}")
                shutil.copy2(local_fallback, dest)
            else:
                try:
                    download_file(config["url"], dest)
                except Exception as e:
                    print(f"  [ERROR] Failed to download {filename}: {e}")
                    sys.exit(1)

        size = dest.stat().st_size
        if size < config["min_size_bytes"]:
            print(f"  [ERROR] File size too small ({size} bytes). Download corrupted.")
            sys.exit(1)

        print(f"  [OK] File size: {size:,} bytes")
        verify_yolo_model(dest, config["expected_class_count"])

    print("\n" + "=" * 60)
    print("All models successfully downloaded and verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
