"""
Unit Tests for OCR Engine, Enhancements, Multi-Frame Voting & Device Resolution
===============================================================================
"""

import cv2
import numpy as np
import pytest
from app.core.device import resolve_device
from app.vision.image_processor import ImageProcessor
from app.vision.plate_ocr import PlateOCR
from app.recognition.recognition_verifier import MultiFrameRecognitionVerifier


def _make_plate_image(text: str, bg=(255, 255, 255), fg=(0, 0, 0), w=320, h=76):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), fg, 2)
    cv2.putText(img, text, (16, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 1.35, fg, 3, cv2.LINE_AA)
    return img


def test_device_resolution():
    dev, diag = resolve_device("cpu")
    assert dev == "cpu"
    assert diag["device"] == "cpu"

    dev_auto, diag_auto = resolve_device("auto")
    assert dev_auto in ("cuda:0", "cpu", "mps")
    assert "cuda_available" in diag_auto

    dev_fake, _ = resolve_device("cuda:999")
    assert dev_fake.startswith("cuda") or dev_fake == "cpu"


def test_image_processor_crop_and_enhance():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:150, 200:350] = (255, 255, 255)

    crop = ImageProcessor.crop_with_padding(frame, (200, 100, 350, 150), padding_ratio=0.1)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0

    candidates = ImageProcessor.enhance_for_ocr(crop)
    assert len(candidates) >= 2
    for cand in candidates:
        assert len(cand.shape) == 2

    # Two-line splitting on a square-ish plate
    sq = np.full((120, 200, 3), 255, dtype=np.uint8)
    cv2.putText(sq, "KA01AB", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    cv2.putText(sq, "1234", (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
    assert ImageProcessor.is_two_line(sq) is True
    top, bot = ImageProcessor.split_two_line(sq)
    assert top.shape[0] > 10 and bot.shape[0] > 10


def test_plate_ocr_accuracy_and_speed():
    ocr = PlateOCR.get_instance()
    samples = [
        ("KA01AB1234", (255, 255, 255), (0, 0, 0)),     # Private white
        ("MH12DE4567", (35, 200, 245), (0, 0, 0)),      # Commercial yellow (BGR)
        ("22BH1234AA", (50, 130, 45), (255, 255, 255)), # EV green (BGR)
    ]
    for expected, bg, fg in samples:
        crop = _make_plate_image(expected, bg=bg, fg=fg)
        res = ocr.recognize(crop)
        assert res is not None
        assert res.is_valid is True
        assert res.normalized_plate == expected
        assert res.confidence >= 0.70


def test_multi_frame_verifier_voting():
    verifier = MultiFrameRecognitionVerifier(window_size=5, min_votes=2)
    tracking_id = 42

    # Frame 1: 1-char typo
    verifier.add_observation(tracking_id, "KA01AB1238", confidence=0.72, raw_text="KA01AB1238")
    assert verifier.get_verified(tracking_id) is None

    # Frames 2 & 3: Correct reading outvotes the typo
    verifier.add_observation(tracking_id, "KA01AB1230", confidence=0.95, raw_text="KA01AB1230")
    verifier.add_observation(tracking_id, "KA01AB1230", confidence=0.96, raw_text="KA01AB1230")

    verified = verifier.get_verified(tracking_id)
    assert verified is not None
    assert verified.plate_number == "KA01AB1230"
    assert verified.vote_count >= 2
    assert verified.average_confidence > 0.90
