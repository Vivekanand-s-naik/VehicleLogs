"""
OCR Accuracy & Latency Benchmark
================================
Generates a reproducible set of synthetic Indian number plate crops with
realistic degradations, runs them through PlateOCR (preprocessing + EasyOCR +
validation) and reports:

- Exact-match accuracy (single frame)
- Character accuracy (1 - normalized edit distance)
- Acceptance rate (valid + above OCR_CONFIDENCE)
- FALSE ACCEPTS (accepted but wrong) -> the metric that creates wrong DB records
- Multi-frame consensus accuracy (3 frames per plate through the verifier)
- Latency p50 / p95 / mean

NOTE: This benchmarks the OCR stage on plate crops (what YOLO hands to OCR).
It does not benchmark plate *detection*. Synthetic plates are a proxy; always
validate on footage from your actual camera position.

Usage:
  python scripts/benchmark_ocr.py --samples 200 --seed 7
"""

import sys
import time
import random
import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

STATES = ["KA", "MH", "DL", "TN", "KL", "AP", "TS", "GJ", "RJ", "UP", "HR", "PB", "WB", "MP", "OD", "GA"]
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # Indian series avoid I/O
FONT_CANDIDATES = [
    "C:/Windows/Fonts/bahnschrift.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

STYLES = {
    "private": ((255, 255, 255), (0, 0, 0)),
    "commercial": ((245, 200, 35), (0, 0, 0)),   # yellow background (RGB), black text
    "ev": ((45, 130, 50), (255, 255, 255)),      # green background (RGB), white text
}


def random_plate(rng: random.Random) -> str:
    if rng.random() < 0.12:
        year = rng.randint(21, 26)
        return f"{year}BH{rng.randint(0, 9999):04d}{''.join(rng.choice(LETTERS) for _ in range(rng.choice([1, 2])))}"
    state = rng.choice(STATES)
    if state == "DL":
        district = str(rng.randint(1, 13))
        series = "".join(rng.choice(LETTERS) for _ in range(rng.choice([1, 2, 3])))
    else:
        district = f"{rng.randint(1, 70):02d}"
        series = "".join(rng.choice(LETTERS) for _ in range(rng.choice([1, 2, 2, 2])))
    return f"{state}{district}{series}{rng.randint(1, 9999):04d}"


def split_display(plate: str) -> str:
    """Insert human-style spacing (KA 01 AB 1234)."""
    if "BH" in plate[2:4]:
        return f"{plate[:2]} BH {plate[4:8]} {plate[8:]}"
    num = plate[-4:]
    prefix = plate[:-4]
    state, rest = prefix[:2], prefix[2:]
    digits = "".join(c for c in rest if c.isdigit())
    letters = rest[len(digits):]
    return " ".join(p for p in [state, digits, letters, num] if p)


def _fit_font(draw: ImageDraw.ImageDraw, text: str, font_path: str, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    for size in range(max_h, 16, -2):
        f = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= max_w and th <= max_h:
            return f
    return ImageFont.truetype(font_path, 16)


def render_plate(plate: str, rng: random.Random, fonts):
    style = rng.choice(list(STYLES.keys()))
    bg, fg = STYLES[style]
    two_line = rng.random() < 0.2 and "BH" not in plate[2:4]
    font_path = rng.choice(fonts)

    if two_line:
        W, H = 400, 220
        num = plate[-4:]
        top = plate[:-4]
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        f_top = _fit_font(d, top, font_path, W - 40, 82)
        f_bot = _fit_font(d, num, font_path, W - 40, 82)
        d.text((W // 2, 58), top, font=f_top, fill=fg, anchor="mm")
        d.text((W // 2, 160), num, font=f_bot, fill=fg, anchor="mm")
    else:
        W, H = 520, 120
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        # IND strip
        d.rectangle([0, 0, 42, H], fill=(20, 60, 170))
        fsmall = ImageFont.truetype(font_path, 14)
        d.text((21, H - 22), "IND", font=fsmall, fill=(255, 255, 255), anchor="mm")
        display = split_display(plate)
        f = _fit_font(d, display, font_path, W - 64, 76)
        d.text(((W + 42) // 2, H // 2), display, font=f, fill=fg, anchor="mm")
    d.rectangle([1, 1, img.width - 2, img.height - 2], outline=fg if style != "ev" else (255, 255, 255), width=4)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), style, two_line


def degrade(plate_img: np.ndarray, rng: random.Random, level: str) -> np.ndarray:
    img = plate_img.copy()
    h, w = img.shape[:2]

    # Perspective / rotation
    if level in ("medium", "hard"):
        max_shift = 0.06 if level == "medium" else 0.12
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jitter = lambda: rng.uniform(-max_shift, max_shift)
        dst = np.float32([
            [w * jitter(), h * jitter()],
            [w + w * jitter(), h * jitter()],
            [w + w * jitter(), h + h * jitter()],
            [w * jitter(), h + h * jitter()],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        angle = rng.uniform(-4, 4) if level == "medium" else rng.uniform(-9, 9)
        R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, R, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Embed in a "car body" background with margin (like a YOLO crop with padding)
    pad_x, pad_y = int(w * 0.08), int(h * 0.15)
    canvas = np.full((h + 2 * pad_y, w + 2 * pad_x, 3), rng.randint(30, 160), dtype=np.uint8)
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = img
    img = canvas

    # Resolution: real webcam crops are small
    target_h = {"easy": rng.randint(60, 90), "medium": rng.randint(32, 55), "hard": rng.randint(22, 36)}[level]
    scale = target_h / img.shape[0]
    img = cv2.resize(img, (max(8, int(img.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)

    # Illumination
    alpha = rng.uniform(0.6, 1.2) if level != "easy" else rng.uniform(0.85, 1.1)
    beta = rng.uniform(-40, 30) if level != "easy" else rng.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if level == "hard" and rng.random() < 0.4:
        # Glare blob
        gx, gy = rng.randint(0, img.shape[1] - 1), rng.randint(0, img.shape[0] - 1)
        overlay = img.copy()
        cv2.circle(overlay, (gx, gy), max(3, img.shape[0] // 3), (255, 255, 255), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # Blur
    if level != "easy" or rng.random() < 0.3:
        k = rng.choice([3, 3, 5]) if level == "hard" else 3
        img = cv2.GaussianBlur(img, (k, k), 0)
    if level == "hard" and rng.random() < 0.4:
        # Motion blur
        k = rng.choice([3, 5])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    # Sensor noise
    sigma = {"easy": 3, "medium": 7, "hard": 12}[level]
    noise = np.random.default_rng(rng.randint(0, 10**9)).normal(0, sigma, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # JPEG compression
    q = {"easy": 85, "medium": 55, "hard": 35}[level]
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def build_dataset(n: int, seed: int):
    rng = random.Random(seed)
    fonts = [f for f in FONT_CANDIDATES if Path(f).exists()]
    if not fonts:
        raise RuntimeError("No TrueType fonts found for synthetic plate rendering.")
    levels = ["easy", "medium", "hard"]
    data = []
    for i in range(n):
        plate = random_plate(rng)
        base, style, two_line = render_plate(plate, rng, fonts)
        level = levels[i % 3]
        frames = [degrade(base, rng, level) for _ in range(3)]
        data.append({"plate": plate, "style": style, "two_line": two_line, "level": level, "frames": frames})
    return data


def main():
    parser = argparse.ArgumentParser(description="ANPR OCR benchmark")
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None, help="auto / cuda / cpu")
    parser.add_argument("--save-failures", action="store_true", help="Write failing crops to scratch folder")
    args = parser.parse_args()

    from app.core.config import settings
    from app.vision.plate_ocr import PlateOCR
    from app.recognition.recognition_verifier import MultiFrameRecognitionVerifier

    print(f"Building {args.samples} synthetic plates (seed={args.seed})...")
    data = build_dataset(args.samples, args.seed)

    ocr = PlateOCR(device=args.device)
    # Warm-up (CUDA kernels / cudnn autotune)
    for _ in range(3):
        ocr.recognize(data[0]["frames"][0])

    latencies, results = [], []
    fail_dir = BASE_DIR / "logs" / "ocr_failures"
    if args.save_failures:
        fail_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(data):
        per_frame = []
        for f_idx, frame in enumerate(item["frames"]):
            t0 = time.perf_counter()
            res = ocr.recognize(frame)
            latencies.append((time.perf_counter() - t0) * 1000)
            pred = res.normalized_plate if res else ""
            valid = bool(res and res.is_valid)
            conf = res.confidence if res else 0.0
            accepted = valid and conf >= settings.OCR_CONFIDENCE
            per_frame.append((pred, valid, conf, accepted, res.raw_text if res else ""))
            if args.save_failures and f_idx == 0 and pred != item["plate"]:
                cv2.imwrite(str(fail_dir / f"{idx:03d}_{item['level']}_{item['plate']}_got_{pred or 'NONE'}.png"), frame)

        # Multi-frame consensus over the 3 frames (accepted readings only)
        verifier = MultiFrameRecognitionVerifier(window_size=5, min_votes=2)
        for pred, valid, conf, accepted, raw in per_frame:
            if accepted:
                verifier.add_observation("t", pred, conf, raw)
        verified = verifier.get_verified("t")
        results.append({**item, "per_frame": per_frame, "consensus": verified.plate_number if verified else None})

    # ---------------- Report ----------------
    def summarize(subset, label):
        frames = [(r["plate"], pf) for r in subset for pf in r["per_frame"]]
        if not frames:
            return
        exact = sum(1 for truth, pf in frames if pf[0] == truth and pf[1])
        char_acc = statistics.mean(
            1 - edit_distance(pf[0], truth) / max(len(truth), 1) for truth, pf in frames
        )
        accepted = [(truth, pf) for truth, pf in frames if pf[3]]
        false_acc = sum(1 for truth, pf in accepted if pf[0] != truth)
        cons_correct = sum(1 for r in subset if r["consensus"] == r["plate"])
        cons_wrong = sum(1 for r in subset if r["consensus"] and r["consensus"] != r["plate"])
        cons_none = sum(1 for r in subset if not r["consensus"])
        print(
            f"{label:<12} frames={len(frames):<4} exact={exact / len(frames) * 100:5.1f}%  "
            f"char={char_acc * 100:5.1f}%  accepted={len(accepted) / len(frames) * 100:5.1f}%  "
            f"FALSE-ACCEPT={false_acc / len(frames) * 100:4.1f}%  | consensus: "
            f"correct={cons_correct / len(subset) * 100:5.1f}% wrong={cons_wrong / len(subset) * 100:4.1f}% "
            f"undecided={cons_none / len(subset) * 100:4.1f}%"
        )

    print("\n" + "=" * 120)
    print(f"OCR BENCHMARK  device={ocr.device}  samples={len(results)} plates x 3 frames")
    print("=" * 120)
    summarize(results, "ALL")
    for lvl in ("easy", "medium", "hard"):
        summarize([r for r in results if r["level"] == lvl], lvl)
    summarize([r for r in results if r["two_line"]], "two-line")
    summarize([r for r in results if r["style"] == "ev"], "ev-green")
    summarize([r for r in results if r["style"] == "commercial"], "commercial")
    summarize([r for r in results if "BH" in r["plate"][2:4]], "bharat")
    lat_sorted = sorted(latencies)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))]
    print("-" * 120)
    print(f"Latency per crop: mean={statistics.mean(latencies):.1f}ms  p50={p(0.5):.1f}ms  p95={p(0.95):.1f}ms")
    print("=" * 120)

    wrong_examples = [
        (r["plate"], pf[0], pf[4]) for r in results for pf in r["per_frame"] if pf[3] and pf[0] != r["plate"]
    ][:10]
    if wrong_examples:
        print("Sample false accepts (truth -> predicted | raw):")
        for t, pr, raw in wrong_examples:
            print(f"  {t} -> {pr} | {raw}")


if __name__ == "__main__":
    main()

