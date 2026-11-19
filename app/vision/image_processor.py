"""
Plate Image Preprocessing for OCR
=================================
Fast, quality-aware preprocessing designed for real-time ANPR:

1. `crop_with_padding` - bounds-safe crop with configurable fractional margin.
2. `quality_score` / `sharpness` - ranks crops across frames so OCR gets the
   sharpest, largest plate view instead of blurry motion frames.
3. `split_two_line` - locates the horizontal gap between the two rows of
   square/two-wheeler plates using vertical edge-energy projection.
4. `prepare_fast` - lightweight (~1-2 ms) deskew + polarity normalization +
   CLAHE/unsharp enhancement used when the raw pass needs help.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import cv2
import numpy as np


@dataclass
class PreparedPlate:
    bgr: np.ndarray                       # deskewed, contrast-enhanced BGR image
    gray: np.ndarray                      # deskewed, polarity-normalized grayscale
    variants: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    two_line: bool = False
    skew_angle: float = 0.0
    inverted: bool = False
    scale: float = 1.0


class ImageProcessor:
    SINGLE_LINE_HEIGHT = 64
    TWO_LINE_HEIGHT = 128
    MAX_WIDTH = 640

    # ------------------------------------------------------------------ crops
    @staticmethod
    def crop_with_padding(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_ratio: float = 0.06,
    ) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_w = int((x2 - x1) * padding_ratio)
        pad_h = int((y2 - y1) * padding_ratio)
        cx1, cy1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        cx2, cy2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return image[cy1:cy2, cx1:cx2]

    # ---------------------------------------------------------------- quality
    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    @classmethod
    def sharpness(cls, img: np.ndarray) -> float:
        """Variance of Laplacian on a height-normalized grayscale image."""
        if img is None or img.size == 0:
            return 0.0
        gray = cls.to_gray(img)
        h, w = gray.shape[:2]
        if h < 4 or w < 4:
            return 0.0
        norm = cv2.resize(gray, (max(8, int(w * 48 / h)), 48), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(norm, cv2.CV_64F).var())

    @classmethod
    def quality_score(cls, crop: np.ndarray, det_conf: float = 1.0) -> float:
        """Higher is better: sharp, large, confidently detected crops."""
        if crop is None or crop.size == 0:
            return 0.0
        h = crop.shape[0]
        size_factor = min(1.0, h / 60.0)
        sharp_factor = min(1.0, cls.sharpness(crop) / 800.0)
        return round((0.5 * sharp_factor + 0.3 * size_factor + 0.2 * det_conf), 4)

    # --------------------------------------------------------------- two-line
    @classmethod
    def upscale_if_small(cls, img: np.ndarray, min_height: int = 56) -> np.ndarray:
        """
        Upscale distant/small plate crops cleanly (Lanczos4 + light edge sharpening)
        so plates captured further from the camera remain readable.
        """
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        if h >= min_height or h <= 0:
            return img
        scale = min(3.5, min_height / float(h))
        up = cv2.resize(
            img,
            (max(16, int(w * scale)), max(16, int(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )
        # Gentle unsharp mask to restore character edges after upscaling
        blur = cv2.GaussianBlur(up, (0, 0), 1.0)
        return cv2.addWeighted(up, 1.35, blur, -0.35, 0)

    @classmethod
    def is_two_line(cls, img: np.ndarray) -> bool:
        h, w = img.shape[:2]
        return (w / max(1.0, float(h))) < 2.25

    @classmethod
    def split_two_line(cls, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a two-line plate into (top_row, bottom_row) by finding the row
        of minimum horizontal-gradient energy in the central vertical band.
        Adds a small overlap margin so descenders/ascenders are never clipped.
        """
        h, w = img.shape[:2]
        gray = cls.to_gray(img)
        # Focus on horizontal inner 80% so side borders don't dominate
        inner = gray[:, int(w * 0.1):int(w * 0.9)]
        # Horizontal edges (Sobel X) highlight vertical strokes of characters
        grad_x = np.abs(cv2.Sobel(inner, cv2.CV_32F, 1, 0, ksize=3))
        row_energy = grad_x.mean(axis=1)

        # Search for the valley between top and bottom text rows (35%..65% of height)
        lo, hi = max(1, int(h * 0.35)), min(h - 1, int(h * 0.65))
        if hi <= lo:
            split_y = h // 2
        else:
            split_y = lo + int(np.argmin(row_energy[lo:hi]))

        margin = max(2, int(h * 0.04))
        # Trim outer vertical padding slightly if plate has thick top/bottom border
        top_start = int(h * 0.04)
        bot_end = int(h * 0.96)
        top = img[top_start:min(h, split_y + margin), :]
        bot = img[max(0, split_y - margin):bot_end, :]
        return top, bot

    # --------------------------------------------------------------- geometry
    @staticmethod
    def estimate_skew(gray: np.ndarray) -> float:
        """Median angle (degrees) of near-horizontal line segments; 0 if unsure."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            h, w = gray.shape[:2]
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=max(16, w // 8),
                minLineLength=max(16, w // 4), maxLineGap=max(4, w // 35),
            )
            if lines is None:
                return 0.0
            angles = []
            for x1, y1, x2, y2 in lines[:, 0]:
                if x2 == x1:
                    continue
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(ang) < 22:
                    angles.append(ang)
            if len(angles) < 2:
                return 0.0
            return float(np.median(angles))
        except Exception:
            return 0.0

    @staticmethod
    def rotate(img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # --------------------------------------------------------------- polarity
    @staticmethod
    def is_light_text(gray: np.ndarray) -> bool:
        """True if characters are lighter than background (e.g. green EV plates)."""
        h, w = gray.shape[:2]
        core = gray[int(h * 0.15):int(h * 0.85), int(w * 0.12):int(w * 0.95)]
        if core.size == 0:
            return False
        _, binary = cv2.threshold(core, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_ratio = float(np.count_nonzero(binary)) / binary.size
        return bright_ratio < 0.42

    # --------------------------------------------------------------- pipeline
    @classmethod
    def prepare(cls, plate_crop: np.ndarray) -> Optional[PreparedPlate]:
        """Fast (~1-2 ms) deskew, polarity normalization, CLAHE and unsharp variants."""
        if plate_crop is None or plate_crop.size == 0:
            return None
        h, w = plate_crop.shape[:2]
        if h < 8 or w < 16:
            return None

        two_line = cls.is_two_line(plate_crop)
        target_h = cls.TWO_LINE_HEIGHT if two_line else cls.SINGLE_LINE_HEIGHT
        scale = target_h / float(h)
        if w * scale > cls.MAX_WIDTH:
            scale = cls.MAX_WIDTH / float(w)
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(
            plate_crop,
            (max(16, int(w * scale)), max(8, int(h * scale))),
            interpolation=interp,
        )
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        angle = cls.estimate_skew(gray)
        if 1.5 < abs(angle) < 20:
            resized = cls.rotate(resized, angle)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            angle = 0.0

        inverted = cls.is_light_text(gray)
        if inverted:
            gray = cv2.bitwise_not(gray)

        # Variant 1: CLAHE local contrast normalization (very fast on 64px height)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 8))
        v_clahe = clahe.apply(gray)

        # Variant 2: Unsharp mask (recovers slightly blurred strokes)
        blur = cv2.GaussianBlur(v_clahe, (0, 0), 1.4)
        v_sharp = cv2.addWeighted(v_clahe, 1.5, blur, -0.5, 0)

        # Deskewed + contrast-boosted color image for color-capable recognizers
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        enhanced_bgr = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        return PreparedPlate(
            bgr=enhanced_bgr,
            gray=v_clahe,
            variants=[("clahe", v_clahe), ("sharp", v_sharp)],
            two_line=two_line,
            skew_angle=angle,
            inverted=inverted,
            scale=scale,
        )

    @classmethod
    def enhance_for_ocr(cls, plate_crop: np.ndarray) -> List[np.ndarray]:
        """Backwards-compatible helper returning grayscale variants."""
        prepared = cls.prepare(plate_crop)
        return [img for _, img in prepared.variants] if prepared else []
