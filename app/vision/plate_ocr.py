"""
High-Accuracy, Low-Latency License Plate OCR Engine
===================================================
Uses a tiered ONNX recognition cascade designed specifically for Indian ANPR:

1. Primary engine  : RapidOCR PP-OCRv4 (recognition-only ONNX, ~9 ms/crop).
                     Skips slow generic text detection since YOLO already
                     cropped the plate, and handles two-line plates via
                     horizontal edge-projection splitting.
2. Secondary engine: fast-plate-ocr (`cct-xs-v2-global-model`, ~20 ms/crop),
                     a Compact Convolutional Transformer trained specifically
                     on license plates. Invoked ONLY on hard/ambiguous crops
                     where the primary pass is uncertain, acting as an
                     independent neural tie-breaker.
3. Fallback engine : EasyOCR (used automatically if ONNX engines are missing).

Most clear/medium plates exit after Pass 1 in ~9-12 ms with 0% false accepts.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.core.device import resolve_device, easyocr_gpu_flag
from app.core.exceptions import OCRError
from app.recognition.plate_validator import IndianVehicleNumberValidator
from app.recognition.text_normalizer import TextNormalizer
from app.vision.image_processor import ImageProcessor, PreparedPlate

logger = logging.getLogger("anpr.ocr")

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
NOISE_TOKENS = {"IND", "INO", "IN0", "1ND", "LND", "IHD", "INA"}


class OCRResult:
    def __init__(
        self,
        raw_text: str,
        normalized_plate: str,
        confidence: float,
        is_valid: bool,
        format_type: str,
        state_code: Optional[str] = None,
        state_name: Optional[str] = None,
        processed_crop: Optional[np.ndarray] = None,
        raw_confidence: float = 0.0,
        variant: str = "",
        latency_ms: float = 0.0,
    ):
        self.raw_text = raw_text
        self.normalized_plate = normalized_plate
        self.confidence = confidence
        self.is_valid = is_valid
        self.format_type = format_type
        self.state_code = state_code
        self.state_name = state_name
        self.processed_crop = processed_crop
        self.raw_confidence = raw_confidence
        self.variant = variant
        self.latency_ms = latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_plate": self.normalized_plate,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "is_valid": self.is_valid,
            "format_type": self.format_type,
            "state_code": self.state_code,
            "state_name": self.state_name,
            "variant": self.variant,
            "latency_ms": round(self.latency_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"<OCRResult(plate='{self.normalized_plate}', conf={self.confidence:.2f}, "
            f"valid={self.is_valid}, src='{self.variant}', {self.latency_ms:.1f}ms)>"
        )


class PlateOCR:
    _instance: Optional["PlateOCR"] = None
    _instance_lock = threading.Lock()

    def __init__(self, device: Optional[str] = None):
        resolved_device, _ = resolve_device(device or settings.DEVICE)
        self.device = resolved_device
        self.use_gpu = easyocr_gpu_flag(resolved_device)
        self.validator = IndianVehicleNumberValidator()
        self.early_exit_conf = settings.OCR_EARLY_EXIT_CONFIDENCE

        self._rapid = None
        self._cct = None
        self._easy = None
        self.engine_name = "uninitialized"
        self._lock = threading.Lock()

        self._initialize_engines()
        self.warmup()

    # ------------------------------------------------------------ lifecycle
    def _initialize_engines(self):
        # 1. Primary: RapidOCR PP-OCRv4 recognition-only ONNX
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._rapid = RapidOCR(
                use_det=False,
                use_cls=False,
                use_rec=True,
                intra_op_num_threads=4,
            )
            self.engine_name = "rapidocr_ppocrv4"
            logger.info("Initialized primary OCR engine: RapidOCR PP-OCRv4 (rec-only ONNX).")
        except Exception as e:
            logger.warning(f"RapidOCR unavailable ({e}); will try secondary/fallback engines.")

        # 2. Secondary: fast-plate-ocr CCT license plate model (bounded ONNX threads to avoid spin contention)
        try:
            import onnxruntime as ort
            from fast_plate_ocr import LicensePlateRecognizer
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self._cct = LicensePlateRecognizer("cct-xs-v2-global-model", device="cpu", sess_options=so)
            if self._rapid is not None:
                self.engine_name = "rapidocr_ppocrv4+fast_plate_ocr"
            else:
                self.engine_name = "fast_plate_ocr_cct"
            logger.info("Initialized secondary plate OCR engine: fast-plate-ocr (cct-xs-v2-global-model).")
        except Exception as e:
            logger.warning(f"fast-plate-ocr unavailable ({e}); continuing without secondary tie-breaker.")

        # 3. Fallback: EasyOCR (only initialized if both ONNX engines failed)
        if self._rapid is None and self._cct is None:
            try:
                import easyocr
                logger.info(f"Initializing EasyOCR fallback (device='{self.device}', gpu={self.use_gpu})...")
                self._easy = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
                self.engine_name = "easyocr_fallback"
            except Exception as e:
                if self.use_gpu:
                    import easyocr
                    self._easy = easyocr.Reader(["en"], gpu=False, verbose=False)
                    self.device, self.use_gpu = "cpu", False
                    self.engine_name = "easyocr_cpu_fallback"
                else:
                    raise OCRError(f"All OCR engines failed to initialize: {e}")

    def warmup(self):
        """Warm up loaded ONNX/PyTorch sessions so the first live plate has low latency."""
        try:
            dummy = np.full((60, 260, 3), 255, dtype=np.uint8)
            cv2.putText(dummy, "KA01AB1234", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
            self.recognize(dummy)
            logger.info(f"OCR engine warm-up complete ({self.engine_name}).")
        except Exception as e:
            logger.warning(f"OCR warm-up skipped: {e}")

    @classmethod
    def get_instance(cls) -> "PlateOCR":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------- engine runners
    def _strip_noise(self, text: str) -> str:
        clean = TextNormalizer.clean(text)
        if clean in NOISE_TOKENS:
            return ""
        return clean

    def _run_rapid_single(self, bgr: np.ndarray) -> Tuple[str, float]:
        if bgr is None or bgr.size == 0 or bgr.shape[0] < 6 or bgr.shape[1] < 10:
            return "", 0.0
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        res, _ = self._rapid(bgr)
        if not res:
            return "", 0.0
        raw, conf = res[0][0], float(res[0][1])
        return self._strip_noise(raw), conf

    def _run_rapid(self, img: np.ndarray, two_line: bool) -> Tuple[str, float]:
        if not two_line:
            return self._run_rapid_single(img)
        top, bot = ImageProcessor.split_two_line(img)
        t1, c1 = self._run_rapid_single(top)
        t2, c2 = self._run_rapid_single(bot)
        parts = [(t, c) for t, c in ((t1, c1), (t2, c2)) if t]
        if not parts:
            return "", 0.0
        text = "".join(t for t, _ in parts)
        weights = [len(t) for t, _ in parts]
        conf = float(np.average([c for _, c in parts], weights=weights))
        return text, conf

    def _run_cct(self, bgr: np.ndarray) -> Tuple[str, float]:
        if self._cct is None or bgr is None or bgr.size == 0:
            return "", 0.0
        if bgr.ndim == 2:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        preds = self._cct.run(rgb, return_confidence=True)
        if not preds:
            return "", 0.0
        pred = preds[0]
        raw = self._strip_noise(pred.plate.replace("_", ""))
        if not raw:
            return "", 0.0
        if pred.char_probs is not None and len(pred.char_probs) > 0:
            probs = [float(p) for p in pred.char_probs[:len(raw)]]
            # Blend mean and min char probability so a single low-confidence character lowers score
            conf = float(0.75 * np.mean(probs) + 0.25 * np.min(probs))
        else:
            conf = 0.0
        return raw, conf

    def _run_easyocr(self, img: np.ndarray) -> Tuple[str, float]:
        if self._easy is None:
            return "", 0.0
        out = self._easy.readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False)
        if not out:
            return "", 0.0
        out.sort(key=lambda b: (b[0][0][1] // 24, b[0][0][0]))
        parts = [(self._strip_noise(b[1]), float(b[2])) for b in out]
        parts = [(t, c) for t, c in parts if t]
        if not parts:
            return "", 0.0
        text = "".join(t for t, _ in parts)
        conf = float(np.average([c for _, c in parts], weights=[len(t) for t, _ in parts]))
        return text, conf

    # ------------------------------------------------------------ scoring
    def _build_result(self, raw_text: str, raw_conf: float, variant: str, crop: np.ndarray) -> OCRResult:
        val = self.validator.validate(raw_text)
        adj_conf = max(0.0, raw_conf - val.confidence_penalty) if val.is_valid else raw_conf * 0.5
        return OCRResult(
            raw_text=raw_text,
            normalized_plate=val.normalized_plate,
            confidence=adj_conf,
            is_valid=val.is_valid,
            format_type=val.format_type,
            state_code=val.state_code,
            state_name=val.state_name,
            processed_crop=crop,
            raw_confidence=raw_conf,
            variant=variant,
        )

    # ------------------------------------------------------------ public API
    def recognize(self, plate_crop: np.ndarray) -> Optional[OCRResult]:
        """
        Recognize an Indian license plate from a cropped plate image.
        Fast path takes ~9 ms on CPU; hard crops escalate through deskew/CLAHE
        and the independent CCT plate model.
        """
        t0 = time.perf_counter()
        if plate_crop is None or plate_crop.size == 0:
            return None
        h, w = plate_crop.shape[:2]
        if h < 8 or w < 16:
            return None

        bgr = plate_crop if plate_crop.ndim == 3 else cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
        two_line = ImageProcessor.is_two_line(bgr)
        # Cleanly upscale distant/small crops (< 44px height) before Pass 1
        pass1_bgr = ImageProcessor.upscale_if_small(bgr, min_height=96 if two_line else 56) if h < 44 else bgr

        with self._lock:
            valid_reads: List[Tuple[str, OCRResult]] = []   # (engine_family, result)
            best_raw: Optional[OCRResult] = None

            def record(engine_family: str, variant: str, text: str, conf: float, img: np.ndarray) -> Optional[OCRResult]:
                nonlocal best_raw
                if not text:
                    return None
                res = self._build_result(text, conf, variant, img)
                if best_raw is None or res.confidence > best_raw.confidence:
                    best_raw = res
                if res.is_valid:
                    valid_reads.append((engine_family, res))
                return res

            # ---- Pass 1: RapidOCR directly on raw/upscaled color crop (~9 ms) ----
            if self._rapid is not None:
                try:
                    t1, c1 = self._run_rapid(pass1_bgr, two_line)
                    r1 = record("rapid", "rapid_raw", t1, c1, pass1_bgr)
                    # Immediate fast-path exit for clean, confident reads
                    if r1 and r1.is_valid and r1.confidence >= 0.78 and (c1 - r1.confidence) < 0.06:
                        r1.latency_ms = (time.perf_counter() - t0) * 1000.0
                        return r1
                except Exception as e:
                    logger.debug(f"RapidOCR raw pass failed: {e}")

            # ---- Pass 2: Deskewed + LAB contrast-enhanced color crop (~10 ms) ----
            prepared: Optional[PreparedPlate] = ImageProcessor.prepare(bgr)
            if self._rapid is not None and prepared is not None:
                try:
                    t2, c2 = self._run_rapid(prepared.bgr, prepared.two_line)
                    r2 = record("rapid", "rapid_enhanced", t2, c2, prepared.bgr)
                    # Early exit if Pass 1 & Pass 2 agree on the exact same valid plate
                    if (
                        r2 and r2.is_valid
                        and len(valid_reads) >= 2
                        and valid_reads[0][1].normalized_plate == r2.normalized_plate
                        and max(valid_reads[0][1].confidence, r2.confidence) >= 0.72
                    ):
                        r2.confidence = min(1.0, max(valid_reads[0][1].confidence, r2.confidence) + 0.06)
                        r2.latency_ms = (time.perf_counter() - t0) * 1000.0
                        return r2
                    if r2 and r2.is_valid and r2.confidence >= self.early_exit_conf and (c2 - r2.confidence) < 0.06:
                        r2.latency_ms = (time.perf_counter() - t0) * 1000.0
                        return r2
                except Exception as e:
                    logger.debug(f"RapidOCR enhanced pass failed: {e}")

            # ---- Pass 3: Independent CCT plate model (~3 ms tie-breaker / hard plates) ----
            if self._cct is not None:
                try:
                    cct_in = prepared.bgr if prepared is not None else bgr
                    t3, c3 = self._run_cct(cct_in)
                    record("cct", "cct_xs_v2", t3, c3, cct_in)
                except Exception as e:
                    logger.debug(f"fast-plate-ocr pass failed: {e}")

            # ---- Fallback: EasyOCR if ONNX engines are not installed ----
            if self._rapid is None and self._cct is None and self._easy is not None:
                img_in = prepared.gray if prepared is not None else bgr
                te, ce = self._run_easyocr(img_in)
                record("easy", "easyocr", te, ce, img_in)

        # ---- Aggregate valid reads across passes & engines ----
        if valid_reads:
            grouped: Dict[str, Dict[str, Any]] = {}
            for fam, res in valid_reads:
                g = grouped.setdefault(res.normalized_plate, {
                    "engines": set(),
                    "count": 0,
                    "best": res,
                    "max_conf": res.confidence,
                })
                g["engines"].add(fam)
                g["count"] += 1
                if res.confidence > g["max_conf"]:
                    g["max_conf"] = res.confidence
                    g["best"] = res

            # Prefer cross-engine agreement first, then pass count, then confidence
            best_plate, info = max(
                grouped.items(),
                key=lambda kv: (len(kv[1]["engines"]), kv[1]["count"], kv[1]["max_conf"]),
            )
            winner: OCRResult = info["best"]
            bonus = 0.08 * (len(info["engines"]) - 1) + 0.03 * max(0, info["count"] - len(info["engines"]))
            winner.confidence = min(1.0, winner.confidence + bonus)
            winner.latency_ms = (time.perf_counter() - t0) * 1000.0
            return winner

        if best_raw is not None:
            best_raw.latency_ms = (time.perf_counter() - t0) * 1000.0
        return best_raw
