"""
Event-Triggered Video Clip Recorder & Deferred Clip Processor
=============================================================
Implements the Trigger -> Record -> Process architecture:

1. Lightweight live trigger:
   - Maintains a short pre-roll ring buffer of raw frames.
   - As soon as ANY license plate is detected (including small/distant plates),
     recording starts immediately without stalling the live feed for OCR.
   - Records to `captures/events/clip_<timestamp>.mp4` until either:
       (a) no plate is seen for `RECORD_POST_ROLL_SECONDS`, OR
       (b) the clip reaches `RECORD_MAX_DURATION` (handles stopped vehicles).

2. Deferred background video stream processing (`ClipProcessorWorker`):
   - Opens the finished video clip and runs high-resolution plate detection
     (`CLIP_DETECT_IMGSZ`, e.g. 960px) across sampled frames.
   - Ranks every plate crop in the clip by sharpness x area x confidence and
     runs the tiered OCR engine on the Top-K clearest frames.
   - Uses character-level multi-frame voting across the entire clip to confirm
     the exact Indian registration number, persists the vehicle event + best
     snapshots + clip path atomically, and dispatches push notifications.
"""

import os
import time
import queue
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set, Tuple
import cv2
import numpy as np
from app.core.config import settings
from app.vision.plate_detector import PlateDetector, PlateDetection
from app.vision.vehicle_detector import VehicleDetector, VehicleDetection
from app.vision.image_processor import ImageProcessor
from app.vision.plate_ocr import PlateOCR, OCRResult
from app.recognition.recognition_verifier import MultiFrameRecognitionVerifier, VerifiedPlate
from app.services.vehicle_state_manager import VehicleStateManager, StateTransitionResult

logger = logging.getLogger("anpr.clip_recorder")


@dataclass
class ActiveClip:
    clip_id: str
    clip_path: Path
    rel_clip_path: str
    writer: cv2.VideoWriter
    fps: float
    frame_size: Tuple[int, int]  # (width, height)
    started_at: float
    last_plate_seen_at: float
    frame_count: int = 0
    track_ids: Set[int] = field(default_factory=set)
    vehicle_track_id: Optional[int] = None
    direction_hint: Optional[str] = None
    is_stationary: bool = False


@dataclass
class ClipJob:
    clip_id: str
    clip_path: Path
    rel_clip_path: str
    duration_s: float
    frame_count: int
    track_ids: Set[int]
    vehicle_track_id: Optional[int]
    direction_hint: Optional[str]
    is_stationary: bool


@dataclass
class ClipProcessOutcome:
    clip_id: str
    rel_clip_path: str
    plate_number: Optional[str] = None
    confidence: float = 0.0
    votes: int = 0
    event_type: Optional[str] = None
    event_recorded: bool = False
    frames_scanned: int = 0
    crops_evaluated: int = 0
    processing_ms: float = 0.0
    reason: str = ""


class ClipRecorderService:
    """Manages plate-triggered video recording and asynchronous clip OCR processing."""

    def __init__(
        self,
        plate_detector: PlateDetector,
        vehicle_detector: VehicleDetector,
        plate_ocr: PlateOCR,
        state_manager: VehicleStateManager,
        on_clip_processed: Optional[Callable[[ClipJob, ClipProcessOutcome], None]] = None,
    ):
        self.plate_detector = plate_detector
        self.vehicle_detector = vehicle_detector
        self.plate_ocr = plate_ocr
        self.state_manager = state_manager
        self.on_clip_processed = on_clip_processed

        settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)

        pre_roll_cap = max(5, int(settings.RECORD_PRE_ROLL_SECONDS * max(10, min(30, settings.CAMERA_FPS))))
        self._pre_roll: deque = deque(maxlen=pre_roll_cap)
        self._active_clip: Optional[ActiveClip] = None
        self._rec_lock = threading.Lock()

        self._clip_queue: queue.Queue = queue.Queue(maxsize=max(2, settings.CLIP_QUEUE_SIZE))
        self._stop_event = threading.Event()
        self._processor_thread: Optional[threading.Thread] = None

        # Telemetry
        self.clips_recorded: int = 0
        self.clips_processed: int = 0
        self.clips_confirmed: int = 0
        self.is_processing_clip: bool = False
        self.last_clip_ms: float = 0.0
        self.last_clip_result: str = "IDLE"

    # ------------------------------------------------------------ lifecycle
    def start(self):
        self._stop_event.clear()
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(
                target=self._processor_loop,
                name="ANPRClipProcessor",
                daemon=True,
            )
            self._processor_thread.start()
            logger.info("Event-triggered ClipProcessorWorker started.")

    def stop(self):
        self._stop_event.set()
        with self._rec_lock:
            if self._active_clip is not None:
                try:
                    self._active_clip.writer.release()
                except Exception:
                    pass
                self._active_clip = None
        if self._processor_thread and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=3.0)
        logger.info("ClipRecorderService stopped.")

    def reset(self):
        """Reset recording telemetry and drop pending clips (used during admin purge)."""
        with self._rec_lock:
            if self._active_clip is not None:
                try:
                    self._active_clip.writer.release()
                except Exception:
                    pass
                self._active_clip = None
            self._pre_roll.clear()
        while not self._clip_queue.empty():
            try:
                self._clip_queue.get_nowait()
                self._clip_queue.task_done()
            except queue.Empty:
                break
        self.clips_recorded = 0
        self.clips_processed = 0
        self.clips_confirmed = 0
        self.last_clip_result = "RESET"

    # -------------------------------------------------------- live feed hook
    def feed_frame(
        self,
        raw_frame: np.ndarray,
        plates: List[PlateDetection],
        unconfirmed_plates: List[PlateDetection],
        vehicle_track_id: Optional[int] = None,
        direction_hint: Optional[str] = None,
        is_stationary: bool = False,
        actual_fps: float = 20.0,
    ) -> Optional[ClipJob]:
        """
        Called by the live pipeline on every fresh frame (non-blocking).
        Starts recording when any unconfirmed plate appears, appends frames while
        active, and finalizes + enqueues the video clip once the plate leaves
        or max duration elapses.
        """
        now = time.time()
        h, w = raw_frame.shape[:2]
        finished_job: Optional[ClipJob] = None

        with self._rec_lock:
            if self._active_clip is None:
                if unconfirmed_plates:
                    self._start_recording_locked(
                        raw_frame, unconfirmed_plates, vehicle_track_id,
                        direction_hint, is_stationary, actual_fps, now,
                    )
                else:
                    self._pre_roll.append(raw_frame.copy())
                return None

            clip = self._active_clip
            # Write frame (resize only if camera resolution changed mid-clip)
            out_frame = raw_frame
            if (w, h) != clip.frame_size:
                out_frame = cv2.resize(raw_frame, clip.frame_size)
            clip.writer.write(out_frame)
            clip.frame_count += 1

            if plates:
                clip.last_plate_seen_at = now
                for p in plates:
                    if p.tracking_id is not None:
                        clip.track_ids.add(p.tracking_id)
                if vehicle_track_id is not None:
                    clip.vehicle_track_id = vehicle_track_id
                if direction_hint is not None:
                    clip.direction_hint = direction_hint
                clip.is_stationary = is_stationary

            elapsed = now - clip.started_at
            absence = now - clip.last_plate_seen_at

            if absence >= settings.RECORD_POST_ROLL_SECONDS or elapsed >= settings.RECORD_MAX_DURATION:
                finished_job = self._finalize_recording_locked(now)

        if finished_job is not None:
            self._enqueue_clip_job(finished_job)

        return finished_job

    def force_finalize(self) -> Optional[ClipJob]:
        """Finalize an active clip immediately (useful in tests or shutdown)."""
        with self._rec_lock:
            if self._active_clip is None:
                return None
            job = self._finalize_recording_locked(time.time())
        if job is not None:
            self._enqueue_clip_job(job)
        return job

    def _start_recording_locked(
        self,
        frame: np.ndarray,
        plates: List[PlateDetection],
        vehicle_track_id: Optional[int],
        direction_hint: Optional[str],
        is_stationary: bool,
        actual_fps: float,
        now: float,
    ):
        settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        fps = max(10.0, min(30.0, actual_fps if actual_fps > 1.0 else float(settings.CAMERA_FPS)))

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = f"{int(now * 1000) % 10000:04d}"
        clip_id = f"clip_{ts}_{suffix}"
        clip_path = settings.EVENTS_DIR / f"{clip_id}.mp4"
        rel_path = str(clip_path.relative_to(settings.CAPTURES_DIR.parent)).replace("\\", "/")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))
        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {clip_path}")
            return

        clip = ActiveClip(
            clip_id=clip_id,
            clip_path=clip_path,
            rel_clip_path=rel_path,
            writer=writer,
            fps=fps,
            frame_size=(w, h),
            started_at=now,
            last_plate_seen_at=now,
            track_ids={p.tracking_id for p in plates if p.tracking_id is not None},
            vehicle_track_id=vehicle_track_id,
            direction_hint=direction_hint,
            is_stationary=is_stationary,
        )

        # Flush pre-roll frames so the beginning of the vehicle approach is preserved
        for pre_f in self._pre_roll:
            pf = pre_f if (pre_f.shape[1], pre_f.shape[0]) == (w, h) else cv2.resize(pre_f, (w, h))
            clip.writer.write(pf)
            clip.frame_count += 1
        self._pre_roll.clear()

        clip.writer.write(frame)
        clip.frame_count += 1
        self._active_clip = clip
        self.clips_recorded += 1
        logger.info(f"[REC START] Plate detected -> recording video clip '{clip_id}' ({w}x{h} @ {fps:.0f}fps)")

    def _finalize_recording_locked(self, now: float) -> Optional[ClipJob]:
        clip = self._active_clip
        self._active_clip = None
        if clip is None:
            return None

        try:
            clip.writer.release()
        except Exception as e:
            logger.warning(f"Error releasing VideoWriter for {clip.clip_id}: {e}")

        duration = max(0.05, now - clip.started_at)
        if clip.frame_count < settings.RECORD_MIN_FRAMES or not clip.clip_path.exists():
            logger.debug(f"Discarding short clip {clip.clip_id} ({clip.frame_count} frames)")
            try:
                if clip.clip_path.exists():
                    clip.clip_path.unlink()
            except Exception:
                pass
            return None

        logger.info(
            f"[REC STOP] Finalized '{clip.clip_id}' ({clip.frame_count} frames, {duration:.2f}s) "
            f"-> queued for video OCR processing."
        )
        return ClipJob(
            clip_id=clip.clip_id,
            clip_path=clip.clip_path,
            rel_clip_path=clip.rel_clip_path,
            duration_s=round(duration, 2),
            frame_count=clip.frame_count,
            track_ids=set(clip.track_ids),
            vehicle_track_id=clip.vehicle_track_id,
            direction_hint=clip.direction_hint,
            is_stationary=clip.is_stationary,
        )

    def _enqueue_clip_job(self, job: ClipJob):
        try:
            self._clip_queue.put_nowait(job)
        except queue.Full:
            logger.warning(f"Clip processing queue full! Dropping oldest clip to process '{job.clip_id}'.")
            try:
                self._clip_queue.get_nowait()
                self._clip_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._clip_queue.put_nowait(job)
            except queue.Full:
                pass

    # --------------------------------------------------- background processing
    def _processor_loop(self):
        while not self._stop_event.is_set():
            try:
                job: ClipJob = self._clip_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            self.is_processing_clip = True
            try:
                outcome = self.process_video_clip(
                    clip_path=job.clip_path,
                    rel_clip_path=job.rel_clip_path,
                    clip_id=job.clip_id,
                    duration_s=job.duration_s,
                    tracking_id=job.vehicle_track_id or (next(iter(job.track_ids), None)),
                    direction_hint=job.direction_hint,
                    is_stationary=job.is_stationary,
                )
                self.clips_processed += 1
                self.last_clip_ms = outcome.processing_ms
                if outcome.plate_number:
                    self.clips_confirmed += 1
                    self.last_clip_result = f"{outcome.plate_number} ({outcome.confidence * 100:.0f}%)"
                else:
                    self.last_clip_result = f"No readable plate ({outcome.reason})"

                if self.on_clip_processed is not None:
                    self.on_clip_processed(job, outcome)
            except Exception as e:
                logger.error(f"Unhandled error processing clip '{job.clip_id}': {e}", exc_info=True)
                self.last_clip_result = f"Error: {e}"
            finally:
                self.is_processing_clip = False
                self._clip_queue.task_done()

    def process_video_clip(
        self,
        clip_path: Path,
        rel_clip_path: Optional[str] = None,
        clip_id: Optional[str] = None,
        duration_s: float = 0.0,
        tracking_id: Optional[int] = None,
        direction_hint: Optional[str] = None,
        is_stationary: bool = False,
        record_event: bool = True,
    ) -> ClipProcessOutcome:
        """
        Open a recorded video clip, detect plates across frames at high resolution
        (including distant/small plates), select the Top-K sharpest crops, run OCR
        + character-level multi-frame voting, and record the confirmed vehicle event.
        """
        t0 = time.perf_counter()
        cid = clip_id or clip_path.stem
        rel_path = rel_clip_path or str(clip_path).replace("\\", "/")

        if not clip_path.exists():
            return ClipProcessOutcome(cid, rel_path, reason="Clip file missing")

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return ClipProcessOutcome(cid, rel_path, reason="Failed to open video clip")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # Sample up to ~45 evenly spaced frames so every clip finishes rapidly
        max_scan = 45
        stride = max(1, total_frames // max_scan) if total_frames > max_scan else 1

        # Candidate list: (quality_score, frame_idx, det_conf, plate_crop, full_frame, plate_bbox)
        candidates: List[Tuple[float, int, float, np.ndarray, np.ndarray, Tuple[int, int, int, int]]] = []
        fallback_poster: Optional[np.ndarray] = None
        frames_scanned = 0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if fallback_poster is None or frame_idx == max(1, total_frames // 2):
                    fallback_poster = frame.copy()
                if frame_idx % stride == 0:
                    frames_scanned += 1
                    dets = self.plate_detector.detect(
                        frame,
                        imgsz=settings.CLIP_DETECT_IMGSZ,
                        conf=min(0.20, settings.DETECTION_CONFIDENCE),
                    )
                    for d in dets:
                        crop = ImageProcessor.crop_with_padding(
                            frame, d.bbox, padding_ratio=settings.PLATE_PADDING_RATIO
                        )
                        if crop is None or crop.size == 0:
                            continue
                        q = ImageProcessor.quality_score(crop, d.confidence)
                        # Slight preference for larger plate area within the clip
                        area_bonus = min(0.25, (d.width * d.height) / 25000.0)
                        candidates.append((q + area_bonus, frame_idx, d.confidence, crop, frame, d.bbox))
                frame_idx += 1
        finally:
            cap.release()

        if not candidates:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if settings.KEEP_ALL_CLIPS:
                self._save_clip_thumbnail(clip_path, fallback_poster)
            else:
                self._remove_unusable_clip(clip_path)
            return ClipProcessOutcome(
                cid, rel_path, frames_scanned=frames_scanned, processing_ms=round(elapsed_ms, 2),
                reason="No plate detections in clip",
            )

        # Pick Top-K sharpest/largest crops across the entire recorded video
        candidates.sort(key=lambda item: item[0], reverse=True)
        top_k = candidates[:max(3, settings.CLIP_TOP_K_CROPS)]

        verifier = MultiFrameRecognitionVerifier(
            window_size=len(top_k),
            min_votes=settings.RECOGNITION_MIN_VOTES,
            max_age_seconds=60.0,
            min_support_ratio=0.50,
        )

        best_single: Optional[Tuple[OCRResult, float, np.ndarray, np.ndarray, Tuple[int, int, int, int]]] = None
        state_names: Dict[str, Optional[str]] = {}
        det_confs: Dict[str, float] = {}
        best_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        crops_evaluated = 0

        for _score, _fidx, det_conf, plate_crop, full_frame, pbbox in top_k:
            ocr_res = self.plate_ocr.recognize(plate_crop)
            crops_evaluated += 1
            if ocr_res is None or not ocr_res.is_valid:
                continue
            if ocr_res.confidence < settings.OCR_CONFIDENCE:
                continue

            plate_str = ocr_res.normalized_plate
            state_names[plate_str] = ocr_res.state_name
            if det_conf > det_confs.get(plate_str, 0.0):
                det_confs[plate_str] = det_conf
                best_bboxes[plate_str] = pbbox

            if best_single is None or ocr_res.confidence > best_single[0].confidence:
                best_single = (ocr_res, det_conf, plate_crop, full_frame, pbbox)

            verifier.add_observation(
                tracking_id=cid,
                plate_number=plate_str,
                confidence=ocr_res.confidence,
                raw_text=ocr_res.raw_text,
                plate_crop=plate_crop,
                vehicle_crop=full_frame,
            )

            # Fast exit once 3 crops strongly agree with high confidence
            v_check = verifier.get_verified(cid)
            if v_check is not None and v_check.vote_count >= 3 and v_check.average_confidence >= 0.85:
                break

        verified: Optional[VerifiedPlate] = verifier.get_verified(cid)
        chosen_plate: Optional[str] = None
        chosen_conf: float = 0.0
        chosen_votes: int = 0
        chosen_raw: str = ""
        chosen_plate_crop: Optional[np.ndarray] = None
        chosen_frame: Optional[np.ndarray] = None

        if verified is not None:
            chosen_plate = verified.plate_number
            chosen_conf = verified.average_confidence
            chosen_votes = verified.vote_count
            chosen_raw = verified.best_raw_text
            chosen_plate_crop = verified.best_plate_crop
            chosen_frame = verified.best_vehicle_crop
        elif best_single is not None and best_single[0].confidence >= 0.80:
            # Short/distant pass-by with 1 crisp, high-confidence valid read
            ocr_res, _dc, pc, ff, _pb = best_single
            chosen_plate = ocr_res.normalized_plate
            chosen_conf = ocr_res.confidence
            chosen_votes = 1
            chosen_raw = ocr_res.raw_text
            chosen_plate_crop = pc
            chosen_frame = ff

        if not chosen_plate:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if settings.KEEP_ALL_CLIPS:
                poster = top_k[0][4] if top_k else fallback_poster
                self._save_clip_thumbnail(clip_path, poster)
            else:
                self._remove_unusable_clip(clip_path)
            return ClipProcessOutcome(
                cid, rel_path, frames_scanned=frames_scanned, crops_evaluated=crops_evaluated,
                processing_ms=round(elapsed_ms, 2), reason="No valid Indian plate reached threshold",
            )

        # Rename stored clip to include confirmed plate number (e.g. KA01AB1234_20261119_122500_1234.mp4)
        final_clip_path, rel_path = self._rename_clip_with_plate(clip_path, chosen_plate)

        # Extract tight vehicle context crop from the winning frame if a vehicle is detected
        vehicle_img = chosen_frame
        if chosen_frame is not None and chosen_plate in best_bboxes:
            try:
                vehs = self.vehicle_detector.detect(chosen_frame)
                matched = self.vehicle_detector.associate_plate_to_vehicle(best_bboxes[chosen_plate], vehs)
                if matched is not None:
                    vc = ImageProcessor.crop_with_padding(chosen_frame, matched.bbox, padding_ratio=0.05)
                    if vc is not None and vc.size > 0:
                        vehicle_img = vc
            except Exception as e:
                logger.debug(f"Vehicle crop association skipped on clip frame: {e}")

        # Save poster thumbnail photo right beside the stored clip
        thumb_src = chosen_frame if chosen_frame is not None else vehicle_img
        rel_thumb_path = self._save_clip_thumbnail(final_clip_path, thumb_src)

        event_type = None
        event_recorded = False
        reason = "Confirmed from recorded clip"

        if record_event:
            transition: StateTransitionResult = self.state_manager.process_detection(
                plate_number=chosen_plate,
                confidence=chosen_conf,
                detection_confidence=det_confs.get(chosen_plate, 0.0),
                raw_ocr_text=chosen_raw,
                tracking_id=tracking_id,
                direction_hint=direction_hint,
                is_stationary=is_stationary,
                vehicle_image=vehicle_img,
                plate_image=chosen_plate_crop,
                state_name=state_names.get(chosen_plate),
                video_clip_path=rel_path,
                extra_metadata={
                    "source": "recorded_clip",
                    "clip_id": cid,
                    "video_clip_path": rel_path,
                    "clip_thumbnail_path": rel_thumb_path,
                    "clip_frames": total_frames or frames_scanned,
                    "clip_duration_s": duration_s,
                    "ocr_votes": chosen_votes,
                    "crops_evaluated": crops_evaluated,
                },
            )
            event_recorded = transition.success
            event_type = transition.event_type
            if not transition.success:
                reason = transition.reason or "Suppressed by cooldown/presence"
                if not settings.KEEP_ALL_CLIPS:
                    self._remove_unusable_clip(final_clip_path)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"[CLIP DONE] '{final_clip_path.name}' -> plate={chosen_plate} conf={chosen_conf:.2f} "
            f"votes={chosen_votes} recorded={event_recorded} ({elapsed_ms:.0f}ms)"
        )
        return ClipProcessOutcome(
            clip_id=final_clip_path.stem,
            rel_clip_path=rel_path,
            plate_number=chosen_plate,
            confidence=chosen_conf,
            votes=chosen_votes,
            event_type=event_type,
            event_recorded=event_recorded,
            frames_scanned=frames_scanned,
            crops_evaluated=crops_evaluated,
            processing_ms=round(elapsed_ms, 2),
            reason=reason,
        )

    @staticmethod
    def _rename_clip_with_plate(clip_path: Path, plate_number: str) -> Tuple[Path, str]:
        """Rename clip_<ts>.mp4 -> <PLATE>_<ts>.mp4 cleanly without overwriting existing files."""
        safe_plate = "".join(c for c in plate_number.upper() if c.isalnum())
        if not safe_plate or not clip_path.exists():
            rel = str(clip_path).replace("\\", "/")
            try:
                rel = str(clip_path.relative_to(settings.CAPTURES_DIR.parent)).replace("\\", "/")
            except ValueError:
                pass
            return clip_path, rel

        stem = clip_path.stem
        suffix_part = stem[5:] if stem.startswith("clip_") else stem
        new_path = clip_path.with_name(f"{safe_plate}_{suffix_part}{clip_path.suffix}")
        try:
            if new_path != clip_path and not new_path.exists():
                clip_path.rename(new_path)
                clip_path = new_path
        except Exception as e:
            logger.debug(f"Could not rename clip {clip_path} -> {new_path}: {e}")

        try:
            rel = str(clip_path.relative_to(settings.CAPTURES_DIR.parent)).replace("\\", "/")
        except ValueError:
            rel = str(clip_path).replace("\\", "/")
        return clip_path, rel

    @staticmethod
    def _save_clip_thumbnail(clip_path: Path, frame: Optional[np.ndarray]) -> Optional[str]:
        """Save a JPEG thumbnail photo alongside the recorded .mp4 clip."""
        if frame is None or frame.size == 0 or not clip_path.exists():
            return None
        thumb_path = clip_path.with_suffix(".jpg")
        try:
            h, w = frame.shape[:2]
            out = frame
            if w > 960:
                out = cv2.resize(frame, (960, max(64, int(h * 960 / w))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            try:
                return str(thumb_path.relative_to(settings.CAPTURES_DIR.parent)).replace("\\", "/")
            except ValueError:
                return str(thumb_path).replace("\\", "/")
        except Exception as e:
            logger.debug(f"Could not save clip thumbnail {thumb_path}: {e}")
            return None

    @staticmethod
    def _remove_unusable_clip(clip_path: Path):
        for p in (clip_path, clip_path.with_suffix(".jpg")):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        with self._rec_lock:
            is_rec = self._active_clip is not None
            rec_elapsed = round(time.time() - self._active_clip.started_at, 1) if self._active_clip else 0.0
            rec_id = self._active_clip.clip_id if self._active_clip else None
        return {
            "enabled": settings.CLIP_RECORDING_ENABLED,
            "is_recording": is_rec,
            "active_clip_id": rec_id,
            "recording_elapsed_s": rec_elapsed,
            "is_processing_clip": self.is_processing_clip,
            "clips_pending": self._clip_queue.qsize(),
            "clips_recorded": self.clips_recorded,
            "clips_processed": self.clips_processed,
            "clips_confirmed": self.clips_confirmed,
            "last_clip_ms": round(self.last_clip_ms, 1),
            "last_clip_result": self.last_clip_result,
        }
