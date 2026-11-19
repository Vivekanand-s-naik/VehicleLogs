"""
ANPR Pipeline Event Service
===========================
Implements Event-Triggered Video Clip Recording & Deferred Best-Frame Processing:

1. Capture thread (CameraManager) -> latest-frame buffer with monotonic frame_id.
2. Live Pipeline thread (ANPRPipelineWorker):
   - Runs fast plate detection (including small/distant plates down to 14x8px)
     and direct plate tracking at full camera FPS — NO instant blocking OCR.
   - The moment ANY unconfirmed license plate is detected, starts recording
     the raw video stream via `ClipRecorderService` (with pre-roll).
   - Once the vehicle passes (post-roll) or `RECORD_MAX_DURATION` elapses,
     hands the recorded video clip to `ClipProcessorWorker`.
3. Deferred Video Clip Processor (ANPRClipProcessor):
   - Scans the recorded clip at high resolution (`CLIP_DETECT_IMGSZ=960`),
     selects the Top-K sharpest/largest crops across the entire pass-by,
     runs tiered RapidOCR + fast-plate-ocr + multi-frame voting, and logs
     the confirmed IN/OUT event + best snapshots + video clip atomically.
"""

import time
import queue
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from app.core.config import settings
from app.core.device import get_device_diagnostics
from app.vision.camera_manager import CameraManager
from app.vision.vehicle_detector import VehicleDetector, VehicleDetection
from app.vision.plate_detector import PlateDetector, PlateDetection
from app.vision.object_tracker import ObjectTracker, TrackedObject
from app.vision.image_processor import ImageProcessor
from app.vision.plate_ocr import PlateOCR, OCRResult
from app.recognition.recognition_verifier import MultiFrameRecognitionVerifier, VerifiedPlate
from app.services.vehicle_state_manager import VehicleStateManager
from app.services.clip_recorder import ClipRecorderService, ClipJob, ClipProcessOutcome

logger = logging.getLogger("anpr.pipeline")


@dataclass
class OCRJob:
    plate_track_id: int
    vehicle_track_id: Optional[int]
    plate_crop: np.ndarray
    vehicle_crop: np.ndarray
    det_confidence: float
    quality: float
    direction_hint: Optional[str]
    is_stationary: bool
    enqueued_at: float = field(default_factory=time.time)


@dataclass
class TrackOCRState:
    confirmed_plate: Optional[str] = None
    confirmed_conf: float = 0.0
    event_type: Optional[str] = None
    last_tentative: str = ""
    last_tentative_conf: float = 0.0
    last_ocr_submit: float = 0.0
    attempts: int = 0
    best_quality: float = 0.0
    clip_retry_after: float = 0.0
    last_seen: float = field(default_factory=time.time)


class PipelineMetrics:
    def __init__(self):
        self.fps: float = 0.0
        self.inference_time_ms: float = 0.0
        self.ocr_time_ms: float = 0.0
        self.total_cycle_ms: float = 0.0
        self.total_plates_detected: int = 0
        self.total_ocr_runs: int = 0
        self.total_events_triggered: int = 0
        self.ocr_queue_depth: int = 0
        self.ocr_engine: str = ""
        self.mode: str = "clip_recording"
        self.clip_status: Dict[str, Any] = {}
        self.device_info: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processing_fps": round(self.fps, 1),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "ocr_time_ms": round(self.ocr_time_ms, 2),
            "total_cycle_ms": round(self.total_cycle_ms, 2),
            "total_plates_detected": self.total_plates_detected,
            "total_ocr_runs": self.total_ocr_runs,
            "total_events_triggered": self.total_events_triggered,
            "ocr_queue_depth": self.ocr_queue_depth,
            "ocr_engine": self.ocr_engine,
            "mode": self.mode,
            "clip_recorder": self.clip_status,
            "hardware": self.device_info,
        }


class EventService:
    _instance: Optional["EventService"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        camera_manager: Optional[CameraManager] = None,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[PlateDetector] = None,
        plate_ocr: Optional[PlateOCR] = None,
        state_manager: Optional[VehicleStateManager] = None,
    ):
        self.camera_manager = camera_manager or CameraManager()
        self.vehicle_detector = vehicle_detector or VehicleDetector.get_instance()
        self.plate_detector = plate_detector or PlateDetector.get_instance()
        self.plate_ocr = plate_ocr or PlateOCR.get_instance()
        self.state_manager = state_manager or VehicleStateManager()

        self.tracker = ObjectTracker(max_age=30, min_hits=2, iou_threshold=0.25)
        self.plate_tracker = ObjectTracker(max_age=25, min_hits=1, iou_threshold=0.20, centroid_ratio=1.25)
        self.verifier = MultiFrameRecognitionVerifier(
            window_size=settings.RECOGNITION_WINDOW,
            min_votes=settings.RECOGNITION_MIN_VOTES,
        )

        self.clip_recorder = ClipRecorderService(
            plate_detector=self.plate_detector,
            vehicle_detector=self.vehicle_detector,
            plate_ocr=self.plate_ocr,
            state_manager=self.state_manager,
            on_clip_processed=self._handle_clip_outcome,
        )

        self.metrics = PipelineMetrics()
        self.metrics.ocr_engine = getattr(self.plate_ocr, "engine_name", "unknown")
        self.metrics.mode = "clip_recording" if settings.CLIP_RECORDING_ENABLED else "instant_ocr"

        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._track_states: Dict[int, TrackOCRState] = {}
        self._cached_vehicles: List[VehicleDetection] = []

        self._annotated_frame: Optional[np.ndarray] = None
        self._annotated_frame_id: int = 0
        self._last_processed_frame_id: int = 0
        self._frame_count: int = 0

        self._ocr_queue: queue.Queue = queue.Queue(maxsize=max(2, settings.OCR_QUEUE_SIZE))
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._ocr_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(cls) -> "EventService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def start(self):
        """Start camera capture, clip processor, OCR worker, and live pipeline threads."""
        self.camera_manager.start()
        self._stop_event.clear()
        self.clip_recorder.start()

        if self._ocr_thread is None or not self._ocr_thread.is_alive():
            self._ocr_thread = threading.Thread(
                target=self._ocr_worker_loop,
                name="ANPROCRWorker",
                daemon=True,
            )
            self._ocr_thread.start()

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._pipeline_loop,
                name="ANPRPipelineWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info(
                f"ANPR pipeline started (mode={self.metrics.mode}, "
                f"max_clip={settings.RECORD_MAX_DURATION}s, post_roll={settings.RECORD_POST_ROLL_SECONDS}s)."
            )

    def stop(self):
        """Stop all worker threads, recorder, and camera cleanly."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        if self._ocr_thread and self._ocr_thread.is_alive():
            self._ocr_thread.join(timeout=3.0)
        self.clip_recorder.stop()
        self.camera_manager.stop()
        logger.info("ANPR Event processing pipeline stopped.")

    def reset_runtime_state(self):
        """Clear in-memory track locks, cooldowns, clip stats, and counters after admin data purge."""
        self.state_manager.reset_memory_state()
        self.clip_recorder.reset()
        with self._state_lock:
            self._track_states.clear()
        self.metrics.total_plates_detected = 0
        self.metrics.total_ocr_runs = 0
        self.metrics.total_events_triggered = 0

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._annotated_frame.copy() if self._annotated_frame is not None else None

    def get_latest_annotated_frame_with_id(self):
        with self._lock:
            if self._annotated_frame is None:
                return 0, None
            return self._annotated_frame_id, self._annotated_frame.copy()

    # ------------------------------------------------- Clip Outcome Callback
    def _handle_clip_outcome(self, job: ClipJob, outcome: ClipProcessOutcome):
        """Update live track annotations & metrics once a recorded video clip finishes processing."""
        self.metrics.total_ocr_runs += outcome.crops_evaluated
        if outcome.processing_ms > 0:
            self.metrics.ocr_time_ms = outcome.processing_ms
        if outcome.event_recorded:
            self.metrics.total_events_triggered += 1

        now = time.time()
        with self._state_lock:
            for tid in job.track_ids:
                st = self._track_states.get(tid)
                if st is None:
                    continue
                if outcome.plate_number:
                    st.confirmed_plate = outcome.plate_number
                    st.confirmed_conf = outcome.confidence
                    st.event_type = outcome.event_type
                else:
                    # Clip had no readable plate (e.g. still too far/blurry); retry briefly if vehicle nears
                    st.clip_retry_after = now + 2.0

    # ------------------------------------------------------- Fallback OCR Worker
    def _enqueue_ocr_job(self, job: OCRJob):
        try:
            self._ocr_queue.put_nowait(job)
        except queue.Full:
            try:
                self._ocr_queue.get_nowait()
                self._ocr_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._ocr_queue.put_nowait(job)
            except queue.Full:
                pass
        self.metrics.ocr_queue_depth = self._ocr_queue.qsize()

    def _ocr_worker_loop(self):
        while not self._stop_event.is_set():
            try:
                job: OCRJob = self._ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._execute_ocr_job(job)
            except Exception as e:
                logger.error(f"Unhandled error in OCR worker for track #{job.plate_track_id}: {e}", exc_info=True)
            finally:
                self._ocr_queue.task_done()
                self.metrics.ocr_queue_depth = self._ocr_queue.qsize()

    def _execute_ocr_job(self, job: OCRJob):
        with self._state_lock:
            st = self._track_states.get(job.plate_track_id)
            if st is None or st.confirmed_plate is not None:
                return

        t0 = time.perf_counter()
        ocr_res: Optional[OCRResult] = self.plate_ocr.recognize(job.plate_crop)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.metrics.total_ocr_runs += 1
        self.metrics.ocr_time_ms = (
            elapsed_ms if self.metrics.ocr_time_ms == 0.0
            else (self.metrics.ocr_time_ms * 0.8 + elapsed_ms * 0.2)
        )

        if ocr_res is None:
            return

        with self._state_lock:
            st = self._track_states.get(job.plate_track_id)
            if st is None or st.confirmed_plate is not None:
                return
            st.last_tentative = ocr_res.normalized_plate or ocr_res.raw_text
            st.last_tentative_conf = ocr_res.confidence

        if not (ocr_res.is_valid and ocr_res.confidence >= settings.OCR_CONFIDENCE):
            return

        vote_key = f"plate_{job.plate_track_id}"
        self.verifier.add_observation(
            tracking_id=vote_key,
            plate_number=ocr_res.normalized_plate,
            confidence=ocr_res.confidence,
            raw_text=ocr_res.raw_text,
            plate_crop=job.plate_crop,
            vehicle_crop=job.vehicle_crop,
        )

        verified: Optional[VerifiedPlate] = self.verifier.get_verified(vote_key)
        if verified is None:
            return

        transition = self.state_manager.process_detection(
            plate_number=verified.plate_number,
            confidence=verified.average_confidence,
            detection_confidence=job.det_confidence,
            raw_ocr_text=verified.best_raw_text,
            tracking_id=job.vehicle_track_id if job.vehicle_track_id is not None else job.plate_track_id,
            direction_hint=job.direction_hint,
            is_stationary=job.is_stationary,
            vehicle_image=verified.best_vehicle_crop,
            plate_image=verified.best_plate_crop,
            state_name=ocr_res.state_name,
        )

        with self._state_lock:
            st = self._track_states.get(job.plate_track_id)
            if st is not None:
                st.confirmed_plate = verified.plate_number
                st.confirmed_conf = verified.average_confidence
                if transition.success:
                    st.event_type = transition.event_type

        if transition.success:
            self.metrics.total_events_triggered += 1

    # ------------------------------------------------------- Detection Pipeline
    def _pipeline_loop(self):
        last_time = time.time()
        last_diag_time = 0.0

        while not self._stop_event.is_set():
            try:
                frame_id, frame = self.camera_manager.get_latest_frame_with_id()
                if frame is None or frame_id == self._last_processed_frame_id:
                    time.sleep(0.005)
                    continue
                self._last_processed_frame_id = frame_id
                self._frame_count += 1

                if settings.FRAME_SKIP > 1 and (self._frame_count % settings.FRAME_SKIP) != 0:
                    continue

                start_t = time.perf_counter()
                annotated = self.process_frame(frame, sync_ocr=False)

                now = time.time()
                if (now - last_diag_time) >= 1.5:
                    self.metrics.device_info = get_device_diagnostics()
                    last_diag_time = now
                self.metrics.clip_status = self.clip_recorder.get_status()

                with self._lock:
                    self._annotated_frame = annotated
                    self._annotated_frame_id += 1

                dt = now - last_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.metrics.fps = inst_fps if self.metrics.fps == 0.0 else (self.metrics.fps * 0.9 + inst_fps * 0.1)
                last_time = now
                self.metrics.total_cycle_ms = (time.perf_counter() - start_t) * 1000.0
            except Exception as e:
                logger.error(f"Error in pipeline loop (recovering cleanly): {e}", exc_info=True)
                time.sleep(0.05)

    def _prune_stale_track_states(self):
        active_plate_ids = set(self.plate_tracker.active_ids())
        now = time.time()
        with self._state_lock:
            for tid in list(self._track_states.keys()):
                if tid not in active_plate_ids and (now - self._track_states[tid].last_seen) > 4.0:
                    del self._track_states[tid]
                    self.verifier.clear(f"plate_{tid}")

    def process_frame(self, frame: np.ndarray, sync_ocr: bool = True) -> np.ndarray:
        """
        Detect vehicles & plates on a frame.
        - In live clip-recording mode (`sync_ocr=False` and `CLIP_RECORDING_ENABLED=True`):
          immediately triggers/feeds video clip recording on ANY detected plate
          (even small/distant ones) without stalling the feed for instant OCR.
        - In direct single-frame mode (`sync_ocr=True` or `CLIP_RECORDING_ENABLED=False`):
          runs OCR directly on detected plates.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        virtual_line_y = int(h * settings.VIRTUAL_LINE_Y)
        now = time.time()

        t_infer_start = time.perf_counter()

        # 1. Vehicle Detection (every VEHICLE_DETECT_INTERVAL frames)
        run_veh = (
            not self._cached_vehicles
            or (self._frame_count % max(1, settings.VEHICLE_DETECT_INTERVAL)) == 0
        )
        if run_veh:
            vehicles: List[VehicleDetection] = self.vehicle_detector.detect(frame)
            self.tracker.update(vehicles)
            self._cached_vehicles = vehicles
        else:
            vehicles = self._cached_vehicles

        # 2. License Plate Detection & Direct Plate Tracking (every frame, catches distant plates)
        plates: List[PlateDetection] = self.plate_detector.detect(frame)
        self.plate_tracker.update(plates)

        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
        self.metrics.inference_time_ms = (
            infer_ms if self.metrics.inference_time_ms == 0.0
            else (self.metrics.inference_time_ms * 0.8 + infer_ms * 0.2)
        )

        if not self.metrics.device_info:
            self.metrics.device_info = get_device_diagnostics()

        # Virtual gate line overlay
        if settings.EVENT_STRATEGY.lower() == "line_crossing":
            cv2.line(annotated, (0, virtual_line_y), (w, virtual_line_y), (0, 165, 255), 2)
            cv2.putText(
                annotated, "VIRTUAL ENTRY/EXIT GATE LINE", (15, virtual_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
            )

        # Vehicle boxes
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v.bbox
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 140, 0), 2)
            label = f"{v.class_name} #{v.tracking_id}" if v.tracking_id else v.class_name
            cv2.putText(annotated, label, (vx1, max(18, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)

        # 3. Inspect tracked plates & separate unconfirmed vs confirmed
        unconfirmed_plates: List[PlateDetection] = []
        primary_vtid: Optional[int] = None
        primary_direction: Optional[str] = None
        primary_stationary: bool = False
        use_clip_mode = settings.CLIP_RECORDING_ENABLED and not sync_ocr

        for plate in plates:
            px1, py1, px2, py2 = plate.bbox
            self.metrics.total_plates_detected += 1
            ptid = plate.tracking_id or 0

            matched_vehicle = self.vehicle_detector.associate_plate_to_vehicle(plate.bbox, vehicles)
            vtid = matched_vehicle.tracking_id if matched_vehicle else None

            direction_hint = None
            is_stationary = False
            track_obj: Optional[TrackedObject] = None
            if vtid and vtid in self.tracker.tracks:
                track_obj = self.tracker.tracks[vtid]
            elif ptid in self.plate_tracker.tracks:
                track_obj = self.plate_tracker.tracks[ptid]

            if track_obj is not None:
                direction_hint = track_obj.check_line_crossing(virtual_line_y)
                is_stationary = track_obj.is_stationary

            with self._state_lock:
                st = self._track_states.setdefault(ptid, TrackOCRState())
                st.last_seen = now
                confirmed_plate = st.confirmed_plate
                confirmed_conf = st.confirmed_conf
                event_tag = st.event_type
                tentative_text = st.last_tentative
                retry_ready = now >= st.clip_retry_after
                can_submit_instant = (
                    confirmed_plate is None
                    and st.attempts < settings.OCR_MAX_ATTEMPTS_PER_TRACK
                    and (now - st.last_ocr_submit) >= settings.OCR_INTERVAL_PER_TRACK
                )

            if confirmed_plate is not None:
                self.state_manager.mark_seen(confirmed_plate)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 230, 80), 2)
                tag = f" [{event_tag}]" if event_tag else ""
                plate_label = f"{confirmed_plate}{tag} ({confirmed_conf * 100:.0f}%)"
                cv2.putText(
                    annotated, plate_label, (px1, max(20, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA,
                )
                continue

            if use_clip_mode:
                if retry_ready:
                    unconfirmed_plates.append(plate)
                    if primary_vtid is None and vtid is not None:
                        primary_vtid = vtid
                    if primary_direction is None and direction_hint is not None:
                        primary_direction = direction_hint
                    primary_stationary = primary_stationary or is_stationary

                # Annotate plate box as recording/queued
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 120, 255), 2)
                lbl = f"REC PLATE #{ptid}"
                cv2.putText(
                    annotated, lbl, (px1, max(18, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2, cv2.LINE_AA,
                )
            else:
                # Direct / instant OCR mode (also used by single-frame tests)
                if can_submit_instant and plate.width >= settings.OCR_MIN_PLATE_WIDTH:
                    plate_crop = ImageProcessor.crop_with_padding(
                        frame, plate.bbox, padding_ratio=settings.PLATE_PADDING_RATIO
                    )
                    if plate_crop is not None and plate_crop.size > 0:
                        quality = ImageProcessor.quality_score(plate_crop, plate.confidence)
                        vehicle_crop = (
                            ImageProcessor.crop_with_padding(frame, matched_vehicle.bbox, padding_ratio=0.05)
                            if matched_vehicle else None
                        )
                        if vehicle_crop is None:
                            vehicle_crop = frame
                        with self._state_lock:
                            st = self._track_states.get(ptid)
                            if st is not None:
                                st.last_ocr_submit = now
                                st.attempts += 1
                                st.best_quality = max(st.best_quality, quality)
                        job = OCRJob(
                            plate_track_id=ptid,
                            vehicle_track_id=vtid,
                            plate_crop=plate_crop,
                            vehicle_crop=vehicle_crop,
                            det_confidence=plate.confidence,
                            quality=quality,
                            direction_hint=direction_hint,
                            is_stationary=is_stationary,
                        )
                        if sync_ocr:
                            self._execute_ocr_job(job)
                        else:
                            self._enqueue_ocr_job(job)

                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 220, 255), 2)
                lbl = tentative_text if tentative_text else f"SCANNING #{ptid}"
                cv2.putText(
                    annotated, lbl, (px1, max(18, py1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA,
                )

        # 4. Feed raw frame into the Event-Triggered Clip Recorder
        if use_clip_mode:
            self.clip_recorder.feed_frame(
                raw_frame=frame,
                plates=plates,
                unconfirmed_plates=unconfirmed_plates,
                vehicle_track_id=primary_vtid,
                direction_hint=primary_direction,
                is_stationary=primary_stationary,
                actual_fps=self.camera_manager.fps_actual or self.metrics.fps or 20.0,
            )
            self.metrics.clip_status = self.clip_recorder.get_status()

        self._prune_stale_track_states()
        self._draw_hud(annotated)
        return annotated

    def _draw_hud(self, img: np.ndarray):
        h, w = img.shape[:2]
        hud_h = 38
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

        dev_name = self.metrics.device_info.get("device_name", "CPU")
        active_dev = self.metrics.device_info.get("active_device", "cpu")
        clip_st = self.metrics.clip_status

        rec_tag = ""
        if clip_st.get("is_recording"):
            rec_tag = f" | REC {clip_st.get('recording_elapsed_s', 0):.1f}s"
            # Draw pulsing red recording indicator on right side
            cv2.circle(img, (w - 22, 19), 7, (0, 0, 255), -1)
        elif clip_st.get("is_processing_clip"):
            rec_tag = " | PROCESSING CLIP"
            cv2.circle(img, (w - 22, 19), 7, (0, 200, 255), -1)

        hud_text = (
            f"FPS: {self.metrics.fps:.1f} | "
            f"YOLO: {self.metrics.inference_time_ms:.1f}ms | "
            f"ClipOCR: {self.metrics.ocr_time_ms:.0f}ms | "
            f"Dev: {active_dev.upper()} ({dev_name}) | "
            f"Events: {self.metrics.total_events_triggered}{rec_tag}"
        )
        cv2.putText(img, hud_text, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
