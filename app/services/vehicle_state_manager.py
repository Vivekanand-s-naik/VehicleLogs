"""
Vehicle Entry/Exit State Manager
================================
State machine governing vehicle transitions (IN <-> OUT) with:
- Atomic database transactions
- Configurable event strategies: 'toggle' and 'line_crossing'
- Recognition cooldown enforcement (prevents duplicate records)
- Tracking session lock (prevents stationary vehicles from toggling state)
- Captures and archives images
- Mobile push notifications dispatch
"""

import time
import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
import numpy as np
from sqlalchemy.orm import Session
from app.core.config import settings
from app.database.connection import transactional_session
from app.database.models import Vehicle, VehicleEvent, utc_now
from app.services.capture_service import CaptureService
from app.services.notification_service import NotificationService, build_notification_photo

logger = logging.getLogger("anpr.state")


class StateTransitionResult:
    def __init__(
        self,
        success: bool,
        plate_number: str,
        event_type: Optional[str] = None,
        reason: Optional[str] = None,
        vehicle_id: Optional[int] = None,
        event_id: Optional[int] = None,
        vehicle_image_path: Optional[str] = None,
        plate_image_path: Optional[str] = None
    ):
        self.success = success
        self.plate_number = plate_number
        self.event_type = event_type
        self.reason = reason
        self.vehicle_id = vehicle_id
        self.event_id = event_id
        self.vehicle_image_path = vehicle_image_path
        self.plate_image_path = plate_image_path

    def __repr__(self) -> str:
        return f"<StateTransitionResult(success={self.success}, plate='{self.plate_number}', type={self.event_type}, reason='{self.reason}')>"


class VehicleStateManager:
    def __init__(
        self,
        capture_service: Optional[CaptureService] = None,
        notification_service: Optional[NotificationService] = None,
        cooldown_seconds: Optional[int] = None,
        presence_timeout: Optional[float] = None
    ):
        self.capture_service = capture_service or CaptureService()
        self.notification_service = notification_service or NotificationService.get_instance()
        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else settings.RECOGNITION_COOLDOWN
        self.presence_timeout = presence_timeout if presence_timeout is not None else settings.PRESENCE_TIMEOUT

        # Serializes state transitions (OCR worker, API manual entries) to prevent conflicting writes
        self._lock = threading.RLock()

        # plate -> timestamp of last confirmed event (cooldown)
        self._last_event_timestamps: Dict[str, float] = {}
        # plate -> timestamp the plate was last seen by the camera (presence)
        self._last_seen: Dict[str, float] = {}
        # tracking_id -> plate: a tracked object can trigger at most one event
        self._active_tracking_events: Dict[int, str] = {}

    def mark_seen(self, plate_number: str):
        """Record that a plate is (still) visible. Called every frame a confirmed track is visible."""
        with self._lock:
            self._last_seen[plate_number.upper().strip()] = time.time()

    def reset_memory_state(self):
        """Clear in-memory cooldown, presence, and tracking locks (used on data purge)."""
        with self._lock:
            self._last_event_timestamps.clear()
            self._last_seen.clear()
            self._active_tracking_events.clear()

    def forget_plate(self, plate_number: str):
        """Clear cooldown/presence memory for a deleted plate so it can re-register fresh."""
        key = plate_number.upper().strip()
        with self._lock:
            self._last_event_timestamps.pop(key, None)
            self._last_seen.pop(key, None)
            for tid in [k for k, v in self._active_tracking_events.items() if v == key]:
                del self._active_tracking_events[tid]

    def is_in_cooldown(self, plate_number: str) -> bool:
        """Check if plate was processed within cooldown_seconds."""
        now = time.time()
        last_time = self._last_event_timestamps.get(plate_number.upper().strip(), 0.0)
        return (now - last_time) < self.cooldown_seconds

    def remaining_cooldown(self, plate_number: str) -> float:
        """Return remaining seconds of cooldown."""
        now = time.time()
        last_time = self._last_event_timestamps.get(plate_number.upper().strip(), 0.0)
        remaining = self.cooldown_seconds - (now - last_time)
        return max(0.0, remaining)

    def process_detection(
        self,
        plate_number: str,
        confidence: float,
        detection_confidence: float = 0.0,
        raw_ocr_text: Optional[str] = None,
        tracking_id: Optional[int] = None,
        direction_hint: Optional[str] = None,  # "IN" or "OUT" (from line crossing)
        is_stationary: bool = False,
        vehicle_image: Optional[np.ndarray] = None,
        plate_image: Optional[np.ndarray] = None,
        state_name: Optional[str] = None,
        camera_id: int = 1,
        forced_event_type: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        video_clip_path: Optional[str] = None,
    ) -> StateTransitionResult:
        """
        Process a confirmed plate recognition, apply duplicate-prevention rules,
        execute an atomic database transition, save images, and dispatch notifications.

        Duplicate prevention (in order):
          1. Tracking lock  - one event per tracked object.
          2. Cooldown       - no second event for the same plate within RECOGNITION_COOLDOWN.
          3. Presence       - a plate that has stayed visible (seen within PRESENCE_TIMEOUT)
                              cannot toggle, even after the cooldown. A parked/stationary
                              car therefore produces exactly one event; it must leave the
                              camera view before it can produce the next one.
        `forced_event_type` (manual operator entries) bypasses 2 and 3.
        """
        with self._lock:
            return self._process_locked(
                plate_number, confidence, detection_confidence, raw_ocr_text, tracking_id,
                direction_hint, is_stationary, vehicle_image, plate_image, state_name,
                camera_id, forced_event_type, extra_metadata, video_clip_path,
            )

    def _process_locked(
        self, plate_number, confidence, detection_confidence, raw_ocr_text, tracking_id,
        direction_hint, is_stationary, vehicle_image, plate_image, state_name, camera_id,
        forced_event_type, extra_metadata=None, video_clip_path=None,
    ) -> StateTransitionResult:
        plate = plate_number.upper().strip()
        now = time.time()
        previously_seen = self._last_seen.get(plate)
        self._last_seen[plate] = now
        forced = forced_event_type is not None and forced_event_type.upper() in ("IN", "OUT")

        # 1. Tracking session lock
        if tracking_id is not None and tracking_id in self._active_tracking_events:
            return StateTransitionResult(
                success=False,
                plate_number=plate,
                reason=f"Active tracking session {tracking_id} already registered."
            )

        if not forced:
            # 2. Cooldown
            if self.is_in_cooldown(plate):
                rem = self.remaining_cooldown(plate)
                if tracking_id is not None:
                    self._active_tracking_events[tracking_id] = plate
                return StateTransitionResult(
                    success=False,
                    plate_number=plate,
                    reason=f"Cooldown active ({rem:.1f}s remaining)."
                )

            # 3. Continuous presence (only relevant once the plate already has an event)
            if (
                plate in self._last_event_timestamps
                and previously_seen is not None
                and (now - previously_seen) < self.presence_timeout
            ):
                if tracking_id is not None:
                    self._active_tracking_events[tracking_id] = plate
                return StateTransitionResult(
                    success=False,
                    plate_number=plate,
                    reason="Vehicle continuously present in view; no new movement."
                )

        if is_stationary:
            logger.debug(f"{plate}: stationary vehicle - allowed once per presence session.")

        # 4. Atomic database transaction
        try:
            with transactional_session() as db:
                # Query existing vehicle record
                vehicle = db.query(Vehicle).filter(Vehicle.plate_number == plate).first()

                # Determine next state according to configured strategy
                if forced:
                    next_event_type = forced_event_type.upper()
                else:
                    next_event_type = self._determine_next_state(vehicle, direction_hint)
                if next_event_type is None:
                    return StateTransitionResult(
                        success=False,
                        plate_number=plate,
                        reason="No state transition warranted by current strategy."
                    )

                # Save captured images
                veh_path, plate_path = self.capture_service.save_event_images(
                    plate_number=plate,
                    event_type=next_event_type,
                    vehicle_image=vehicle_image,
                    plate_image=plate_image
                )

                utc_timestamp = utc_now()

                # Update or create vehicle record
                if vehicle is None:
                    # First time vehicle seen: create record
                    vehicle = Vehicle(
                        plate_number=plate,
                        first_seen_at=utc_timestamp,
                        last_seen_at=utc_timestamp,
                        current_status=next_event_type,
                        total_entries=1 if next_event_type == "IN" else 0,
                        total_exits=1 if next_event_type == "OUT" else 0
                    )
                    db.add(vehicle)
                    db.flush()  # Populates vehicle.id
                else:
                    vehicle.last_seen_at = utc_timestamp
                    vehicle.current_status = next_event_type
                    if next_event_type == "IN":
                        vehicle.total_entries += 1
                    else:
                        vehicle.total_exits += 1

                resolved_clip_path = video_clip_path
                if not resolved_clip_path and isinstance(extra_metadata, dict):
                    resolved_clip_path = extra_metadata.get("video_clip_path")

                metadata = {
                    "strategy": settings.EVENT_STRATEGY,
                    "direction_hint": direction_hint,
                    "state_name": state_name,
                }
                if extra_metadata:
                    metadata.update(extra_metadata)
                if resolved_clip_path:
                    metadata["video_clip_path"] = resolved_clip_path

                # Create movement event record
                event = VehicleEvent(
                    vehicle_id=vehicle.id,
                    plate_number=plate,
                    event_type=next_event_type,
                    event_timestamp=utc_timestamp,
                    raw_ocr_text=raw_ocr_text or plate,
                    ocr_confidence=round(confidence, 4),
                    detection_confidence=round(detection_confidence, 4),
                    tracking_id=tracking_id,
                    vehicle_image_path=veh_path,
                    plate_image_path=plate_path,
                    video_clip_path=resolved_clip_path,
                    camera_id=camera_id,
                    processing_metadata=metadata
                )
                db.add(event)
                db.flush()

                event_id = event.id
                veh_id = vehicle.id

            # Database transaction committed successfully
            # Update memory state
            self._last_event_timestamps[plate] = now
            if tracking_id is not None:
                self._active_tracking_events[tracking_id] = plate

            # Clean tracking memory periodically
            if len(self._active_tracking_events) > 500:
                self._active_tracking_events.clear()

            # Build composite clip photo (vehicle frame + zoomed plate inset) and dispatch notification
            photo_bytes = build_notification_photo(
                vehicle_image=vehicle_image,
                plate_image=plate_image,
                plate_number=plate,
                event_type=next_event_type,
                confidence=confidence,
            )
            time_display = utc_timestamp.strftime("%d-%b-%Y %H:%M:%S UTC")
            self.notification_service.dispatch_event(
                plate_number=plate,
                event_type=next_event_type,
                timestamp_str=time_display,
                confidence=confidence,
                state_name=state_name,
                image_bytes=photo_bytes,
                image_path=veh_path or plate_path,
                clip_path=resolved_clip_path,
            )

            logger.info(
                f"[STATE EVENT] Vehicle '{plate}' transitioned to '{next_event_type}' "
                f"(Event #{event_id}, Conf: {confidence:.2f})"
            )

            return StateTransitionResult(
                success=True,
                plate_number=plate,
                event_type=next_event_type,
                vehicle_id=veh_id,
                event_id=event_id,
                vehicle_image_path=veh_path,
                plate_image_path=plate_path
            )

        except Exception as e:
            logger.error(f"Failed to record state transition for {plate}: {e}", exc_info=True)
            return StateTransitionResult(
                success=False,
                plate_number=plate,
                reason=f"Database error: {str(e)}"
            )

    def _determine_next_state(
        self,
        vehicle: Optional[Vehicle],
        direction_hint: Optional[str]
    ) -> Optional[str]:
        """
        Determine next state based on configured strategy:
        - 'toggle': alternates between IN and OUT (first seen = IN)
        - 'line_crossing': uses physical trajectory crossing direction
        """
        strategy = settings.EVENT_STRATEGY.lower()

        if strategy == "line_crossing":
            if direction_hint in ("IN", "OUT"):
                return direction_hint
            # Fallback to toggle if line crossing hasn't registered yet
            return self._toggle_state(vehicle)

        # Default strategy: toggle
        return self._toggle_state(vehicle)

    @staticmethod
    def _toggle_state(vehicle: Optional[Vehicle]) -> str:
        """
        Standard toggle logic:
        No record -> IN
        IN -> OUT
        OUT -> IN
        """
        if vehicle is None:
            return "IN"
        if vehicle.current_status == "IN":
            return "OUT"
        return "IN"
