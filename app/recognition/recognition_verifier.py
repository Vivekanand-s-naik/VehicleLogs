"""
Multi-Frame Recognition Verifier
================================
Confirms a plate only after consistent evidence across frames.

Voting
------
1. Observations for a track are kept in a bounded, time-limited window.
2. Readings are grouped by length; the heaviest length group (sum of
   confidences) is selected.
3. A confidence-weighted vote is taken at every character position, so a
   single misread character in one frame (e.g. 8 vs 6) is outvoted by the
   other frames.
4. The consensus string is re-validated. It is confirmed when at least
   `min_votes` readings agree with it exactly and its share of total weight
   dominates competing readings.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from app.recognition.plate_validator import IndianVehicleNumberValidator


@dataclass
class PlateObservation:
    timestamp: float
    plate_number: str
    confidence: float
    raw_text: str
    plate_crop: Optional[np.ndarray] = None
    vehicle_crop: Optional[np.ndarray] = None


@dataclass
class VerifiedPlate:
    plate_number: str
    average_confidence: float
    vote_count: int
    best_raw_text: str
    best_plate_crop: Optional[np.ndarray] = None
    best_vehicle_crop: Optional[np.ndarray] = None
    support_ratio: float = 1.0


class MultiFrameRecognitionVerifier:
    def __init__(
        self,
        window_size: int = 5,
        min_votes: int = 2,
        max_age_seconds: float = 8.0,
        min_support_ratio: float = 0.6,
    ):
        self.window_size = window_size
        self.min_votes = min_votes
        self.max_age_seconds = max_age_seconds
        self.min_support_ratio = min_support_ratio
        self._buffers: Dict[Any, List[PlateObservation]] = {}
        self._lock = threading.Lock()

    def add_observation(
        self,
        tracking_id: Any,
        plate_number: str,
        confidence: float,
        raw_text: str,
        plate_crop: Optional[np.ndarray] = None,
        vehicle_crop: Optional[np.ndarray] = None,
    ):
        obs = PlateObservation(time.time(), plate_number, float(confidence), raw_text, plate_crop, vehicle_crop)
        with self._lock:
            buf = self._buffers.setdefault(tracking_id, [])
            buf.append(obs)
            if len(buf) > self.window_size:
                buf.pop(0)
            self._cleanup_stale_locked()

    def observation_count(self, tracking_id: Any) -> int:
        with self._lock:
            return len(self._buffers.get(tracking_id, []))

    def get_verified(self, tracking_id: Any) -> Optional[VerifiedPlate]:
        with self._lock:
            observations = list(self._buffers.get(tracking_id, []))
        if not observations:
            return None

        # 1. Heaviest length group
        groups: Dict[int, List[PlateObservation]] = {}
        for o in observations:
            groups.setdefault(len(o.plate_number), []).append(o)
        group = max(groups.values(), key=lambda g: sum(o.confidence for o in g))

        # 2. Position-wise confidence-weighted vote
        length = len(group[0].plate_number)
        consensus_chars = []
        for i in range(length):
            weights: Dict[str, float] = {}
            for o in group:
                ch = o.plate_number[i]
                weights[ch] = weights.get(ch, 0.0) + o.confidence
            consensus_chars.append(max(weights.items(), key=lambda kv: kv[1])[0])
        consensus = "".join(consensus_chars)

        # 3. Consensus must itself be a valid plate; otherwise fall back to best exact string
        if not IndianVehicleNumberValidator.validate(consensus).is_valid or \
                IndianVehicleNumberValidator.validate(consensus).normalized_plate != consensus:
            totals: Dict[str, float] = {}
            for o in observations:
                totals[o.plate_number] = totals.get(o.plate_number, 0.0) + o.confidence
            consensus = max(totals.items(), key=lambda kv: kv[1])[0]

        agreeing = [o for o in observations if o.plate_number == consensus]
        total_weight = sum(o.confidence for o in observations) or 1.0
        support = sum(o.confidence for o in agreeing) / total_weight

        if len(agreeing) < self.min_votes or support < self.min_support_ratio:
            return None

        best = max(agreeing, key=lambda o: o.confidence)
        best_plate_crop = best.plate_crop
        best_vehicle_crop = best.vehicle_crop
        if best_vehicle_crop is None:
            best_vehicle_crop = next((o.vehicle_crop for o in agreeing if o.vehicle_crop is not None), None)

        return VerifiedPlate(
            plate_number=consensus,
            average_confidence=round(float(np.mean([o.confidence for o in agreeing])), 4),
            vote_count=len(agreeing),
            best_raw_text=best.raw_text,
            best_plate_crop=best_plate_crop,
            best_vehicle_crop=best_vehicle_crop,
            support_ratio=round(support, 3),
        )

    def clear(self, tracking_id: Any):
        with self._lock:
            self._buffers.pop(tracking_id, None)

    def _cleanup_stale_locked(self):
        now = time.time()
        for tid in list(self._buffers.keys()):
            active = [o for o in self._buffers[tid] if (now - o.timestamp) <= self.max_age_seconds]
            if active:
                self._buffers[tid] = active
            else:
                del self._buffers[tid]
