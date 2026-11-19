"""
Lightweight Object Tracker with Trajectory Analysis & Virtual Line Crossing
===========================================================================
Tracks vehicles across consecutive frames using IoU & Centroid association.
Maintains movement history, detects stationary vehicles, and determines
virtual entry/exit line crossing direction.
"""

import time
from typing import List, Tuple, Dict, Optional, Any


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class TrackedObject:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], class_id: int):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.created_at = time.time()
        self.last_seen = time.time()
        self.hits = 1
        self.time_since_update = 0
        self.history: List[Tuple[int, int, float]] = []  # (cx, cy, timestamp)
        self.line_crossing_event: Optional[str] = None  # "IN" or "OUT"
        self.is_stationary = False
        self._add_history(bbox)

    def _add_history(self, bbox: Tuple[int, int, int, int]):
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        now = time.time()
        self.history.append((cx, cy, now))
        if len(self.history) > 30:
            self.history.pop(0)

        # Check stationary condition:
        if len(self.history) >= 10:
            xs = [p[0] for p in self.history[-10:]]
            ys = [p[1] for p in self.history[-10:]]
            displacement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            self.is_stationary = displacement < 8.0

    def update(self, bbox: Tuple[int, int, int, int]):
        self.bbox = bbox
        self.last_seen = time.time()
        self.hits += 1
        self.time_since_update = 0
        self._add_history(bbox)

    def check_line_crossing(self, line_y: int) -> Optional[str]:
        """
        Check if trajectory crossed virtual horizontal line.
        Downwards crossing (top to bottom) -> IN
        Upwards crossing (bottom to top) -> OUT
        """
        if len(self.history) < 2:
            return None

        # Compare centroid a few frames ago to current centroid
        prev_cy = self.history[0][1]
        curr_cy = self.history[-1][1]

        # Top to bottom crossing
        if prev_cy < line_y <= curr_cy:
            return "IN"
        # Bottom to top crossing
        elif prev_cy > line_y >= curr_cy:
            return "OUT"

        return None


class ObjectTracker:
    """
    Lightweight IoU tracker with a centroid-distance fallback.
    Works for any detection object exposing `.bbox`, `.class_id` and a writable
    `.tracking_id` (vehicles and license plates).
    """

    def __init__(self, max_age: int = 25, min_hits: int = 2, iou_threshold: float = 0.3,
                 centroid_ratio: float = 0.8):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.centroid_ratio = centroid_ratio
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    @staticmethod
    def _centroid_close(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], ratio: float) -> float:
        """Return a pseudo-score in (0, iou_threshold) if centroids are close, else 0."""
        acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
        bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        scale = max(a[2] - a[0], b[2] - b[0], 1)
        dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
        return 1.0 - dist / (ratio * scale) if dist < ratio * scale else 0.0

    def update(self, detections: List[Any]) -> List[TrackedObject]:
        """
        Update active tracks with new detections.
        Matches greedily by IoU, then by centroid proximity for fast-moving
        objects whose boxes no longer overlap. Returns confirmed active tracks.
        """
        for track in self.tracks.values():
            track.time_since_update += 1

        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            pairs = []
            for tid in track_ids:
                tb = self.tracks[tid].bbox
                for d_idx, det in enumerate(detections):
                    iou = calculate_iou(tb, det.bbox)
                    if iou >= self.iou_threshold:
                        pairs.append((1.0 + iou, tid, d_idx))  # IoU matches always rank first
                    else:
                        c = self._centroid_close(tb, det.bbox, self.centroid_ratio)
                        if c > 0:
                            pairs.append((c, tid, d_idx))
            pairs.sort(key=lambda p: p[0], reverse=True)
            for _score, tid, d_idx in pairs:
                if tid in matched_tracks or d_idx not in unmatched_dets:
                    continue
                det = detections[d_idx]
                self.tracks[tid].update(det.bbox)
                det.tracking_id = tid
                matched_tracks.add(tid)
                unmatched_dets.remove(d_idx)

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_id = self.next_id
            self.next_id += 1
            self.tracks[new_id] = TrackedObject(new_id, det.bbox, getattr(det, "class_id", 0))
            det.tracking_id = new_id

        dead_ids = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]

        return [
            t for t in self.tracks.values()
            if t.hits >= self.min_hits and t.time_since_update == 0
        ]

    def active_ids(self) -> List[int]:
        return list(self.tracks.keys())
