"""
Tests for Event-Triggered Video Clip Recording, Deferred Clip OCR Processing,
and RBAC-Protected Data Deletion Endpoints.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.database.connection import init_db, SessionLocal
from app.database.repository import VehicleRepository, EventRepository
from app.vision.plate_detector import PlateDetection
from app.vision.plate_ocr import PlateOCR
from app.services.clip_recorder import ClipRecorderService
from app.services.event_service import EventService

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_env():
    init_db()


def _login(username: str, password: str) -> str:
    res = client.post("/api/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200
    return res.json()["access_token"]


def _draw_approaching_plate_frame(plate_text: str, plate_w: int, plate_h: int, frame_w=640, frame_h=360):
    """Render a vehicle scene with a plate of configurable size (simulating far -> medium distance)."""
    frame = np.full((frame_h, frame_w, 3), (40, 45, 50), dtype=np.uint8)
    cx, cy = frame_w // 2, frame_h // 2
    x1, y1 = cx - plate_w // 2, cy - plate_h // 2
    x2, y2 = x1 + plate_w, y1 + plate_h

    # Car bumper background around the plate
    cv2.rectangle(frame, (x1 - 40, y1 - 35), (x2 + 40, y2 + 35), (90, 60, 40), -1)
    # Crisp white plate with high-res text resized to target distance size
    hi = np.full((80, 340, 3), (255, 255, 255), dtype=np.uint8)
    cv2.rectangle(hi, (2, 2), (337, 77), (0, 0, 0), 3)
    cv2.putText(hi, plate_text, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 0, 0), 3, cv2.LINE_AA)
    scaled = cv2.resize(hi, (plate_w, plate_h), interpolation=cv2.INTER_AREA)
    frame[y1:y2, x1:x2] = scaled
    return frame, (x1, y1, x2, y2)


def test_trigger_record_and_process_clip_from_distance():
    """
    Verify that:
    1. A small/distant plate immediately starts video clip recording.
    2. Finalizing the clip yields a readable .mp4 file.
    3. Background clip processing reads the video stream, extracts the plate from
       the best frames, and persists the vehicle event + video_clip_path in DB.
    """
    unique_num = f"{int(time.time() * 1000) % 8999 + 1000}"
    plate_text = f"KA01AB{unique_num}"

    # Mock detectors to return exact plate bbox at every distance (so test isolates recording + clip OCR)
    mock_plate_det = MagicMock()
    mock_veh_det = MagicMock()
    mock_veh_det.detect.return_value = []
    mock_veh_det.associate_plate_to_vehicle.return_value = None

    event_service = EventService.get_instance()
    recorder = ClipRecorderService(
        plate_detector=mock_plate_det,
        vehicle_detector=mock_veh_det,
        plate_ocr=PlateOCR.get_instance(),
        state_manager=event_service.state_manager,
    )

    # Simulate vehicle approaching from far (small plate 110x26) to medium distance (220x52)
    sizes = [(110, 26), (140, 34), (175, 42), (210, 50), (220, 52), (190, 46)]
    bboxes_by_frame = []
    for i, (pw, ph) in enumerate(sizes):
        f, bbox = _draw_approaching_plate_frame(plate_text, pw, ph)
        bboxes_by_frame.append(bbox)
        det = PlateDetection(bbox=bbox, confidence=0.82, class_id=0)
        det.tracking_id = 9001
        recorder.feed_frame(
            raw_frame=f,
            plates=[det],
            unconfirmed_plates=[det],
            actual_fps=15.0,
        )

    assert recorder.get_status()["is_recording"] is True
    job = recorder.force_finalize()
    assert job is not None
    assert job.clip_path.exists()
    assert job.frame_count >= len(sizes)

    # Configure detector mock to return plate bbox for each frame read back from the recorded video clip
    call_idx = {"i": 0}

    def _detect_from_clip(frame, imgsz=None, conf=None):
        idx = min(call_idx["i"], len(bboxes_by_frame) - 1)
        call_idx["i"] += 1
        return [PlateDetection(bbox=bboxes_by_frame[idx], confidence=0.85, class_id=0)]

    mock_plate_det.detect.side_effect = _detect_from_clip

    outcome = recorder.process_video_clip(
        clip_path=job.clip_path,
        rel_clip_path=job.rel_clip_path,
        clip_id=job.clip_id,
        duration_s=job.duration_s,
    )

    assert outcome.plate_number == plate_text
    assert outcome.event_recorded is True
    assert outcome.event_type == "IN"
    assert outcome.confidence >= 0.60

    final_clip_path = (settings.CAPTURES_DIR.parent / outcome.rel_clip_path).resolve()
    assert final_clip_path.exists()
    assert plate_text in final_clip_path.name
    assert final_clip_path.with_suffix(".jpg").exists()  # Poster thumbnail stored beside clip

    # Verify DB persisted video_clip_path column + processing_metadata
    with SessionLocal() as db:
        history = EventRepository.get_vehicle_history(db, plate_text)
        assert len(history) == 1
        assert history[0].video_clip_path == outcome.rel_clip_path
        meta = history[0].processing_metadata or {}
        assert meta.get("video_clip_path") == outcome.rel_clip_path

    # Verify Stored Clips API lists this clip with thumbnail and replay URLs
    clips_res = client.get("/api/clips")
    assert clips_res.status_code == 200
    found = [c for c in clips_res.json()["items"] if c["filename"] == final_clip_path.name]
    assert len(found) == 1
    assert found[0]["plate_number"] == plate_text
    assert found[0]["thumbnail_url"] is not None

    # Non-admin CANNOT delete a stored clip
    guard_token = _login("guard", "guard123")
    assert client.delete(
        f"/api/clips/{final_clip_path.name}",
        headers={"Authorization": f"Bearer {guard_token}"},
    ).status_code == 403

    # Clean up via Admin DELETE vehicle -> removes DB record, renamed clip .mp4, AND thumbnail .jpg
    admin_token = _login("admin", "admin123")
    del_res = client.delete(
        f"/api/vehicles/{plate_text}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_res.status_code == 200
    assert del_res.json()["deleted_plate"] == plate_text
    assert not final_clip_path.exists()
    assert not final_clip_path.with_suffix(".jpg").exists()


def test_rbac_and_delete_event_and_vehicle():
    admin_token = _login("admin", "admin123")
    guard_token = _login("guard", "guard123")
    auditor_token = _login("auditor", "auditor123")

    unique_num = f"{int(time.time() * 1000) % 8999 + 1000}"
    plate = f"MH12DE{unique_num}"

    # Create two events for this plate via guard manual entry
    r1 = client.post(
        f"/api/vehicles/{plate}/manual-entry",
        json={"event_type": "IN", "notes": "Test delete 1"},
        headers={"Authorization": f"Bearer {guard_token}"},
    )
    assert r1.status_code == 200
    ev1_id = r1.json()["event_id"]

    r2 = client.post(
        f"/api/vehicles/{plate}/manual-entry",
        json={"event_type": "OUT", "notes": "Test delete 2"},
        headers={"Authorization": f"Bearer {guard_token}"},
    )
    assert r2.status_code == 200
    ev2_id = r2.json()["event_id"]

    # Non-admin roles MUST be blocked (403) from deleting events, vehicles, or purging data
    for tok in (guard_token, auditor_token):
        assert client.delete(f"/api/events/{ev1_id}", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
        assert client.delete(f"/api/vehicles/{plate}", headers={"Authorization": f"Bearer {tok}"}).status_code == 403
        assert client.delete("/api/system/data?scope=events", headers={"Authorization": f"Bearer {tok}"}).status_code == 403

    # Admin deletes single event -> parent vehicle counters reconcile cleanly
    res_del_ev = client.delete(f"/api/events/{ev2_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_del_ev.status_code == 200
    assert res_del_ev.json()["deleted_event_id"] == ev2_id

    veh_after = client.get(f"/api/vehicles/{plate}").json()
    assert veh_after["total_entries"] == 1
    assert veh_after["total_exits"] == 0
    assert veh_after["current_status"] == "IN"

    # Admin deletes entire vehicle -> vehicle and remaining events disappear
    res_del_v = client.delete(f"/api/vehicles/{plate}", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_del_v.status_code == 200
    assert res_del_v.json()["deleted_events"] == 1
    assert client.get(f"/api/vehicles/{plate}").status_code == 404


def test_admin_purge_preserves_rbac_users():
    admin_token = _login("admin", "admin123")

    # Invalid scope returns 400
    bad = client.delete("/api/system/data?scope=invalid", headers={"Authorization": f"Bearer {admin_token}"})
    assert bad.status_code == 400

    # Seed one record then purge all
    plate = "DL04CA7777"
    client.post(
        f"/api/vehicles/{plate}/manual-entry",
        json={"event_type": "IN", "notes": "Purge test"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    purge = client.delete("/api/system/data?scope=all", headers={"Authorization": f"Bearer {admin_token}"})
    assert purge.status_code == 200
    assert purge.json()["status"] == "purged"

    # Database has 0 vehicles and 0 events
    summary = client.get("/api/dashboard/summary").json()
    assert summary["total_registered_vehicles"] == 0
    assert summary["recent_activity_count"] == 0

    # RBAC users are still intact after full data purge
    assert _login("admin", "admin123")
    assert _login("guard", "guard123")
    assert _login("auditor", "auditor123")
