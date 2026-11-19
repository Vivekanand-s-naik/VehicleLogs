"""
End-to-End System Integration Tests
===================================
Tests full integration: Synthetic Frame -> CV Pipeline -> OCR -> State Manager -> Database.
"""

import time
import cv2
import numpy as np
import pytest
from app.services.event_service import EventService
from app.database.connection import init_db
from app.database.repository import VehicleRepository, EventRepository, DashboardRepository
from app.database.connection import SessionLocal


@pytest.fixture(scope="module", autouse=True)
def setup_integration_env():
    init_db()


def test_end_to_end_frame_processing():
    event_service = EventService.get_instance()

    # Generate synthetic camera frame with a car and plate
    frame = event_service.camera_manager._generate_synthetic_frame()
    assert frame is not None
    assert frame.shape[0] > 0

    # Process frame through the entire pipeline
    annotated = event_service.process_frame(frame)
    assert annotated is not None
    assert annotated.shape == frame.shape

    # Metrics should be populated
    metrics = event_service.metrics.to_dict()
    assert "inference_time_ms" in metrics
    assert "hardware" in metrics
    assert "processing_fps" in metrics


def test_end_to_end_state_and_database_persistence():
    event_service = EventService.get_instance()
    # Use unique plate for isolation across test runs
    unique_suffix = f"{int(time.time() * 1000) % 9000 + 1000}"
    plate = f"KA05XX{unique_suffix}"

    # Create dummy crops
    veh_crop = np.zeros((200, 300, 3), dtype=np.uint8)
    plate_crop = np.zeros((60, 180, 3), dtype=np.uint8)

    # 1. Process detection (first time seen -> IN)
    result = event_service.state_manager.process_detection(
        plate_number=plate,
        confidence=0.95,
        detection_confidence=0.88,
        raw_ocr_text=plate,
        tracking_id=int(unique_suffix),
        vehicle_image=veh_crop,
        plate_image=plate_crop,
        state_name="Karnataka"
    )

    assert result.success is True
    assert result.event_type == "IN"

    # 2. Verify database records
    with SessionLocal() as db:
        vehicle = VehicleRepository.get_by_plate(db, plate)
        assert vehicle is not None
        assert vehicle.current_status == "IN"
        assert vehicle.total_entries == 1

        history = EventRepository.get_vehicle_history(db, plate)
        assert len(history) >= 1
        latest_event = history[0]
        assert latest_event.event_type == "IN"
        assert latest_event.ocr_confidence >= 0.95
        assert latest_event.plate_image_path is not None
