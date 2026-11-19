"""
Unit & State Machine Tests for Vehicle State Transitions
========================================================
Mandatory Scenarios:
1. First detection creates an IN event.
2. An IN vehicle subsequently generates an OUT event (after cooldown + leaving view).
3. An OUT vehicle subsequently generates an IN event.
4. Repeated detections during cooldown / same tracking session are rejected as duplicates.
5. A stationary vehicle continuously in front of the camera registers once and NEVER toggles IN/OUT.
6. Manual override (forced_event_type) works even during cooldown.
"""

import time
import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.models import Base
from app.services.vehicle_state_manager import VehicleStateManager
import app.services.vehicle_state_manager as vsm_module


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _patch_session(monkeypatch, test_db):
    @contextmanager
    def mock_transactional_session():
        session = test_db()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(vsm_module, "transactional_session", mock_transactional_session)


def test_state_transitions_and_cooldown(monkeypatch, test_db):
    _patch_session(monkeypatch, test_db)
    mock_capture = MagicMock()
    mock_capture.save_event_images.return_value = ("vehicles/dummy.jpg", "plates/dummy.jpg")
    mock_notif = MagicMock()

    # Short 1-second cooldown and presence timeout for fast deterministic testing
    manager = VehicleStateManager(
        capture_service=mock_capture,
        notification_service=mock_notif,
        cooldown_seconds=1,
        presence_timeout=0.8,
    )

    plate = "KA01AB9999"

    # Scenario 1: First detection creates IN event
    res1 = manager.process_detection(plate_number=plate, confidence=0.92, tracking_id=101)
    assert res1.success is True
    assert res1.event_type == "IN"

    # Scenario 4: Immediate repeated detection rejected due to active tracking / cooldown
    res_repeat = manager.process_detection(plate_number=plate, confidence=0.95, tracking_id=101)
    assert res_repeat.success is False
    assert "already registered" in res_repeat.reason or "Cooldown active" in res_repeat.reason

    # Scenario 5: Wait for cooldown & presence timeout (vehicle left view and returned)
    time.sleep(1.1)

    # Scenario 2: IN vehicle generates OUT event on new tracking session
    res2 = manager.process_detection(plate_number=plate, confidence=0.94, tracking_id=102)
    assert res2.success is True
    assert res2.event_type == "OUT"

    # Wait for cooldown & presence timeout
    time.sleep(1.1)

    # Scenario 3: OUT vehicle generates IN event
    res3 = manager.process_detection(plate_number=plate, confidence=0.91, tracking_id=103)
    assert res3.success is True
    assert res3.event_type == "IN"


def test_stationary_vehicle_presence_suppression(monkeypatch, test_db):
    """A parked vehicle continuously visible in view registers once and never toggles even after cooldown."""
    _patch_session(monkeypatch, test_db)
    mock_capture = MagicMock()
    mock_capture.save_event_images.return_value = (None, None)
    mock_notif = MagicMock()

    manager = VehicleStateManager(
        capture_service=mock_capture,
        notification_service=mock_notif,
        cooldown_seconds=1,
        presence_timeout=1.5,
    )

    plate = "KA01AB1111"

    # Vehicle pulls up and stops at the gate -> registers IN once
    first = manager.process_detection(plate_number=plate, confidence=0.92, tracking_id=201, is_stationary=True)
    assert first.success is True
    assert first.event_type == "IN"

    # Cooldown elapses while vehicle stays parked and visible (mark_seen keeps presence alive)
    time.sleep(0.6)
    manager.mark_seen(plate)
    time.sleep(0.6)

    # Even if a tracker ID flickers (new tracking_id=202), presence suppression blocks duplicate toggle
    second = manager.process_detection(plate_number=plate, confidence=0.93, tracking_id=202, is_stationary=True)
    assert second.success is False
    assert "continuously present" in second.reason.lower()
