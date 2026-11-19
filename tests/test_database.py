"""
Database Repository, Integrity & Transaction Tests
==================================================
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.models import Base, Vehicle, VehicleEvent, Camera
from app.database.repository import VehicleRepository, EventRepository, DashboardRepository


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def test_vehicle_creation_and_uniqueness(session):
    now = datetime.now(timezone.utc)
    v1 = Vehicle(
        plate_number="KA01AB1234",
        first_seen_at=now,
        last_seen_at=now,
        current_status="IN",
        total_entries=1,
        total_exits=0
    )
    session.add(v1)
    session.commit()

    retrieved = VehicleRepository.get_by_plate(session, "KA01AB1234")
    assert retrieved is not None
    assert retrieved.plate_number == "KA01AB1234"
    assert retrieved.current_status == "IN"

    # Case insensitive lookup
    assert VehicleRepository.get_by_plate(session, "ka01ab1234") is not None

    # Uniqueness constraint violation
    v_dup = Vehicle(
        plate_number="KA01AB1234",
        first_seen_at=now,
        last_seen_at=now,
        current_status="IN"
    )
    session.add(v_dup)
    with pytest.raises(Exception):
        session.commit()
    session.rollback()


def test_event_listing_and_filtering(session):
    now = datetime.now(timezone.utc)
    v = Vehicle(plate_number="MH12AB1234", first_seen_at=now, last_seen_at=now, current_status="OUT")
    session.add(v)
    session.commit()

    e1 = VehicleEvent(
        vehicle_id=v.id,
        plate_number="MH12AB1234",
        event_type="IN",
        event_timestamp=now,
        ocr_confidence=0.95
    )
    e2 = VehicleEvent(
        vehicle_id=v.id,
        plate_number="MH12AB1234",
        event_type="OUT",
        event_timestamp=now,
        ocr_confidence=0.91
    )
    session.add_all([e1, e2])
    session.commit()

    events, total = EventRepository.list_events(session, plate_number="MH12AB1234")
    assert total == 2
    assert len(events) == 2

    # Filter by event_type
    in_events, in_total = EventRepository.list_events(session, event_type="IN")
    assert in_total == 1
    assert in_events[0].event_type == "IN"


def test_dashboard_summary_metrics(session):
    now = datetime.now(timezone.utc)
    v1 = Vehicle(plate_number="KA01AA1111", first_seen_at=now, last_seen_at=now, current_status="IN")
    v2 = Vehicle(plate_number="KA01AA2222", first_seen_at=now, last_seen_at=now, current_status="OUT")
    session.add_all([v1, v2])
    session.commit()

    summary = DashboardRepository.get_summary(session)
    assert summary["total_registered_vehicles"] == 2
    assert summary["vehicles_currently_inside"] == 1
    assert summary["vehicles_currently_outside"] == 1
