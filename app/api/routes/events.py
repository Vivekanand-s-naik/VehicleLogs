"""
Vehicle Movement Event API Routes
=================================
Endpoints to query, filter, and inspect IN/OUT movement events.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.api.dependencies import get_db, require_role, get_event_service
from app.database.repository import EventRepository
from app.schemas.event import VehicleEventResponse, PaginatedEvents
from app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["Events"])


@router.get("", response_model=PaginatedEvents)
def list_events(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    plate: Optional[str] = Query(None, description="Filter by plate number"),
    event_type: Optional[str] = Query(None, description="Filter by event type: IN or OUT"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_db)
):
    """Retrieve filtered vehicle movement events."""
    items, total = EventRepository.list_events(
        db,
        skip=skip,
        limit=limit,
        plate_number=plate,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date
    )
    return PaginatedEvents(total=total, skip=skip, limit=limit, items=items)


@router.get("/{event_id}", response_model=VehicleEventResponse)
def get_event(event_id: int, db: Session = Depends(get_db)):
    """Retrieve full details of an individual movement event."""
    event = EventRepository.get_by_id(db, event_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event #{event_id} not found"
        )
    return event


@router.delete(
    "/{event_id}",
    dependencies=[Depends(require_role(["ADMIN"]))]
)
def delete_event(
    event_id: int,
    db: Session = Depends(get_db),
    event_service: EventService = Depends(get_event_service),
):
    """Delete an individual movement event and its captured images/clip (ADMIN role only)."""
    res = EventRepository.delete_by_id(db, event_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event #{event_id} not found",
        )
    event_service.state_manager.forget_plate(res["plate_number"])
    return {"status": "deleted", **res}
