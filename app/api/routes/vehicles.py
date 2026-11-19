"""
Vehicle API Routes with RBAC Controls
====================================
Endpoints for vehicle directory, details, history, manual gate actions,
and administrative status overrides.
"""

from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.api.dependencies import get_db, require_role, get_event_service
from app.database.repository import VehicleRepository, EventRepository
from app.database.models import Vehicle, utc_now
from app.schemas.vehicle import VehicleResponse, PaginatedVehicles
from app.schemas.event import VehicleEventResponse
from app.services.event_service import EventService

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])


class ManualEntryRequest(BaseModel):
    event_type: str  # "IN" or "OUT"
    notes: Optional[str] = "Manual entry by gate operator"


class StatusOverrideRequest(BaseModel):
    new_status: str  # "IN" or "OUT"
    reason: Optional[str] = "Administrative status correction"


@router.get("", response_model=PaginatedVehicles)
def list_vehicles(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by plate number"),
    status: Optional[str] = Query(None, description="Filter by status: IN or OUT"),
    db: Session = Depends(get_db)
):
    """List registered vehicles with search and pagination."""
    items, total = VehicleRepository.list_vehicles(
        db, skip=skip, limit=limit, search=search, status=status
    )
    return PaginatedVehicles(total=total, skip=skip, limit=limit, items=items)


@router.get("/{plate_number}", response_model=VehicleResponse)
def get_vehicle(plate_number: str, db: Session = Depends(get_db)):
    """Retrieve details of a vehicle by its registration plate number."""
    vehicle = VehicleRepository.get_by_plate(db, plate_number)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle '{plate_number}' not found"
        )
    return vehicle


@router.get("/{plate_number}/history", response_model=List[VehicleEventResponse])
def get_vehicle_history(plate_number: str, db: Session = Depends(get_db)):
    """Retrieve complete chronological entry/exit history for a specific vehicle."""
    vehicle = VehicleRepository.get_by_plate(db, plate_number)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle '{plate_number}' not found"
        )
    events = EventRepository.get_vehicle_history(db, plate_number)
    return events


@router.post(
    "/{plate_number}/manual-entry",
    dependencies=[Depends(require_role(["ADMIN", "OPERATOR"]))]
)
def manual_gate_entry(
    plate_number: str,
    req: ManualEntryRequest,
    event_service: EventService = Depends(get_event_service)
):
    """
    Operator/Admin manual entry or exit trigger (e.g., physical gate buzzer override).
    Registers movement event, captures snapshot, and updates vehicle status.
    """
    ev_type = req.event_type.upper().strip()
    if ev_type not in ("IN", "OUT"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="event_type must be 'IN' or 'OUT'"
        )

    # Get current camera frame for proof of manual entry
    _, frame = event_service.camera_manager.get_latest_frame()

    res = event_service.state_manager.process_detection(
        plate_number=plate_number,
        confidence=1.0,
        detection_confidence=1.0,
        raw_ocr_text=f"MANUAL_OVERRIDE_{ev_type}",
        direction_hint=ev_type,
        vehicle_image=frame,
        state_name=f"Manual Override: {req.notes}",
        forced_event_type=ev_type,
    )

    if not res.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Manual entry rejected: {res.reason}"
        )

    return {
        "status": "success",
        "message": f"Vehicle {plate_number} recorded as {res.event_type} successfully.",
        "event_id": res.event_id,
        "event_type": res.event_type
    }


@router.post(
    "/{plate_number}/override-status",
    dependencies=[Depends(require_role(["ADMIN"]))]
)
def override_vehicle_status(
    plate_number: str,
    req: StatusOverrideRequest,
    db: Session = Depends(get_db)
):
    """Administrative status override (ADMIN role only)."""
    new_status = req.new_status.upper().strip()
    if new_status not in ("IN", "OUT"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="new_status must be 'IN' or 'OUT'"
        )

    vehicle = VehicleRepository.get_by_plate(db, plate_number)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle '{plate_number}' not found"
        )

    vehicle.current_status = new_status
    vehicle.updated_at = utc_now()
    db.commit()
    db.refresh(vehicle)

    return {
        "status": "success",
        "plate_number": vehicle.plate_number,
        "current_status": vehicle.current_status,
        "reason": req.reason
    }


@router.delete(
    "/{plate_number}",
    dependencies=[Depends(require_role(["ADMIN"]))]
)
def delete_vehicle(
    plate_number: str,
    db: Session = Depends(get_db),
    event_service: EventService = Depends(get_event_service),
):
    """Delete a vehicle, all its movement history, and archived media (ADMIN role only)."""
    res = VehicleRepository.delete_by_plate(db, plate_number)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle '{plate_number}' not found",
        )
    event_service.state_manager.forget_plate(plate_number)
    return {"status": "deleted", **res}
