"""
Pydantic Schemas: Vehicle Events
================================
Request/Response models for Vehicle Movement Event endpoints.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict


class VehicleEventResponse(BaseModel):
    id: int
    vehicle_id: int
    plate_number: str
    event_type: str
    event_timestamp: datetime
    raw_ocr_text: Optional[str] = None
    ocr_confidence: float
    detection_confidence: float
    tracking_id: Optional[int] = None
    vehicle_image_path: Optional[str] = None
    plate_image_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    camera_id: Optional[int] = None
    processing_metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class PaginatedEvents(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[VehicleEventResponse]
