"""
Pydantic Schemas: Vehicles
==========================
Request/Response models for Vehicle endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class VehicleBase(BaseModel):
    plate_number: str
    current_status: str


class VehicleResponse(VehicleBase):
    id: int
    first_seen_at: datetime
    last_seen_at: datetime
    total_entries: int
    total_exits: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedVehicles(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[VehicleResponse]
