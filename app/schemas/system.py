"""
Pydantic Schemas: System, Cameras & Dashboard
=============================================
Request/Response models for system monitoring, camera operations,
and dashboard summary statistics.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict
from app.schemas.event import VehicleEventResponse


class CameraStatusResponse(BaseModel):
    source: str
    is_connected: bool
    fps_actual: float
    total_frames_captured: int
    reconnect_attempts: int
    configured_resolution: str
    configured_fps: int


class DashboardSummaryResponse(BaseModel):
    total_registered_vehicles: int
    vehicles_currently_inside: int
    vehicles_currently_outside: int
    today_entries: int
    today_exits: int
    recent_activity_count: int
    recent_events: List[VehicleEventResponse]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    timestamp: datetime
    device: str
    device_name: str
    camera_connected: bool
    database_ok: bool


class SystemStatusResponse(BaseModel):
    health: HealthResponse
    camera: CameraStatusResponse
    pipeline_metrics: Dict[str, Any]
    notifications: Dict[str, Any]
    hardware_diagnostics: Dict[str, Any]
