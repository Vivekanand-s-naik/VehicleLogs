"""
SQLAlchemy ORM Database Models
==============================
Defines schema for Users (RBAC), Vehicles, VehicleEvents, Cameras, and SystemEvents.
All timestamps are stored in UTC.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON, Index, Boolean
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(32), default="OPERATOR", nullable=False)  # "ADMIN", "OPERATOR", "AUDITOR"
    full_name = Column(String(128), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"


class Vehicle(Base):
    __tablename__ = "vehicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(32), unique=True, nullable=False, index=True)
    first_seen_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_seen_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    current_status = Column(String(16), default="IN", nullable=False, index=True)  # "IN" or "OUT"
    total_entries = Column(Integer, default=0, nullable=False)
    total_exits = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    events = relationship("VehicleEvent", back_populates="vehicle", cascade="all, delete-orphan", order_by="desc(VehicleEvent.event_timestamp)")

    def __repr__(self) -> str:
        return f"<Vehicle(id={self.id}, plate='{self.plate_number}', status='{self.current_status}')>"


class VehicleEvent(Base):
    __tablename__ = "vehicle_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), nullable=False, index=True)
    plate_number = Column(String(32), nullable=False, index=True)
    event_type = Column(String(16), nullable=False, index=True)  # "IN" or "OUT"
    event_timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    raw_ocr_text = Column(String(64), nullable=True)
    ocr_confidence = Column(Float, default=0.0, nullable=False)
    detection_confidence = Column(Float, default=0.0, nullable=False)
    tracking_id = Column(Integer, nullable=True, index=True)
    vehicle_image_path = Column(String(255), nullable=True)
    plate_image_path = Column(String(255), nullable=True)
    video_clip_path = Column(String(255), nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    processing_metadata = Column(JSON, nullable=True)

    vehicle = relationship("Vehicle", back_populates="events")
    camera = relationship("Camera", back_populates="events")

    __table_args__ = (
        Index("ix_plate_timestamp", "plate_number", "event_timestamp"),
        Index("ix_event_type_timestamp", "event_type", "event_timestamp"),
    )

    def __repr__(self) -> str:
        return f"<VehicleEvent(id={self.id}, plate='{self.plate_number}', type='{self.event_type}', time='{self.event_timestamp}')>"


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_name = Column(String(64), default="Main Entrance Camera", nullable=False)
    camera_index = Column(Integer, default=0, nullable=False)
    location = Column(String(128), default="Gate 1 Entrance/Exit", nullable=False)
    status = Column(String(32), default="ACTIVE", nullable=False)  # ACTIVE, INACTIVE, DISCONNECTED
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    events = relationship("VehicleEvent", back_populates="camera")

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name='{self.camera_name}', status='{self.status}')>"


class SystemEvent(Base):
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), default="INFO", nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    metadata_json = Column("metadata", JSON, nullable=True)

    def __repr__(self) -> str:
        return f"<SystemEvent(id={self.id}, type='{self.event_type}', sev='{self.severity}', time='{self.timestamp}')>"
