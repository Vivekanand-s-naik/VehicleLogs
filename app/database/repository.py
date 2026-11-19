"""
Database Repository: Clean Data Access Layer
============================================
Handles all queries, inserts, transactions, aggregations, and pagination.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from app.core.config import settings
from app.database.models import Vehicle, VehicleEvent, Camera, SystemEvent, utc_now


def _safe_delete_capture_file(rel_path: Optional[str]) -> bool:
    """Safely delete a file only if it resolves inside settings.CAPTURES_DIR."""
    if not rel_path:
        return False
    try:
        base_root = settings.CAPTURES_DIR.parent.resolve()
        captures_root = settings.CAPTURES_DIR.resolve()
        candidate = (base_root / rel_path).resolve()
        candidate.relative_to(captures_root)
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
            return True
    except Exception:
        pass
    return False


def _collect_event_media(ev: VehicleEvent) -> List[Optional[str]]:
    paths: List[Optional[str]] = [ev.vehicle_image_path, ev.plate_image_path, ev.video_clip_path]
    clip_paths = set()
    if ev.video_clip_path:
        clip_paths.add(ev.video_clip_path)
    if isinstance(ev.processing_metadata, dict):
        meta_clip = ev.processing_metadata.get("video_clip_path")
        if meta_clip:
            clip_paths.add(meta_clip)
            paths.append(meta_clip)
        meta_thumb = ev.processing_metadata.get("clip_thumbnail_path")
        if meta_thumb:
            paths.append(meta_thumb)
    # Also clean up paired thumbnail .jpg next to any .mp4 clip
    for cp in clip_paths:
        if cp.lower().endswith(".mp4"):
            paths.append(cp[:-4] + ".jpg")
    return paths


class VehicleRepository:
    @staticmethod
    def get_by_plate(db: Session, plate_number: str) -> Optional[Vehicle]:
        return db.query(Vehicle).filter(Vehicle.plate_number == plate_number.upper().strip()).first()

    @staticmethod
    def list_vehicles(
        db: Session,
        skip: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None
    ) -> Tuple[List[Vehicle], int]:
        query = db.query(Vehicle)
        if search:
            query = query.filter(Vehicle.plate_number.ilike(f"%{search.strip()}%"))
        if status:
            query = query.filter(Vehicle.current_status == status.upper().strip())
        
        total = query.count()
        vehicles = query.order_by(desc(Vehicle.last_seen_at)).offset(skip).limit(limit).all()
        return vehicles, total

    @staticmethod
    def delete_by_plate(db: Session, plate_number: str) -> Optional[Dict[str, Any]]:
        """Delete a vehicle, all its movement events, and associated media files."""
        vehicle = VehicleRepository.get_by_plate(db, plate_number)
        if not vehicle:
            return None

        events = list(vehicle.events)
        media_paths: List[Optional[str]] = []
        for ev in events:
            media_paths.extend(_collect_event_media(ev))

        deleted_events = len(events)
        plate = vehicle.plate_number
        db.delete(vehicle)
        db.commit()

        deleted_files = sum(1 for p in media_paths if _safe_delete_capture_file(p))
        return {
            "deleted_plate": plate,
            "deleted_events": deleted_events,
            "deleted_files": deleted_files,
        }


class EventRepository:
    @staticmethod
    def get_by_id(db: Session, event_id: int) -> Optional[VehicleEvent]:
        return db.query(VehicleEvent).filter(VehicleEvent.id == event_id).first()

    @staticmethod
    def list_events(
        db: Session,
        skip: int = 0,
        limit: int = 20,
        plate_number: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[VehicleEvent], int]:
        query = db.query(VehicleEvent)
        if plate_number:
            query = query.filter(VehicleEvent.plate_number.ilike(f"%{plate_number.strip()}%"))
        if event_type:
            query = query.filter(VehicleEvent.event_type == event_type.upper().strip())
        if start_date:
            query = query.filter(VehicleEvent.event_timestamp >= start_date)
        if end_date:
            query = query.filter(VehicleEvent.event_timestamp <= end_date)

        total = query.count()
        events = query.order_by(desc(VehicleEvent.event_timestamp)).offset(skip).limit(limit).all()
        return events, total

    @staticmethod
    def get_vehicle_history(db: Session, plate_number: str) -> List[VehicleEvent]:
        return (
            db.query(VehicleEvent)
            .filter(VehicleEvent.plate_number == plate_number.upper().strip())
            .order_by(desc(VehicleEvent.event_timestamp))
            .all()
        )

    @staticmethod
    def get_latest_event_for_plate(db: Session, plate_number: str) -> Optional[VehicleEvent]:
        return (
            db.query(VehicleEvent)
            .filter(VehicleEvent.plate_number == plate_number.upper().strip())
            .order_by(desc(VehicleEvent.event_timestamp))
            .first()
        )

    @staticmethod
    def delete_by_id(db: Session, event_id: int) -> Optional[Dict[str, Any]]:
        """Delete a single movement event, clean up its files, and reconcile vehicle counters."""
        ev = EventRepository.get_by_id(db, event_id)
        if not ev:
            return None

        plate = ev.plate_number
        vehicle_id = ev.vehicle_id
        media_paths = _collect_event_media(ev)

        db.delete(ev)
        db.flush()

        # Reconcile parent vehicle counts & latest status
        vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
        if vehicle is not None:
            remaining = (
                db.query(VehicleEvent)
                .filter(VehicleEvent.vehicle_id == vehicle_id)
                .order_by(desc(VehicleEvent.event_timestamp))
                .all()
            )
            vehicle.total_entries = sum(1 for r in remaining if r.event_type == "IN")
            vehicle.total_exits = sum(1 for r in remaining if r.event_type == "OUT")
            if remaining:
                vehicle.current_status = remaining[0].event_type
                vehicle.last_seen_at = remaining[0].event_timestamp
            else:
                vehicle.current_status = "OUT"
            vehicle.updated_at = utc_now()

        db.commit()
        deleted_files = sum(1 for p in media_paths if _safe_delete_capture_file(p))
        return {
            "deleted_event_id": event_id,
            "plate_number": plate,
            "deleted_files": deleted_files,
        }


class DashboardRepository:
    @staticmethod
    def get_summary(db: Session) -> Dict[str, Any]:
        """Aggregate high-level system summary statistics."""
        total_vehicles = db.query(func.count(Vehicle.id)).scalar() or 0
        vehicles_inside = db.query(func.count(Vehicle.id)).filter(Vehicle.current_status == "IN").scalar() or 0
        vehicles_outside = db.query(func.count(Vehicle.id)).filter(Vehicle.current_status == "OUT").scalar() or 0

        # Today's boundaries in UTC
        now = utc_now()
        start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc)

        today_entries = (
            db.query(func.count(VehicleEvent.id))
            .filter(VehicleEvent.event_type == "IN", VehicleEvent.event_timestamp >= start_of_day)
            .scalar() or 0
        )
        today_exits = (
            db.query(func.count(VehicleEvent.id))
            .filter(VehicleEvent.event_type == "OUT", VehicleEvent.event_timestamp >= start_of_day)
            .scalar() or 0
        )

        recent_events = (
            db.query(VehicleEvent)
            .order_by(desc(VehicleEvent.event_timestamp))
            .limit(10)
            .all()
        )

        return {
            "total_registered_vehicles": total_vehicles,
            "vehicles_currently_inside": vehicles_inside,
            "vehicles_currently_outside": vehicles_outside,
            "today_entries": today_entries,
            "today_exits": today_exits,
            "recent_activity_count": len(recent_events),
            "recent_events": recent_events
        }


class SystemRepository:
    @staticmethod
    def log_event(
        db: Session,
        event_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemEvent:
        event = SystemEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            metadata_json=metadata or {}
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event

    @staticmethod
    def list_system_events(db: Session, limit: int = 50) -> List[SystemEvent]:
        return db.query(SystemEvent).order_by(desc(SystemEvent.timestamp)).limit(limit).all()

    @staticmethod
    def purge_data(db: Session, scope: str = "all") -> Dict[str, Any]:
        """
        Purge system data by scope (preserves RBAC users and camera records):
        - 'events':   Delete all VehicleEvent rows + captured media files; reset vehicle counters.
        - 'vehicles': Delete all Vehicle & VehicleEvent rows + captured media files.
        - 'all':      Delete all Vehicles, VehicleEvents, SystemEvents + captured media files.
        - 'captures': Delete only archived images & recorded clips on disk.
        - 'logs':     Truncate application log file.
        """
        normalized = (scope or "all").strip().lower()
        result: Dict[str, Any] = {
            "scope": normalized,
            "deleted_vehicles": 0,
            "deleted_events": 0,
            "deleted_system_events": 0,
            "deleted_files": 0,
            "logs_cleared": False,
        }

        if normalized in ("events", "vehicles", "all"):
            result["deleted_events"] = db.query(VehicleEvent).delete(synchronize_session=False)
            if normalized == "events":
                # Reset vehicle counters while keeping vehicle registry rows
                for v in db.query(Vehicle).all():
                    v.total_entries = 0
                    v.total_exits = 0
                    v.current_status = "OUT"
                    v.updated_at = utc_now()
            else:
                result["deleted_vehicles"] = db.query(Vehicle).delete(synchronize_session=False)

            if normalized == "all":
                result["deleted_system_events"] = db.query(SystemEvent).delete(synchronize_session=False)

            db.commit()

        if normalized in ("events", "vehicles", "all", "captures"):
            deleted_files = 0
            for folder in (settings.VEHICLES_DIR, settings.PLATES_DIR, settings.EVENTS_DIR):
                if not folder.exists():
                    continue
                for f in folder.iterdir():
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".webm"):
                        try:
                            f.unlink()
                            deleted_files += 1
                        except Exception:
                            pass
            result["deleted_files"] = deleted_files

        if normalized in ("logs", "all"):
            log_file = settings.LOG_FILE_PATH
            if log_file.exists():
                try:
                    with open(log_file, "w", encoding="utf-8") as lf:
                        lf.truncate(0)
                    result["logs_cleared"] = True
                except Exception:
                    pass

        return result


class CameraRepository:
    @staticmethod
    def get_camera(db: Session, camera_id: int = 1) -> Optional[Camera]:
        return db.query(Camera).filter(Camera.id == camera_id).first()

    @staticmethod
    def update_status(db: Session, camera_id: int, status: str) -> Optional[Camera]:
        cam = db.query(Camera).filter(Camera.id == camera_id).first()
        if cam:
            cam.status = status
            cam.updated_at = utc_now()
            db.commit()
            db.refresh(cam)
        return cam
