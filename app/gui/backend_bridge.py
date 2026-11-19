"""
In-Process Backend Bridge for the Native PyQt6 Desktop GUI
==========================================================
Directly invokes the existing RBAC, SQLAlchemy repositories, NotificationService,
CameraService, and EventService in-process — zero network port exposure and zero
browser serialization overhead while enforcing the exact same JWT + RBAC rules.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

from app.core.config import settings
from app.core.device import get_device_diagnostics
from app.core.logging_config import get_recent_logs
from app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from app.database.connection import SessionLocal, init_db
from app.database.models import User, utc_now
from app.database.repository import (
    DashboardRepository,
    EventRepository,
    SystemRepository,
    VehicleRepository,
)
from app.services.camera_service import CameraService
from app.services.event_service import EventService
from app.services.notification_service import NotificationService

logger = logging.getLogger("anpr.gui.bridge")


class PermissionDeniedError(PermissionError):
    """Raised when an authenticated user lacks the required RBAC role."""


@dataclass
class AuthenticatedSession:
    access_token: str
    username: str
    full_name: str
    email: str
    role: str  # "ADMIN", "OPERATOR", "AUDITOR"


class BackendBridge:
    """
    Facade connecting the PyQt6 desktop UI to the in-process ANPR engine,
    SQLite database, and RBAC authentication/authorization layer.
    """

    def __init__(self, event_service: Optional[EventService] = None):
        init_db()
        self._event_service: Optional[EventService] = event_service
        self._camera_service: Optional[CameraService] = None
        self._notification_service: NotificationService = NotificationService.get_instance()

    # ------------------------------------------------------------ Engine lifecycle
    def init_engine(self) -> EventService:
        """Initialize EventService (loads ONNX models + OCR) and CameraService."""
        if self._event_service is None:
            self._event_service = EventService.get_instance()
        self._camera_service = CameraService.get_instance(self._event_service.camera_manager)
        return self._event_service

    @property
    def event_service(self) -> Optional[EventService]:
        return self._event_service

    @property
    def camera_service(self) -> Optional[CameraService]:
        if self._camera_service is None and self._event_service is not None:
            self._camera_service = CameraService.get_instance(self._event_service.camera_manager)
        return self._camera_service

    def start_pipeline(self) -> bool:
        svc = self.init_engine()
        svc.start()
        return svc.is_running

    def stop_pipeline(self) -> None:
        if self._event_service is not None:
            self._event_service.stop()

    def is_pipeline_running(self) -> bool:
        return bool(self._event_service is not None and self._event_service.is_running)

    # -------------------------------------------------------- RBAC Authentication
    def authenticate(self, username: str, password: str) -> AuthenticatedSession:
        """Authenticate against SQLite RBAC users using bcrypt + JWT."""
        clean_user = (username or "").strip().lower()
        with SessionLocal() as db:
            user = db.query(User).filter(User.username == clean_user).first()
            if not user or not verify_password(password or "", user.hashed_password):
                raise ValueError("Incorrect username or password")
            if not user.is_active:
                raise ValueError("Account has been deactivated")

            token = create_access_token(data={"sub": user.username, "role": user.role})
            return AuthenticatedSession(
                access_token=token,
                username=user.username,
                full_name=user.full_name,
                email=user.email,
                role=user.role.upper(),
            )

    @staticmethod
    def _enforce_roles(session: AuthenticatedSession, allowed_roles: Sequence[str]) -> None:
        payload = decode_access_token(session.access_token)
        if not payload:
            raise PermissionDeniedError("Session expired or invalid token. Please sign in again.")
        token_role = str(payload.get("role", "")).upper()
        if token_role not in {r.upper() for r in allowed_roles}:
            raise PermissionDeniedError(
                f"Action requires role {', '.join(allowed_roles)} (current role: {token_role})."
            )

    # ------------------------------------------------------------ Camera controls
    def list_cameras(self, max_index: int = 6) -> List[Dict[str, str]]:
        if self.camera_service is not None:
            return self.camera_service.list_cameras(max_index=max_index)
        from app.vision.camera_manager import enumerate_video_devices
        return enumerate_video_devices(max_index=max_index, active_source=settings.CAMERA_SOURCE)

    def switch_camera(self, new_source: str) -> bool:
        if self.camera_service is None:
            settings.CAMERA_SOURCE = str(new_source).strip()
            return True
        return self.camera_service.switch_source(new_source)

    def reconnect_camera(self) -> bool:
        if self.camera_service is None:
            return False
        return self.camera_service.reconnect()

    # -------------------------------------------------------- Dashboard & Status
    def get_dashboard_summary(self) -> Dict[str, Any]:
        with SessionLocal() as db:
            return DashboardRepository.get_summary(db)

    def get_system_status(self) -> Dict[str, Any]:
        db_ok = True
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1"))
        except Exception:
            db_ok = False

        diag = get_device_diagnostics()
        cam_status = (
            self.camera_service.get_status()
            if self.camera_service is not None
            else {
                "source": settings.CAMERA_SOURCE,
                "is_connected": False,
                "fps_actual": 0.0,
                "total_frames_captured": 0,
                "reconnect_attempts": 0,
                "configured_resolution": f"{settings.CAMERA_WIDTH}x{settings.CAMERA_HEIGHT}",
                "configured_fps": settings.CAMERA_FPS,
            }
        )
        metrics = (
            self._event_service.metrics.to_dict()
            if self._event_service is not None
            else {
                "processing_fps": 0.0,
                "inference_time_ms": 0.0,
                "ocr_time_ms": 0.0,
                "clip_recorder": {},
            }
        )
        return {
            "health": {
                "status": "HEALTHY" if (db_ok and cam_status.get("is_connected")) else "DEGRADED",
                "database_ok": db_ok,
                "camera_connected": bool(cam_status.get("is_connected")),
                "pipeline_running": self.is_pipeline_running(),
            },
            "camera": cam_status,
            "pipeline_metrics": metrics,
            "notifications": self._notification_service.get_status(),
            "hardware_diagnostics": diag,
        }

    # --------------------------------------------------------------- Vehicles
    def list_vehicles(
        self,
        search: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        with SessionLocal() as db:
            items, total = VehicleRepository.list_vehicles(
                db,
                skip=0,
                limit=limit,
                search=(search or "").strip() or None,
                status=(status_filter or "").strip() or None,
            )
            serialized = [
                {
                    "id": v.id,
                    "plate_number": v.plate_number,
                    "current_status": v.current_status,
                    "total_entries": v.total_entries,
                    "total_exits": v.total_exits,
                    "first_seen_at": v.first_seen_at,
                    "last_seen_at": v.last_seen_at,
                }
                for v in items
            ]
            return {"total": total, "items": serialized}

    def get_vehicle_history(self, plate_number: str) -> List[Dict[str, Any]]:
        with SessionLocal() as db:
            events = EventRepository.get_vehicle_history(db, plate_number)
            return [self._serialize_event(ev) for ev in events]

    def manual_gate_entry(
        self,
        session: AuthenticatedSession,
        plate_number: str,
        event_type: str,
        notes: str = "Gate Operator Manual Trigger",
    ) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN", "OPERATOR"])
        plate = (plate_number or "").strip().upper()
        ev_type = (event_type or "").strip().upper()
        if not plate:
            raise ValueError("Plate number is required")
        if ev_type not in ("IN", "OUT"):
            raise ValueError("Event type must be 'IN' or 'OUT'")

        frame = None
        if self._event_service is not None:
            _, frame = self._event_service.camera_manager.get_latest_frame()
            state_mgr = self._event_service.state_manager
        else:
            from app.services.vehicle_state_manager import VehicleStateManager
            state_mgr = VehicleStateManager()

        res = state_mgr.process_detection(
            plate_number=plate,
            confidence=1.0,
            detection_confidence=1.0,
            raw_ocr_text=f"MANUAL_OVERRIDE_{ev_type}",
            direction_hint=ev_type,
            vehicle_image=frame,
            state_name=f"Manual Override: {notes or 'Gate Operator Manual Trigger'}",
            forced_event_type=ev_type,
        )
        if not res.success:
            raise ValueError(f"Manual entry rejected: {res.reason}")
        return {
            "status": "success",
            "plate_number": plate,
            "event_id": res.event_id,
            "event_type": res.event_type,
        }

    def override_vehicle_status(
        self,
        session: AuthenticatedSession,
        plate_number: str,
        new_status: str,
        reason: str = "Manual admin toggle",
    ) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN"])
        st = (new_status or "").strip().upper()
        if st not in ("IN", "OUT"):
            raise ValueError("new_status must be 'IN' or 'OUT'")
        with SessionLocal() as db:
            vehicle = VehicleRepository.get_by_plate(db, plate_number)
            if not vehicle:
                raise ValueError(f"Vehicle '{plate_number}' not found")
            vehicle.current_status = st
            vehicle.updated_at = utc_now()
            db.commit()
            return {"status": "success", "plate_number": vehicle.plate_number, "current_status": st, "reason": reason}

    def delete_vehicle(self, session: AuthenticatedSession, plate_number: str) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN"])
        with SessionLocal() as db:
            res = VehicleRepository.delete_by_plate(db, plate_number)
            if not res:
                raise ValueError(f"Vehicle '{plate_number}' not found")
        if self._event_service is not None:
            self._event_service.state_manager.forget_plate(plate_number)
        return {"status": "deleted", **res}

    # ----------------------------------------------------------------- Events
    @staticmethod
    def _serialize_event(ev) -> Dict[str, Any]:
        meta = ev.processing_metadata or {}
        clip_path = getattr(ev, "video_clip_path", None) or meta.get("video_clip_path") or ""
        return {
            "id": ev.id,
            "plate_number": ev.plate_number,
            "event_type": ev.event_type,
            "event_timestamp": ev.event_timestamp,
            "ocr_confidence": float(ev.ocr_confidence or 0.0),
            "raw_ocr_text": ev.raw_ocr_text or "",
            "plate_image_path": ev.plate_image_path or "",
            "vehicle_image_path": ev.vehicle_image_path or "",
            "video_clip_path": clip_path,
            "clip_filename": Path(clip_path).name if clip_path else "",
        }

    def list_events(
        self,
        plate: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        with SessionLocal() as db:
            items, total = EventRepository.list_events(
                db,
                skip=0,
                limit=limit,
                plate_number=(plate or "").strip() or None,
                event_type=(event_type or "").strip() or None,
            )
            return {"total": total, "items": [self._serialize_event(ev) for ev in items]}

    def delete_event(self, session: AuthenticatedSession, event_id: int) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN"])
        with SessionLocal() as db:
            res = EventRepository.delete_by_id(db, event_id)
            if not res:
                raise ValueError(f"Event #{event_id} not found")
        if self._event_service is not None:
            self._event_service.state_manager.forget_plate(res["plate_number"])
        return {"status": "deleted", **res}

    # ------------------------------------------------------------------ Clips
    def list_clips(self, limit: int = 100) -> Dict[str, Any]:
        settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        mp4_files = sorted(
            (f for f in settings.EVENTS_DIR.glob("*.mp4") if f.is_file()),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        items = []
        for f in mp4_files[:limit]:
            st = f.stat()
            thumb = f.with_suffix(".jpg")
            stem = f.stem
            plate_guess = None if stem.startswith("clip_") else stem.split("_")[0]
            items.append({
                "filename": f.name,
                "file_path": str(f),
                "plate_number": plate_guess,
                "size_kb": round(st.st_size / 1024.0, 1),
                "created_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                "thumbnail_path": str(thumb) if thumb.exists() else "",
            })
        return {"total": len(mp4_files), "items": items}

    def delete_clip(self, session: AuthenticatedSession, filename: str) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN"])
        clean_name = Path(filename).name
        target = (settings.EVENTS_DIR / clean_name).resolve()
        target.relative_to(settings.EVENTS_DIR.resolve())
        if not target.exists() or not target.is_file():
            raise ValueError("Clip not found")
        deleted = 0
        for p in (target, target.with_suffix(".jpg")):
            if p.exists() and p.is_file():
                p.unlink()
                deleted += 1
        return {"status": "deleted", "filename": clean_name, "deleted_files": deleted}

    # ------------------------------------------------------------------ Purge
    def purge_system_data(self, session: AuthenticatedSession, scope: str) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN"])
        norm_scope = scope.strip().lower()
        if norm_scope not in {"all", "events", "vehicles", "captures", "logs"}:
            raise ValueError(f"Invalid purge scope: {scope}")
        with SessionLocal() as db:
            summary = SystemRepository.purge_data(db, scope=norm_scope)
        if self._event_service is not None and norm_scope in ("all", "events", "vehicles"):
            self._event_service.reset_runtime_state()
        return {"status": "purged", **summary}

    # ------------------------------------------------------------- RBAC Users
    def list_users(self, session: AuthenticatedSession) -> List[Dict[str, Any]]:
        self._enforce_roles(session, ["ADMIN"])
        with SessionLocal() as db:
            users = db.query(User).order_by(User.id).all()
            return [
                {
                    "id": u.id,
                    "full_name": u.full_name,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role,
                    "is_active": u.is_active,
                }
                for u in users
            ]

    def create_user(
        self,
        session: AuthenticatedSession,
        full_name: str,
        username: str,
        email: str,
        role: str,
        password: str,
    ) -> Dict[str, Any]:
        self._enforce_roles(session, ["ADMIN"])
        u_clean = username.strip().lower()
        e_clean = email.strip().lower()
        r_clean = role.strip().upper()
        if not full_name.strip() or not u_clean or not e_clean or not password:
            raise ValueError("All fields are required")
        if r_clean not in ("ADMIN", "OPERATOR", "AUDITOR"):
            raise ValueError("Role must be ADMIN, OPERATOR, or AUDITOR")

        with SessionLocal() as db:
            existing = db.query(User).filter(
                (User.username == u_clean) | (User.email == e_clean)
            ).first()
            if existing:
                raise ValueError("Username or email already exists")

            new_user = User(
                username=u_clean,
                email=e_clean,
                hashed_password=get_password_hash(password),
                role=r_clean,
                full_name=full_name.strip(),
                is_active=True,
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            return {
                "id": new_user.id,
                "username": new_user.username,
                "full_name": new_user.full_name,
                "role": new_user.role,
            }

    # --------------------------------------------------------- Logs & Settings
    def get_logs(self, max_lines: int = 120) -> List[str]:
        return get_recent_logs(max_lines=max_lines)

    def get_license_logs(self, max_lines: int = 80) -> List[str]:
        p = Path(settings.LICENSE_LOG_PATH)
        if not p.exists():
            return ["(No license audit log entries recorded yet.)"]
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-max_lines:]
        except Exception as exc:
            return [f"Error reading license log: {exc}"]

    def update_runtime_settings(
        self,
        session: AuthenticatedSession,
        detection_confidence: float,
        recognition_cooldown: int,
        event_strategy: str,
        clip_recording_enabled: bool,
        notifications_enabled: bool,
        ntfy_topic: str,
    ) -> None:
        self._enforce_roles(session, ["ADMIN"])
        settings.DETECTION_CONFIDENCE = max(0.05, min(0.95, float(detection_confidence)))
        settings.RECOGNITION_COOLDOWN = max(0, int(recognition_cooldown))
        if event_strategy in ("toggle", "line_crossing"):
            settings.EVENT_STRATEGY = event_strategy
        settings.CLIP_RECORDING_ENABLED = bool(clip_recording_enabled)
        settings.NOTIFICATIONS_ENABLED = bool(notifications_enabled)
        if ntfy_topic.strip():
            settings.NTFY_TOPIC = ntfy_topic.strip()

        if self._event_service is not None:
            self._event_service.plate_detector.conf_threshold = settings.DETECTION_CONFIDENCE
            self._event_service.metrics.mode = (
                "clip_recording" if settings.CLIP_RECORDING_ENABLED else "instant_ocr"
            )
        self._notification_service.enabled = settings.NOTIFICATIONS_ENABLED
        self._notification_service.topic = settings.NTFY_TOPIC
        logger.info(
            f"Admin '{session.username}' updated live settings: "
            f"conf={settings.DETECTION_CONFIDENCE}, cooldown={settings.RECOGNITION_COOLDOWN}s, "
            f"strategy={settings.EVENT_STRATEGY}, clip_mode={settings.CLIP_RECORDING_ENABLED}, "
            f"ntfy={settings.NOTIFICATIONS_ENABLED}/{settings.NTFY_TOPIC}"
        )

    # -------------------------------------------------------------- CSV Export
    def export_vehicles_csv(
        self,
        filepath: str,
        search: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> int:
        data = self.list_vehicles(search=search, status_filter=status_filter, limit=5000)
        rows = data["items"]
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Plate Number", "Current Status", "Total Entries", "Total Exits", "Last Seen"])
            for v in rows:
                writer.writerow([
                    v["plate_number"],
                    v["current_status"],
                    v["total_entries"],
                    v["total_exits"],
                    v["last_seen_at"].isoformat() if hasattr(v["last_seen_at"], "isoformat") else str(v["last_seen_at"]),
                ])
        return len(rows)

    def export_events_csv(
        self,
        filepath: str,
        plate: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> int:
        data = self.list_events(plate=plate, event_type=event_type, limit=5000)
        rows = data["items"]
        with open(filepath, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Event ID", "Type", "Plate Number", "Timestamp", "OCR Confidence", "Raw OCR", "Clip File"])
            for ev in rows:
                writer.writerow([
                    ev["id"],
                    ev["event_type"],
                    ev["plate_number"],
                    ev["event_timestamp"].isoformat() if hasattr(ev["event_timestamp"], "isoformat") else str(ev["event_timestamp"]),
                    f"{ev['ocr_confidence']:.4f}",
                    ev["raw_ocr_text"],
                    ev["clip_filename"],
                ])
        return len(rows)

    @staticmethod
    def resolve_capture_path(rel_or_abs: str) -> Optional[Path]:
        """Resolve a DB capture path safely against CAPTURES_DIR / BASE_DIR."""
        if not rel_or_abs:
            return None
        p = Path(rel_or_abs)
        if p.is_absolute() and p.exists():
            return p
        # Try relative to BASE_DIR or CAPTURES_DIR
        from app.core.config import BASE_DIR
        for candidate in (BASE_DIR / p, settings.CAPTURES_DIR / p):
            if candidate.exists() and candidate.is_file():
                return candidate
        return None
