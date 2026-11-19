"""
System, Dashboard & Health API Routes
====================================
Provides system status, health check, dashboard summaries, log retrieval,
and secure image retrieval.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import time
import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging_config import get_recent_logs
from app.core.device import get_device_diagnostics
from app.api.dependencies import (
    get_db, get_event_service, get_camera_service, get_notification_service, require_role
)
from app.database.repository import DashboardRepository, SystemRepository
from app.services.event_service import EventService
from app.services.camera_service import CameraService
from app.services.notification_service import NotificationService
from app.schemas.system import (
    HealthResponse, SystemStatusResponse, DashboardSummaryResponse, CameraStatusResponse
)

router = APIRouter(tags=["System & Dashboard"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    db: Session = Depends(get_db),
    camera_service: CameraService = Depends(get_camera_service)
):
    """Application health endpoint."""
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    diag = get_device_diagnostics()

    return HealthResponse(
        status="HEALTHY" if (db_ok and camera_service.camera_manager.is_connected) else "DEGRADED",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        device=diag.get("active_device", "cpu"),
        device_name=diag.get("device_name", "CPU"),
        camera_connected=camera_service.camera_manager.is_connected,
        database_ok=db_ok
    )


@router.get("/api/system/status", response_model=SystemStatusResponse)
def get_system_status(
    db: Session = Depends(get_db),
    event_service: EventService = Depends(get_event_service),
    camera_service: CameraService = Depends(get_camera_service),
    notification_service: NotificationService = Depends(get_notification_service)
):
    """Complete diagnostic status of database, camera, pipeline, hardware, and notifications."""
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    diag = get_device_diagnostics()
    health = HealthResponse(
        status="HEALTHY" if (db_ok and camera_service.camera_manager.is_connected) else "DEGRADED",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        device=diag.get("active_device", "cpu"),
        device_name=diag.get("device_name", "CPU"),
        camera_connected=camera_service.camera_manager.is_connected,
        database_ok=db_ok
    )

    return SystemStatusResponse(
        health=health,
        camera=CameraStatusResponse(**camera_service.get_status()),
        pipeline_metrics=event_service.metrics.to_dict(),
        notifications=notification_service.get_status(),
        hardware_diagnostics=diag
    )


@router.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary(db: Session = Depends(get_db)):
    """High-level summary cards and recent activity for the monitoring dashboard."""
    return DashboardRepository.get_summary(db)


@router.get("/api/system/logs", response_model=List[str])
def get_application_logs(max_lines: int = Query(100, ge=10, le=500)):
    """Retrieve recent lines from application log file."""
    return get_recent_logs(max_lines=max_lines)


@router.delete(
    "/api/system/data",
    dependencies=[Depends(require_role(["ADMIN"]))]
)
def purge_system_data(
    scope: str = Query("all", description="Scope to delete: 'all', 'events', 'vehicles', 'captures', or 'logs'"),
    db: Session = Depends(get_db),
    event_service: EventService = Depends(get_event_service),
):
    """
    Purge stored ANPR data safely (ADMIN role only).
    Always preserves RBAC user accounts and camera definitions.
    """
    allowed_scopes = {"all", "events", "vehicles", "captures", "logs"}
    norm_scope = scope.strip().lower()
    if norm_scope not in allowed_scopes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scope '{scope}'. Allowed: {sorted(allowed_scopes)}",
        )

    summary = SystemRepository.purge_data(db, scope=norm_scope)
    if norm_scope in ("all", "events", "vehicles"):
        event_service.reset_runtime_state()
    return {"status": "purged", **summary}


@router.get("/api/clips")
def list_stored_clips(limit: int = Query(50, ge=1, le=200)):
    """List all recorded video clips stored in captures/events/ with thumbnail & replay URLs."""
    settings.EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    mp4_files = sorted(
        (f for f in settings.EVENTS_DIR.glob("*.mp4") if f.is_file()),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    for f in mp4_files[:limit]:
        stat = f.stat()
        thumb = f.with_suffix(".jpg")
        stem = f.stem
        plate_guess = None
        if not stem.startswith("clip_"):
            plate_guess = stem.split("_")[0]
        clips.append({
            "filename": f.name,
            "plate_number": plate_guess,
            "size_kb": round(stat.st_size / 1024.0, 1),
            "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "thumbnail_url": f"/api/captures/events/{thumb.name}" if thumb.exists() else None,
            "replay_url": f"/api/captures/replay/{f.name}",
            "download_url": f"/api/captures/events/{f.name}",
        })
    return {"total": len(mp4_files), "items": clips}


@router.delete(
    "/api/clips/{filename}",
    dependencies=[Depends(require_role(["ADMIN"]))]
)
def delete_stored_clip(filename: str):
    """Delete a specific stored video clip and its poster thumbnail (ADMIN role only)."""
    clean_name = Path(filename).name
    target = (settings.EVENTS_DIR / clean_name).resolve()
    try:
        target.relative_to(settings.EVENTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found")

    deleted = 0
    for p in (target, target.with_suffix(".jpg")):
        if p.exists() and p.is_file():
            p.unlink()
            deleted += 1

    return {"status": "deleted", "filename": clean_name, "deleted_files": deleted}


@router.get("/api/captures/replay/{filename}")
def replay_recorded_clip(filename: str):
    """Stream a recorded clip in captures/events/ as universal browser-compatible MJPEG."""
    clean_filename = Path(filename).name
    target_path = (settings.EVENTS_DIR / clean_filename).resolve()
    try:
        target_path.relative_to(settings.EVENTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Clip not found")

    def _stream_clip():
        cap = cv2.VideoCapture(str(target_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        delay = 1.0 / max(8.0, min(30.0, fps))
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                time.sleep(delay)
        finally:
            cap.release()

    return StreamingResponse(_stream_clip(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/api/captures/{subfolder}/{filename}")
def get_capture_image(subfolder: str, filename: str):
    """
    Secure media retrieval endpoint for captured images and recorded clips.
    Prevents directory traversal and restricts access to captures directory.
    """
    allowed_folders = {"vehicles", "plates", "events"}
    if subfolder not in allowed_folders:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid folder")

    # Guard against directory traversal attacks
    clean_filename = Path(filename).name
    target_path = (settings.CAPTURES_DIR / subfolder / clean_filename).resolve()

    # Verify path is inside CAPTURES_DIR
    try:
        target_path.relative_to(settings.CAPTURES_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found")

    ext = target_path.suffix.lower()
    media_type = "video/mp4" if ext == ".mp4" else ("image/png" if ext == ".png" else "image/jpeg")
    return FileResponse(str(target_path), media_type=media_type)
