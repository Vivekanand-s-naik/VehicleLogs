"""
Camera Operation & Video Stream API Routes
==========================================
Endpoints for camera status, reconnection, and live MJPEG video streaming.
"""

import time
import cv2
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.dependencies import get_camera_service, get_event_service, verify_api_key
from app.services.camera_service import CameraService
from app.services.event_service import EventService
from app.schemas.system import CameraStatusResponse

router = APIRouter(prefix="/camera", tags=["Camera"])


@router.get("/status", response_model=CameraStatusResponse)
def get_camera_status(camera_service: CameraService = Depends(get_camera_service)):
    """Retrieve live camera hardware and capture metrics."""
    return camera_service.get_status()


@router.post("/reconnect", dependencies=[Depends(verify_api_key)])
def reconnect_camera(camera_service: CameraService = Depends(get_camera_service)):
    """Request immediate reconnection of video source."""
    success = camera_service.reconnect()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to connect to camera source"
        )
    return {"status": "reconnected", "details": camera_service.get_status()}


def _generate_mjpeg_stream(event_service: EventService):
    """Generator yielding multipart MJPEG frames only when a new annotated frame is ready."""
    last_id = -1
    while True:
        frame_id, frame = event_service.get_latest_annotated_frame_with_id()
        if frame is None or frame_id == last_id:
            time.sleep(0.015)
            continue
        last_id = frame_id

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            time.sleep(0.02)
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@router.get("/stream")
def live_stream(event_service: EventService = Depends(get_event_service)):
    """Live MJPEG video stream of annotated camera feed."""
    return StreamingResponse(
        _generate_mjpeg_stream(event_service),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
