"""
FastAPI Dependencies: Database, Services & Role-Based Access Control (RBAC)
===========================================================================
Enforces JWT authentication, role verification (ADMIN, OPERATOR, AUDITOR),
and provides service singletons.
"""

from typing import Generator, Optional, List
from fastapi import Depends, HTTPException, Security, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import decode_access_token
from app.database.connection import get_db
from app.database.models import User
from app.services.event_service import EventService
from app.services.camera_service import CameraService
from app.services.notification_service import NotificationService

security_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    token_query: Optional[str] = Query(None, alias="token"),
    db: Session = Depends(get_db)
) -> User:
    """
    Extract and validate JWT token from Bearer header or '?token=' query parameter.
    Returns authenticated User model.
    """
    token = None
    if auth and auth.credentials:
        token = auth.credentials
    elif token_query:
        token = token_query

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or corrupted authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload["sub"]
    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account no longer exists or is inactive"
        )
    return user


def require_role(allowed_roles: List[str]):
    """
    Dependency factory checking if current authenticated user has an authorized role.
    Roles: "ADMIN", "OPERATOR", "AUDITOR".
    """
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = current_user.role.upper()
        normalized_allowed = [r.upper() for r in allowed_roles]
        if user_role not in normalized_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Role '{user_role}' does not have sufficient permission for this operation. Required: {allowed_roles}"
            )
        return current_user
    return role_checker


def verify_api_key(api_key: Optional[str] = None) -> bool:
    """Optional backwards-compatibility check for API keys."""
    if settings.API_KEY and api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
    return True


def get_event_service() -> EventService:
    return EventService.get_instance()


def get_camera_service() -> CameraService:
    event_service = EventService.get_instance()
    return CameraService.get_instance(camera_manager=event_service.camera_manager)


def get_notification_service() -> NotificationService:
    return NotificationService.get_instance()
