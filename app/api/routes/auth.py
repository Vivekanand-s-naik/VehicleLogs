"""
Authentication & RBAC User Management Routes
============================================
Endpoints for login, user verification, and user management.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.dependencies import get_db, get_current_user, require_role
from app.core.security import verify_password, get_password_hash, create_access_token
from app.database.models import User
from app.schemas.auth import LoginRequest, TokenResponse, UserResponse, UserCreate

router = APIRouter(prefix="/auth", tags=["Authentication & RBAC"])


@router.post("/login", response_model=TokenResponse)
def login(login_req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with username and password; returns JWT token with role claim."""
    user = db.query(User).filter(User.username == login_req.username.strip().lower()).first()
    if not user or not verify_password(login_req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been deactivated"
        )

    # Encode token with role claim
    token = create_access_token(data={"sub": user.username, "role": user.role})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        username=user.username,
        full_name=user.full_name
    )


@router.get("/me", response_model=UserResponse)
def get_current_profile(current_user: User = Depends(get_current_user)):
    """Retrieve current authenticated user profile and assigned role."""
    return current_user


@router.get("/users", response_model=List[UserResponse], dependencies=[Depends(require_role(["ADMIN"]))])
def list_system_users(db: Session = Depends(get_db)):
    """List all user accounts in system (ADMIN role only)."""
    return db.query(User).order_by(User.id).all()


@router.post("/users", response_model=UserResponse, dependencies=[Depends(require_role(["ADMIN"]))])
def create_system_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Create a new role-based user account (ADMIN role only)."""
    existing = db.query(User).filter(
        (User.username == user_in.username.strip().lower()) | (User.email == user_in.email.strip().lower())
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )

    new_user = User(
        username=user_in.username.strip().lower(),
        email=user_in.email.strip().lower(),
        hashed_password=get_password_hash(user_in.password),
        role=user_in.role.upper(),
        full_name=user_in.full_name,
        is_active=user_in.is_active
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

