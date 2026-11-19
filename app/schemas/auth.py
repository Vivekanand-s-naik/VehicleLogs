"""
Pydantic Schemas: Authentication & RBAC Users
=============================================
Request and response models for JWT authentication and user management.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, ConfigDict


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    full_name: str


class UserBase(BaseModel):
    username: str
    email: str
    role: str  # "ADMIN", "OPERATOR", "AUDITOR"
    full_name: str
    is_active: bool = True


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

