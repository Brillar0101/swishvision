"""
Pydantic schemas for User API.
"""
from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=255)


class UserCreate(UserBase):
    """Schema for user registration."""

    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    """Schema for user login."""

    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    """Schema for updating user profile."""

    full_name: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = Field(None, min_length=8, max_length=100)


class UserResponse(UserBase):
    """User response schema."""

    id: str
    role: str
    plan: str
    usage_minutes_remaining: int
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class UserAdminResponse(UserResponse):
    """User response for admin views with additional fields."""

    last_login_at: Optional[str] = None
    total_jobs: int = 0
    total_processing_minutes: float = 0


class TokenResponse(BaseModel):
    """Authentication token response."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse
