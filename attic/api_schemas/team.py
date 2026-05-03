"""
Pydantic schemas for Team API.
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator
import re


class TeamRoster(BaseModel):
    """Team roster mapping jersey numbers to player names."""

    # Dynamic dict of jersey number -> player name
    # Validated in parent schema
    pass


class TeamBase(BaseModel):
    """Base team schema with shared fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Team name")
    color: str = Field(..., description="Team color in hex format (e.g., #E85D04)")
    roster: Dict[str, str] = Field(default_factory=dict, description="Jersey number to player name mapping")

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: str) -> str:
        """Validate hex color format."""
        if not re.match(r"^#[0-9A-Fa-f]{6}$", v):
            raise ValueError("Color must be a valid hex color (e.g., #E85D04)")
        return v.upper()

    @field_validator("roster")
    @classmethod
    def validate_roster(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate roster format."""
        validated = {}
        for jersey, name in v.items():
            # Jersey number should be string digits or "00"
            jersey_str = str(jersey).strip()
            if not jersey_str.isdigit() and jersey_str != "00":
                raise ValueError(f"Invalid jersey number: {jersey}")
            name_str = str(name).strip()
            if not name_str:
                raise ValueError(f"Player name cannot be empty for jersey {jersey}")
            validated[jersey_str] = name_str
        return validated


class TeamCreate(TeamBase):
    """Schema for creating a new team."""

    logo_url: Optional[str] = Field(None, max_length=500)
    league: Optional[str] = Field(None, max_length=100)


class TeamUpdate(BaseModel):
    """Schema for updating a team."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    color: Optional[str] = Field(None)
    roster: Optional[Dict[str, str]] = None
    logo_url: Optional[str] = Field(None, max_length=500)
    league: Optional[str] = Field(None, max_length=100)

    @field_validator("color")
    @classmethod
    def validate_color(cls, v: Optional[str]) -> Optional[str]:
        """Validate hex color format if provided."""
        if v is not None and not re.match(r"^#[0-9A-Fa-f]{6}$", v):
            raise ValueError("Color must be a valid hex color (e.g., #E85D04)")
        return v.upper() if v else None


class TeamResponse(TeamBase):
    """Schema for team response."""

    id: str
    logo_url: Optional[str] = None
    league: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class TeamInJob(TeamBase):
    """Team configuration as used in job requests."""

    pass
