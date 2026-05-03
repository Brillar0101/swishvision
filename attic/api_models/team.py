"""
Team database model for storing user's team configurations.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


class Team(Base):
    """
    Team model for storing team configurations.

    Each user can save multiple team configurations with rosters
    that can be reused across video analyses.
    """

    __tablename__ = "teams"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Team info
    name = Column(String(255), nullable=False)
    color = Column(String(7), nullable=False)  # Hex color like #E85D04

    # Roster: {"0": "Haliburton", "23": "Nesmith", ...}
    roster = Column(JSON, nullable=False, default=dict)

    # Optional metadata
    logo_url = Column(String(500), nullable=True)
    league = Column(String(100), nullable=True)  # NBA, NCAA, etc.

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="teams")

    def __repr__(self):
        return f"<Team {self.name}>"

    def to_dict(self) -> dict:
        """Convert team to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "color": self.color,
            "roster": self.roster,
            "logo_url": self.logo_url,
            "league": self.league,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
