"""
Job database model for tracking video analysis jobs.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, JSON, Integer, Float, Text
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class JobStatus(str, enum.Enum):
    """Job status enumeration."""
    PENDING = "pending"
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """
    Job model for tracking video analysis jobs.

    Stores all configuration, status, and results for each video analysis.
    """

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Status
    status = Column(String(20), default=JobStatus.PENDING.value, nullable=False, index=True)
    progress = Column(Integer, default=0, nullable=False)  # 0-100
    current_stage = Column(String(100), nullable=True)  # "Detecting players..."
    error_message = Column(Text, nullable=True)

    # Input video
    input_video_s3_key = Column(String(500), nullable=True)
    input_video_filename = Column(String(255), nullable=True)
    input_video_duration_seconds = Column(Float, nullable=True)
    input_video_size_bytes = Column(Integer, nullable=True)

    # Team configuration
    # {
    #   "name": "Lakers",
    #   "color": "#552583",
    #   "roster": {"23": "LeBron James", "3": "Anthony Davis"}
    # }
    team_a = Column(JSON, nullable=False)
    team_b = Column(JSON, nullable=False)

    # Processing configuration
    # {
    #   "max_duration_seconds": 60,
    #   "start_time_seconds": 0,
    #   "enable_segmentation": true,
    #   "enable_jersey_detection": true,
    #   "enable_tactical_view": true
    # }
    processing_config = Column(JSON, nullable=False)

    # Visualization configuration
    # {
    #   "mask_display": "all",
    #   "selected_player_ids": [],
    #   "show_bounding_boxes": true,
    #   "show_player_names": true,
    #   "show_jersey_numbers": true,
    #   "tactical_view_position": "bottom-right",
    #   "tactical_view_scale": 0.35
    # }
    visualization_config = Column(JSON, nullable=False)

    # Output
    output_s3_prefix = Column(String(500), nullable=True)  # jobs/{job_id}/

    # Raw tracking data for re-rendering without reprocessing
    # Stores masks, positions, team assignments, jersey numbers
    tracking_data_s3_key = Column(String(500), nullable=True)

    # Detected players list
    # [{"id": 1, "team": "Lakers", "jersey": "23", "name": "LeBron James", "frames": 287}]
    detected_players = Column(JSON, nullable=True)

    # Output video keys
    output_videos = Column(JSON, nullable=True)
    # {
    #   "stage1_detection": "jobs/{id}/stage1.mp4",
    #   "stage2_tracking": "jobs/{id}/stage2.mp4",
    #   ...
    #   "final": "jobs/{id}/final.mp4"
    # }

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Performance metrics
    processing_time_seconds = Column(Float, nullable=True)
    gpu_type = Column(String(20), nullable=True)  # A10G, A100, etc.

    # Billing
    gpu_cost_cents = Column(Integer, nullable=True)

    # Relationships
    user = relationship("User", back_populates="jobs")

    def __repr__(self):
        return f"<Job {self.id} - {self.status}>"

    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status == JobStatus.COMPLETED.value

    @property
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status == JobStatus.FAILED.value

    @property
    def is_processing(self) -> bool:
        """Check if job is currently processing."""
        return self.status in [
            JobStatus.UPLOADING.value,
            JobStatus.QUEUED.value,
            JobStatus.PROCESSING.value,
            JobStatus.RENDERING.value,
        ]

    def to_summary_dict(self) -> dict:
        """Convert job to summary dictionary for list views."""
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "team_a_name": self.team_a.get("name") if self.team_a else None,
            "team_b_name": self.team_b.get("name") if self.team_b else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "input_video_filename": self.input_video_filename,
        }

    def to_full_dict(self) -> dict:
        """Convert job to full dictionary for detail views."""
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "processing_config": self.processing_config,
            "visualization_config": self.visualization_config,
            "detected_players": self.detected_players,
            "output_videos": self.output_videos,
            "input_video_filename": self.input_video_filename,
            "input_video_duration_seconds": self.input_video_duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time_seconds": self.processing_time_seconds,
        }
