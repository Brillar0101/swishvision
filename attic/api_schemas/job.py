"""
Pydantic schemas for Job API.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from app.schemas.team import TeamInJob


class ProcessingConfig(BaseModel):
    """Video processing configuration."""

    max_duration_seconds: Optional[int] = Field(
        None,
        ge=1,
        le=600,
        description="Maximum duration to process (null = full video)"
    )
    start_time_seconds: float = Field(
        0,
        ge=0,
        description="Start time for processing"
    )
    enable_segmentation: bool = Field(
        True,
        description="Enable player outline masks"
    )
    enable_jersey_detection: bool = Field(
        True,
        description="Enable jersey number OCR"
    )
    enable_tactical_view: bool = Field(
        True,
        description="Enable court overview"
    )


class VisualizationConfig(BaseModel):
    """Output visualization configuration."""

    mask_display: Literal["all", "team_a", "team_b", "selected", "none"] = Field(
        "all",
        description="Which players to show outlines for"
    )
    selected_player_ids: List[int] = Field(
        default_factory=list,
        description="Player IDs to show (when mask_display='selected')"
    )
    show_bounding_boxes: bool = Field(
        True,
        description="Show boxes around players"
    )
    show_player_names: bool = Field(
        True,
        description="Show player names"
    )
    show_jersey_numbers: bool = Field(
        True,
        description="Show jersey numbers"
    )
    tactical_view_position: Literal["bottom-right", "bottom-left", "top-right", "top-left", "none"] = Field(
        "bottom-right",
        description="Position of court overview"
    )
    tactical_view_scale: float = Field(
        0.35,
        ge=0.2,
        le=0.5,
        description="Size of court overview (fraction of frame width)"
    )


class JobCreate(BaseModel):
    """Schema for creating a new analysis job."""

    team_a: TeamInJob = Field(..., description="Home team configuration")
    team_b: TeamInJob = Field(..., description="Away team configuration")
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig,
        description="Processing options"
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Output options"
    )


class JobStatusResponse(BaseModel):
    """Simple job status response."""

    id: str
    status: str
    progress: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None


class DetectedPlayer(BaseModel):
    """Detected player information."""

    id: int
    team: str
    jersey_number: Optional[str] = None
    player_name: Optional[str] = None
    frames_tracked: int


class JobVideoUrls(BaseModel):
    """Presigned URLs for job videos."""

    stage1_detection: Optional[str] = None
    stage2_tracking: Optional[str] = None
    stage3_segmentation: Optional[str] = None
    stage4_teams: Optional[str] = None
    stage5_jerseys: Optional[str] = None
    final: Optional[str] = None


class JobSummaryResponse(BaseModel):
    """Job summary for list views."""

    id: str
    status: str
    progress: int
    team_a_name: Optional[str] = None
    team_b_name: Optional[str] = None
    input_video_filename: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class JobDetailResponse(BaseModel):
    """Full job details response."""

    id: str
    status: str
    progress: int
    current_stage: Optional[str] = None
    error_message: Optional[str] = None

    team_a: Dict
    team_b: Dict
    processing_config: Dict
    visualization_config: Dict

    detected_players: Optional[List[DetectedPlayer]] = None
    video_urls: Optional[JobVideoUrls] = None

    input_video_filename: Optional[str] = None
    input_video_duration_seconds: Optional[float] = None

    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    class Config:
        from_attributes = True


class JobRerenderRequest(BaseModel):
    """Request to re-render a job with new visualization settings."""

    visualization: VisualizationConfig = Field(
        ...,
        description="New visualization settings"
    )


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: List[JobSummaryResponse]
    total: int
    page: int
    per_page: int
    total_pages: int
