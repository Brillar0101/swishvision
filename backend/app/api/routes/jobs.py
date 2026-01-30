"""
Jobs API routes for video analysis.
"""
import os
import tempfile
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc
import json

from app.core.config import settings
from app.core.database import get_db
from app.core.s3 import s3_client
from app.models import User, Job, JobStatus
from app.schemas import (
    JobCreate,
    JobStatusResponse,
    JobSummaryResponse,
    JobDetailResponse,
    JobRerenderRequest,
    JobListResponse,
    JobVideoUrls,
    ProcessingConfig,
    VisualizationConfig,
)
from app.api.deps import get_current_user

router = APIRouter(prefix="/jobs", tags=["Jobs"])


def process_video_task(job_id: str, db_url: str):
    """
    Background task to process a video.

    This would typically call Modal or another GPU service.
    For now, it's a placeholder that simulates processing.
    """
    # TODO: Implement actual GPU processing via Modal
    # This is where we'd call the ML pipeline
    pass


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    page: int = 1,
    per_page: int = 20,
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all jobs for the current user with pagination.
    """
    query = db.query(Job).filter(Job.user_id == current_user.id)

    if status_filter:
        query = query.filter(Job.status == status_filter)

    total = query.count()
    total_pages = (total + per_page - 1) // per_page

    jobs = query.order_by(desc(Job.created_at)).offset((page - 1) * per_page).limit(per_page).all()

    return JobListResponse(
        jobs=[
            JobSummaryResponse(
                id=job.id,
                status=job.status,
                progress=job.progress,
                team_a_name=job.team_a.get("name") if job.team_a else None,
                team_b_name=job.team_b.get("name") if job.team_b else None,
                input_video_filename=job.input_video_filename,
                created_at=job.created_at.isoformat(),
            )
            for job in jobs
        ],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@router.post("/", response_model=JobStatusResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to analyze"),
    team_a: str = Form(..., description="Team A configuration as JSON"),
    team_b: str = Form(..., description="Team B configuration as JSON"),
    processing: str = Form(default="{}", description="Processing config as JSON"),
    visualization: str = Form(default="{}", description="Visualization config as JSON"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new video analysis job.

    Upload a video file along with team configurations and processing options.
    The video will be uploaded to S3 and processing will begin in the background.
    """
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video format. Supported: MP4, MOV, AVI",
        )

    # Parse JSON configurations
    try:
        team_a_data = json.loads(team_a)
        team_b_data = json.loads(team_b)
        processing_data = json.loads(processing) if processing else {}
        visualization_data = json.loads(visualization) if visualization else {}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON in configuration: {str(e)}",
        )

    # Validate with Pydantic (will raise if invalid)
    processing_config = ProcessingConfig(**processing_data)
    visualization_config = VisualizationConfig(**visualization_data)

    # Create job record
    job = Job(
        user_id=current_user.id,
        status=JobStatus.UPLOADING.value,
        team_a=team_a_data,
        team_b=team_b_data,
        processing_config=processing_config.model_dump(),
        visualization_config=visualization_config.model_dump(),
        input_video_filename=video.filename,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Upload video to S3
    try:
        s3_key = f"jobs/{job.id}/input.mp4"
        s3_client.upload_file(video.file, s3_key, content_type=video.content_type or "video/mp4")
        job.input_video_s3_key = s3_key
        job.status = JobStatus.QUEUED.value
        db.commit()
    except Exception as e:
        job.status = JobStatus.FAILED.value
        job.error_message = f"Failed to upload video: {str(e)}"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload video",
        )

    # TODO: Trigger GPU processing via Modal
    # background_tasks.add_task(process_video_task, job.id, settings.DATABASE_URL)

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage="Queued for processing",
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get detailed information about a specific job.
    """
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Generate presigned URLs for videos if job is complete
    video_urls = None
    if job.status == JobStatus.COMPLETED.value and job.output_videos:
        video_urls = JobVideoUrls(
            stage1_detection=s3_client.generate_presigned_url(job.output_videos.get("stage1_detection")),
            stage2_tracking=s3_client.generate_presigned_url(job.output_videos.get("stage2_tracking")),
            stage3_segmentation=s3_client.generate_presigned_url(job.output_videos.get("stage3_segmentation")),
            stage4_teams=s3_client.generate_presigned_url(job.output_videos.get("stage4_teams")),
            stage5_jerseys=s3_client.generate_presigned_url(job.output_videos.get("stage5_jerseys")),
            final=s3_client.generate_presigned_url(job.output_videos.get("final")),
        )

    return JobDetailResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
        team_a=job.team_a,
        team_b=job.team_b,
        processing_config=job.processing_config,
        visualization_config=job.visualization_config,
        detected_players=job.detected_players,
        video_urls=video_urls,
        input_video_filename=job.input_video_filename,
        input_video_duration_seconds=job.input_video_duration_seconds,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        processing_time_seconds=job.processing_time_seconds,
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get just the status of a job (lightweight endpoint for polling).
    """
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error_message=job.error_message,
    )


@router.post("/{job_id}/rerender", response_model=JobStatusResponse)
async def rerender_job(
    job_id: str,
    rerender_request: JobRerenderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Re-render a completed job with new visualization settings.

    This does not re-run the ML pipeline, just applies new visualization
    settings to the cached tracking data.
    """
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only re-render completed jobs",
        )

    if not job.tracking_data_s3_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tracking data available for re-rendering",
        )

    # Update visualization config
    job.visualization_config = rerender_request.visualization.model_dump()
    job.status = JobStatus.RENDERING.value
    job.progress = 0
    db.commit()

    # TODO: Trigger re-render task (can run on CPU)
    # background_tasks.add_task(rerender_video_task, job.id, settings.DATABASE_URL)

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage="Preparing to re-render",
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a job and all associated files.
    """
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Delete S3 files
    if job.output_s3_prefix:
        s3_client.delete_folder(job.output_s3_prefix)
    if job.input_video_s3_key:
        s3_client.delete_file(job.input_video_s3_key)

    # Delete job record
    db.delete(job)
    db.commit()


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Cancel a job that is currently processing.
    """
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id,
    ).first()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    if job.status not in [JobStatus.QUEUED.value, JobStatus.PROCESSING.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only cancel queued or processing jobs",
        )

    job.status = JobStatus.CANCELLED.value
    db.commit()

    # TODO: Actually cancel the GPU job via Modal

    return JobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage="Cancelled",
    )
