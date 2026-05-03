"""
Webhook endpoints for receiving callbacks from GPU workers.
"""
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import get_db
from app.models import Job, JobStatus

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


class JobProgressUpdate(BaseModel):
    """Progress update from GPU worker."""
    job_id: str
    progress: int  # 0-100
    current_stage: str


class JobCompletedPayload(BaseModel):
    """Completion payload from GPU worker."""
    job_id: str
    status: str  # "completed" or "failed"
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    gpu_cost_cents: Optional[int] = None
    detected_players: Optional[List[Dict]] = None
    output_videos: Optional[Dict[str, str]] = None
    tracking_data_s3_key: Optional[str] = None


def verify_webhook_secret(x_webhook_secret: str = Header(...)) -> bool:
    """Verify the webhook secret to authenticate GPU worker callbacks."""
    # In production, use a secure secret
    expected_secret = settings.SECRET_KEY
    if x_webhook_secret != expected_secret:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    return True


@router.post("/job/progress")
async def update_job_progress(
    payload: JobProgressUpdate,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_webhook_secret),
):
    """
    Receive progress updates from GPU worker.

    Called periodically during video processing to update progress.
    """
    job = db.query(Job).filter(Job.id == payload.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job.progress = payload.progress
    job.current_stage = payload.current_stage

    if job.status == JobStatus.QUEUED.value:
        job.status = JobStatus.PROCESSING.value
        job.started_at = datetime.utcnow()

    db.commit()

    return {"status": "ok"}


@router.post("/job/completed")
async def job_completed(
    payload: JobCompletedPayload,
    db: Session = Depends(get_db),
    _: bool = Depends(verify_webhook_secret),
):
    """
    Receive job completion notification from GPU worker.

    Called when video processing finishes (success or failure).
    """
    job = db.query(Job).filter(Job.id == payload.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if payload.status == "completed":
        job.status = JobStatus.COMPLETED.value
        job.progress = 100
        job.current_stage = "Complete"
        job.detected_players = payload.detected_players
        job.output_videos = payload.output_videos
        job.tracking_data_s3_key = payload.tracking_data_s3_key
        job.output_s3_prefix = f"jobs/{job.id}/"
    else:
        job.status = JobStatus.FAILED.value
        job.error_message = payload.error_message

    job.completed_at = datetime.utcnow()
    job.processing_time_seconds = payload.processing_time_seconds
    job.gpu_cost_cents = payload.gpu_cost_cents

    db.commit()

    # TODO: Send email notification to user
    # TODO: Update user's usage minutes

    return {"status": "ok"}


@router.post("/modal")
async def modal_webhook(
    payload: dict,
    db: Session = Depends(get_db),
    x_webhook_secret: str = Header(None),
):
    """
    Receive webhooks from Modal.

    Modal sends various webhook events - we handle job completion here.
    """
    # Modal might send different event types
    event_type = payload.get("event_type")

    if event_type == "function.completed":
        job_id = payload.get("input", {}).get("job_id")
        result = payload.get("result", {})

        if job_id:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                if result.get("status") == "success":
                    job.status = JobStatus.COMPLETED.value
                    job.progress = 100
                    job.detected_players = result.get("detected_players")
                    job.output_videos = result.get("output_videos")
                else:
                    job.status = JobStatus.FAILED.value
                    job.error_message = result.get("error")

                job.completed_at = datetime.utcnow()
                db.commit()

    return {"status": "ok"}
