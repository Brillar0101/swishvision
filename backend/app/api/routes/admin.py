"""
Admin API routes.
"""
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from app.core.config import settings
from app.core.database import get_db
from app.models import User, Job, Team, JobStatus, UserRole
from app.schemas import UserAdminResponse, JobSummaryResponse
from app.api.deps import get_current_admin

router = APIRouter(prefix="/admin", tags=["Admin"])


# ============================================================================
# Dashboard Stats
# ============================================================================

@router.get("/stats")
async def get_dashboard_stats(
    period_hours: int = Query(24, ge=1, le=720),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Get dashboard statistics for the admin panel.
    """
    cutoff = datetime.utcnow() - timedelta(hours=period_hours)

    # Job stats
    total_jobs = db.query(Job).filter(Job.created_at >= cutoff).count()
    completed_jobs = db.query(Job).filter(
        Job.created_at >= cutoff,
        Job.status == JobStatus.COMPLETED.value,
    ).count()
    processing_jobs = db.query(Job).filter(
        Job.status.in_([JobStatus.QUEUED.value, JobStatus.PROCESSING.value]),
    ).count()
    failed_jobs = db.query(Job).filter(
        Job.created_at >= cutoff,
        Job.status == JobStatus.FAILED.value,
    ).count()

    # User stats
    total_users = db.query(User).count()
    new_users = db.query(User).filter(User.created_at >= cutoff).count()
    active_users = db.query(func.count(func.distinct(Job.user_id))).filter(
        Job.created_at >= cutoff
    ).scalar()

    # Cost stats (if tracking)
    total_cost_cents = db.query(func.sum(Job.gpu_cost_cents)).filter(
        Job.created_at >= cutoff,
        Job.gpu_cost_cents.isnot(None),
    ).scalar() or 0

    # Processing time stats
    avg_processing_time = db.query(func.avg(Job.processing_time_seconds)).filter(
        Job.created_at >= cutoff,
        Job.processing_time_seconds.isnot(None),
    ).scalar()

    return {
        "period_hours": period_hours,
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "processing": processing_jobs,
            "failed": failed_jobs,
            "success_rate": round(completed_jobs / total_jobs * 100, 1) if total_jobs > 0 else 0,
        },
        "users": {
            "total": total_users,
            "new": new_users,
            "active": active_users,
        },
        "costs": {
            "total_cents": total_cost_cents,
            "total_dollars": round(total_cost_cents / 100, 2),
        },
        "performance": {
            "avg_processing_time_seconds": round(avg_processing_time, 1) if avg_processing_time else None,
        },
    }


# ============================================================================
# User Management
# ============================================================================

@router.get("/users", response_model=List[UserAdminResponse])
async def list_users(
    page: int = 1,
    per_page: int = 50,
    search: Optional[str] = None,
    role: Optional[str] = None,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    List all users with admin details.
    """
    query = db.query(User)

    if search:
        query = query.filter(
            User.email.ilike(f"%{search}%") | User.full_name.ilike(f"%{search}%")
        )

    if role:
        query = query.filter(User.role == role)

    users = query.order_by(desc(User.created_at)).offset((page - 1) * per_page).limit(per_page).all()

    result = []
    for user in users:
        # Get job stats for this user
        total_jobs = db.query(Job).filter(Job.user_id == user.id).count()
        total_minutes = db.query(func.sum(Job.processing_time_seconds)).filter(
            Job.user_id == user.id,
            Job.processing_time_seconds.isnot(None),
        ).scalar() or 0

        result.append(UserAdminResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            plan=user.plan,
            usage_minutes_remaining=user.usage_minutes_remaining,
            is_active=user.is_active,
            created_at=user.created_at.isoformat(),
            last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
            total_jobs=total_jobs,
            total_processing_minutes=round(total_minutes / 60, 1),
        ))

    return result


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    role: Optional[str] = None,
    plan: Optional[str] = None,
    is_active: Optional[bool] = None,
    usage_minutes_remaining: Optional[int] = None,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Update a user's role, plan, or status.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent self-demotion from super_admin
    if user.id == current_admin.id and role and role != UserRole.SUPER_ADMIN.value:
        if current_admin.role == UserRole.SUPER_ADMIN.value:
            raise HTTPException(
                status_code=400,
                detail="Cannot demote yourself from super_admin",
            )

    if role is not None:
        user.role = role
    if plan is not None:
        user.plan = plan
    if is_active is not None:
        user.is_active = is_active
    if usage_minutes_remaining is not None:
        user.usage_minutes_remaining = usage_minutes_remaining

    db.commit()
    return {"message": "User updated successfully"}


# ============================================================================
# Job Management
# ============================================================================

@router.get("/jobs")
async def list_all_jobs(
    page: int = 1,
    per_page: int = 50,
    status_filter: Optional[str] = None,
    user_id: Optional[str] = None,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    List all jobs across all users.
    """
    query = db.query(Job)

    if status_filter:
        query = query.filter(Job.status == status_filter)
    if user_id:
        query = query.filter(Job.user_id == user_id)

    total = query.count()
    jobs = query.order_by(desc(Job.created_at)).offset((page - 1) * per_page).limit(per_page).all()

    # Get user emails for display
    user_ids = list(set(job.user_id for job in jobs))
    users = db.query(User).filter(User.id.in_(user_ids)).all()
    user_map = {u.id: u.email for u in users}

    return {
        "jobs": [
            {
                "id": job.id,
                "user_email": user_map.get(job.user_id, "Unknown"),
                "status": job.status,
                "progress": job.progress,
                "team_a_name": job.team_a.get("name") if job.team_a else None,
                "team_b_name": job.team_b.get("name") if job.team_b else None,
                "gpu_type": job.gpu_type,
                "gpu_cost_cents": job.gpu_cost_cents,
                "processing_time_seconds": job.processing_time_seconds,
                "created_at": job.created_at.isoformat(),
                "error_message": job.error_message,
            }
            for job in jobs
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    gpu_type: Optional[str] = None,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Retry a failed job, optionally with a different GPU type.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail="Can only retry failed or cancelled jobs")

    job.status = JobStatus.QUEUED.value
    job.progress = 0
    job.error_message = None
    job.current_stage = "Queued for retry"
    if gpu_type:
        job.gpu_type = gpu_type

    db.commit()

    # TODO: Trigger GPU processing via Modal

    return {"message": "Job queued for retry", "job_id": job.id}


@router.post("/jobs/{job_id}/cancel")
async def admin_cancel_job(
    job_id: str,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Cancel any job (admin override).
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in [JobStatus.COMPLETED.value, JobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail="Job is already finished")

    job.status = JobStatus.CANCELLED.value
    db.commit()

    # TODO: Actually cancel the GPU job via Modal

    return {"message": "Job cancelled", "job_id": job.id}


# ============================================================================
# System Settings
# ============================================================================

@router.get("/settings")
async def get_system_settings(
    current_admin: User = Depends(get_current_admin),
):
    """
    Get current system settings.
    """
    return {
        "processing": {
            "default_max_duration_seconds": settings.DEFAULT_MAX_DURATION_SECONDS,
            "default_gpu_type": settings.DEFAULT_GPU_TYPE,
            "max_concurrent_jobs": settings.MAX_CONCURRENT_JOBS,
            "job_timeout_seconds": settings.JOB_TIMEOUT_SECONDS,
        },
        "ml_pipeline": {
            "detection_confidence": settings.DETECTION_CONFIDENCE,
            "detection_iou_threshold": settings.DETECTION_IOU_THRESHOLD,
            "jersey_ocr_interval": settings.JERSEY_OCR_INTERVAL,
        },
        "storage": {
            "s3_bucket": settings.S3_BUCKET_NAME,
            "video_retention_days": settings.VIDEO_RETENTION_DAYS,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        },
    }
