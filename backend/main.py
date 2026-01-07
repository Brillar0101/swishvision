"""
Swish Vision API - AI-powered basketball game analysis
"""
import os
import uuid
import shutil
from datetime import datetime
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import ML modules
from app.ml.player_tracker import PlayerTracker
from app.ml.court_detector import CourtDetector

# ============================================================================
# Configuration
# ============================================================================

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job storage (in production, use Redis or database)
jobs = {}

# ============================================================================
# Models
# ============================================================================

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[dict] = None

class AnalysisResult(BaseModel):
    job_id: str
    video_url: str
    sample_frames: list
    stats: dict

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Swish Vision API",
    description="AI-powered basketball game analysis - Player tracking, team classification, and tactical views",
    version="0.2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (outputs)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ============================================================================
# ML Models (lazy loading)
# ============================================================================

_tracker = None
_court_detector = None

def get_tracker():
    global _tracker, _court_detector
    if _tracker is None:
        print("Loading ML models...")
        _court_detector = CourtDetector()
        _tracker = PlayerTracker()
        _tracker.set_court_detector(_court_detector)
        print("ML models loaded!")
    return _tracker

# ============================================================================
# Background Processing
# ============================================================================

def process_video_task(job_id: str, video_path: str):
    """Background task to process video."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Loading ML models..."
        jobs[job_id]["progress"] = 10
        
        # Get tracker
        tracker = get_tracker()
        
        jobs[job_id]["message"] = "Processing video..."
        jobs[job_id]["progress"] = 20
        
        # Create output directory for this job
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(exist_ok=True)
        
        # Process video
        result = tracker.process_video_with_tracking(
            video_path=video_path,
            output_dir=str(job_output_dir),
            keyframe_interval=30,
            max_total_objects=15,
            num_sample_frames=3,
            max_seconds=30.0,  # Limit to 30 seconds for MVP
        )
        
        jobs[job_id]["progress"] = 90
        jobs[job_id]["message"] = "Finalizing..."
        
        # Build result
        if "error" in result:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = result["error"]
            return
        
        # Get relative URLs for frontend
        video_filename = Path(result.get("video_path", "")).name
        sample_frame_urls = [
            f"/outputs/{job_id}/{Path(f).name}" 
            for f in result.get("sample_frames", [])
        ]
        
        # Build stats from tracking info
        tracking_info = result.get("tracking_info", {})
        team_counts = {"team_0": 0, "team_1": 0, "referee": 0}
        players = []
        
        for obj_id, info in tracking_info.items():
            team = info.get("team", 0)
            team_name = info.get("team_name", "Unknown")
            
            if team == -1:
                team_counts["referee"] += 1
            elif team == 0:
                team_counts["team_0"] += 1
            elif team == 1:
                team_counts["team_1"] += 1
            
            players.append({
                "id": obj_id,
                "team": team,
                "team_name": team_name,
                "class": info.get("class", "player"),
                "first_seen_frame": info.get("first_seen_frame", 0),
            })
        
        jobs[job_id]["result"] = {
            "video_url": f"/outputs/{job_id}/{video_filename}",
            "sample_frames": sample_frame_urls,
            "stats": {
                "frames_processed": result.get("frames_processed", 0),
                "players_tracked": result.get("players_tracked", 0),
                "team_counts": team_counts,
                "players": players,
            }
        }
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Analysis complete!"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Clean up uploaded video
        try:
            os.remove(video_path)
        except:
            pass
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["message"] = str(e)
        jobs[job_id]["progress"] = 0
        print(f"Error processing job {job_id}: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "Swish Vision API",
        "status": "running",
        "version": "0.2.0",
        "endpoints": {
            "upload": "POST /api/upload",
            "status": "GET /api/status/{job_id}",
            "results": "GET /api/results/{job_id}",
        }
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a video for analysis.
    
    Returns a job_id to track processing status.
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", "video/mpeg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    file_ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Video uploaded, queued for processing",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None,
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, job_id, str(video_path))
    
    return {
        "job_id": job_id,
        "message": "Video uploaded successfully",
        "status_url": f"/api/status/{job_id}",
    }


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """
    Get the processing status of a job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/api/results/{job_id}")
def get_results(job_id: str):
    """
    Get the analysis results for a completed job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    return job["result"]


@app.get("/api/jobs")
def list_jobs():
    """
    List all jobs (for debugging).
    """
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "progress": j["progress"],
                "created_at": j["created_at"],
            }
            for j in jobs.values()
        ]
    }


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """
    Delete a job and its outputs.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete output directory
    job_output_dir = OUTPUT_DIR / job_id
    if job_output_dir.exists():
        shutil.rmtree(job_output_dir)
    
    # Remove from jobs dict
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted"}


# ============================================================================
# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# ============================================================================
