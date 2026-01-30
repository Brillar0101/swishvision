"""
SwishVision API - Production-ready FastAPI application.

Run with: uvicorn app_main:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.config import settings
from app.core.database import init_db
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    init_db()
    print("Database initialized")
    yield
    # Shutdown
    print("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered basketball video analysis - Player tracking, team classification, and tactical views",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router with /api prefix
app.include_router(api_router, prefix="/api")

# Serve static files for local development
outputs_dir = Path("outputs")
if outputs_dir.exists():
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "Disabled in production",
    }


@app.get("/health")
async def health_check():
    """Health check for load balancers."""
    return {"status": "healthy"}
