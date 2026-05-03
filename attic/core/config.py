"""
Application configuration using Pydantic Settings.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    APP_NAME: str = "SwishVision"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    # Database
    DATABASE_URL: str = "sqlite:///./swishvision.db"

    # AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "swishvision-uploads"

    # Roboflow
    ROBOFLOW_API_KEY: Optional[str] = None

    # Modal (GPU compute)
    MODAL_TOKEN_ID: Optional[str] = None
    MODAL_TOKEN_SECRET: Optional[str] = None

    # Processing defaults
    DEFAULT_MAX_DURATION_SECONDS: int = 300
    DEFAULT_GPU_TYPE: str = "A10G"
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT_SECONDS: int = 1800

    # ML Pipeline defaults
    DETECTION_CONFIDENCE: float = 0.4
    DETECTION_IOU_THRESHOLD: float = 0.9
    JERSEY_OCR_INTERVAL: int = 5

    # Storage
    VIDEO_RETENTION_DAYS: int = 30
    MAX_UPLOAD_SIZE_MB: int = 500

    # URLs
    FRONTEND_URL: str = "http://localhost:3000"
    BACKEND_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
