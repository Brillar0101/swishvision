"""
API package.
"""
from fastapi import APIRouter

from app.api.routes import auth, teams, jobs, admin, webhooks

# Main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router)
api_router.include_router(teams.router)
api_router.include_router(jobs.router)
api_router.include_router(admin.router)
api_router.include_router(webhooks.router)
