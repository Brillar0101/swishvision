"""
Teams API routes.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models import User, Team
from app.schemas import TeamCreate, TeamUpdate, TeamResponse
from app.api.deps import get_current_user

router = APIRouter(prefix="/teams", tags=["Teams"])


@router.get("/", response_model=List[TeamResponse])
async def list_teams(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    List all teams for the current user.
    """
    teams = db.query(Team).filter(Team.user_id == current_user.id).order_by(Team.name).all()
    return [
        TeamResponse(
            id=team.id,
            name=team.name,
            color=team.color,
            roster=team.roster,
            logo_url=team.logo_url,
            league=team.league,
            created_at=team.created_at.isoformat(),
        )
        for team in teams
    ]


@router.post("/", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    team_data: TeamCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new team for the current user.
    """
    team = Team(
        user_id=current_user.id,
        name=team_data.name,
        color=team_data.color,
        roster=team_data.roster,
        logo_url=team_data.logo_url,
        league=team_data.league,
    )
    db.add(team)
    db.commit()
    db.refresh(team)

    return TeamResponse(
        id=team.id,
        name=team.name,
        color=team.color,
        roster=team.roster,
        logo_url=team.logo_url,
        league=team.league,
        created_at=team.created_at.isoformat(),
    )


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get a specific team by ID.
    """
    team = db.query(Team).filter(
        Team.id == team_id,
        Team.user_id == current_user.id,
    ).first()

    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    return TeamResponse(
        id=team.id,
        name=team.name,
        color=team.color,
        roster=team.roster,
        logo_url=team.logo_url,
        league=team.league,
        created_at=team.created_at.isoformat(),
    )


@router.patch("/{team_id}", response_model=TeamResponse)
async def update_team(
    team_id: str,
    team_data: TeamUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update a team.
    """
    team = db.query(Team).filter(
        Team.id == team_id,
        Team.user_id == current_user.id,
    ).first()

    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    # Update fields if provided
    update_data = team_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(team, field, value)

    db.commit()
    db.refresh(team)

    return TeamResponse(
        id=team.id,
        name=team.name,
        color=team.color,
        roster=team.roster,
        logo_url=team.logo_url,
        league=team.league,
        created_at=team.created_at.isoformat(),
    )


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team(
    team_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a team.
    """
    team = db.query(Team).filter(
        Team.id == team_id,
        Team.user_id == current_user.id,
    ).first()

    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    db.delete(team)
    db.commit()
