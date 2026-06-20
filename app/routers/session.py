from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.sessions import Session, get_session

router = APIRouter(prefix="/api/session", tags=["session"])


class SessionSettingsResponse(BaseModel):
    active_tag_type: str | None


class SessionSettingsUpdateRequest(BaseModel):
    active_tag_type: str | None


@router.get("/settings", response_model=SessionSettingsResponse)
def get_session_settings(session: Session = Depends(get_session)):
    return SessionSettingsResponse(
        active_tag_type=session.config.get("active_tag_type")
    )


@router.put("/settings", response_model=SessionSettingsResponse)
def update_session_settings(req: SessionSettingsUpdateRequest, session: Session = Depends(get_session)):
    if req.active_tag_type is not None:
        session.config["active_tag_type"] = req.active_tag_type
    return SessionSettingsResponse(
        active_tag_type=session.config.get("active_tag_type")
    )