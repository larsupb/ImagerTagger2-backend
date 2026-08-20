import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.services.project_service import (
    find_registry_entry,
    read_project_config,
    update_project_config,
)
from app.sessions import Session, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/session", tags=["session"])


class SessionSettingsResponse(BaseModel):
    active_tag_type: str | None


class SessionSettingsUpdateRequest(BaseModel):
    active_tag_type: str | None


def _persist_active_tag_type(session: Session, active_tag_type: str) -> None:
    """Mirror the choice into the project config so it survives a restart."""
    entry = find_registry_entry(session.project_id)
    if entry is None:
        return
    config = read_project_config(entry["path"])
    overrides = dict((config or {}).get("settings_overrides") or {})
    if overrides.get("active_tag_type") == active_tag_type:
        return
    overrides["active_tag_type"] = active_tag_type
    update_project_config(entry["path"], settings_overrides=overrides)


@router.get("/settings", response_model=SessionSettingsResponse)
def get_session_settings(session: Session = Depends(get_session)):
    return SessionSettingsResponse(
        active_tag_type=session.config.get("active_tag_type")
    )


@router.put("/settings", response_model=SessionSettingsResponse)
def update_session_settings(req: SessionSettingsUpdateRequest, session: Session = Depends(get_session)):
    if req.active_tag_type is not None:
        session.config["active_tag_type"] = req.active_tag_type
        try:
            _persist_active_tag_type(session, req.active_tag_type)
        except Exception as e:
            logger.warning("Could not persist active_tag_type: %s", e)
    return SessionSettingsResponse(
        active_tag_type=session.config.get("active_tag_type")
    )
