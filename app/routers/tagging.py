from fastapi import APIRouter, Depends, HTTPException

from app.sessions import Session, get_session
from app.models.schemas import CaptionGenerateRequest, CaptionGenerateResponse
from app.services import tagger_service

router = APIRouter(prefix="/api/tagging", tags=["tagging"])


@router.post("/generate", response_model=CaptionGenerateResponse)
def generate(req: CaptionGenerateRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        caption = tagger_service.generate_caption(session, req.index, req.tagger)
        return CaptionGenerateResponse(caption=caption)
    except Exception as e:
        raise HTTPException(500, str(e))
