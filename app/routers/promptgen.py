import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.sessions import Session, get_session
from app.services import promptgen_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/promptgen", tags=["promptgen"])


class GenerateRequest(BaseModel):
    caption_type: str
    example_count: int = 20
    user_prompt: str = ""


class GenerateResponse(BaseModel):
    prompt: str


@router.post("/generate", response_model=GenerateResponse)
def generate_prompt(req: GenerateRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    try:
        prompt = promptgen_service.generate_prompt(
            session, req.caption_type, req.example_count, req.user_prompt
        )
        return GenerateResponse(prompt=prompt)
    except ValueError as e:
        logger.warning(f"PromptGen ValueError: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error generating prompt: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to generate prompt: {str(e)}")
