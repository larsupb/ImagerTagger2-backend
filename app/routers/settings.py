import json
import os
from fastapi import APIRouter, Depends, HTTPException

from app.sessions import Session, get_session
from app.models.schemas import (
    SettingsResponse,
    SettingsUpdateRequest,
)
from app.config import read_settings, update_setting

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/", response_model=SettingsResponse)
def get_settings():
    settings = read_settings()
    return SettingsResponse(**settings)


@router.put("/")
def update_settings(req: SettingsUpdateRequest):
    update_setting(req.key, req.value)
    return {"ok": True}


@router.get("/upscalers")
def list_upscalers():
    upscalers_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "upscalers.json"
    )
    with open(upscalers_path) as f:
        return json.load(f)


@router.get("/background-removers")
def list_background_removers():
    return [
        {"name": "u2net", "description": "General purpose"},
        {"name": "u2net_human_seg", "description": "Human segmentation (default)"},
        {"name": "u2netp", "description": "Portable u2net"},
        {"name": "isnet", "description": "Interactive segmentation"},
        {"name": "bisenet_v1", "description": "Business scene v1"},
        {"name": "bisenet_v2", "description": "Business scene v2"},
    ]


@router.get("/taggers")
def list_taggers():
    settings = read_settings()
    default_tagger = settings.get("default_tagger", "florence")
    return {
        "taggers": [
            {"id": "joytag", "name": "JoyTag", "description": "Simple tag generation"},
            {"id": "wd14", "name": "WD14", "description": "Waifu Diffusion tags"},
            {
                "id": "florence",
                "name": "Florence-2",
                "description": "Detailed descriptions",
            },
            {
                "id": "vlm-tagger",
                "name": "VLM Tagger",
                "description": "API-based (Ollama, etc.)",
            },
            {
                "id": "ideogram",
                "name": "Ideogram JSON",
                "description": "Structured JSON via VLM",
            },
            {
                "id": "combo",
                "name": "Combo",
                "description": "Combination of selected taggers",
            },
        ],
        "default_tagger": default_tagger,
    }
