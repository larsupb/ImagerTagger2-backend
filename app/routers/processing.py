import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from app.sessions import Session, get_session

logger = logging.getLogger(__name__)
from app.models.schemas import (
    UpscaleRequest,
    MaskGenerateRequest,
    ImageVersionEntry,
    ComfyUIUpscaleRequest,
    CropRequest,
    WhiteBalanceRequest,
)
from app.services import processing_service
from app.services.version_service import get_versions, restore_version, delete_version

router = APIRouter(prefix="/api/processing", tags=["processing"])


@router.post("/upscale")
def upscale(req: UpscaleRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        processing_service.upscale_image(
            session, req.index, req.upscaler, req.target_megapixels
        )
        return {"status": "upscaled", "index": req.index}
    except Exception as e:
        logger.error("upscale failed for index %s: %s", req.index, e, exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/upscale-comfyui")
async def upscale_comfyui(
    req: ComfyUIUpscaleRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        import asyncio
        from app.services import comfyui_service

        await asyncio.to_thread(
            comfyui_service.upscale_with_comfyui, session, req.index, req.target_megapixels
        )
        return {"status": "upscaled", "index": req.index}
    except Exception as e:
        logger.error("comfyui upscale failed for index %s: %s", req.index, e, exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/upscale/save")
def save_upscaled(index: int, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        processing_service.save_upscaled(session, index)
        return {"status": "saved"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/remove-background")
def remove_background(
    index: int, model: str | None = None, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        processing_service.remove_background(session, index, model)
        return {"status": "done", "index": index}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/mask/generate")
def generate_mask(req: MaskGenerateRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        mask_path = processing_service.generate_mask(session, req.index)
        return {"status": "done", "mask_url": f"/api/media/mask/{req.index}"}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/crop")
def crop(req: CropRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        processing_service.crop_image(
            session, req.index, req.x, req.y, req.width, req.height
        )
        return {"status": "cropped", "index": req.index}
    except Exception as e:
        logger.error("crop failed for index %s: %s", req.index, e, exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/white-balance")
def white_balance(req: WhiteBalanceRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        processing_service.white_balance_image(session, req.index, req.method)
        return {"status": "balanced", "index": req.index}
    except Exception as e:
        logger.error(
            "white balance failed for index %s: %s", req.index, e, exc_info=True
        )
        raise HTTPException(500, str(e))


@router.post("/paint")
async def paint(
    index: int = Form(...),
    paint_png: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        data = await paint_png.read()
        processing_service.paint_image(session, index, data)
        return {"status": "painted", "index": index}
    except Exception as e:
        logger.error("paint failed for index %s: %s", index, e, exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/mask/save")
async def save_mask_endpoint(
    index: int = Form(...),
    mask_png: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    try:
        data = await mask_png.read()
        processing_service.save_mask(session, index, data)
        return {"status": "saved", "index": index}
    except Exception as e:
        logger.error("mask save failed for index %s: %s", index, e, exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/versions/{index}", response_model=list[ImageVersionEntry])
def list_versions(index: int, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    return get_versions(session, index)


@router.post("/versions/{version_id}/restore")
def restore_ver(
    version_id: int,
    index: Optional[int] = Query(None),
    session: Session = Depends(get_session),
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    ok = restore_version(session, version_id, index=index)
    if not ok:
        raise HTTPException(404, "Version not found or restore failed")
    return {"ok": True}


@router.delete("/versions/{version_id}")
def delete_ver(version_id: int, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    ok = delete_version(session, version_id)
    if not ok:
        raise HTTPException(404, "Version not found")
    return {"ok": True}
