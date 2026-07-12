from fastapi import APIRouter, Depends, HTTPException

from app.sessions import Session, get_session
from app.models.schemas import (
    CaptionSaveRequest,
    TagCloudEntry,
    TagOperationRequest,
    AppendTagRequest,
    SearchReplaceRequest,
    SearchReplacePreview,
    MoveToSubdirRequest,
    ExportResponse,
    SetActiveCaptionRequest,
    ExportCaptionsTxtRequest,
    CaptionTypeRequest,
    RenameCaptionTypeRequest,
)
from app.services import caption_service

router = APIRouter(prefix="/api/captions", tags=["captions"])


@router.put("/save")
def save_caption(req: CaptionSaveRequest, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    caption_service.save_caption(session, req.index, req.caption, req.caption_type)
    return {"ok": True}


@router.put("/set-active")
def set_active(req: SetActiveCaptionRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    caption_service.set_active_caption_type(session, req.index, req.caption_type)
    return {"ok": True}


@router.get("/types")
def get_caption_types(session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    return caption_service.get_caption_types(session)


@router.post("/types/rename")
def rename_caption_type(req: RenameCaptionTypeRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    if not req.new_type.strip():
        raise HTTPException(400, "New type name cannot be empty")
    updated = caption_service.rename_caption_type(session, req.old_type, req.new_type.strip())
    return {"updated": updated}


@router.post("/types/delete")
def delete_caption_type(req: CaptionTypeRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    deleted = caption_service.delete_caption_type(session, req.caption_type)
    return {"deleted": deleted}


@router.get("/tags", response_model=list[TagCloudEntry])
def tag_cloud(sort_by: str = "frequency", caption_type: str = "tags", session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    return caption_service.get_tag_cloud(session, sort_by, caption_type)


@router.post("/tags/remove")
def remove_tags(req: TagOperationRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    modified = caption_service.remove_tags(session, req.tags, req.caption_type)
    return {"modified": modified}


@router.post("/tags/append")
def append_tag(req: AppendTagRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    modified = caption_service.append_tag(session, req.tag, req.caption_type, req.categories)
    return {"modified": modified}


@router.post("/tags/prepend")
def prepend_tag(req: AppendTagRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    modified = caption_service.prepend_tag(session, req.tag, req.caption_type, req.categories)
    return {"modified": modified}


@router.post("/tags/cleanup")
def cleanup_tags_ep(req: CaptionTypeRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    modified = caption_service.cleanup_tags(session, req.caption_type)
    return {"modified": modified}


@router.post("/tags/replace-underscores")
def replace_underscores_ep(req: CaptionTypeRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    modified = caption_service.replace_underscores(session, req.caption_type)
    return {"modified": modified}


@router.post("/search-replace/preview", response_model=SearchReplacePreview)
def search_replace_preview(
    req: SearchReplaceRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    return caption_service.search_replace_preview(session, req.search, req.replace, req.caption_type)


@router.post("/search-replace/apply")
def search_replace_apply(
    req: SearchReplaceRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    modified = caption_service.search_replace_apply(session, req.search, req.replace, req.caption_type)
    return {"modified": modified}


@router.post("/export", response_model=ExportResponse)
def export_jsonl(session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    return caption_service.export_to_jsonl(session)


@router.post("/export-txt")
def export_captions_txt(req: ExportCaptionsTxtRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    count = caption_service.export_captions_to_txt(session, req.caption_type)
    return {"count": count}


@router.post("/move-to-subdir")
def move_to_subdirectory(
    req: MoveToSubdirRequest, session: Session = Depends(get_session)
):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    moved = caption_service.move_to_subdirectory(
        session, req.tags, req.inverse, req.subdirectory_name, req.caption_type
    )
    return {"moved": moved}
