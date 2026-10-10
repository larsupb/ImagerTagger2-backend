import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.db.repository import ImageRepository
from app.sessions import Session, get_session_flexible

router = APIRouter(prefix="/api/media", tags=["media"])

IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
VIDEO_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


@router.get("/file/{index}")
def serve_media(index: int, session: Session = Depends(get_session_flexible)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    if index < 0 or index >= len(ds):
        raise HTTPException(404, "Index out of range")

    item = ds.get_item(index)
    if not item.media_exists():
        raise HTTPException(404, "Media file not found")

    ext = item.extension.lower()
    media_type = IMAGE_MEDIA_TYPES.get(ext) or VIDEO_MEDIA_TYPES.get(
        ext, "application/octet-stream"
    )
    return FileResponse(item.media_path, media_type=media_type)


@router.get("/thumbnail/{index}")
def serve_thumbnail(index: int, session: Session = Depends(get_session_flexible)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    if index < 0 or index >= len(ds):
        raise HTTPException(404, "Index out of range")

    item = ds.get_item(index)
    if item.thumbnail_exists():
        return FileResponse(item.thumbnail_path, media_type="image/jpeg")
    elif item.media_exists():
        ext = item.extension.lower()
        media_type = IMAGE_MEDIA_TYPES.get(ext, "image/jpeg")
        return FileResponse(item.media_path, media_type=media_type)
    else:
        raise HTTPException(404, "No thumbnail or media file found")


@router.get("/version/{version_id}")
def serve_version(version_id: int, session: Session = Depends(get_session_flexible)):
    from app.db.repository import VersionRepository

    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")

    version = VersionRepository.get_by_id(session.db, version_id)
    if not version:
        raise HTTPException(404, "Version not found")

    version_full_path = os.path.join(
        ds.base_dir, ".imagetagger", version["version_path"]
    )
    if not os.path.exists(version_full_path):
        raise HTTPException(404, "Version file not found")

    ext = os.path.splitext(version_full_path)[1].lower()
    media_type = IMAGE_MEDIA_TYPES.get(ext, "image/jpeg")
    return FileResponse(version_full_path, media_type=media_type)


@router.get("/mask/{index}")
def serve_mask(index: int, session: Session = Depends(get_session_flexible)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    if index < 0 or index >= len(ds):
        raise HTTPException(404, "Index out of range")

    item = ds.get_item(index)
    if not item.mask_exists():
        raise HTTPException(404, "Mask file not found")
    return FileResponse(item.mask_path, media_type="image/png")


@router.get("/by-id/{image_id}")
def serve_media_by_id(image_id: int, session: Session = Depends(get_session_flexible)):
    if session.db is None:
        raise HTTPException(400, "No project database open")
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    row = ImageRepository.get_by_id(session.db, image_id)
    if row is None:
        raise HTTPException(404, "Image not found")
    try:
        index = session.dataset.find_index(
            os.path.join(session.dataset.base_dir, row["filename"])
        )
    except ValueError:
        raise HTTPException(404, "Image not in current dataset")
    return serve_media(index, session)


@router.get("/thumbnail-by-id/{image_id}")
def serve_thumbnail_by_id(image_id: int, session: Session = Depends(get_session_flexible)):
    if session.db is None:
        raise HTTPException(400, "No project database open")
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    row = ImageRepository.get_by_id(session.db, image_id)
    if row is None:
        raise HTTPException(404, "Image not found")
    try:
        index = session.dataset.find_index(
            os.path.join(session.dataset.base_dir, row["filename"])
        )
    except ValueError:
        raise HTTPException(404, "Image not in current dataset")
    return serve_thumbnail(index, session)
