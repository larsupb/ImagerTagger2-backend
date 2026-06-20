import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image

from app.models.schemas import (
    DedupRemoveRequest,
    DedupScanRequest,
    DedupScanResponse,
    DedupGroup,
    DedupImageInfo,
)
from app.services import dedup_service
from app.sessions import Session, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dedup", tags=["dedup"])


def _image_dimensions(path: str) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except Exception:
        return None, None


@router.post("/scan", response_model=DedupScanResponse)
def scan(req: DedupScanRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    image_items = []
    dataset_indices = []
    for i in range(len(ds)):
        item = ds.get_item(i)
        if item and item.is_image:
            image_items.append(item)
            dataset_indices.append(i)

    try:
        groups = dedup_service.group_duplicates(
            image_items, req.phash_threshold, req.ssim_threshold
        )
    except Exception as e:
        logger.error("dedup scan failed: %s", e, exc_info=True)
        raise HTTPException(500, str(e))

    result_groups = []
    for group_indices in groups:
        images = []
        for filtered_idx in group_indices:
            ds_idx = dataset_indices[filtered_idx]
            item = image_items[filtered_idx]
            width, height = _image_dimensions(item.media_path)
            file_size = os.path.getsize(item.media_path) if os.path.exists(item.media_path) else None
            images.append(
                DedupImageInfo(
                    index=ds_idx,
                    filename=item.filename,
                    width=width,
                    height=height,
                    file_size=file_size,
                    thumbnail_url=f"/api/media/thumbnail/{ds_idx}",
                    path=item.media_path,
                )
            )
        keep_idx = max(
            range(len(images)),
            key=lambda i: (images[i].width or 0) * (images[i].height or 0),
        )
        result_groups.append(DedupGroup(images=images, keep_index=keep_idx))

    return DedupScanResponse(groups=result_groups)


@router.delete("/remove")
def remove(req: DedupRemoveRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    valid_paths = {
        item.media_path
        for i in range(len(ds))
        if (item := ds.get_item(i)) is not None
    }
    invalid = [p for p in req.paths if p not in valid_paths]
    if invalid:
        raise HTTPException(400, f"Paths not in current session: {invalid[:3]}")

    deleted = 0
    for path in req.paths:
        try:
            idx = ds.find_index(path)
            if ds.delete_item(idx):
                deleted += 1
        except ValueError:
            logger.warning("remove: path already gone: %s", path)

    return {"deleted": deleted}
