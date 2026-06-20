import hashlib
import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.sessions import Session, get_session, session_manager
from app.db.repository import ImageRepository, CaptionRepository, CategoryRepository
from app.models.schemas import (
    DatasetLoadRequest,
    DatasetInfo,
    MediaItemResponse,
    NavigationResponse,
    GalleryResponse,
    GalleryItem,
)
from app.services.dataset_service import load_dataset, get_media_item_response
from lib.media_item import is_image, is_video

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/session")
def create_session():
    session = session_manager.create()
    return {"session_id": session.id}


@router.post("/load", response_model=DatasetInfo)
def load(req: DatasetLoadRequest, session: Session = Depends(get_session)):
    load_dataset(
        session,
        req.path,
        req.only_missing_captions,
        req.include_subdirectory,
    )
    ds = session.dataset
    return DatasetInfo(
        total_items=len(ds),
        base_dir=ds._base_dir,
        masks_dir=ds._masks_dir,
    )


@router.get("/info", response_model=DatasetInfo)
def get_info(session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    return DatasetInfo(
        total_items=len(ds),
        base_dir=ds._base_dir,
        masks_dir=ds._masks_dir,
    )


@router.get("/histogram/{index}")
def get_histogram(index: int, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    if index < 0 or index >= len(ds):
        raise HTTPException(400, "Invalid index")

    from lib.color_matching import compute_lab_histogram
    from PIL import Image

    item = ds.get_item(index)
    img = Image.open(item.media_path)
    histogram = compute_lab_histogram(img)
    return histogram
    ds = session.dataset
    return DatasetInfo(
        total_items=len(ds),
        base_dir=ds._base_dir,
        masks_dir=ds._masks_dir,
    )


@router.get("/item/{index}", response_model=MediaItemResponse)
def get_item(index: int, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    if index < 0 or index >= len(ds):
        raise HTTPException(404, f"Index {index} out of range [0, {len(ds)})")
    return get_media_item_response(session, index)


@router.get("/gallery", response_model=GalleryResponse)
def gallery(
    page: int = 0, page_size: int = 50, session: Session = Depends(get_session)
):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")

    start = page * page_size
    end = min(start + page_size, len(ds))

    category_map: dict[str, str] = (
        CategoryRepository.get_filename_to_category(session.db) if session.db else {}
    )

    items = []
    for i in range(start, end):
        item = ds.get_item(i)
        rel_path = os.path.relpath(item.media_path, ds.base_dir)
        is_bm = (
            ImageRepository.is_bookmarked(session.db, rel_path) if session.db else False
        )
        img_row = (
            ImageRepository.get_by_filename(session.db, rel_path)
            if session.db
            else None
        )
        has_caption = False
        if img_row:
            captions = CaptionRepository.get_all_for_image(session.db, img_row["id"])
            has_caption = any(c.get("content") for c in captions)
        items.append(
            GalleryItem(
                index=i,
                thumbnail_url=f"/api/media/thumbnail/{i}",
                thumb_token=hashlib.md5(item.media_path.encode()).hexdigest()[:8],
                filename=item.filename,
                is_bookmarked=is_bm,
                has_caption=has_caption,
                width=item.width if hasattr(item, "width") else None,
                height=item.height if hasattr(item, "height") else None,
                category=category_map.get(rel_path),
            )
        )

    return GalleryResponse(items=items, total=len(ds), page=page, page_size=page_size)


@router.post("/bookmark/{index}")
def toggle_bookmark(index: int, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    item = ds.get_item(index)
    if not item:
        raise HTTPException(404, "Item not found")
    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    current = ImageRepository.is_bookmarked(session.db, rel_path)
    row = ImageRepository.get_by_filename(session.db, rel_path)
    if row:
        ImageRepository.set_bookmarked(session.db, row["id"], not current)
    return {"is_bookmarked": not current}


@router.delete("/item/{index}")
def delete_item(index: int, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    item = ds.get_item(index)
    if item:
        rel_path = os.path.relpath(item.media_path, ds.base_dir)
        ImageRepository.delete_by_filename(session.db, rel_path)
    ds.delete_item(index)
    return {"total_items": len(ds)}


@router.post("/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    session: Session = Depends(get_session),
):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")

    added = []
    for file in files:
        filename = file.filename or ""
        if not (is_image(filename) or is_video(filename)):
            continue

        content = await file.read()
        suffix = os.path.splitext(filename)[1].lower()
        file_hash = hashlib.sha256(content).hexdigest()[:16]
        hashed_name = f"{file_hash}{suffix}"
        dest_path = os.path.join(ds._base_dir, hashed_name)

        if os.path.exists(dest_path):
            existing_index = ds.find_index(dest_path)
            added.append({"index": existing_index, "filename": hashed_name})
            continue

        with open(dest_path, "wb") as f:
            f.write(content)

        index = ds.add_item(dest_path)

        if session.db:
            rel_path = os.path.relpath(dest_path, ds._base_dir)
            ImageRepository.ensure_image(session.db, rel_path)

        added.append({"index": index, "filename": hashed_name})

    return {"added": added, "total_items": len(ds)}


@router.put("/item/{index}/rename")
def rename_item(index: int, new_name: str, session: Session = Depends(get_session)):
    ds = session.dataset
    if ds is None:
        raise HTTPException(400, "No dataset loaded")
    item = ds.get_item(index)
    old_rel = os.path.relpath(item.media_path, ds.base_dir) if item else None
    ds.rename_item(index, new_name)
    if old_rel and session.db:
        new_item = ds.get_item(index)
        new_rel = os.path.relpath(new_item.media_path, ds.base_dir)
        ImageRepository.rename(session.db, old_rel, new_rel)
    return get_media_item_response(session, index)
