import os
from typing import Optional
from PIL import Image

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from lib.image_dataset import ImageDataSet
from lib.media_item import MediaItem

from app.sessions import Session
from app.config import read_settings
from app.db import open_db, sync_db_with_filesystem
from app.db.repository import ImageRepository, CaptionRepository, CategoryRepository
from app.services.caption_service import read_caption, get_all_captions


def load_dataset(
    session: Session,
    path: str,
    only_missing_captions: bool = False,
    include_subdirectories: bool = False,
):
    masks_dir = os.path.join(path, ".imagetagger", "masks")
    dataset = ImageDataSet()
    dataset.load(
        path,
        masks_dir=masks_dir,
        subdirectories=include_subdirectories,
        only_missing_captions=only_missing_captions,
    )
    session.dataset = dataset
    session.config = read_settings()

    session.db = open_db(path)
    sync_db_with_filesystem(session.db, dataset)


def get_media_item_response(session: Session, index: int) -> dict:
    ds = session.dataset
    item: MediaItem = ds.get_item(index)
    caption = read_caption(session, index)
    all_captions = get_all_captions(session, index)

    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    is_bookmarked = False
    category = None
    if session.db:
        is_bookmarked = ImageRepository.is_bookmarked(session.db, rel_path)
        img_row = ImageRepository.get_by_filename(session.db, rel_path)
        if img_row:
            category = CategoryRepository.get_for_image(session.db, img_row["id"])

    width, height, file_size = None, None, None
    try:
        file_size = os.path.getsize(item.media_path)
        if item.is_image:
            with Image.open(item.media_path) as img:
                width, height = img.size
    except Exception:
        pass

    return {
        "index": index,
        "filename": item.filename,
        "basename": item.basename,
        "extension": item.extension,
        "is_video": item.is_video,
        "is_image": item.is_image,
        "has_caption": bool(caption),
        "has_mask": item.mask_exists(),
        "is_bookmarked": is_bookmarked,
        "width": width,
        "height": height,
        "file_size": file_size,
        "media_url": f"/api/media/file/{index}",
        "thumbnail_url": f"/api/media/thumbnail/{index}",
        "caption": caption,
        "captions": [
            {"caption_type": c["caption_type"], "content": c["content"], "is_active": bool(c["is_active"])}
            for c in all_captions
        ],
        "category": category,
    }
