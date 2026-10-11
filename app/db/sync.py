import json
import logging
import os

from PIL import Image

from app.db.repository import ImageRepository, CaptionRepository

logger = logging.getLogger(__name__)


def _file_meta(media_path: str) -> tuple:
    try:
        file_size = os.path.getsize(media_path)
    except OSError:
        return None, None, None
    width = height = None
    try:
        with Image.open(media_path) as img:
            width, height = img.size
    except Exception:
        pass
    return width, height, file_size


def sync_db_with_filesystem(db, dataset) -> dict:
    existing_filenames = ImageRepository.get_all_filenames(db)
    # rows created before metadata was populated at sync time; file_size is the
    # durable marker since videos legitimately keep width/height NULL
    missing_meta = {
        r["filename"]: r["id"]
        for r in db.execute(
            "SELECT id, filename FROM images WHERE file_size IS NULL"
        ).fetchall()
    }
    dataset_filenames = set()
    added = 0
    removed = 0
    backfilled = 0

    for i in range(len(dataset)):
        item = dataset.get_item(i)
        if not item:
            continue
        rel_path = os.path.relpath(item.media_path, dataset.base_dir)
        dataset_filenames.add(rel_path)

        if rel_path not in existing_filenames:
            width, height, file_size = _file_meta(item.media_path)
            image_id = ImageRepository.ensure_image(
                db, rel_path, width=width, height=height, file_size=file_size
            )
            _import_caption_file(db, image_id, item)
            added += 1
        elif rel_path in missing_meta:
            width, height, file_size = _file_meta(item.media_path)
            ImageRepository.update_metadata(
                db, missing_meta[rel_path], width=width, height=height, file_size=file_size
            )
            backfilled += 1

    stale = existing_filenames - dataset_filenames
    for filename in stale:
        ImageRepository.delete_by_filename(db, filename)
        removed += 1

    _import_bookmarks(db, dataset)

    logger.info(
        f"DB sync: {added} added, {removed} removed, {backfilled} backfilled, "
        f"{len(dataset_filenames)} total"
    )
    return {"added": added, "removed": removed, "total": len(dataset_filenames)}


def _import_caption_file(db, image_id: int, item) -> None:
    if not os.path.exists(item.caption_path):
        return
    try:
        with open(item.caption_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            CaptionRepository.upsert(db, image_id, "tags", content, is_active=True)
    except Exception as e:
        logger.warning(f"Failed to import caption for {item.filename}: {e}")


def _import_bookmarks(db, dataset) -> None:
    bookmarks_path = os.path.join(dataset.base_dir, "bookmarks.json")
    if not os.path.exists(bookmarks_path):
        return
    try:
        with open(bookmarks_path, "r", encoding="utf-8") as f:
            bookmarks = json.load(f)
        for filename, is_bm in bookmarks.items():
            if not is_bm:
                continue
            row = db.execute(
                "SELECT id FROM images WHERE filename = ?", (filename,)
            ).fetchone()
            if row:
                ImageRepository.set_bookmarked(db, row["id"], True)
    except Exception as e:
        logger.warning(f"Failed to import bookmarks: {e}")
