import json
import logging
import os

from app.db.repository import ImageRepository, CaptionRepository

logger = logging.getLogger(__name__)


def sync_db_with_filesystem(db, dataset) -> dict:
    existing_filenames = ImageRepository.get_all_filenames(db)
    dataset_filenames = set()
    added = 0
    removed = 0

    for i in range(len(dataset)):
        item = dataset.get_item(i)
        if not item:
            continue
        rel_path = os.path.relpath(item.media_path, dataset.base_dir)
        dataset_filenames.add(rel_path)

        if rel_path not in existing_filenames:
            image_id = ImageRepository.ensure_image(db, rel_path)
            _import_caption_file(db, image_id, item)
            added += 1

    stale = existing_filenames - dataset_filenames
    for filename in stale:
        ImageRepository.delete_by_filename(db, filename)
        removed += 1

    _import_bookmarks(db, dataset)

    logger.info(
        f"DB sync: {added} added, {removed} removed, {len(dataset_filenames)} total"
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
