import logging
import os
import shutil
from datetime import datetime
from typing import Optional

from PIL import Image

from app.sessions import Session
from app.db.repository import ImageRepository, VersionRepository

logger = logging.getLogger(__name__)

VERSIONS_DIR = "versions"


def _versions_dir(dataset_path: str) -> str:
    d = os.path.join(dataset_path, ".imagetagger", VERSIONS_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def create_version_backup(session: Session, index: int, operation: str) -> Optional[int]:
    ds = session.dataset
    item = ds.get_item(index)
    if not item or not os.path.exists(item.media_path):
        return None

    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    image_id = ImageRepository.ensure_image(session.db, rel_path)
    img_row = ImageRepository.get_by_id(session.db, image_id)

    width, height = None, None
    try:
        if item.is_image:
            with Image.open(item.media_path) as img:
                width, height = img.size
    except Exception:
        pass

    file_size = os.path.getsize(item.media_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    basename = os.path.splitext(os.path.basename(rel_path))[0]
    ext = os.path.splitext(rel_path)[1]
    version_filename = f"{basename}_{operation}_{timestamp}{ext}"

    versions_dir = _versions_dir(ds.base_dir)
    version_full_path = os.path.join(versions_dir, version_filename)
    shutil.copy2(item.media_path, version_full_path)

    version_rel_path = os.path.join(VERSIONS_DIR, version_filename)
    version_id = VersionRepository.create(
        session.db,
        img_row["id"],
        version_rel_path,
        operation,
        original_width=width,
        original_height=height,
        original_size=file_size,
    )

    logger.info(f"Created version backup: {version_rel_path} for {rel_path} ({operation})")
    return version_id


def restore_version(session: Session, version_id: int, index: Optional[int] = None) -> bool:
    from lib.media_cache import compute_cache_path, generate_thumbnail

    version = VersionRepository.get_by_id(session.db, version_id)
    if not version:
        return False

    ds = session.dataset
    img_row = ImageRepository.get_by_id(session.db, version["image_id"])
    if not img_row:
        return False

    version_full_path = os.path.join(ds.base_dir, ".imagetagger", version["version_path"])
    if not os.path.exists(version_full_path):
        logger.warning(f"Version file not found: {version_full_path}")
        return False

    version_ext = os.path.splitext(version["version_path"])[1]
    current_filename = img_row["filename"]
    current_ext = os.path.splitext(current_filename)[1]

    if version_ext != current_ext:
        target_filename = os.path.splitext(current_filename)[0] + version_ext
        target_path = os.path.join(ds.base_dir, target_filename)
        ImageRepository.rename(session.db, current_filename, target_filename)
    else:
        target_filename = current_filename
        target_path = os.path.join(ds.base_dir, current_filename)

    if index is not None:
        item = ds.get_item(index)
        if item and item.media_path != target_path:
            if os.path.exists(item.media_path):
                os.remove(item.media_path)
            item.update_paths_after_rename(target_path, ds.masks_dir)

    shutil.copy2(version_full_path, target_path)

    if index is not None:
        item = ds.get_item(index)
        if item:
            stale_cache = compute_cache_path(target_path)
            if os.path.exists(stale_cache):
                os.remove(stale_cache)
            item.thumbnail_path = generate_thumbnail(target_path)

    if version["original_width"] and version["original_height"]:
        ImageRepository.update_metadata(
            session.db,
            img_row["id"],
            width=version["original_width"],
            height=version["original_height"],
            file_size=version["original_size"],
        )

    logger.info(f"Restored version {version_id} for {target_filename}")
    return True


def get_versions(session: Session, index: int) -> list[dict]:
    ds = session.dataset
    item = ds.get_item(index)
    if not item:
        return []
    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    img_row = ImageRepository.get_by_filename(session.db, rel_path)
    if not img_row:
        return []
    return VersionRepository.get_for_image(session.db, img_row["id"])


def delete_version(session: Session, version_id: int) -> bool:
    version = VersionRepository.get_by_id(session.db, version_id)
    if not version:
        return False

    ds = session.dataset
    version_full_path = os.path.join(ds.base_dir, ".imagetagger", version["version_path"])
    if os.path.exists(version_full_path):
        os.remove(version_full_path)

    VersionRepository.delete(session.db, version_id)
    return True


def _delete_all_version_files(session: Session, image_id: int) -> None:
    ds = session.dataset
    for version in VersionRepository.get_for_image(session.db, image_id):
        version_full_path = os.path.join(
            ds.base_dir, ".imagetagger", version["version_path"]
        )
        if os.path.exists(version_full_path):
            os.remove(version_full_path)
    VersionRepository.delete_for_image(session.db, image_id)


def revert_to_original(session: Session, index: int) -> str:
    """Restore an image to its earliest backup and drop its version history.

    Returns "reverted", "no_versions" or "failed".
    """
    ds = session.dataset
    item = ds.get_item(index)
    if not item:
        return "failed"

    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    img_row = ImageRepository.get_by_filename(session.db, rel_path)
    if not img_row:
        return "no_versions"

    oldest = VersionRepository.get_oldest_for_image(session.db, img_row["id"])
    if not oldest:
        return "no_versions"

    if not restore_version(session, oldest["id"], index=index):
        return "failed"

    _delete_all_version_files(session, img_row["id"])
    logger.info(f"Reverted {rel_path} to original and cleared version history")
    return "reverted"
