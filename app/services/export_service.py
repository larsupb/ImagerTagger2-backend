import os
import zipfile
import tempfile
import shutil
from collections import OrderedDict
from typing import Callable, Optional
from PIL import Image

from app.sessions import Session
from app.db.repository import ImageRepository, CategoryRepository
from lib.bucketing import (
    generate_buckets,
    find_nearest_bucket,
    resize_and_crop_to_bucket,
)


MUSUBI_CONTROL_CONFIG = """[general]
resolution = [1024, 1024]
batch_size = 1

[[datasets]]
image_directory = "img"
caption_extension = ".txt"
control_directory = "control"
"""

MUSUBI_CONTROL_JSON_CONFIG = """[general]
resolution = [1024, 1024]
batch_size = 1

[[datasets]]
image_directory = "img"
caption_directory = "captions"
caption_extension = ".json"
control_directory = "control"
"""


class ExportOptions:
    def __init__(
        self,
        format: str = "standard",
        caption_type: str = "tags",
        bucket_resize: bool = False,
        bucket_resolution: int = 1024,
        bucket_step: int = 128,
        bucket_max_steps: int = 2,
        categories: list[str] | None = None,
    ):
        self.format = format
        self.caption_type = caption_type
        self.bucket_resize = bucket_resize
        self.bucket_resolution = bucket_resolution
        self.bucket_step = bucket_step
        self.bucket_max_steps = bucket_max_steps
        self.categories = categories


ProgressCallback = Callable[[int, str, str], bool]


def export_dataset(
    session: Session,
    options: ExportOptions,
    on_progress: Optional[ProgressCallback] = None,
) -> str:
    """
    Export dataset to standard format.

    Creates:
    - output_dir/img/ with images and .txt caption files
    - output_dir/masks/ with mask files (if exist)

    on_progress is called once per dataset item with (index, filename, message);
    returning False aborts the export and leaves a partial output directory.

    Returns path to output directory.
    """
    from app.services import caption_service

    ds = session.dataset
    if ds is None:
        raise ValueError("No dataset loaded")

    output_dir = tempfile.mkdtemp(prefix="imagetagger_export_")
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(img_dir)

    if options.format in ("musubi_control", "musubi_control_json"):
        control_dir = os.path.join(output_dir, "control")
        os.makedirs(control_dir)
    else:
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir)

    buckets = None
    if options.bucket_resize:
        buckets = generate_buckets(
            options.bucket_resolution,
            options.bucket_step,
            options.bucket_max_steps,
        )

    def report(index: int, filename: str, message: str) -> bool:
        return True if on_progress is None else on_progress(index, filename, message)

    total = len(ds)
    for i in range(total):
        item = ds.get_item(i)
        if item is None:
            if not report(i, "", "Skipped (item missing)"):
                break
            continue

        filename = os.path.basename(item.media_path)

        if options.categories is not None:
            rel_path = os.path.relpath(item.media_path, ds.base_dir)
            img_row = ImageRepository.get_by_filename(session.db, rel_path)
            category = CategoryRepository.get_for_image(session.db, img_row["id"]) if img_row else None
            if category not in options.categories:
                if category is None and "Uncategorized" not in options.categories:
                    if not report(i, filename, "Skipped (uncategorized)"):
                        break
                    continue
                elif category is not None:
                    if not report(i, filename, "Skipped (category)"):
                        break
                    continue

        new_filename = f"{i:05d}"

        img = Image.open(item.media_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if options.bucket_resize and buckets:
            bucket = find_nearest_bucket(img.width, img.height, buckets)
            img = resize_and_crop_to_bucket(img, bucket)

        img_path = os.path.join(img_dir, f"{new_filename}.jpg")
        img.save(img_path, format="JPEG", quality=95)

        caption = caption_service.read_caption(session, i, options.caption_type)

        if options.format == "musubi_control_json":
            captions_dir = os.path.join(output_dir, "captions")
            os.makedirs(captions_dir, exist_ok=True)
            caption_path = os.path.join(captions_dir, f"{new_filename}.json")
            if caption.startswith("{"):
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption)
            else:
                import json as _json
                with open(caption_path, "w", encoding="utf-8") as f:
                    _json.dump({"caption": caption}, f, ensure_ascii=False)
        else:
            caption_path = os.path.join(img_dir, f"{new_filename}.txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)

        if options.format == "musubi_control":
            if item.mask_path and os.path.exists(item.mask_path):
                mask = Image.open(item.mask_path)
                control_path = os.path.join(control_dir, f"{new_filename}.png")
                mask.save(control_path, format="PNG")
        elif options.format == "musubi_control_json":
            if item.mask_path and os.path.exists(item.mask_path):
                mask = Image.open(item.mask_path)
                control_path = os.path.join(control_dir, f"{new_filename}.png")
                mask.save(control_path, format="PNG")
        else:
            if item.mask_path and os.path.exists(item.mask_path):
                mask = Image.open(item.mask_path)
                mask_path = os.path.join(masks_dir, f"{new_filename}.png")
                mask.save(mask_path, format="PNG")

        if not report(i, filename, f"Exported as {new_filename}"):
            break

    if options.format == "musubi_control":
        toml_path = os.path.join(output_dir, "dataset_config.toml")
        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(MUSUBI_CONTROL_CONFIG)
    elif options.format == "musubi_control_json":
        toml_path = os.path.join(output_dir, "dataset_config.toml")
        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(MUSUBI_CONTROL_JSON_CONFIG)

    return output_dir


def create_zip_archive(source_dir: str, zip_path: str) -> str:
    """Create ZIP archive from source directory."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    return zip_path


MAX_RETAINED_ARCHIVES = 5

_archives: "OrderedDict[str, str]" = OrderedDict()


def create_archive(source_dir: str, filename: str) -> str:
    """Zip source_dir into its own temp directory, so archives never collide."""
    archive_dir = tempfile.mkdtemp(prefix="imagetagger_archive_")
    zip_path = os.path.join(archive_dir, filename)
    try:
        create_zip_archive(source_dir, zip_path)
    except Exception:
        shutil.rmtree(archive_dir, ignore_errors=True)
        raise
    return zip_path


def store_archive(task_id: str, zip_path: str) -> None:
    """Keep the newest archives only; older ones are deleted from temp."""
    _archives[task_id] = zip_path
    while len(_archives) > MAX_RETAINED_ARCHIVES:
        _, evicted_path = _archives.popitem(last=False)
        shutil.rmtree(os.path.dirname(evicted_path), ignore_errors=True)


def get_archive(task_id: str) -> Optional[str]:
    return _archives.get(task_id)


def discard_archives() -> None:
    """Drop every archive; called on shutdown so no export zips outlive the server."""
    while _archives:
        _, zip_path = _archives.popitem()
        shutil.rmtree(os.path.dirname(zip_path), ignore_errors=True)


def cleanup_export_dir(output_dir: str):
    """Remove temporary export directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
