import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PIL import Image
from app.sessions import Session
from app.config import read_settings
from app.services.version_service import create_version_backup


_UPSCALERS_CACHE: list[dict] | None = None


def _get_upscalers() -> list[dict]:
    global _UPSCALERS_CACHE
    if _UPSCALERS_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "upscalers.json")
        with open(path) as f:
            _UPSCALERS_CACHE = json.load(f)
    return _UPSCALERS_CACHE


def _is_comfyui_upscaler(name: str) -> bool:
    if name in ("comfyui-backend", "comfyui-seedvr2"):
        return True
    if name.startswith("comfyui-"):
        return True
    return False


def _get_workflow_file(upscaler_name: str) -> str | None:
    for entry in _get_upscalers():
        if entry.get("name") == upscaler_name:
            return entry.get("workflow_file")
    return None


def upscale_image(
    session: Session,
    index: int,
    upscaler_name: str = None,
    target_megapixels: float = None,
    cancel_check=None,
) -> str:
    ds = session.dataset
    item = ds.get_item(index)
    settings = read_settings()

    upscaler = upscaler_name or settings.get("upscaler", "NMKD_Siax_200k_4x")
    target_mp = target_megapixels or settings.get("upscale_target_megapixels", 2.0)

    if _is_comfyui_upscaler(upscaler):
        from app.services import comfyui_service

        workflow_file = _get_workflow_file(upscaler)
        return comfyui_service.upscale_with_comfyui(
            session, index, target_mp, workflow_file=workflow_file, upscaler_name=upscaler, cancel_check=cancel_check
        )

    from lib.upscaling import upscale_image as do_upscale

    img = Image.open(item.media_path)
    upscaled = do_upscale(
        img, upscaler, state_dict=session.config, target_megapixels=target_mp
    )

    session.upscaled_image = upscaled
    session.upscaled_index = index

    return "ok"


def save_upscaled(session: Session, index: int):
    if session.upscaled_image is None or session.upscaled_index != index:
        raise ValueError("No upscaled image cached for this index")

    create_version_backup(session, index, "upscale")

    ds = session.dataset
    item = ds.get_item(index)
    session.upscaled_image.save(item.media_path)
    session.upscaled_image = None
    session.upscaled_index = None


def remove_background(session: Session, index: int, model: str | None = None) -> str:
    from lib.masking import remove_background as do_remove_bg
    from lib.media_cache import generate_thumbnail
    from app.db.repository import ImageRepository

    ds = session.dataset
    item = ds.get_item(index)
    settings = read_settings()
    model_name = (
        model if model else settings.get("rembg", {}).get("model", "u2net_human_seg")
    )

    create_version_backup(session, index, "rembg")

    img = Image.open(item.media_path)
    result = do_remove_bg(img, model_name, session.config)

    png_path = os.path.splitext(item.media_path)[0] + ".png"
    result.save(png_path)

    if png_path != item.media_path:
        old_rel = os.path.relpath(item.media_path, ds.base_dir)
        os.remove(item.media_path)
        item.update_paths_after_rename(png_path, ds.masks_dir)
        new_rel = os.path.relpath(item.media_path, ds.base_dir)
        if session.db:
            ImageRepository.rename(session.db, old_rel, new_rel)

    item.thumbnail_path = generate_thumbnail(png_path)
    return png_path


def generate_mask(session: Session, index: int) -> str:
    from lib.masking import ask_mask_from_model

    ds = session.dataset
    item = ds.get_item(index)
    settings = read_settings()
    model_name = settings.get("rembg", {}).get("model", "u2net_human_seg")

    img = Image.open(item.media_path)
    mask = ask_mask_from_model(img, model_name)

    mask_path = item.mask_path
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    mask.save(mask_path)
    return mask_path


def crop_image(session: Session, index: int, x: int, y: int, width: int, height: int):
    from lib.media_cache import generate_thumbnail
    from app.db.repository import ImageRepository

    create_version_backup(session, index, "crop")

    ds = session.dataset
    item = ds.get_item(index)

    with Image.open(item.media_path) as img:
        cropped = img.crop((x, y, x + width, y + height))
        cropped.save(item.media_path)

    item.thumbnail_path = generate_thumbnail(item.media_path)

    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    file_size = os.path.getsize(item.media_path)
    if session.db:
        ImageRepository.update_metadata(
            session.db,
            ImageRepository.get_by_filename(session.db, rel_path)["id"],
            width=width,
            height=height,
            file_size=file_size,
        )


def white_balance_image(session: Session, index: int, method: str = "gray_world"):
    from lib.media_cache import generate_thumbnail
    from lib.white_balance import white_balance_image as do_white_balance
    from app.db.repository import ImageRepository

    create_version_backup(session, index, "white_balance")

    ds = session.dataset
    item = ds.get_item(index)

    with Image.open(item.media_path) as img:
        balanced = do_white_balance(img, method)
        balanced.save(item.media_path)

    item.thumbnail_path = generate_thumbnail(item.media_path)

    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    file_size = os.path.getsize(item.media_path)
    if session.db:
        ImageRepository.update_metadata(
            session.db,
            ImageRepository.get_by_filename(session.db, rel_path)["id"],
            file_size=file_size,
        )


def paint_image(session: Session, index: int, paint_png_bytes: bytes):
    import io
    from lib.media_cache import generate_thumbnail
    from app.db.repository import ImageRepository

    create_version_backup(session, index, "paint")

    ds = session.dataset
    item = ds.get_item(index)

    with Image.open(item.media_path) as img:
        original_mode = img.mode
        base = img.convert("RGBA")

    paint_layer = Image.open(io.BytesIO(paint_png_bytes)).convert("RGBA")
    composited = Image.alpha_composite(base, paint_layer)
    composited.convert(original_mode).save(item.media_path)

    item.thumbnail_path = generate_thumbnail(item.media_path)

    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    file_size = os.path.getsize(item.media_path)
    if session.db:
        ImageRepository.update_metadata(
            session.db,
            ImageRepository.get_by_filename(session.db, rel_path)["id"],
            file_size=file_size,
        )


def save_mask(session: Session, index: int, mask_png_bytes: bytes):
    import io
    create_version_backup(session, index, "mask")
    ds = session.dataset
    item = ds.get_item(index)
    mask_layer = Image.open(io.BytesIO(mask_png_bytes)).convert("RGBA")
    if item.mask_path is None:
        raise ValueError("Dataset has no mask directory configured")
    os.makedirs(os.path.dirname(item.mask_path), exist_ok=True)
    mask_layer.save(item.mask_path)
