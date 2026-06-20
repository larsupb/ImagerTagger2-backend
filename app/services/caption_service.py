import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

from app.sessions import Session
from app.db.repository import ImageRepository, CaptionRepository

logger = logging.getLogger(__name__)


def _get_image_id(session: Session, index: int) -> Optional[int]:
    ds = session.dataset
    item = ds.get_item(index)
    if not item:
        return None
    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    row = ImageRepository.get_by_filename(session.db, rel_path)
    return row["id"] if row else None


def _rel_path(session: Session, index: int) -> str:
    ds = session.dataset
    item = ds.get_item(index)
    return os.path.relpath(item.media_path, ds.base_dir)


def save_caption(session: Session, index: int, caption: str, caption_type: str = "tags") -> bool:
    image_id = _get_image_id(session, index)
    if image_id is None:
        return False
    is_new_type = len(CaptionRepository.get_all_by_type(session.db, caption_type)) == 0
    CaptionRepository.upsert(session.db, image_id, caption_type, caption)
    if is_new_type:
        _propagate_new_caption_type(session, caption_type, exclude_image_id=image_id)
    return True


def _propagate_new_caption_type(session: Session, caption_type: str, exclude_image_id: int) -> None:
    ds = session.dataset
    for i in range(len(ds)):
        item = ds.get_item(i)
        if not item:
            continue
        rel_path = os.path.relpath(item.media_path, ds.base_dir)
        img_id = ImageRepository.ensure_image(session.db, rel_path)
        if img_id != exclude_image_id:
            CaptionRepository.upsert(session.db, img_id, caption_type, "")


def read_caption(session: Session, index: int, caption_type: str = None) -> str:
    image_id = _get_image_id(session, index)
    if image_id is None:
        return ""
    if caption_type:
        content = CaptionRepository.get(session.db, image_id, caption_type)
        return content if content else ""
    active = CaptionRepository.get_active(session.db, image_id)
    return active[1] if active else ""


def get_caption_types(session: Session) -> list[str]:
    return CaptionRepository.get_all_types(session.db)


def delete_caption_type(session: Session, caption_type: str) -> int:
    return CaptionRepository.delete_by_type(session.db, caption_type)


def rename_caption_type(session: Session, old_type: str, new_type: str) -> int:
    return CaptionRepository.rename_type(session.db, old_type, new_type)


def get_all_captions(session: Session, index: int) -> list[dict]:
    image_id = _get_image_id(session, index)
    if image_id is None:
        return []
    return CaptionRepository.get_all_for_image(session.db, image_id)


def set_active_caption_type(session: Session, index: int, caption_type: str) -> bool:
    image_id = _get_image_id(session, index)
    if image_id is None:
        return False
    CaptionRepository.set_active(session.db, image_id, caption_type)
    return True


def get_tag_cloud(session: Session, sort_by: str = "frequency", caption_type: str = "tags") -> list[dict]:
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    tag_counts: dict[str, int] = {}
    for row in rows:
        content = row["content"]
        if not content:
            continue
        tags = [t.strip() for t in content.split(",") if t.strip()]
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    items = [{"tag": tag, "count": count} for tag, count in tag_counts.items()]
    if sort_by == "frequency":
        items.sort(key=lambda x: x["count"], reverse=True)
    else:
        items.sort(key=lambda x: x["tag"])
    return items


def remove_tags(session: Session, tags_to_remove: list[str], caption_type: str = "tags") -> int:
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    tags_set = set(t.lower() for t in tags_to_remove)
    modified = 0
    for row in rows:
        content = row["content"]
        if not content:
            continue
        tags = [t.strip() for t in content.split(",") if t.strip()]
        new_tags = [t for t in tags if t.lower() not in tags_set]
        if len(new_tags) != len(tags):
            img = ImageRepository.get_by_filename(session.db, row["filename"])
            if img:
                CaptionRepository.upsert(session.db, img["id"], caption_type, ", ".join(new_tags))
                modified += 1
    return modified


def append_tag(session: Session, tag: str, caption_type: str = "tags") -> int:
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    modified = 0
    for row in rows:
        content = row["content"]
        existing_tags = [t.strip() for t in content.split(",") if t.strip()] if content else []
        if tag not in existing_tags:
            existing_tags.append(tag)
            img = ImageRepository.get_by_filename(session.db, row["filename"])
            if img:
                CaptionRepository.upsert(session.db, img["id"], caption_type, ", ".join(existing_tags))
                modified += 1
    return modified


def prepend_tag(session: Session, tag: str, caption_type: str = "tags") -> int:
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    modified = 0
    for row in rows:
        content = row["content"]
        existing_tags = [t.strip() for t in content.split(",") if t.strip()] if content else []
        if tag not in existing_tags:
            existing_tags.insert(0, tag)
            img = ImageRepository.get_by_filename(session.db, row["filename"])
            if img:
                CaptionRepository.upsert(session.db, img["id"], caption_type, ", ".join(existing_tags))
                modified += 1
    return modified


def search_replace_preview(session: Session, search: str, replace: str, caption_type: str = "tags") -> dict:
    ds = session.dataset
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    matches = []
    for row in rows:
        content = row["content"]
        if not content or search not in content:
            continue
        new_content = content.replace(search, replace)
        matches.append({
            "index": _index_for_filename(ds, row["filename"]),
            "filename": os.path.basename(row["filename"]),
            "before": content,
            "after": new_content,
        })
    return {"matches": matches, "total_matches": len(matches)}


def search_replace_apply(session: Session, search: str, replace: str, caption_type: str = "tags") -> int:
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    modified = 0
    for row in rows:
        content = row["content"]
        if not content or search not in content:
            continue
        new_content = content.replace(search, replace)
        img = ImageRepository.get_by_filename(session.db, row["filename"])
        if img:
            CaptionRepository.upsert(session.db, img["id"], caption_type, new_content)
            modified += 1
    return modified


def replace_underscores(session: Session, caption_type: str = "tags") -> int:
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    modified = 0
    for row in rows:
        content = row["content"]
        if not content or "_" not in content:
            continue
        img = ImageRepository.get_by_filename(session.db, row["filename"])
        if img:
            CaptionRepository.upsert(session.db, img["id"], caption_type, content.replace("_", " "))
            modified += 1
    return modified


def cleanup_tags(session: Session, caption_type: str = "tags") -> int:
    unwanted_patterns = [
        r"\d+girl",
        r"\d+boy",
        "solo",
        "looking at viewer",
    ]
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    modified = 0
    for row in rows:
        content = row["content"]
        if not content:
            continue
        tags = [t.strip() for t in content.split(",") if t.strip()]
        new_tags = []
        for tag in tags:
            skip = False
            for pattern in unwanted_patterns:
                if re.match(pattern, tag.lower()):
                    skip = True
                    break
            if not skip:
                new_tags.append(tag)
        if len(new_tags) != len(tags):
            img = ImageRepository.get_by_filename(session.db, row["filename"])
            if img:
                CaptionRepository.upsert(session.db, img["id"], caption_type, ", ".join(new_tags))
                modified += 1
    return modified


def export_to_jsonl(session: Session) -> dict:
    ds = session.dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(ds.base_dir, f"captions_export_{timestamp}.jsonl")
    count = 0
    with open(export_path, "w", encoding="utf-8") as f:
        for i in range(len(ds)):
            item = ds.get_item(i)
            caption = read_caption(session, i)
            record = {
                "filename": item.filename,
                "caption": caption,
                "has_mask": item.mask_exists(),
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    return {"path": export_path, "count": count}


def export_captions_to_txt(session: Session, caption_type: str = "tags") -> int:
    ds = session.dataset
    rows = CaptionRepository.get_all_by_type(session.db, caption_type)
    count = 0
    for row in rows:
        media_path = os.path.join(ds.base_dir, row["filename"])
        txt_path = os.path.splitext(media_path)[0] + ".txt"
        content = row["content"] or ""
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
    return count


def move_to_subdirectory(
    session: Session, tags: list[str], inverse: bool, subdirectory_name: str, caption_type: str = "tags"
) -> int:
    ds = session.dataset
    tags_set = set(t.lower() for t in tags)
    target_dir = os.path.join(ds.base_dir, subdirectory_name)
    os.makedirs(target_dir, exist_ok=True)

    moved = 0
    indices_to_move = []
    for i in range(len(ds)):
        caption = read_caption(session, i, caption_type)
        if not caption:
            continue
        image_tags = set(t.strip().lower() for t in caption.split(",") if t.strip())
        has_match = bool(image_tags & tags_set)
        if (has_match and not inverse) or (not has_match and inverse):
            indices_to_move.append(i)

    for i in reversed(indices_to_move):
        item = ds.get_item(i)
        old_rel = os.path.relpath(item.media_path, ds.base_dir)
        ds.copy_item(i, target_dir)
        ds.delete_item(i)
        ImageRepository.delete_by_filename(session.db, old_rel)
        moved += 1

    return moved


def _index_for_filename(ds, rel_filename: str) -> int:
    basename = os.path.basename(rel_filename)
    for i in range(len(ds)):
        item = ds.get_item(i)
        if item and item.filename == basename:
            return i
    return -1
