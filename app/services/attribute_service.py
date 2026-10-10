import os
import sqlite3

from app.attributes import ATTRIBUTES, validate_attributes
from app.db.repository import AttributeRepository, CaptionRepository, CategoryRepository, ImageRepository


class UnknownAttributeError(ValueError):
    pass


def set_attributes(db: sqlite3.Connection, image_id: int, attributes: dict) -> tuple[dict, dict]:
    valid, errors = validate_attributes(attributes)
    if valid:
        AttributeRepository.set_many(db, image_id, valid)
    for key, value in attributes.items():
        if value is None and key in ATTRIBUTES:
            AttributeRepository.unset(db, image_id, key)
    return valid, errors


def compute_footprint(db: sqlite3.Connection) -> dict:
    total = db.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
    footprint: dict = {name: {} for name in ATTRIBUTES}
    for key, value, count in AttributeRepository.get_all_rows(db):
        if key in footprint and value in ATTRIBUTES[key]["values"]:
            footprint[key][value] = count
    for key in footprint:
        footprint[key]["_unset"] = total - sum(footprint[key].values())
    footprint["_total"] = total
    return footprint


def query_images(
    db: sqlite3.Connection,
    filters: dict[str, list[str]],
    limit: int = 100,
    offset: int = 0,
) -> dict:
    for key in filters:
        if key not in ATTRIBUTES:
            raise UnknownAttributeError(key)

    where: list[str] = []
    params: list = []
    joins: list[str] = []
    for n, (key, values) in enumerate(filters.items()):
        alias = f"m{n}"
        if values == ["unset"]:
            joins.append(
                f"LEFT JOIN metadata {alias} ON {alias}.image_id = i.id AND {alias}.key = ?"
            )
            params.append(key)
            where.append(f"{alias}.key IS NULL")
        else:
            joins.append(
                f"JOIN metadata {alias} ON {alias}.image_id = i.id AND {alias}.key = ?"
                f" AND {alias}.value IN ({', '.join('?' * len(values))})"
            )
            params.append(key)
            params.extend(values)

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    joins_sql = " ".join(joins)
    total = db.execute(
        f"SELECT COUNT(*) AS n FROM images i {joins_sql}{where_sql}", params
    ).fetchone()["n"]
    rows = db.execute(
        f"SELECT i.* FROM images i {joins_sql}{where_sql}"
        " GROUP BY i.id ORDER BY i.id LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()

    attrs_map = AttributeRepository.get_for_image_ids(db, [r["id"] for r in rows])
    items = []
    for row in rows:
        row = dict(row)
        captions = {
            c["caption_type"]: c["content"]
            for c in CaptionRepository.get_all_for_image(db, row["id"])
        }
        items.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "width": row["width"],
                "height": row["height"],
                "file_size": row["file_size"],
                "is_bookmarked": bool(row["is_bookmarked"]),
                "attributes": attrs_map.get(row["id"], {}),
                "captions": captions,
                "category": CategoryRepository.get_for_image(db, row["id"]),
            }
        )
    return {"total": total, "items": items}


def index_for_filename(session, filename: str) -> int | None:
    ds = session.dataset
    if ds is None:
        return None
    try:
        return ds.find_index(os.path.join(ds.base_dir, filename))
    except ValueError:
        return None


async def classify_item(
    ds, db: sqlite3.Connection, index: int, tagger: str, prompt: str
) -> tuple[dict, bool]:
    import asyncio

    from app.attributes import parse_attributes_response

    item = ds.get_item(index)
    rel_path = os.path.relpath(item.media_path, ds.base_dir)
    row = ImageRepository.get_by_filename(db, rel_path)
    if row is None:
        return {}, False

    from lib.captioning import generate_caption

    parsed: dict = {}
    for _attempt in range(2):
        try:
            raw = await asyncio.to_thread(
                generate_caption, index, tagger, {"dataset": ds}, prompt=prompt
            )
        except Exception:
            continue
        parsed, _has_person = parse_attributes_response(raw or "")
        if parsed:
            break

    if not parsed:
        return {}, True

    AttributeRepository.set_many(db, row["id"], parsed)
    return parsed, False


async def run_classification(session, task_id: str, tagger: str, only_missing: bool) -> None:
    from app.attributes import build_classify_prompt
    from app.services import task_service

    ds = session.dataset
    db = session.db
    prompt = build_classify_prompt()
    failures = 0
    task_service.start_task(task_id)

    for index in range(len(ds)):
        if task_service.is_cancelled(task_id):
            break
        await task_service.wait_if_paused(task_id)

        item = ds.get_item(index)
        rel_path = os.path.relpath(item.media_path, ds.base_dir)
        row = ImageRepository.get_by_filename(db, rel_path)
        if row is None:
            continue
        existing = AttributeRepository.get_for_image(db, row["id"])
        if only_missing and existing:
            task_service.update_progress(
                task_id, index, len(ds), index, rel_path, "skipped (already set)"
            )
            continue

        parsed, failed = await classify_item(ds, db, index, tagger, prompt)
        if failed:
            failures += 1
            message = "error"
        else:
            message = "ok" if parsed else "no attributes parsed"
        task_service.update_progress(task_id, index, len(ds), index, rel_path, message)
        session.touch()

    task_service.complete_task(
        task_id, error=f"{failures} images failed to classify" if failures else None
    )
