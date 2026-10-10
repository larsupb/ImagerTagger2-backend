import os

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.attributes import ATTRIBUTES
from app.db.repository import AttributeRepository, ImageRepository
from app.sessions import Session, get_session
from app.services import task_service
from app.services.attribute_service import (
    UnknownAttributeError,
    compute_footprint,
    index_for_filename,
    query_images,
    run_classification,
    set_attributes,
)

router = APIRouter(prefix="/api/attributes", tags=["attributes"])


class AttributeUpdate(BaseModel):
    filename: str
    attributes: dict


def _require_db(session: Session):
    if session.db is None:
        raise HTTPException(400, "No project database open")


@router.get("/schema")
def schema(session: Session = Depends(get_session)):
    return ATTRIBUTES


@router.get("/footprint")
def footprint(session: Session = Depends(get_session)):
    _require_db(session)
    return compute_footprint(session.db)


@router.get("")
def get_attributes(filename: str, session: Session = Depends(get_session)):
    _require_db(session)
    row = ImageRepository.get_by_filename(session.db, filename)
    return {
        "filename": filename,
        "attributes": AttributeRepository.get_for_image(session.db, row["id"]) if row else {},
    }


@router.get("/query")
def query(
    request: Request,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    session: Session = Depends(get_session),
):
    _require_db(session)
    filters = {
        key: values.split("|")
        for key, values in request.query_params.items()
        if key not in ("limit", "offset")
    }
    try:
        result = query_images(session.db, filters, limit=limit, offset=offset)
    except UnknownAttributeError as exc:
        raise HTTPException(422, f"Unknown attribute: {exc}")
    for item in result["items"]:
        item["index"] = index_for_filename(session, item["filename"])
    return result


@router.put("")
def put(update: AttributeUpdate, session: Session = Depends(get_session)):
    _require_db(session)
    # ensure_image, not get_by_filename: uploads after project open have no DB row until next sync
    image_id = ImageRepository.ensure_image(session.db, update.filename)
    valid, errors = set_attributes(session.db, image_id, update.attributes)
    if errors:
        rendered = ", ".join(f"{k}={v}" for k, v in errors.items())
        raise HTTPException(422, f"Invalid attribute values: {rendered}")
    return {"id": image_id, "filename": update.filename, "attributes": valid}


@router.post("/classify")
async def classify(
    tagger: str = Query(default="vlm-tagger"),
    only_missing: bool = False,
    session: Session = Depends(get_session),
):
    _require_db(session)
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")
    task_id = task_service.create_task(
        len(session.dataset),
        name="Attribute classification",
        operation="attribute_classification",
        project_name=os.path.basename(session.dataset.base_dir)
        if session.dataset.base_dir
        else None,
    )

    async def run():
        await run_classification(session, task_id, tagger, only_missing)

    task_service.submit(task_id, run)
    return {"task_id": task_id}
