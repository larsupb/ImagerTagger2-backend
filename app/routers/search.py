import os

from fastapi import APIRouter, Depends, HTTPException

from app.config import embedding_endpoint_settings
from app.sessions import Session, get_session
from app.services import task_service
from app.services.embedding_service import (
    compute_index_plan,
    index_dataset,
    indexed_count,
    search,
)

router = APIRouter(prefix="/api/search", tags=["search"])


def _require_db(session: Session):
    if session.db is None:
        raise HTTPException(400, "No project database open")


@router.get("")
def search_captions(
    q: str = "",
    limit: int = 200,
    min_score: float = 0.0,
    session: Session = Depends(get_session),
):
    _require_db(session)
    model = embedding_endpoint_settings()["model"]
    stats = {
        "indexed": indexed_count(session.db, model),
        "pending": len(compute_index_plan(session.db, model)[0]),
        "model": model,
    }
    if not q.strip():
        return {"results": [], **stats}
    try:
        results = search(session, q.strip(), min(limit, 500), min_score)
    except Exception as e:
        raise HTTPException(502, f"embedding endpoint error: {e}")
    if results is None:
        raise HTTPException(400, "search index is empty, index captions first")
    return {"results": results, **stats}


@router.post("/index")
async def index(session: Session = Depends(get_session)):
    _require_db(session)
    model = embedding_endpoint_settings()["model"]
    stale, _fresh = compute_index_plan(session.db, model)
    task_id = task_service.create_task(
        max(len(stale), 1),
        name="Caption embedding index",
        operation="search_index",
        project_name=os.path.basename(session.dataset.base_dir)
        if session.dataset
        else None,
    )

    async def run():
        await index_dataset(session, task_id)

    task_service.submit(task_id, run)
    return {"task_id": task_id}
