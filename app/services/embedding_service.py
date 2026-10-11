"""Caption embedding + semantic search over the caption_vectors table."""

import asyncio
import hashlib
import json
import os
import urllib.request

import numpy as np

from app.config import embedding_endpoint_settings
from app.db.repository import ImageRepository, VectorRepository
from app.services import task_service

BATCH_SIZE = 32


def _create_embeddings(texts: list[str], settings: dict) -> list[list[float]]:
    """Raw call to the OpenAI-compatible embeddings endpoint (test seam).

    Plain HTTP instead of the openai client: the client always sends
    encoding_format, which LiteLLM/ollama frontends reject outright."""
    req = urllib.request.Request(
        settings["base_url"].rstrip("/") + "/embeddings",
        data=json.dumps({"model": settings["model"], "input": texts}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + (settings.get("api_key") or "none"),
        },
    )
    with urllib.request.urlopen(req, timeout=settings.get("timeout", 120)) as resp:
        data = json.load(resp)
    return [d["embedding"] for d in data.get("data", [])]


def embed_texts(texts: list[str], settings: dict) -> np.ndarray:
    """Embed texts in batches, one retry per batch, L2-normalized float32 rows."""
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        chunk = texts[start:start + BATCH_SIZE]
        vecs = _create_embeddings(chunk, settings)
        if len(vecs) != len(chunk):
            vecs = _create_embeddings(chunk, settings)
        if len(vecs) != len(chunk):
            raise RuntimeError(
                f"embedding endpoint returned {len(vecs)}/{len(chunk)} vectors"
            )
        out.extend(vecs)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def caption_text(db, image_id: int) -> str:
    """All non-empty captions of an image, tags and descriptions alike, in
    stable order so the hash does not depend on insertion order."""
    rows = db.execute(
        "SELECT content FROM captions WHERE image_id = ? AND content != ''"
        " ORDER BY caption_type",
        (image_id,),
    ).fetchall()
    return "\n\n".join(r["content"] for r in rows)


def compute_index_plan(db, model: str) -> tuple[list[tuple[int, str]], int]:
    """Images needing (re-)embedding + count already fresh for `model`.

    Staleness is derived, never stored: hash or model mismatch."""
    stored = VectorRepository.hashes(db)
    todo: list[tuple[int, str]] = []
    fresh = 0
    rows = db.execute(
        "SELECT DISTINCT image_id FROM captions WHERE content != ''"
    ).fetchall()
    for r in rows:
        text = caption_text(db, r["image_id"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if stored.get(r["image_id"]) == (model, digest):
            fresh += 1
        else:
            todo.append((r["image_id"], text))
    return todo, fresh


def indexed_count(db, model: str) -> int:
    return sum(1 for m, _ in VectorRepository.hashes(db).values() if m == model)


# keyed by sqlite file path: {"rev", "model", "ids", "mat"}
# ponytail: brute-force cosine, fine to ~50k images/dataset; upgrade = sqlite-vec/ANN
_matrix_cache: dict[str, dict] = {}


async def index_dataset(session, task_id: str) -> None:
    db = session.db
    settings = embedding_endpoint_settings()
    model = settings["model"]
    task_service.start_task(task_id)
    todo, _fresh = compute_index_plan(db, model)
    total = len(todo)
    failures = 0
    done = 0
    for start in range(0, total, BATCH_SIZE):
        if task_service.is_cancelled(task_id):
            break
        await task_service.wait_if_paused(task_id)
        chunk = todo[start:start + BATCH_SIZE]
        try:
            vecs = await asyncio.to_thread(embed_texts, [t for _, t in chunk], settings)
        except Exception:
            failures += len(chunk)
            done += len(chunk)
            continue
        for j, ((image_id, text), vec) in enumerate(zip(chunk, vecs)):
            row = ImageRepository.get_by_id(db, image_id)
            VectorRepository.upsert(
                db, image_id, model, len(vec),
                np.asarray(vec, dtype=np.float32).tobytes(),
                hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
            if row:
                task_service.update_progress(
                    task_id, start + j, total, start + j, row["filename"], "ok")
        db.commit()
        session.touch()
        done += len(chunk)
    keep = {
        r["image_id"]
        for r in db.execute("SELECT DISTINCT image_id FROM captions WHERE content != ''")
    }
    VectorRepository.delete_missing(db, keep)
    db.commit()
    task_service.complete_task(
        task_id, error=f"{failures} images failed to embed" if failures else None
    )


async def embed_one(session, image_id: int) -> None:
    """Embed (or refresh) a single image's combined caption — batch-hook entry."""
    settings = embedding_endpoint_settings()
    text = caption_text(session.db, image_id)
    if not text:
        return
    vec = await asyncio.to_thread(embed_texts, [text], settings)
    VectorRepository.upsert(
        session.db, image_id, settings["model"], len(vec[0]),
        np.asarray(vec[0], dtype=np.float32).tobytes(),
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    session.db.commit()


def _matrix(db, model: str) -> tuple[list[int], np.ndarray]:
    path = db.execute("PRAGMA database_list").fetchone()[2]
    rev = VectorRepository.revision(db)
    entry = _matrix_cache.get(path)
    if entry and entry["rev"] == rev and entry["model"] == model:
        return entry["ids"], entry["mat"]
    ids: list[int] = []
    vecs: list[np.ndarray] = []
    for r in VectorRepository.load_all(db):
        if r["model"] != model:
            continue
        ids.append(r["image_id"])
        vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))
    mat = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)
    _matrix_cache[path] = {"rev": rev, "model": model, "ids": ids, "mat": mat}
    return ids, mat


def search(session, query: str, limit: int = 200, min_score: float = 0.0):
    """Ranked hits or None when the index is empty."""
    from app.services.attribute_service import index_for_filename

    db = session.db
    settings = embedding_endpoint_settings()
    ids, mat = _matrix(db, settings["model"])
    if mat.shape[0] == 0:
        return None
    q = embed_texts([query], settings)
    sims = mat @ q[0]
    results = []
    for i in np.argsort(-sims):
        if len(results) >= limit or sims[i] < min_score:
            break
        row = ImageRepository.get_by_id(db, ids[int(i)])
        if row is None:
            continue
        idx = index_for_filename(session, row["filename"])
        if idx is None:
            continue
        results.append({"index": idx, "filename": row["filename"], "score": float(sims[i])})
    return results
