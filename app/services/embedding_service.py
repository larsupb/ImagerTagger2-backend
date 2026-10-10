"""Caption embedding + semantic search over the caption_vectors table."""

import hashlib
import os

import numpy as np

from app.db.repository import ImageRepository, VectorRepository

BATCH_SIZE = 32


def _create_embeddings(texts: list[str], settings: dict) -> list[list[float]]:
    """Raw call to the OpenAI-compatible embeddings endpoint (test seam)."""
    from openai import OpenAI

    client = OpenAI(
        base_url=settings["base_url"],
        api_key=settings.get("api_key") or "none",
        timeout=settings.get("timeout", 120),
    )
    resp = client.embeddings.create(model=settings["model"], input=texts)
    return [d.embedding for d in resp.data]


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
