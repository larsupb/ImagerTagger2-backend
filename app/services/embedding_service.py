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
