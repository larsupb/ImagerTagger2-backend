import io
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dataset(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "dataset")
        response = client.post("/api/projects/create", json={"path": path})
        session_id = response.json()["session_id"]
        files = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buf, format="PNG")
            files.append(("files", (f"img{i}.png", buf.getvalue(), "image/png")))
        client.post("/api/dataset/upload", files=files, headers={"X-Session-ID": session_id})
        # upload content-hash-renames files; rename back to stable names
        for i in range(3):
            client.put(
                f"/api/dataset/item/{i}/rename",
                params={"new_name": f"img{i}"},
                headers={"X-Session-ID": session_id},
            )
        client.delete(f"/api/projects/{session_id}")
        response = client.post("/api/projects/open", json={"path": path})
        session_id = response.json()["session_id"]
        yield {"session_id": session_id, "path": path, "headers": {"X-Session-ID": session_id}}


def _wait_task(client, task_id, headers):
    for _ in range(200):
        status = client.get(f"/api/tasks/{task_id}", headers=headers).json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def _set_caption(client, headers, filename, content, caption_type="tags"):
    gallery = client.get("/api/dataset/gallery?page=0&page_size=100", headers=headers).json()
    index = next(i["index"] for i in gallery["items"] if i["filename"] == filename)
    r = client.put(
        "/api/captions/save",
        json={"index": index, "caption": content, "caption_type": caption_type},
        headers=headers,
    )
    assert r.status_code == 200, r.text


def _fake_embed():
    """Fake _create_embeddings: deterministic 2-d vectors from a lookup."""
    table = {
        "a person by the sea": [1.0, 0.0],
        "a car on the road": [0.0, 1.0],
        "sea": [0.9, 0.1],
        "car": [0.1, 0.9],
    }

    def fake(texts, settings):
        out = []
        for t in texts:
            if t not in table:
                return []
            out.append(table[t])
        return out

    return fake


def test_index_job_and_search_ranking(client, dataset, monkeypatch):
    headers = dataset["headers"]
    _set_caption(client, headers, "img0.png", "a person by the sea")
    _set_caption(client, headers, "img1.png", "a car on the road")
    monkeypatch.setattr("app.services.embedding_service._create_embeddings", _fake_embed())

    empty = client.get("/api/search?q=sea", headers=headers)
    assert empty.status_code == 400        # index empty -> 400 pointing at index job

    stats = client.get("/api/search", headers=headers)   # no q: stats only
    assert stats.json()["indexed"] == 0 and stats.json()["pending"] == 2

    task_id = client.post("/api/search/index", headers=headers).json()["task_id"]
    status = _wait_task(client, task_id, headers)
    assert status["status"] == "completed"

    body = client.get("/api/search?q=sea", headers=headers).json()
    assert [r["filename"] for r in body["results"]] == ["img0.png", "img1.png"]
    assert body["results"][0]["score"] > body["results"][1]["score"]
    assert body["indexed"] == 2 and body["pending"] == 0

    limited = client.get("/api/search?q=sea&min_score=0.5", headers=headers).json()
    assert [r["filename"] for r in limited["results"]] == ["img0.png"]


def test_reindex_after_caption_edit(client, dataset, monkeypatch):
    headers = dataset["headers"]
    _set_caption(client, headers, "img0.png", "a person by the sea")
    monkeypatch.setattr("app.services.embedding_service._create_embeddings", _fake_embed())
    _wait_task(client, client.post("/api/search/index", headers=headers).json()["task_id"], headers)
    assert client.get("/api/search", headers=headers).json()["pending"] == 0

    _set_caption(client, headers, "img0.png", "a car on the road")
    assert client.get("/api/search", headers=headers).json()["pending"] == 1
    _wait_task(client, client.post("/api/search/index", headers=headers).json()["task_id"], headers)

    body = client.get("/api/search?q=car", headers=headers).json()
    assert body["results"][0]["filename"] == "img0.png"
    assert body["pending"] == 0
