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
        for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buf, format="PNG")
            files.append(("files", (f"img{i}.png", buf.getvalue(), "image/png")))
        client.post(
            "/api/dataset/upload", files=files, headers={"X-Session-ID": session_id}
        )
        # upload content-hash-renames files; rename back to stable names
        for i in range(2):
            client.put(
                f"/api/dataset/item/{i}/rename",
                params={"new_name": f"img{i}"},
                headers={"X-Session-ID": session_id},
            )
        client.delete(f"/api/projects/{session_id}")
        response = client.post("/api/projects/open", json={"path": path})
        session_id = response.json()["session_id"]
        yield {
            "session_id": session_id,
            "path": path,
            "headers": {"X-Session-ID": session_id},
        }


def _wait_task(client, task_id, headers):
    for _ in range(200):
        status = client.get(f"/api/tasks/{task_id}", headers=headers).json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("batch task did not finish")


def test_batch_embed_captions(client, dataset, monkeypatch):
    headers = dataset["headers"]
    embedded = []

    def fake(texts, settings):
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def fake_caption(index, tagger, state_dict, prompt=None):
        return f"a description {index}"

    monkeypatch.setattr("app.services.embedding_service._create_embeddings", fake)
    monkeypatch.setattr("lib.captioning.generate_caption", fake_caption)
    r = client.post(
        "/api/batch/process",
        json={"caption": True, "classify_attributes": False, "embed_captions": True},
        headers=headers,
    )
    assert r.status_code == 200
    status = _wait_task(client, r.json()["task_id"], headers)
    assert status["status"] == "completed"
    assert embedded == ["a description 0", "a description 1"]
    stats = client.get("/api/search", headers=headers).json()
    assert stats["indexed"] == 2 and stats["pending"] == 0


def test_batch_embed_failure_does_not_abort(client, dataset, monkeypatch):
    headers = dataset["headers"]

    def fake_caption(index, tagger, state_dict, prompt=None):
        return f"a description {index}"

    def boom(texts, settings):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr("app.services.embedding_service._create_embeddings", boom)
    monkeypatch.setattr("lib.captioning.generate_caption", fake_caption)
    r = client.post(
        "/api/batch/process",
        json={"caption": True, "classify_attributes": False, "embed_captions": True},
        headers=headers,
    )
    assert r.status_code == 200
    status = _wait_task(client, r.json()["task_id"], headers)
    assert status["status"] == "completed"
    item = client.get("/api/dataset/item/1", headers=headers).json()
    tags = next(c for c in item["captions"] if c["caption_type"] == "tags")
    assert tags["content"] == "a description 1"
    stats = client.get("/api/search", headers=headers).json()
    assert stats["indexed"] == 0 and stats["pending"] == 2
