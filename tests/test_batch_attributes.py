import io
import json
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


def _run_batch(client, dataset, body: dict) -> dict:
    response = client.post("/api/batch/process", json=body, headers=dataset["headers"])
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    for _ in range(200):
        status = client.get(f"/api/tasks/{task_id}", headers=dataset["headers"]).json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("batch task did not finish")


def _fake_captioner(monkeypatch):
    calls = []

    def fake_generate_caption(index, tagger, state_dict, prompt=None):
        calls.append({"index": index, "tagger": tagger, "prompt": prompt})
        if "has_person" in (prompt or ""):
            return json.dumps(
                {"has_person": False, "shot_type": "product", "medium": "photo"}
            )
        return "a caption"

    monkeypatch.setattr("lib.captioning.generate_caption", fake_generate_caption)
    return calls


def test_batch_caption_with_attribute_detection(client, dataset, monkeypatch):
    calls = _fake_captioner(monkeypatch)
    status = _run_batch(
        client, dataset, {"caption": True, "classify_attributes": True}
    )
    assert status["status"] == "completed"

    classify_calls = [c for c in calls if "has_person" in (c["prompt"] or "")]
    assert [c["index"] for c in classify_calls] == [0, 1]
    assert all(c["tagger"] == "vlm-tagger" for c in classify_calls)

    for i in (0, 1):
        response = client.get(
            f"/api/attributes?filename=img{i}.png", headers=dataset["headers"]
        )
        assert response.json()["attributes"]["shot_type"] == "product"
        assert response.json()["attributes"]["medium"] == "photo"

    query = client.get(
        "/api/attributes/query?shot_type=product", headers=dataset["headers"]
    ).json()
    assert sorted(i["filename"] for i in query["items"]) == ["img0.png", "img1.png"]
    assert all(i["captions"]["tags"] == "a caption" for i in query["items"])


def test_batch_caption_without_attribute_detection(client, dataset, monkeypatch):
    calls = _fake_captioner(monkeypatch)
    status = _run_batch(
        client, dataset, {"caption": True, "classify_attributes": False}
    )
    assert status["status"] == "completed"
    assert [c["index"] for c in calls] == [0, 1]
    assert all("has_person" not in (c["prompt"] or "") for c in calls)

    footprint = client.get("/api/attributes/footprint", headers=dataset["headers"]).json()
    assert footprint["shot_type"].get("product", 0) == 0
