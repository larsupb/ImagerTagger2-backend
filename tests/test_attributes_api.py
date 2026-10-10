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


def test_schema_lists_vocabularies(client, dataset):
    response = client.get("/api/attributes/schema", headers=dataset["headers"])
    assert response.status_code == 200
    schema = response.json()
    assert schema["shot_type"]["applies_to"] == "all"
    assert "selfie" in schema["shot_type"]["values"]


def test_put_and_footprint(client, dataset):
    headers = dataset["headers"]
    response = client.put(
        "/api/attributes",
        json={"filename": "img0.png", "attributes": {"shot_type": "Selfie", "pose": "walking"}},
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["attributes"] == {"shot_type": "selfie", "pose": "walking"}

    response = client.put(
        "/api/attributes",
        json={"filename": "img1.png", "attributes": {"framing": "blurry"}},
        headers=headers,
    )
    assert response.status_code == 422
    assert "blurry" in response.json()["detail"]

    response = client.put(
        "/api/attributes",
        json={"filename": "img1.png", "attributes": {"nope": "selfie"}},
        headers=headers,
    )
    assert response.status_code == 422

    footprint = client.get("/api/attributes/footprint", headers=headers).json()
    assert footprint["_total"] == 3
    assert footprint["shot_type"]["selfie"] == 1
    assert footprint["shot_type"]["_unset"] == 2

    response = client.get("/api/attributes?filename=img0.png", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"filename": "img0.png", "attributes": {"shot_type": "selfie", "pose": "walking"}}


def test_query_and_media_by_id(client, dataset):
    headers = dataset["headers"]
    client.put(
        "/api/attributes",
        json={"filename": "img0.png", "attributes": {"shot_type": "selfie", "pose": "walking"}},
        headers=headers,
    )
    put = client.put(
        "/api/attributes",
        json={"filename": "img1.png", "attributes": {"shot_type": "selfie"}},
        headers=headers,
    )
    img1_id = put.json()["id"]

    response = client.get(
        "/api/attributes/query?shot_type=selfie&pose=walking", headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["filename"] == "img0.png"
    assert item["index"] == 0
    assert item["attributes"] == {"shot_type": "selfie", "pose": "walking"}

    response = client.get("/api/attributes/query?shot_typ=selfie", headers=headers)
    assert response.status_code == 422

    response = client.get("/api/attributes/query?shot_type=selfie|headshot", headers=headers)
    assert response.json()["total"] == 2

    response = client.get(f"/api/media/by-id/{img1_id}", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    response = client.get(f"/api/media/thumbnail-by-id/{img1_id}", headers=headers)
    assert response.status_code == 200

    assert client.get("/api/media/by-id/99999", headers=headers).status_code == 404


import json as _json


def _start_classify(client, dataset, monkeypatch, responses, only_missing=False):
    calls = []

    def fake_generate_caption(index, tagger, state_dict, prompt=None):
        calls.append({"index": index, "tagger": tagger, "prompt": prompt})
        return responses[index]

    monkeypatch.setattr("lib.captioning.generate_caption", fake_generate_caption)
    response = client.post(
        f"/api/attributes/classify?tagger=vlm-tagger&only_missing={str(only_missing).lower()}",
        headers=dataset["headers"],
    )
    assert response.status_code == 200
    return response.json()["task_id"], calls


def _wait_task(client, task_id, headers):
    for _ in range(200):
        status = client.get(f"/api/tasks/{task_id}", headers=headers).json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_classify_writes_valid_values(client, dataset, monkeypatch):
    good = _json.dumps({"has_person": True, "shot_type": "Selfie", "pose": "walking"})
    junk = "I cannot answer"
    task_id, calls = _start_classify(client, dataset, monkeypatch, {0: good, 1: junk, 2: ""})
    status = _wait_task(client, task_id, dataset["headers"])
    assert status["status"] == "completed"
    assert all(c["tagger"] == "vlm-tagger" for c in calls)
    assert "has_person" in calls[0]["prompt"]

    query = client.get("/api/attributes/query?shot_type=selfie", headers=dataset["headers"]).json()
    assert [i["filename"] for i in query["items"]] == ["img0.png"]
    assert query["items"][0]["attributes"]["pose"] == "walking"


def test_classify_only_missing_skips_classified_images(client, dataset, monkeypatch):
    client.put(
        "/api/attributes",
        json={"filename": "img0.png", "attributes": {"shot_type": "headshot"}},
        headers=dataset["headers"],
    )
    good = _json.dumps({"has_person": False, "shot_type": "aerial"})
    task_id, calls = _start_classify(
        client, dataset, monkeypatch, {0: good, 1: good, 2: good}, only_missing=True
    )
    _wait_task(client, task_id, dataset["headers"])
    assert 0 not in [c["index"] for c in calls]
    query = client.get("/api/attributes/query?shot_type=headshot", headers=dataset["headers"]).json()
    assert [i["filename"] for i in query["items"]] == ["img0.png"]
    query = client.get("/api/attributes/query?shot_type=aerial", headers=dataset["headers"]).json()
    assert sorted(i["filename"] for i in query["items"]) == ["img1.png", "img2.png"]


def test_classify_failures_reported_in_task(client, dataset, monkeypatch):
    good = _json.dumps({"has_person": False, "medium": "photo"})

    def fake_generate_caption(index, tagger, state_dict, prompt=None):
        if index == 1:
            raise RuntimeError("VLM down")
        return good

    monkeypatch.setattr("lib.captioning.generate_caption", fake_generate_caption)
    response = client.post("/api/attributes/classify?tagger=vlm-tagger", headers=dataset["headers"])
    task_id = response.json()["task_id"]
    status = _wait_task(client, task_id, dataset["headers"])
    assert status["status"] == "failed"
    assert "1" in (status.get("error") or "")
    query = client.get("/api/attributes/query?medium=photo", headers=dataset["headers"]).json()
    assert sorted(i["filename"] for i in query["items"]) == ["img0.png", "img2.png"]
