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
