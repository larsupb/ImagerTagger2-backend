import io
import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def dataset(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "dataset")
        response = client.post("/api/projects/create", json={"path": path})
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        files = [
            ("files", (f"img{i}.png", _png_bytes(color), "image/png"))
            for i, color in enumerate(colors)
        ]
        response = client.post(
            "/api/dataset/upload", files=files, headers={"X-Session-ID": session_id}
        )
        assert response.status_code == 200
        assert response.json()["total_items"] == 4

        yield {"session_id": session_id, "path": path}


def _filenames(client, session_id: str) -> list[str]:
    response = client.get(
        "/api/dataset/gallery?page=0&page_size=100",
        headers={"X-Session-ID": session_id},
    )
    assert response.status_code == 200
    return [item["filename"] for item in response.json()["items"]]


def test_bulk_delete_removes_selected_items(client, dataset):
    session_id = dataset["session_id"]
    before = _filenames(client, session_id)

    response = client.request(
        "DELETE",
        "/api/dataset/items/bulk",
        json={"indices": [0, 2]},
        headers={"X-Session-ID": session_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_items"] == 2
    assert body["deleted"] == 2
    assert _filenames(client, session_id) == [before[1], before[3]]


def test_bulk_delete_removes_files_from_disk(client, dataset):
    session_id = dataset["session_id"]
    before = _filenames(client, session_id)

    response = client.request(
        "DELETE",
        "/api/dataset/items/bulk",
        json={"indices": [1, 2, 3]},
        headers={"X-Session-ID": session_id},
    )

    assert response.status_code == 200
    remaining = os.listdir(dataset["path"])
    assert before[0] in remaining
    for name in before[1:]:
        assert name not in remaining


def test_bulk_delete_ignores_duplicate_indices(client, dataset):
    session_id = dataset["session_id"]

    response = client.request(
        "DELETE",
        "/api/dataset/items/bulk",
        json={"indices": [1, 1, 1]},
        headers={"X-Session-ID": session_id},
    )

    assert response.status_code == 200
    assert response.json()["total_items"] == 3


def test_bulk_delete_out_of_range_index_returns_400(client, dataset):
    session_id = dataset["session_id"]

    response = client.request(
        "DELETE",
        "/api/dataset/items/bulk",
        json={"indices": [0, 99]},
        headers={"X-Session-ID": session_id},
    )

    assert response.status_code == 400
    assert _filenames(client, session_id) != []
    assert len(_filenames(client, session_id)) == 4
