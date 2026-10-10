import io
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

SOURCE_SIZE = (16, 16)
UPSCALED_SIZE = (32, 32)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def dataset(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "dataset")
        response = client.post("/api/projects/create", json={"path": path})
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        files = [
            ("files", (f"img{i}.png", _png_bytes(SOURCE_SIZE, (10 * i, 20, 30)), "image/png"))
            for i in range(2)
        ]
        response = client.post(
            "/api/dataset/upload", files=files, headers={"X-Session-ID": session_id}
        )
        assert response.status_code == 200
        assert response.json()["total_items"] == 2

        yield {"session_id": session_id, "path": path}


def _image_paths(path: str) -> list[str]:
    return sorted(
        os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".png")
    )


def _run_batch(client, session_id: str, body: dict) -> dict:
    response = client.post(
        "/api/batch/process", json=body, headers={"X-Session-ID": session_id}
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        status = client.get(f"/api/tasks/{task_id}").json()
        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        time.sleep(0.05)
    raise AssertionError("batch task did not finish in time")


@pytest.fixture()
def fake_comfyui(monkeypatch):
    """Stand in for the ComfyUI round trip, keeping its real contract.

    The real upscale_with_comfyui does not touch disk: it caches the result on
    the session and returns "ok".
    """
    from app.services import comfyui_service

    calls = []

    def fake_upscale(session, index, target_megapixels=2.0, **kwargs):
        calls.append(index)
        item = session.dataset.get_item(index)
        with Image.open(item.media_path) as img:
            session.upscaled_image = img.resize(UPSCALED_SIZE)
        session.upscaled_index = index
        return "ok"

    monkeypatch.setattr(comfyui_service, "upscale_with_comfyui", fake_upscale)
    return calls


@pytest.fixture()
def fake_spandrel(monkeypatch):
    """Stand in for the local spandrel upscaler, which returns an image."""
    import lib.upscaling

    calls = []

    def fake_upscale(img, upscaler_name, state_dict, **kwargs):
        calls.append(upscaler_name)
        return img.resize(UPSCALED_SIZE)

    monkeypatch.setattr(lib.upscaling, "upscale_image", fake_upscale)
    return calls


def test_batch_upscale_comfyui_writes_result_to_disk(client, dataset, fake_comfyui):
    session_id = dataset["session_id"]

    status = _run_batch(
        client,
        session_id,
        {"upscale": True, "upscaler": "comfyui-flux2.klein.9b-edit"},
    )

    assert status["status"] == "completed"
    assert fake_comfyui == [0, 1]
    for image_path in _image_paths(dataset["path"]):
        with Image.open(image_path) as img:
            assert img.size == UPSCALED_SIZE, f"{image_path} was not overwritten"


def test_batch_upscale_local_writes_result_to_disk(client, dataset, fake_spandrel):
    session_id = dataset["session_id"]

    status = _run_batch(
        client, session_id, {"upscale": True, "upscaler": "NMKD_Siax_200k_4x"}
    )

    assert status["status"] == "completed"
    assert fake_spandrel == ["NMKD_Siax_200k_4x"] * 2
    for image_path in _image_paths(dataset["path"]):
        with Image.open(image_path) as img:
            assert img.size == UPSCALED_SIZE, f"{image_path} was not overwritten"


def test_batch_upscale_backup_version_differs_from_result(client, dataset, fake_comfyui):
    session_id = dataset["session_id"]

    _run_batch(
        client,
        session_id,
        {"upscale": True, "upscaler": "comfyui-flux2.klein.9b-edit"},
    )

    versions = client.get(
        "/api/processing/versions/0", headers={"X-Session-ID": session_id}
    ).json()
    assert len(versions) == 1, "expected exactly one upscale version"

    version_path = os.path.join(
        dataset["path"], ".imagetagger", versions[0]["version_path"]
    )
    with Image.open(version_path) as backup:
        assert backup.size == SOURCE_SIZE, "backup should hold the pre-upscale image"

    for image_path in _image_paths(dataset["path"]):
        with Image.open(image_path) as img:
            assert img.size == UPSCALED_SIZE
