import io
import threading
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services import task_service


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_job_store():
    task_service._task_store.clear()
    yield
    task_service._task_store.clear()


@pytest.fixture(autouse=True)
def clean_archives():
    from app.services import export_service

    export_service.discard_archives()
    yield
    export_service.discard_archives()


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def dataset(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "export-dataset")
        response = client.post("/api/projects/create", json={"path": path})
        session_id = response.json()["session_id"]

        files = [
            ("files", (f"img{i}.png", _png_bytes((10 * i, 20, 30)), "image/png"))
            for i in range(2)
        ]
        client.post(
            "/api/dataset/upload", files=files, headers={"X-Session-ID": session_id}
        )
        yield session_id


def _wait_for_finish(client, task_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        progress = client.get(f"/api/export/progress/{task_id}").json()
        if progress["status"] in ("completed", "failed", "cancelled"):
            return progress
        time.sleep(0.05)
    raise AssertionError(f"export task did not finish: {progress}")


def test_export_completes_without_error(client, dataset):
    response = client.post(
        "/api/export",
        json={"format": "standard", "caption_type": "tags"},
        headers={"X-Session-ID": dataset},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    progress = _wait_for_finish(client, task_id)

    assert progress["error"] is None
    assert progress["status"] == "completed"

    download = client.get(f"/api/export/download/{task_id}")
    assert download.status_code == 200


def test_export_runs_as_named_export_job(client, dataset):
    response = client.post(
        "/api/export",
        json={"format": "standard", "caption_type": "tags"},
        headers={"X-Session-ID": dataset},
    )
    task_id = response.json()["task_id"]

    _wait_for_finish(client, task_id)

    jobs = client.get("/api/jobs").json()
    job = next(j for j in jobs if j["task_id"] == task_id)

    assert job["operation"] == "export"
    assert job["name"] == "Export (standard)"
    assert job["project_name"] == "export-dataset"
    assert [entry["message"] for entry in job["logs"]] == [
        "Exported as 00000",
        "Exported as 00001",
        "Created ZIP archive",
    ]
    assert job["logs"][-1]["filename"] == "export.zip"


def test_export_stops_when_cancelled(client, dataset, monkeypatch):
    from app.services import caption_service

    gate = {"release": False, "started": 0}

    def blocking_read_caption(session, index, caption_type):
        gate["started"] += 1
        while not gate["release"]:
            time.sleep(0.01)
        return ""

    monkeypatch.setattr(caption_service, "read_caption", blocking_read_caption)

    response = client.post(
        "/api/export",
        json={"format": "standard", "caption_type": "tags"},
        headers={"X-Session-ID": dataset},
    )
    task_id = response.json()["task_id"]

    deadline = time.time() + 10
    while gate["started"] == 0 and time.time() < deadline:
        time.sleep(0.01)
    assert gate["started"] > 0, "export never reached the first item"

    threading.Timer(2.0, lambda: gate.__setitem__("release", True)).start()
    assert client.delete(f"/api/jobs/{task_id}").status_code == 200
    gate["release"] = True

    progress = _wait_for_finish(client, task_id)
    assert progress["status"] == "cancelled"
    assert client.get(f"/api/export/download/{task_id}").status_code == 400


def test_export_of_empty_dataset_completes(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "empty-dataset")
        session_id = client.post(
            "/api/projects/create", json={"path": path}
        ).json()["session_id"]

        response = client.post(
            "/api/export",
            json={"format": "standard", "caption_type": "tags"},
            headers={"X-Session-ID": session_id},
        )
        task_id = response.json()["task_id"]

        progress = _wait_for_finish(client, task_id)

        assert progress["error"] is None
        assert progress["status"] == "completed"
        assert progress["logs"][-1]["index"] == 0


def _run_export(client, session_id):
    response = client.post(
        "/api/export",
        json={"format": "standard", "caption_type": "tags"},
        headers={"X-Session-ID": session_id},
    )
    task_id = response.json()["task_id"]
    _wait_for_finish(client, task_id)
    return task_id


def test_repeated_exports_keep_separate_archives(client, dataset):
    from app.services import export_service

    first = _run_export(client, dataset)
    second = _run_export(client, dataset)

    assert export_service.get_archive(first) != export_service.get_archive(second)
    assert client.get(f"/api/export/download/{first}").status_code == 200
    assert client.get(f"/api/export/download/{second}").status_code == 200


def test_old_export_archives_are_removed_from_temp(client, dataset):
    from app.services import export_service

    oldest = _run_export(client, dataset)
    oldest_path = export_service.get_archive(oldest)
    assert oldest_path and os.path.exists(oldest_path)

    for _ in range(export_service.MAX_RETAINED_ARCHIVES):
        newest = _run_export(client, dataset)

    assert export_service.get_archive(oldest) is None
    assert not os.path.exists(oldest_path)
    assert not os.path.exists(os.path.dirname(oldest_path))
    assert client.get(f"/api/export/download/{oldest}").status_code == 404
    assert client.get(f"/api/export/download/{newest}").status_code == 200


def test_shutdown_removes_archives():
    from app.services import export_service

    with TestClient(app) as c:
        with tempfile.TemporaryDirectory() as parent:
            path = os.path.join(parent, "shutdown-dataset")
            session_id = c.post(
                "/api/projects/create", json={"path": path}
            ).json()["session_id"]
            c.post(
                "/api/dataset/upload",
                files=[("files", ("img0.png", _png_bytes((1, 2, 3)), "image/png"))],
                headers={"X-Session-ID": session_id},
            )

            task_id = _run_export(c, session_id)
            zip_path = export_service.get_archive(task_id)
            assert zip_path and os.path.exists(zip_path)

    assert not os.path.exists(zip_path)
