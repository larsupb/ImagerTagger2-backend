import io
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


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def dataset(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "my-dataset")
        response = client.post("/api/projects/create", json={"path": path})
        session_id = response.json()["session_id"]

        files = [
            ("files", (f"img{i}.png", _png_bytes((10 * i, 20, 30)), "image/png"))
            for i in range(4)
        ]
        client.post(
            "/api/dataset/upload", files=files, headers={"X-Session-ID": session_id}
        )
        yield {"session_id": session_id, "path": path}


@pytest.fixture()
def slow_caption(monkeypatch):
    """Captioning that blocks until released, so jobs stay observable."""
    import lib.captioning

    gate = {"release": False, "started": 0}

    def fake_caption(index, tagger, state_dict, prompt=None):
        gate["started"] += 1
        while not gate["release"]:
            time.sleep(0.01)
        return f"caption {index}"

    monkeypatch.setattr(lib.captioning, "generate_caption", fake_caption)
    return gate


@pytest.fixture()
def fast_caption(monkeypatch):
    import lib.captioning

    seen = []

    def fake_caption(index, tagger, state_dict, prompt=None):
        seen.append({"index": index, "tagger": tagger, "prompt": prompt})
        return f"caption {index}"

    monkeypatch.setattr(lib.captioning, "generate_caption", fake_caption)
    return seen


def _start_caption_job(client, session_id: str, **overrides) -> str:
    body = {"caption": True, "tagger": "joytag"}
    body.update(overrides)
    response = client.post(
        "/api/batch/process", json=body, headers={"X-Session-ID": session_id}
    )
    assert response.status_code == 200
    return response.json()["task_id"]


def _wait_for(predicate, timeout: float = 20.0, client=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = client.get("/api/jobs").json()
        if predicate(jobs):
            return jobs
        time.sleep(0.05)
    raise AssertionError(f"condition not met, last jobs: {client.get('/api/jobs').json()}")


def _job(jobs: list[dict], job_id: str) -> dict:
    return next(j for j in jobs if j["task_id"] == job_id)


def test_jobs_list_is_empty_initially(client):
    assert client.get("/api/jobs").json() == []


def test_job_carries_name_project_and_total(client, dataset, fast_caption):
    job_id = _start_caption_job(client, dataset["session_id"])

    jobs = _wait_for(
        lambda js: _job(js, job_id)["status"] == "completed", client=client
    )
    job = _job(jobs, job_id)
    assert job["name"] == "Caption (joytag)"
    assert job["operation"] == "caption"
    assert job["project_name"] == "my-dataset"
    assert job["total"] == 4
    assert job["progress"] == 1.0
    assert len(job["logs"]) == 4


def test_job_restricted_to_selected_indices(client, dataset, fast_caption):
    job_id = _start_caption_job(client, dataset["session_id"], indices=[1, 3])

    jobs = _wait_for(
        lambda js: _job(js, job_id)["status"] == "completed", client=client
    )
    job = _job(jobs, job_id)
    assert job["total"] == 2
    assert job["progress"] == 1.0
    assert [entry["index"] for entry in job["logs"]] == [1, 3]
    assert [c["index"] for c in fast_caption] == [1, 3]


def test_job_passes_prompt_override_to_tagger(client, dataset, fast_caption):
    _start_caption_job(
        client,
        dataset["session_id"],
        indices=[0],
        tagger="florence",
        prompt="describe briefly",
    )

    _wait_for(lambda js: js and js[0]["status"] == "completed", client=client)
    assert fast_caption[0]["prompt"] == "describe briefly"


def test_second_job_stays_queued_while_first_runs(client, dataset, slow_caption):
    session_id = dataset["session_id"]
    first = _start_caption_job(client, session_id, indices=[0])
    second = _start_caption_job(client, session_id, indices=[1])

    jobs = _wait_for(lambda js: _job(js, first)["status"] == "running", client=client)
    assert _job(jobs, second)["status"] == "pending"

    slow_caption["release"] = True
    _wait_for(
        lambda js: _job(js, second)["status"] == "completed", client=client
    )
    assert _job(client.get("/api/jobs").json(), first)["status"] == "completed"


def test_pause_and_resume_running_job(client, dataset, slow_caption):
    job_id = _start_caption_job(client, dataset["session_id"], indices=[0, 1, 2, 3])
    # The first item blocks inside the tagger, so the job is reliably running.
    _wait_for(lambda js: _job(js, job_id)["status"] == "running", client=client)

    assert client.post(f"/api/jobs/{job_id}/pause").status_code == 200
    assert _job(client.get("/api/jobs").json(), job_id)["status"] == "paused"

    # Releasing lets the in-flight item finish; the loop then parks at the pause.
    slow_caption["release"] = True
    _wait_for(lambda js: _job(js, job_id)["progress"] >= 0.25, client=client)

    progress_at_pause = _job(client.get("/api/jobs").json(), job_id)["progress"]
    time.sleep(0.3)
    held = _job(client.get("/api/jobs").json(), job_id)
    assert held["status"] == "paused"
    assert held["progress"] == progress_at_pause, "paused job kept processing"

    assert client.post(f"/api/jobs/{job_id}/resume").status_code == 200
    _wait_for(lambda js: _job(js, job_id)["status"] == "completed", client=client)
    assert _job(client.get("/api/jobs").json(), job_id)["progress"] == 1.0


def test_cancel_queued_job_never_runs_it(client, dataset, slow_caption):
    session_id = dataset["session_id"]
    first = _start_caption_job(client, session_id, indices=[0])
    second = _start_caption_job(client, session_id, indices=[1, 2, 3])

    _wait_for(lambda js: _job(js, first)["status"] == "running", client=client)
    assert client.delete(f"/api/jobs/{second}").status_code == 200

    slow_caption["release"] = True
    _wait_for(lambda js: _job(js, first)["status"] == "completed", client=client)

    cancelled = _job(client.get("/api/jobs").json(), second)
    assert cancelled["status"] == "cancelled"
    assert cancelled["logs"] == []
    assert slow_caption["started"] == 1


def test_pause_rejects_job_that_is_not_running(client, dataset, fast_caption):
    job_id = _start_caption_job(client, dataset["session_id"], indices=[0])
    _wait_for(lambda js: _job(js, job_id)["status"] == "completed", client=client)

    assert client.post(f"/api/jobs/{job_id}/pause").status_code == 409
    assert client.post("/api/jobs/does-not-exist/pause").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}").status_code == 409


def test_failed_job_records_error(client, dataset, monkeypatch):
    import lib.captioning

    def boom(index, tagger, state_dict, prompt=None):
        raise RuntimeError("tagger exploded")

    monkeypatch.setattr(lib.captioning, "generate_caption", boom)
    job_id = _start_caption_job(client, dataset["session_id"], indices=[0])

    jobs = _wait_for(
        lambda js: _job(js, job_id)["status"] == "completed", client=client
    )
    # Per-item failures are logged and the job still finishes.
    assert "ERROR: tagger exploded" in _job(jobs, job_id)["logs"][0]["message"]
