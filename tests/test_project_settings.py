from app.config import florence_settings, vlm_endpoint_settings


def test_helpers_prefer_state_dict_config():
    cfg = {
        "florence_settings": {"prompt": "proj florence"},
        "vlm_endpoint": {"prompt": "proj vlm", "base_url": "http://proj:1/v1"},
    }
    assert florence_settings({"config": cfg})["prompt"] == "proj florence"
    vlm = vlm_endpoint_settings({"config": cfg})
    assert vlm["prompt"] == "proj vlm"
    assert vlm["base_url"] == "http://proj:1/v1"


def test_helpers_fall_back_to_global_without_config():
    assert vlm_endpoint_settings()["base_url"]
    assert vlm_endpoint_settings({"dataset": None})["base_url"]
    assert florence_settings({})["prompt"]


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


@pytest.fixture()
def dataset(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "dataset")
        response = client.post("/api/projects/create", json={"path": path})
        project_id = response.json()["project_id"]
        session_id = response.json()["session_id"]
        files = []
        for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buf, format="PNG")
            files.append(("files", (f"img{i}.png", buf.getvalue(), "image/png")))
        client.post("/api/dataset/upload", files=files, headers={"X-Session-ID": session_id})
        for i in range(3):
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
            "project_id": project_id,
            "path": path,
            "headers": {"X-Session-ID": session_id},
        }


def test_tagging_generate_receives_project_config(client, dataset, monkeypatch):
    captured = {}

    def fake_generate_caption(index, tagger, state_dict, prompt=None):
        captured["state_dict"] = state_dict
        return "a caption"

    monkeypatch.setattr("lib.captioning.generate_caption", fake_generate_caption)
    response = client.put(
        f"/api/projects/{dataset['project_id']}/config",
        json={"settings_overrides": {"vlm_endpoint": {"prompt": "PROJECT PROMPT"}}},
    )
    assert response.status_code == 200
    response = client.post(
        "/api/tagging/generate",
        json={"index": 0, "tagger": "vlm-tagger"},
        headers=dataset["headers"],
    )
    assert response.status_code == 200
    vlm = captured["state_dict"]["config"]["vlm_endpoint"]
    assert vlm["prompt"] == "PROJECT PROMPT"
    assert vlm["base_url"]
