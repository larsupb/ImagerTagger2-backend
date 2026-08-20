import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def project(client):
    with tempfile.TemporaryDirectory() as parent:
        path = os.path.join(parent, "tagtypes")
        response = client.post("/api/projects/create", json={"path": path})
        assert response.status_code == 200
        yield {"session_id": response.json()["session_id"], "path": path}


def _project_overrides(path: str) -> dict:
    config_path = os.path.join(path, ".imagetagger", "project.json")
    with open(config_path) as f:
        return json.load(f).get("settings_overrides", {})


def test_active_tag_type_defaults_to_unset(client, project):
    response = client.get(
        "/api/session/settings", headers={"X-Session-ID": project["session_id"]}
    )
    assert response.status_code == 200
    assert response.json()["active_tag_type"] is None


def test_setting_active_tag_type_is_readable_back(client, project):
    headers = {"X-Session-ID": project["session_id"]}
    response = client.put(
        "/api/session/settings", json={"active_tag_type": "description"}, headers=headers
    )
    assert response.status_code == 200
    assert response.json()["active_tag_type"] == "description"

    read_back = client.get("/api/session/settings", headers=headers)
    assert read_back.json()["active_tag_type"] == "description"


def test_setting_active_tag_type_persists_to_project_config(client, project):
    client.put(
        "/api/session/settings",
        json={"active_tag_type": "description"},
        headers={"X-Session-ID": project["session_id"]},
    )
    assert _project_overrides(project["path"])["active_tag_type"] == "description"


def test_reopening_project_restores_active_tag_type(client, project):
    client.put(
        "/api/session/settings",
        json={"active_tag_type": "description"},
        headers={"X-Session-ID": project["session_id"]},
    )

    reopened = client.post("/api/projects/open", json={"path": project["path"]})
    assert reopened.status_code == 200
    response = client.get(
        "/api/session/settings",
        headers={"X-Session-ID": reopened.json()["session_id"]},
    )
    assert response.json()["active_tag_type"] == "description"


def test_persisting_keeps_other_overrides(client, project):
    project_id = client.get("/api/projects/recent").json()["projects"][0]["project_id"]
    client.put(
        f"/api/projects/{project_id}/config",
        json={"settings_overrides": {"upscaler": "custom-upscaler"}},
    )

    client.put(
        "/api/session/settings",
        json={"active_tag_type": "description"},
        headers={"X-Session-ID": project["session_id"]},
    )

    overrides = _project_overrides(project["path"])
    assert overrides["active_tag_type"] == "description"
    assert overrides["upscaler"] == "custom-upscaler", "clobbered an unrelated override"
