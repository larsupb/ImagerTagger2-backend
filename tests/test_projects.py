import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_create_project_creates_directory(client):
    with tempfile.TemporaryDirectory() as parent:
        new_path = os.path.join(parent, "my-dataset")
        response = client.post("/api/projects/create", json={"path": new_path})
        assert response.status_code == 200
        assert os.path.isdir(new_path)


def test_create_project_returns_open_response_shape(client):
    with tempfile.TemporaryDirectory() as parent:
        new_path = os.path.join(parent, "test-dataset")
        response = client.post("/api/projects/create", json={"path": new_path})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "project_id" in data
        assert "project_name" in data
        assert data["dataset_info"]["total_items"] == 0


def test_create_project_existing_path_returns_409(client):
    with tempfile.TemporaryDirectory() as existing_path:
        response = client.post("/api/projects/create", json={"path": existing_path})
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()


def test_create_project_empty_path_returns_422(client):
    response = client.post("/api/projects/create", json={})
    assert response.status_code == 422


def test_create_project_creates_intermediate_directories(client):
    with tempfile.TemporaryDirectory() as parent:
        nested_path = os.path.join(parent, "a", "b", "my-dataset")
        response = client.post("/api/projects/create", json={"path": nested_path})
        assert response.status_code == 200
        assert os.path.isdir(nested_path)
