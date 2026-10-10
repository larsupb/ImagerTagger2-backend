"""
Pytest configuration for loading test environment variables.

Automatically loads .env file from tests directory if it exists.
"""

import os
from pathlib import Path

import pytest

_env_file = Path(__file__).parent / ".env"

if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


@pytest.fixture(autouse=True)
def isolated_project_registry(tmp_path, monkeypatch):
    """Keep tests out of backend/data/projects.json, the real project registry."""
    from app.services import project_service

    registry_dir = tmp_path / "projects"
    monkeypatch.setattr(project_service, "PROJECTS_DIR", str(registry_dir))
    monkeypatch.setattr(
        project_service, "PROJECTS_FILE", str(registry_dir / "projects.json")
    )
    yield
