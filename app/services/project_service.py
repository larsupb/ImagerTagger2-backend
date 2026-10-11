import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from app.config import read_settings, merge_settings

PROJECTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PROJECTS_FILE = os.path.join(PROJECTS_DIR, "projects.json")
PROJECT_CONFIG_DIR = ".imagetagger"
PROJECT_CONFIG_FILE = "project.json"
PROJECT_CONFIG_VERSION = 1


def _ensure_projects_dir():
    os.makedirs(PROJECTS_DIR, exist_ok=True)


def _read_registry() -> dict:
    _ensure_projects_dir()
    if not os.path.exists(PROJECTS_FILE):
        return {"projects": [], "version": 1}
    with open(PROJECTS_FILE, "r") as f:
        return json.load(f)


def _write_registry(data: dict):
    _ensure_projects_dir()
    with open(PROJECTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def compute_project_id(path: str) -> str:
    """Stable project ID derived from absolute path hash."""
    abs_path = os.path.abspath(path)
    return "proj_" + hashlib.sha256(abs_path.encode()).hexdigest()[:12]


def get_project_config_path(dataset_path: str) -> str:
    return os.path.join(dataset_path, PROJECT_CONFIG_DIR, PROJECT_CONFIG_FILE)


def read_project_config(dataset_path: str) -> Optional[dict]:
    config_path = get_project_config_path(dataset_path)
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def write_project_config(dataset_path: str, config: dict):
    config_dir = os.path.join(dataset_path, PROJECT_CONFIG_DIR)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, PROJECT_CONFIG_FILE)
    config["version"] = PROJECT_CONFIG_VERSION
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def create_or_update_project_config(
    dataset_path: str, name: Optional[str] = None
) -> dict:
    """Create or update project config in dataset directory."""
    existing = read_project_config(dataset_path) or {}
    display_name = (
        name or existing.get("name") or os.path.basename(os.path.abspath(dataset_path))
    )
    config = {
        "name": display_name,
        "settings_overrides": existing.get("settings_overrides", {}),
        "last_opened": datetime.now(timezone.utc).isoformat(),
        "version": PROJECT_CONFIG_VERSION,
    }
    write_project_config(dataset_path, config)
    return config


def get_or_create_registry_entry(path: str, name: Optional[str] = None) -> dict:
    """Get existing registry entry or create new one."""
    project_id = compute_project_id(path)
    registry = _read_registry()
    for entry in registry["projects"]:
        if entry["project_id"] == project_id:
            entry["last_opened"] = datetime.now(timezone.utc).isoformat()
            entry["open_count"] = entry.get("open_count", 0) + 1
            if name:
                entry["name"] = name
            _write_registry(registry)
            return entry
    new_entry = {
        "project_id": project_id,
        "name": name or os.path.basename(os.path.abspath(path)),
        "path": os.path.abspath(path),
        "last_opened": datetime.now(timezone.utc).isoformat(),
        "open_count": 1,
    }
    registry["projects"].append(new_entry)
    _write_registry(registry)
    return new_entry


def find_registry_entry(project_id: str) -> Optional[dict]:
    registry = _read_registry()
    for entry in registry["projects"]:
        if entry["project_id"] == project_id:
            return entry
    return None


def remove_from_registry(project_id: str) -> bool:
    """Remove project from registry. Returns True if found and removed."""
    registry = _read_registry()
    before = len(registry["projects"])
    registry["projects"] = [
        p for p in registry["projects"] if p["project_id"] != project_id
    ]
    if len(registry["projects"]) < before:
        _write_registry(registry)
        return True
    return False


def get_recent_projects(limit: int = 10) -> list[dict]:
    """Get recently opened projects sorted by last_opened descending."""
    registry = _read_registry()
    projects = sorted(
        registry["projects"], key=lambda p: p.get("last_opened", ""), reverse=True
    )
    return projects[:limit]


def update_project_config(
    dataset_path: str,
    name: Optional[str] = None,
    settings_overrides: Optional[dict] = None,
) -> dict:
    """Update project config with partial changes."""
    existing = read_project_config(dataset_path)
    if existing is None:
        return create_or_update_project_config(dataset_path, name=name)
    if name is not None:
        existing["name"] = name
    if settings_overrides is not None:
        existing["settings_overrides"] = settings_overrides
    existing["last_opened"] = datetime.now(timezone.utc).isoformat()
    write_project_config(dataset_path, existing)
    return existing


def get_effective_settings(dataset_path: str) -> dict:
    """Get merged settings: global + project overrides."""
    config = read_project_config(dataset_path)
    overrides = config.get("settings_overrides", {}) if config else {}
    return merge_settings(overrides if overrides else None)
