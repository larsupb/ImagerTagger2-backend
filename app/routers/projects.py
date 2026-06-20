import os
from fastapi import APIRouter, Depends, HTTPException

from app.sessions import Session, get_session, session_manager
from app.models.project import (
    ProjectOpenRequest,
    ProjectCreateRequest,
    ProjectOpenResponse,
    ProjectDatasetInfo,
    ProjectCloseResponse,
    ActiveProjectInfo,
    ActiveProjectsResponse,
    RecentProjectEntry,
    RecentProjectsResponse,
    ProjectConfigUpdateRequest,
    ProjectConfigResponse,
)
from app.services.project_service import (
    compute_project_id,
    get_or_create_registry_entry,
    remove_from_registry,
    get_recent_projects,
    create_or_update_project_config,
    read_project_config,
    update_project_config,
    get_effective_settings,
)
from app.services.dataset_service import load_dataset
from app.config import read_settings

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.post("/open", response_model=ProjectOpenResponse)
def open_project(req: ProjectOpenRequest):
    """Open a project: creates session + loads dataset in one call."""
    path = os.path.abspath(req.path)
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Dataset path not found")
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Dataset path is not a directory")

    # Get or create registry entry
    entry = get_or_create_registry_entry(path)
    project_id = entry["project_id"]
    project_name = entry["name"]

    # Create or update project config
    create_or_update_project_config(path, name=project_name)

    # Create session with project_id
    session = session_manager.create(project_id=project_id)

    # Load dataset into session
    load_dataset(
        session,
        path,
        req.only_missing_captions,
        req.include_subdirectories,
    )

    # Apply project-specific settings overrides
    effective_settings = get_effective_settings(path)
    session.config = effective_settings

    ds = session.dataset
    return ProjectOpenResponse(
        session_id=session.id,
        project_id=project_id,
        project_name=project_name,
        dataset_info=ProjectDatasetInfo(
            total_items=len(ds),
            base_dir=ds._base_dir,
            masks_dir=ds._masks_dir,
        ),
    )


@router.post("/create", response_model=ProjectOpenResponse)
def create_project(req: ProjectCreateRequest):
    """Create a new empty directory and open it as a project."""
    path = os.path.abspath(req.path)
    if os.path.exists(path):
        raise HTTPException(status_code=409, detail="Path already exists")
    try:
        os.makedirs(path)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Path already exists")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))

    entry = get_or_create_registry_entry(path)
    project_id = entry["project_id"]
    project_name = entry["name"]

    create_or_update_project_config(path, name=project_name)

    session = session_manager.create(project_id=project_id)

    load_dataset(session, path, False, False)

    effective_settings = get_effective_settings(path)
    session.config = effective_settings

    ds = session.dataset
    return ProjectOpenResponse(
        session_id=session.id,
        project_id=project_id,
        project_name=project_name,
        dataset_info=ProjectDatasetInfo(
            total_items=len(ds),
            base_dir=ds._base_dir,
            masks_dir=ds._masks_dir,
        ),
    )


@router.delete("/{session_id}", response_model=ProjectCloseResponse)
def close_project(session_id: str):
    """Close a project session."""
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    session_manager.delete(session_id)
    return ProjectCloseResponse(status="closed")


@router.get("/", response_model=ActiveProjectsResponse)
def list_active_projects():
    """List all active project sessions."""
    active = session_manager.list_active()
    projects = []
    for info in active:
        if not info["has_dataset"]:
            continue
        # Get project name from registry
        registry_entries = get_recent_projects(limit=100)
        project_name = "Unknown"
        project_path = ""
        for entry in registry_entries:
            if entry["project_id"] == info["project_id"]:
                project_name = entry["name"]
                project_path = entry["path"]
                break
        projects.append(
            ActiveProjectInfo(
                session_id=info["id"],
                project_id=info["project_id"],
                project_name=project_name,
                path=project_path,
                total_items=info["total_items"],
                current_index=info["current_index"],
                created_at=info["created_at"],
                last_accessed=info["last_accessed"],
            )
        )
    return ActiveProjectsResponse(projects=projects)


@router.get("/recent", response_model=RecentProjectsResponse)
def recent_projects(limit: int = 10):
    """List recently opened projects from registry."""
    limit = min(limit, 50)
    recent = get_recent_projects(limit)
    active_sessions = session_manager.list_active()
    active_project_ids = {s["project_id"] for s in active_sessions}
    projects = [
        RecentProjectEntry(
            project_id=p["project_id"],
            name=p["name"],
            path=p["path"],
            last_opened=p["last_opened"],
            open_count=p.get("open_count", 0),
            is_active=p["project_id"] in active_project_ids,
        )
        for p in recent
    ]
    return RecentProjectsResponse(projects=projects)


@router.put("/{project_id}/config", response_model=ProjectConfigResponse)
def update_project_configuration(project_id: str, req: ProjectConfigUpdateRequest):
    """Update project configuration."""
    # Find the project path from registry
    recent = get_recent_projects(limit=200)
    project_entry = None
    for p in recent:
        if p["project_id"] == project_id:
            project_entry = p
            break
    if project_entry is None:
        raise HTTPException(status_code=404, detail="Project not found")

    config = update_project_config(
        project_entry["path"],
        name=req.name,
        settings_overrides=req.settings_overrides,
    )

    # If session is active, update its config
    active = session_manager.list_active()
    for info in active:
        if info["project_id"] == project_id:
            session = session_manager.get(info["id"])
            if session:
                session.config = get_effective_settings(project_entry["path"])

    return ProjectConfigResponse(
        name=config["name"],
        settings_overrides=config.get("settings_overrides", {}),
        last_opened=config["last_opened"],
        version=config.get("version", 1),
    )


@router.delete("/recent/{project_id}")
def remove_recent_project(project_id: str):
    """Remove a project from the recent projects registry."""
    if not remove_from_registry(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "removed"}
