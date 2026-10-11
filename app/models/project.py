from pydantic import BaseModel
from typing import Optional


class ProjectOpenRequest(BaseModel):
    path: str
    only_missing_captions: bool = False
    include_subdirectories: bool = False


class ProjectCreateRequest(BaseModel):
    path: str


class ProjectDatasetInfo(BaseModel):
    total_items: int
    base_dir: str
    masks_dir: Optional[str]


class ProjectOpenResponse(BaseModel):
    session_id: str
    project_id: str
    project_name: str
    dataset_info: ProjectDatasetInfo


class ProjectCloseResponse(BaseModel):
    status: str


class ActiveProjectInfo(BaseModel):
    session_id: str
    project_id: str
    project_name: str
    path: str
    total_items: int
    current_index: int
    created_at: float
    last_accessed: float


class ActiveProjectsResponse(BaseModel):
    projects: list[ActiveProjectInfo]


class RecentProjectEntry(BaseModel):
    project_id: str
    name: str
    path: str
    last_opened: str
    open_count: int
    is_active: bool


class RecentProjectsResponse(BaseModel):
    projects: list[RecentProjectEntry]


class ProjectConfigUpdateRequest(BaseModel):
    name: Optional[str] = None
    settings_overrides: Optional[dict] = None


class ProjectConfigResponse(BaseModel):
    name: str
    settings_overrides: dict
    last_opened: str
    version: int
    settings: dict = {}
