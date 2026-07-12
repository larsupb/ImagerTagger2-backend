from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from typing import Literal, Optional


# --- Dataset ---
class DatasetLoadRequest(BaseModel):
    path: str
    only_missing_captions: bool = False
    include_subdirectories: bool = False


class DatasetInfo(BaseModel):
    total_items: int
    base_dir: str
    masks_dir: Optional[str]


class CaptionEntry(BaseModel):
    caption_type: str
    content: str
    is_active: bool


class ImageVersionEntry(BaseModel):
    id: int
    version_path: str
    operation: str
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    original_size: Optional[int] = None
    created_at: str


class MediaItemResponse(BaseModel):
    index: int
    filename: str
    basename: str
    extension: str
    is_video: bool
    is_image: bool
    has_caption: bool
    has_mask: bool
    is_bookmarked: bool
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    media_url: str
    thumbnail_url: str
    caption: str
    captions: list[CaptionEntry] = []
    category: Optional[str] = None


class NavigationResponse(BaseModel):
    item: MediaItemResponse
    dataset_info: DatasetInfo


# --- Captions ---
class CaptionSaveRequest(BaseModel):
    index: int
    caption: str
    caption_type: str = "tags"


class SetActiveCaptionRequest(BaseModel):
    index: int
    caption_type: str


class ExportCaptionsTxtRequest(BaseModel):
    caption_type: str = "tags"


class CaptionGenerateRequest(BaseModel):
    index: int
    tagger: str


class CaptionGenerateResponse(BaseModel):
    caption: str


class TagCloudEntry(BaseModel):
    tag: str
    count: int


class SearchReplaceRequest(BaseModel):
    search: str
    replace: str
    caption_type: str = "tags"


class SearchReplacePreview(BaseModel):
    matches: list[dict]
    total_matches: int


class TagOperationRequest(BaseModel):
    tags: list[str]
    caption_type: str = "tags"


class AppendTagRequest(BaseModel):
    tag: str
    caption_type: str = "tags"
    categories: Optional[list[str]] = None


class CaptionTypeRequest(BaseModel):
    caption_type: str


class RenameCaptionTypeRequest(BaseModel):
    old_type: str
    new_type: str


class MoveToSubdirRequest(BaseModel):
    tags: list[str]
    inverse: bool = False
    subdirectory_name: str
    caption_type: str = "tags"


# --- Processing ---
class UpscaleRequest(BaseModel):
    index: int
    upscaler: Optional[str] = None
    target_megapixels: Optional[float] = None


class MaskGenerateRequest(BaseModel):
    index: int


class CropRequest(BaseModel):
    index: int
    x: int
    y: int
    width: int
    height: int


class WhiteBalanceRequest(BaseModel):
    index: int
    method: str = "gray_world"


class ComfyUIUpscaleRequest(BaseModel):
    index: int
    target_megapixels: Optional[float] = 2.0


# --- Batch ---
class BatchProcessRequest(BaseModel):
    rename: bool = False
    rename_offset: int = 0
    upscale: bool = False
    upscaler: Optional[str] = None
    bucket_resize: bool = False
    mask: bool = False
    skip_existing_masks: bool = False
    caption: bool = False
    tagger: str = "joytag"
    unified_caption: str = ""
    caption_type: str = "tags"
    skip_existing_captions: bool = False
    vlm_prompt: Optional[str] = None
    bucket_resolution: int = 1024
    bucket_step: int = 128
    bucket_max_steps: int = 2
    color_match: bool = False
    color_match_method: str = "histogram"
    color_match_reference: int = 0
    white_balance: bool = False
    white_balance_method: str = "gray_world"


class BucketAnalyzeRequest(BaseModel):
    resolution: int = 1024
    step: int = 64
    max_steps: int = 4
    categories: list[str] | None = None


class BucketAnalyzeResponse(BaseModel):
    buckets: list[dict]
    total_images: int


class ColorMatchPreviewRequest(BaseModel):
    method: str = "histogram"
    reference: int = 0
    sample_count: int = 4


class ColorMatchPreviewResult(BaseModel):
    previews: list[dict]


class WhiteBalancePreviewRequest(BaseModel):
    method: str = "gray_world"
    sample_count: int = 4


class WhiteBalancePreviewResult(BaseModel):
    previews: list[dict]


# --- Export ---
class ExportRequest(BaseModel):
    format: Literal["standard", "musubi_control", "musubi_control_json"] = "standard"
    caption_type: str = "tags"
    bucket_resize: bool = False
    bucket_resolution: int = 1024
    bucket_step: int = 128
    bucket_max_steps: int = 2
    categories: list[str] | None = None


# --- Settings ---
class SettingsResponse(BaseModel):
    ignore_list: list[str]
    upscaler: str
    upscale_target_megapixels: float
    default_tagger: str
    tagger_instruction: str
    combo_taggers: list[str]
    florence_settings: dict
    rembg: dict
    vlm_endpoint: dict
    llm_endpoint: dict
    white_balance_method: Optional[str] = "gray_world"


class SettingsUpdateRequest(BaseModel):
    key: str
    value: str | int | float | bool | list | dict


class ExportResponse(BaseModel):
    path: str
    count: int


# --- Gallery ---
class GalleryItem(BaseModel):
    index: int
    thumbnail_url: str
    thumb_token: str
    filename: str
    is_bookmarked: bool
    has_caption: bool = False
    width: int | None = None
    height: int | None = None
    category: str | None = None


class GalleryResponse(BaseModel):
    items: list[GalleryItem]
    total: int
    page: int
    page_size: int


# --- Task ---
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskLogEntry(BaseModel):
    index: int
    filename: str
    message: str


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float
    current_index: int
    current_filename: Optional[str] = None
    total: int
    logs: list[TaskLogEntry]
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


# --- Dedup ---
class DedupScanRequest(BaseModel):
    phash_threshold: int = 10
    ssim_threshold: float = 0.85


class DedupImageInfo(BaseModel):
    index: int
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    thumbnail_url: str
    path: str


class DedupGroup(BaseModel):
    images: list[DedupImageInfo]
    keep_index: int


class DedupScanResponse(BaseModel):
    groups: list[DedupGroup]


class DedupRemoveRequest(BaseModel):
    paths: list[str]
