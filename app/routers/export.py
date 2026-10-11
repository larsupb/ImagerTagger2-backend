import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.sessions import Session, get_session
from app.models.schemas import ExportRequest, TaskResponse
from app.services import export_service, task_service

router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("")
async def export_dataset(req: ExportRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    total = len(ds)

    task_id = task_service.create_task(
        total,
        name=f"Export ({req.format})",
        operation="export",
        project_name=os.path.basename(ds.base_dir) if ds.base_dir else None,
    )

    options = export_service.ExportOptions(
        format=req.format,
        caption_type=req.caption_type,
        bucket_resize=req.bucket_resize,
        bucket_resolution=req.bucket_resolution,
        bucket_step=req.bucket_step,
        bucket_max_steps=req.bucket_max_steps,
        categories=req.categories,
    )

    def on_progress(index: int, filename: str, message: str) -> bool:
        task_service.wait_if_paused_sync(task_id)
        if task_service.is_cancelled(task_id):
            return False
        session.touch()
        task_service.update_progress(task_id, index, total, index, filename, message)
        return True

    def build_archive() -> Optional[str]:
        """Runs off the event loop: image conversion and zipping are blocking work."""
        output_dir = export_service.export_dataset(session, options, on_progress)
        try:
            if task_service.is_cancelled(task_id):
                return None
            return export_service.create_archive(
                output_dir, f"export_{session.id[:8]}.zip"
            )
        finally:
            export_service.cleanup_export_dir(output_dir)

    async def run_export():
        zip_path = await asyncio.to_thread(build_archive)
        if zip_path is None:
            return
        export_service.store_archive(task_id, zip_path)
        last = max(total - 1, 0)
        task_service.update_progress(
            task_id, last, total, last, "export.zip", "Created ZIP archive"
        )

    task_service.submit(task_id, run_export)
    return {"task_id": task_id}


@router.get("/download/{task_id}")
async def download_export(task_id: str):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")

    if task.status != "completed":
        raise HTTPException(400, "Export not completed")

    zip_path = export_service.get_archive(task_id)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(404, "Export file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="dataset_export.zip",
    )


@router.get("/progress/{task_id}", response_model=TaskResponse)
async def get_export_progress(task_id: str):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return task
