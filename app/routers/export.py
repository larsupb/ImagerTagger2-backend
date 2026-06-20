import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.sessions import Session, get_session
from app.models.schemas import ExportRequest
from app.services import export_service, task_service

router = APIRouter(prefix="/api/export", tags=["export"])

_zip_paths: dict[str, str] = {}


@router.post("")
async def export_dataset(req: ExportRequest, session: Session = Depends(get_session)):
    if session.dataset is None:
        raise HTTPException(400, "No dataset loaded")

    ds = session.dataset
    total = len(ds)

    task_id = task_service.create_task(total)
    task_service.start_task(task_id)

    zip_path = None

    async def run_export():
        try:
            output_dir = None
            nonlocal zip_path
            try:
                options = export_service.ExportOptions(
                    format=req.format,
                    caption_type=req.caption_type,
                    bucket_resize=req.bucket_resize,
                    bucket_resolution=req.bucket_resolution,
                    bucket_step=req.bucket_step,
                    bucket_max_steps=req.bucket_max_steps,
                    categories=req.categories,
                )

                output_dir = export_service.export_dataset(session, options)

                temp_dir = os.path.dirname(output_dir)
                zip_filename = f"export_{session.id[:8]}.zip"
                zip_path = os.path.join(temp_dir, zip_filename)

                export_service.create_zip_archive(output_dir, zip_path)

                _zip_paths[task_id] = zip_path

                task_service.update_progress(
                    task_id,
                    total - 1,
                    total,
                    "export.zip",
                    "Creating ZIP archive",
                )

                task_service.complete_task(task_id)
            except Exception as e:
                task_service.complete_task(task_id, str(e))
            finally:
                if output_dir:
                    export_service.cleanup_export_dir(output_dir)
        except Exception as e:
            task_service.complete_task(task_id, str(e))

    asyncio.create_task(run_export())
    return {"task_id": task_id}


@router.get("/download/{task_id}")
async def download_export(task_id: str):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")

    if task.status != "completed":
        raise HTTPException(400, "Export not completed")

    zip_path = _zip_paths.get(task_id)
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(404, "Export file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="dataset_export.zip",
    )


@router.get("/progress/{task_id}")
async def get_export_progress(task_id: str):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return {
        "status": task.status,
        "progress": task.progress,
        "total": task.total,
        "current_index": task.current_index,
        "current_filename": task.current_filename,
        "error": task.error,
    }
