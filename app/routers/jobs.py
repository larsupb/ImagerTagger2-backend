from fastapi import APIRouter, HTTPException

from app.models.schemas import TaskResponse
from app.services import task_service

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=list[TaskResponse])
def list_jobs():
    return task_service.list_tasks()


@router.post("/{job_id}/pause")
def pause_job(job_id: str):
    if task_service.get_task(job_id) is None:
        raise HTTPException(404, "Job not found")
    if not task_service.pause_task(job_id):
        raise HTTPException(409, "Job is not running")
    return {"paused": True}


@router.post("/{job_id}/resume")
def resume_job(job_id: str):
    if task_service.get_task(job_id) is None:
        raise HTTPException(404, "Job not found")
    if not task_service.resume_task(job_id):
        raise HTTPException(409, "Job is not paused")
    return {"paused": False}


@router.delete("/{job_id}")
def cancel_job(job_id: str):
    if task_service.get_task(job_id) is None:
        raise HTTPException(404, "Job not found")
    if not task_service.cancel_task(job_id):
        raise HTTPException(409, "Job already finished")
    return {"cancelled": True}
