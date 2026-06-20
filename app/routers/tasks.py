from fastapi import APIRouter, HTTPException

from app.models.schemas import TaskResponse
from app.services import task_service

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskResponse)
def get_task_status(task_id: str):
    task = task_service.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return task


@router.delete("/{task_id}")
def cancel_task(task_id: str):
    cancelled = task_service.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(404, "Task not found")
    return {"cancelled": True}
