import uuid
from datetime import datetime
from typing import Optional

from app.models.schemas import TaskResponse, TaskStatus, TaskLogEntry

_task_store: dict[str, TaskResponse] = {}


def create_task(total: int) -> str:
    task_id = str(uuid.uuid4())
    task = TaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        progress=0.0,
        current_index=0,
        total=total,
        logs=[],
        started_at=datetime.now(),
    )
    _task_store[task_id] = task
    return task_id


def get_task(task_id: str) -> Optional[TaskResponse]:
    return _task_store.get(task_id)


def start_task(task_id: str) -> None:
    if task_id in _task_store:
        _task_store[task_id].status = TaskStatus.RUNNING


def update_progress(
    task_id: str, index: int, total: int, filename: str, message: str
) -> None:
    if task_id in _task_store:
        task = _task_store[task_id]
        task.current_index = index
        task.current_filename = filename
        task.progress = (index + 1) / total if total > 0 else 0.0
        task.logs.append(TaskLogEntry(index=index, filename=filename, message=message))
        if len(task.logs) > 1000:
            task.logs = task.logs[-1000:]


def complete_task(task_id: str, error: Optional[str] = None) -> None:
    if task_id in _task_store:
        task = _task_store[task_id]
        task.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.error = error


def cancel_task(task_id: str) -> bool:
    if task_id in _task_store:
        _task_store[task_id].status = TaskStatus.CANCELLED
        return True
    return False


def is_cancelled(task_id: str) -> bool:
    if task_id in _task_store:
        return _task_store[task_id].status == TaskStatus.CANCELLED
    return False
