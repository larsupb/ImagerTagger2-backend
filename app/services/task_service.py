import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Awaitable, Callable, Optional

from app.models.schemas import TaskResponse, TaskStatus, TaskLogEntry

logger = logging.getLogger(__name__)

MAX_FINISHED_TASKS = 50

_task_store: dict[str, TaskResponse] = {}
_runners: dict[str, Callable[[], Awaitable[None]]] = {}
_pause_events: dict[str, asyncio.Event] = {}
_queue: Optional[asyncio.Queue] = None
_worker: Optional[asyncio.Task] = None
_loop: Optional[asyncio.AbstractEventLoop] = None

FINISHED_STATUSES = (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


def create_task(
    total: int,
    name: str = "Batch",
    operation: str = "batch",
    project_name: Optional[str] = None,
) -> str:
    task_id = str(uuid.uuid4())
    _task_store[task_id] = TaskResponse(
        task_id=task_id,
        name=name,
        operation=operation,
        project_name=project_name,
        status=TaskStatus.PENDING,
        progress=0.0,
        current_index=0,
        total=total,
        logs=[],
        created_at=datetime.now(),
    )
    _prune_finished()
    return task_id


def get_task(task_id: str) -> Optional[TaskResponse]:
    return _task_store.get(task_id)


def list_tasks() -> list[TaskResponse]:
    """Newest first, so the jobs page shows current work at the top."""
    return sorted(_task_store.values(), key=lambda t: t.created_at, reverse=True)


def _prune_finished() -> None:
    finished = [t for t in list_tasks() if t.status in FINISHED_STATUSES]
    for task in finished[MAX_FINISHED_TASKS:]:
        _task_store.pop(task.task_id, None)
        _runners.pop(task.task_id, None)
        _pause_events.pop(task.task_id, None)


def start_task(task_id: str) -> None:
    task = _task_store.get(task_id)
    if task and task.status == TaskStatus.PENDING:
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()


def update_progress(
    task_id: str,
    position: int,
    total: int,
    index: int,
    filename: str,
    message: str,
) -> None:
    """position drives progress; index identifies the dataset item in the log."""
    task = _task_store.get(task_id)
    if task is None:
        return
    task.current_index = index
    task.current_filename = filename
    task.progress = (position + 1) / total if total > 0 else 0.0
    task.logs.append(TaskLogEntry(index=index, filename=filename, message=message))
    if len(task.logs) > 1000:
        task.logs = task.logs[-1000:]


def complete_task(task_id: str, error: Optional[str] = None) -> None:
    task = _task_store.get(task_id)
    if task is None:
        return
    if task.status == TaskStatus.CANCELLED:
        task.completed_at = task.completed_at or datetime.now()
        return
    task.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
    task.completed_at = datetime.now()
    task.error = error


def cancel_task(task_id: str) -> bool:
    task = _task_store.get(task_id)
    if task is None or task.status in FINISHED_STATUSES:
        return False
    task.status = TaskStatus.CANCELLED
    if task.started_at is None:
        task.completed_at = datetime.now()
    _pause_events.pop(task_id, None)
    return True


def is_cancelled(task_id: str) -> bool:
    task = _task_store.get(task_id)
    return task is not None and task.status == TaskStatus.CANCELLED


def pause_task(task_id: str) -> bool:
    task = _task_store.get(task_id)
    if task is None or task.status != TaskStatus.RUNNING:
        return False
    _pause_events[task_id] = asyncio.Event()
    task.status = TaskStatus.PAUSED
    return True


def resume_task(task_id: str) -> bool:
    task = _task_store.get(task_id)
    if task is None or task.status != TaskStatus.PAUSED:
        return False
    task.status = TaskStatus.RUNNING
    event = _pause_events.pop(task_id, None)
    if event:
        event.set()
    return True


async def wait_if_paused(task_id: str) -> None:
    """Block between items while paused, so no in-flight work is interrupted."""
    while True:
        event = _pause_events.get(task_id)
        if event is None:
            return
        await event.wait()


def wait_if_paused_sync(task_id: str, poll_interval: float = 0.1) -> None:
    """Thread-side counterpart of wait_if_paused, for job bodies run via to_thread."""
    while task_id in _pause_events:
        time.sleep(poll_interval)


def submit(task_id: str, runner: Callable[[], Awaitable[None]]) -> None:
    """Queue a job body. Jobs run one at a time, in submission order."""
    _runners[task_id] = runner
    _ensure_worker()
    _queue.put_nowait(task_id)


def _ensure_worker() -> None:
    """An asyncio.Queue binds to the loop that created it, so rebuild the
    worker whenever the running loop changes (app restart, tests)."""
    global _queue, _worker, _loop
    loop = asyncio.get_running_loop()
    if _loop is not loop or _queue is None:
        _loop = loop
        _queue = asyncio.Queue()
        _worker = None
    if _worker is None or _worker.done():
        _worker = asyncio.create_task(_worker_loop())


async def _worker_loop() -> None:
    while True:
        task_id = await _queue.get()
        runner = _runners.pop(task_id, None)
        try:
            if runner is None or is_cancelled(task_id):
                continue
            start_task(task_id)
            try:
                await runner()
                complete_task(task_id)
            except Exception as e:
                logger.error("Job %s failed: %s", task_id, e, exc_info=True)
                complete_task(task_id, str(e))
        finally:
            _queue.task_done()


def start_worker() -> None:
    _ensure_worker()


def shutdown_worker() -> None:
    global _worker, _queue, _loop
    if _worker is not None:
        _worker.cancel()
    _worker = None
    _queue = None
    _loop = None
