import uuid
import time
import threading
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.image_dataset import ImageDataSet


SESSION_TTL_SECONDS = 3600


@dataclass
class Session:
    id: str
    project_id: str = ""
    dataset: Optional[ImageDataSet] = None
    config: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    upscaled_image: Optional[object] = None
    upscaled_index: Optional[int] = None
    db: Optional[sqlite3.Connection] = None

    def touch(self):
        self.last_accessed = time.time()

    @property
    def is_expired(self) -> bool:
        return time.time() - self.last_accessed > SESSION_TTL_SECONDS


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, project_id: str = "") -> Session:
        session_id = str(uuid.uuid4())
        session = Session(id=session_id, project_id=project_id)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired:
            self.delete(session_id)
            return None
        session.touch()
        return session

    def delete(self, session_id: str):
        with self._lock:
            # deliberately not closing session.db: background tasks keep the
            # connection alive until they finish; GC closes it with the session
            self._sessions.pop(session_id, None)

    def cleanup_all(self):
        with self._lock:
            for s in self._sessions.values():
                if s.db:
                    try:
                        s.db.close()
                    except Exception:
                        pass
            self._sessions.clear()

    def cleanup_expired(self):
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
            for sid in expired:
                # see delete(): running jobs may still hold this connection
                del self._sessions[sid]

    def list_active(self) -> list[dict]:
        """Return list of active (non-expired) sessions with basic info."""
        now = time.time()
        with self._lock:
            return [
                {
                    "id": s.id,
                    "project_id": s.project_id,
                    "created_at": s.created_at,
                    "last_accessed": s.last_accessed,
                    "has_dataset": s.dataset is not None,
                    "total_items": len(s.dataset) if s.dataset else 0,
                    "current_index": getattr(s.dataset, "_current_index", 0)
                    if s.dataset
                    else 0,
                }
                for s in self._sessions.values()
                if now - s.last_accessed <= SESSION_TTL_SECONDS
            ]


session_manager = SessionManager()


from fastapi import Header, HTTPException, Query


def get_session(x_session_id: str = Header(...)) -> Session:
    session = session_manager.get(x_session_id)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session


def get_session_flexible(
    x_session_id: str = Header(None),
    session_id: str = Query(None),
) -> Session:
    sid = x_session_id or session_id
    if not sid:
        raise HTTPException(status_code=401, detail="Session ID required")
    session = session_manager.get(sid)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return session
