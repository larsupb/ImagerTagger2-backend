import logging
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


async def background_cleanup():
    """Periodic cleanup of expired sessions."""
    from app.sessions import session_manager

    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        session_manager.cleanup_expired()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_cleanup())
    yield
    task.cancel()
    from app.sessions import session_manager

    session_manager.cleanup_all()


app = FastAPI(title="ImageTagger API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import (
    dataset,
    media,
    captions,
    categories,
    processing,
    tagging,
    batch,
    export,
    settings,
    projects,
    promptgen,
    tasks,
    dedup,
)

app.include_router(dataset.router)
app.include_router(media.router)
app.include_router(captions.router)
app.include_router(categories.router)
app.include_router(processing.router)
app.include_router(tagging.router)
app.include_router(batch.router)
app.include_router(export.router)
app.include_router(settings.router)
app.include_router(projects.router)
app.include_router(promptgen.router)
app.include_router(tasks.router)
app.include_router(dedup.router)


@app.get("/health")
def health():
    return {"status": "ok"}
