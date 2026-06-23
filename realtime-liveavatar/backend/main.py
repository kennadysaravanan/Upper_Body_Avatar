"""FastAPI application entrypoint for the realtime LiveAvatar platform."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.health import router as health_router
from backend.config.settings import get_settings
from backend.utils.logging import configure_logging, get_logger
from backend.websocket.signaling import manager, router as ws_router

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting realtime-liveavatar engine=%s gpus=%d", settings.avatar_engine, settings.gpu_count)
    if settings.avatar_engine == "liveavatar" and os.getenv("PRELOAD_MODEL", "0") == "1":
        from backend.avatar import build_engine

        await build_engine().warmup()
    yield
    await manager.shutdown()
    log.info("shutdown complete")


app = FastAPI(title="Realtime LiveAvatar", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(ws_router)

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, log_config=None)
