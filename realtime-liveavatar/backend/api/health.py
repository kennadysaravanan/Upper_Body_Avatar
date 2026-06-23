"""Health, readiness, and Prometheus metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Response

from backend.config.settings import get_settings
from backend.utils.metrics import generate_latest, CONTENT_TYPE_LATEST, metrics_enabled
from backend.websocket.signaling import manager

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz() -> dict:
    s = get_settings()
    return {
        "status": "ready",
        "engine": s.avatar_engine,
        "gpu_count": s.gpu_count,
        "active_sessions": manager.count,
        "max_sessions": s.max_sessions,
        "metrics": metrics_enabled(),
    }


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
