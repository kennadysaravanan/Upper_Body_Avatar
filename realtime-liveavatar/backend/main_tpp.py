"""Realtime multi-GPU (TPP) launcher.

Run with:
    GPU_COUNT=5 AVATAR_ENGINE=liveavatar \
    torchrun --nproc_per_node=5 --master_port=29502 -m backend.main_tpp

Model:
  * All 5 ranks build the WanS2V TPP pipeline and join generate()'s collective.
  * Rank 0 additionally runs the FastAPI/WebRTC/OpenAI server (uvicorn) on the
    main thread and a render-server thread that broadcasts each render job to the
    other ranks, so every rank calls generate() together.
  * Ranks 1..N-1 run a worker loop waiting for broadcast jobs.

This is the realtime path: TPP makes per-sentence generate() run at ~45 FPS
throughput, so spoken responses are produced near-realtime and streamed to the
browser over WebRTC.

NOTE: validate the NCCL collective dispatch on the actual 5-GPU pod.
"""
from __future__ import annotations

import os

from backend.config.settings import get_settings
from backend.utils.logging import configure_logging, get_logger


def main() -> None:
    os.environ.setdefault("AVATAR_ENGINE", "liveavatar")
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("main_tpp")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    log.info("main_tpp rank=%d world=%d engine=%s", rank, world, settings.avatar_engine)

    from backend.avatar import build_engine

    engine = build_engine()
    # all ranks build the pipeline + init the process group
    engine.build_blocking()  # type: ignore[attr-defined]

    if rank != 0:
        # worker ranks: participate in generate() collectives forever
        engine.worker_loop()  # type: ignore[attr-defined]
        return

    # rank 0: render server already started inside build_blocking(); serve HTTP
    import uvicorn

    log.info("rank 0 serving on %s:%d", settings.host, settings.port)
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
