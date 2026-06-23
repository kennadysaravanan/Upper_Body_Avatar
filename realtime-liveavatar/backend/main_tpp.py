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
        # worker ranks: participate in generate() collectives forever (main thread)
        engine.worker_loop()  # type: ignore[attr-defined]
        return

    # rank 0: serve HTTP on a BACKGROUND thread, run the render loop (NCCL) on the
    # MAIN thread so collectives share the same thread/device as the worker ranks.
    import asyncio
    import threading

    import uvicorn

    config = uvicorn.Config("backend.main:app", host=settings.host, port=settings.port, log_config=None)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # not the main thread

    def _serve() -> None:
        asyncio.run(server.serve())

    threading.Thread(target=_serve, name="uvicorn", daemon=True).start()
    log.info("rank 0 serving on %s:%d (render loop on main thread)", settings.host, settings.port)
    engine.run_render_server_blocking()  # type: ignore[attr-defined]  blocks forever


if __name__ == "__main__":
    main()
