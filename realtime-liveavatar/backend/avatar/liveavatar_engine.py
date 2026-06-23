"""LiveAvatarEngine — binds the real Alibaba-Quark/LiveAvatar (WanS2V-14B) code,
single-GPU AND 5-GPU realtime TPP.

Grounded in the actual repo (read from source):
  * pipeline class  : `WanS2V`
      single-GPU    : liveavatar.models.wan.causal_s2v_pipeline.WanS2V
      multi-GPU TPP : liveavatar.models.wan.causal_s2v_pipeline_tpp.WanS2V
  * config          : liveavatar.models.wan.wan_2_2.configs.WAN_CONFIGS["s2v-14B"]
  * generation      : WanS2V.generate(input_prompt, ref_image_path, audio_path,
                        sampling_steps=4, sample_solver="euler", generate_size,
                        max_area, infer_frames, num_gpus_dit, enable_vae_parallel,
                        ...) -> (video, info)
  * realtime launch : gradio_multi_gpu.sh = torchrun --nproc_per_node=5
                        minimal_inference/gradio_app.py ... (also calls generate())

REALTIME MODEL (important)
--------------------------
The repo's own multi-GPU "realtime" path calls `generate()` per clip; the 5-GPU
**TPP** pipeline makes generate() run at ~45 FPS *throughput*, so each spoken
sentence is produced near-realtime and streamed. There is no public per-frame
push/pull API. So realtime = run this app under `torchrun --nproc_per_node=5`
(see backend/main_tpp.py): rank 0 serves OpenAI+WebRTC; every rank participates in
generate()'s collective. A central render server (rank 0, one dedicated thread)
broadcasts each render job to the other ranks, then all ranks call generate()
together.

  single-GPU : GPU_COUNT=1, plain `uvicorn backend.main:app`
  realtime   : GPU_COUNT=5, `torchrun --nproc_per_node=5 backend/main_tpp.py`

NOTE: the multi-GPU NCCL dispatch must be validated on the actual 5-GPU pod
(collective ordering / init). The single-GPU path runs without torch.distributed.
"""
from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import threading
from typing import AsyncIterator, Callable, Optional

import numpy as np
import soundfile as sf

from backend.avatar.base import AvatarEngine, AvatarFrame, AvatarSession, EngineConfig
from backend.config.settings import get_settings
from backend.utils.logging import get_logger
from backend.utils.metrics import FRAMES

log = get_logger("avatar.liveavatar")

OnFrames = Callable[[list], None]


def _video_to_frames(video) -> list[np.ndarray]:
    """Convert a WanS2V `generate()` video tensor to a list of HxWx3 uint8 RGB.

    Handles common Wan layouts ([C,T,H,W] / [T,C,H,W] / [T,H,W,C]) and both
    [-1,1] and [0,1] ranges. If colors look off, match save_video() in the repo.
    """
    import torch

    if video is None:
        return []
    x = video.detach().float().cpu() if isinstance(video, torch.Tensor) else torch.as_tensor(np.asarray(video)).float()
    if x.dim() == 5:
        x = x[0]
    if x.dim() != 4:
        raise ValueError(f"unexpected video tensor shape {tuple(x.shape)}")
    shape = list(x.shape)
    if shape[0] == 3:        # (C,T,H,W)
        x = x.permute(1, 2, 3, 0)
    elif shape[1] == 3:      # (T,C,H,W)
        x = x.permute(0, 2, 3, 1)
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return [f.numpy() for f in x]


class LiveAvatarSession(AvatarSession):
    """Buffers a clause of audio; on mark_segment_end submits one render job to the
    engine's central render server and streams the returned frames."""

    def __init__(self, engine: "LiveAvatarEngine", ref_image: np.ndarray, prompt: str, config: EngineConfig) -> None:
        super().__init__(config)
        self._engine = engine
        self._prompt = prompt
        self._settings = get_settings()

        from PIL import Image

        self._tmpdir = tempfile.mkdtemp(prefix="liveavatar_", dir=self._settings.liveavatar_repo + "/tmp" if os.path.isdir(self._settings.liveavatar_repo) else None)
        self._ref_path = os.path.join(self._tmpdir, "ref.png")
        Image.fromarray(ref_image).convert("RGB").save(self._ref_path)

        self._buf: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._seg = 0
        self._frame_q: asyncio.Queue[AvatarFrame] = asyncio.Queue(maxsize=config.fps * 8)
        self._loop = asyncio.get_event_loop()

    def push_audio(self, pcm16k_mono: np.ndarray) -> None:
        with self._lock:
            self._buf.append(pcm16k_mono)

    def mark_segment_end(self) -> None:
        with self._lock:
            if not self._buf:
                return
            audio = np.concatenate(self._buf)
            self._buf.clear()
        rate = self.config.audio_rate
        if audio.size < int(0.1 * rate):
            return
        wav_path = os.path.join(self._tmpdir, f"seg_{self._seg}.wav")
        self._seg += 1
        sf.write(wav_path, audio, rate, subtype="PCM_16")
        self._engine.submit_render(self._prompt, self._ref_path, wav_path, self._on_frames)

    def _on_frames(self, frames: list) -> None:
        # runs on the engine render-server thread; hand frames to the event loop
        for rgb in frames:
            if self.closed:
                break
            frame = AvatarFrame(rgb=rgb, index=self.frame_index, pts_seconds=self.frame_index / self.config.fps)
            self.frame_index += 1
            FRAMES.labels(engine="liveavatar").inc()
            try:
                asyncio.run_coroutine_threadsafe(self._frame_q.put(frame), self._loop).result(timeout=5.0)
            except Exception:
                pass

    async def frames(self) -> AsyncIterator[AvatarFrame]:
        while not self.closed:
            yield await self._frame_q.get()

    async def close(self) -> None:
        self.closed = True


class LiveAvatarEngine(AvatarEngine):
    name = "liveavatar"

    def __init__(self) -> None:
        self._wan = None
        self._cfg = None
        self._modules = None
        self._settings = get_settings()
        self._rank = int(os.environ.get("RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._render_q: "queue.Queue" = queue.Queue()
        self._server_started = False
        self._build_lock = threading.Lock()

    # ---------- lifecycle ----------
    def is_distributed(self) -> bool:
        return self._world_size > 1

    async def warmup(self) -> None:
        if self._wan is None:
            await asyncio.get_event_loop().run_in_executor(None, self.build_blocking)

    def build_blocking(self) -> None:
        with self._build_lock:
            if self._wan is not None:
                return
            self._build_pipeline()
            if self._rank == 0:
                self.start_render_server()

    def _build_pipeline(self) -> None:
        import sys
        import torch

        s = self._settings
        if s.liveavatar_repo not in sys.path:
            sys.path.insert(0, s.liveavatar_repo)
        os.makedirs(os.path.join(s.liveavatar_repo, "tmp"), exist_ok=True)

        from liveavatar.models.wan.wan_2_2.configs import (  # type: ignore
            MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS)

        self._modules = (SIZE_CONFIGS, MAX_AREA_CONFIGS)
        self._cfg = WAN_CONFIGS[s.la_task]
        single_gpu = self._world_size <= 1

        if single_gpu:
            from liveavatar.models.wan.causal_s2v_pipeline import WanS2V  # type: ignore
        else:
            from liveavatar.models.wan.wan_2_2.distributed.util import init_distributed_group  # type: ignore
            from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V  # type: ignore

            try:
                init_distributed_group()
            except Exception as exc:
                log.warning("init_distributed_group() failed (%s); falling back to nccl init", exc)
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

        device = int(os.environ.get("LOCAL_RANK", "0"))
        log.info("rank %d/%d building WanS2V (single_gpu=%s, fp8=%s)", self._rank, self._world_size, single_gpu, s.enable_fp8)
        self._wan = WanS2V(
            config=self._cfg,
            checkpoint_dir=s.liveavatar_ckpt,
            device_id=device,
            rank=self._rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            sp_size=1,
            t5_cpu=False,
            convert_model_dtype=True,
            single_gpu=single_gpu,
            offload_kv_cache=s.offload_kv_cache,
        )
        loader = getattr(self._wan, "load_lora", None)
        if callable(loader):
            try:
                loader(os.path.join(s.liveavatar_repo, s.la_lora_path))
            except Exception as exc:
                log.warning("load_lora() failed (%s); confirm against repo", exc)
        torch.set_grad_enabled(False)
        log.info("rank %d WanS2V ready", self._rank)

    # ---------- render dispatch ----------
    def start_render_server(self) -> None:
        if self._server_started or self._rank != 0:
            return
        self._server_started = True
        threading.Thread(target=self._render_server, name="render-server", daemon=True).start()
        log.info("render server started (world_size=%d)", self._world_size)

    def submit_render(self, prompt: str, ref_path: str, audio_path: str, on_frames: OnFrames) -> None:
        self._render_q.put((prompt, ref_path, audio_path, on_frames))

    def _render_server(self) -> None:
        """Rank-0 dedicated thread. Drains render jobs; in distributed mode it first
        broadcasts the job so all ranks call generate() together."""
        while True:
            job = self._render_q.get()
            if job is None:
                if self.is_distributed():
                    self._broadcast_job(None)
                return
            prompt, ref_path, audio_path, on_frames = job
            try:
                if self.is_distributed():
                    self._broadcast_job((prompt, ref_path, audio_path))
                frames = self.render_clip(prompt, ref_path, audio_path)
            except Exception as exc:
                log.error("render failed: %s", exc, exc_info=True)
                frames = []
            try:
                on_frames(frames)
            except Exception as exc:
                log.error("on_frames callback failed: %s", exc)

    def _broadcast_job(self, job) -> None:
        import torch.distributed as dist

        buf = [job]
        dist.broadcast_object_list(buf, src=0)

    def worker_loop(self) -> None:
        """Ranks > 0: wait for broadcast jobs and participate in generate()."""
        import torch.distributed as dist

        log.info("rank %d entering worker loop", self._rank)
        while True:
            buf = [None]
            dist.broadcast_object_list(buf, src=0)
            job = buf[0]
            if job is None:
                log.info("rank %d worker loop stop", self._rank)
                return
            prompt, ref_path, audio_path = job
            try:
                self.render_clip(prompt, ref_path, audio_path)  # participate; output discarded
            except Exception as exc:
                log.error("rank %d render failed: %s", self._rank, exc)

    def render_clip(self, prompt: str, ref_image_path: str, audio_path: str) -> list[np.ndarray]:
        s = self._settings
        SIZE_CONFIGS, MAX_AREA_CONFIGS = self._modules
        video, _info = self._wan.generate(
            input_prompt=prompt,
            ref_image_path=ref_image_path,
            audio_path=audio_path,
            enable_tts=False,
            num_repeat=1,
            pose_video=None,
            generate_size=SIZE_CONFIGS[s.la_size],
            max_area=MAX_AREA_CONFIGS[s.la_size],
            infer_frames=s.la_infer_frames,
            shift=s.la_sample_shift,
            sample_solver=s.la_sample_solver,
            sampling_steps=s.la_sample_steps,
            guide_scale=s.la_guide_scale,
            seed=42,
            offload_model=s.offload_model,
            init_first_frame=True,
            num_gpus_dit=s.la_num_gpus_dit,
            enable_vae_parallel=self.is_distributed(),
            enable_online_decode=True,
        )
        return _video_to_frames(video)

    async def start_session(self, ref_image: np.ndarray, prompt: str, config: EngineConfig) -> AvatarSession:
        await self.warmup()
        return LiveAvatarSession(self, ref_image, prompt, config)
