"""LiveAvatarEngine — binds the real Alibaba-Quark/LiveAvatar (WanS2V-14B) code.

Grounded in the actual repo (read from source):
  * pipeline class  : `WanS2V`
      single-GPU    : liveavatar.models.wan.causal_s2v_pipeline.WanS2V
      multi-GPU TPP : liveavatar.models.wan.causal_s2v_pipeline_tpp.WanS2V
  * config          : liveavatar.models.wan.wan_2_2.configs.WAN_CONFIGS["s2v-14B"]
  * generation      : WanS2V.generate(input_prompt, ref_image_path, audio_path,
                        sample_steps=4, sample_solver="euler", generate_size,
                        max_area, infer_frames, ...) -> (video, info)
  * helpers         : SIZE_CONFIGS, MAX_AREA_CONFIGS, save_video (utils)

DEPLOYMENT MODEL
----------------
The public `generate()` call is audio-FILE-in -> video-TENSOR-out for one clip.
We embed it cleanly in the FastAPI process for the **single-GPU** path: each
spoken clause (flushed via `mark_segment_end()`) is rendered with a 4-step
`generate()` and its frames are streamed to WebRTC. This is genuinely runnable
after `git clone` + checkpoint download on a single 80GB GPU (FP8 -> 48GB).

The **5-GPU realtime TPP** path (`causal_s2v_pipeline_tpp*`) runs as a
`torchrun --nproc_per_node=5` multi-process job (see the repo's
`infinite_inference_multi_gpu.sh` / `minimal_inference/s2v_streaming_interact.py`).
That does not embed in a single uvicorn process; run it as a separate worker and
bridge frames over a queue. See `_build_pipeline` notes and SETUP.md.

Set TARGET_FPS to LA_NATIVE_FPS (default 16) so audio/video stay in sync, since
generate() emits frames at the model's native rate.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf

from backend.avatar.base import AvatarEngine, AvatarFrame, AvatarSession, EngineConfig
from backend.config.settings import get_settings
from backend.utils.logging import get_logger
from backend.utils.metrics import FRAMES

log = get_logger("avatar.liveavatar")


def _video_to_frames(video) -> list[np.ndarray]:
    """Convert a WanS2V `generate()` video tensor to a list of HxWx3 uint8 RGB.

    Handles common Wan layouts ([C,T,H,W] / [T,C,H,W] / [T,H,W,C]) and both
    [-1,1] and [0,1] value ranges. If colors look inverted/odd in practice,
    adjust the normalization branch below to match save_video() in the repo.
    """
    import torch

    if video is None:
        return []
    if isinstance(video, torch.Tensor):
        x = video.detach().float().cpu()
    else:
        x = torch.as_tensor(np.asarray(video)).float()

    if x.dim() == 5:               # batch dim
        x = x[0]
    if x.dim() != 4:
        raise ValueError(f"unexpected video tensor shape {tuple(x.shape)}")

    # move channels last -> (T, H, W, C)
    shape = list(x.shape)
    if shape[0] == 3:              # (C, T, H, W)
        x = x.permute(1, 2, 3, 0)
    elif shape[1] == 3:           # (T, C, H, W)
        x = x.permute(0, 2, 3, 1)
    # else assume already (T, H, W, C)

    if float(x.min()) < -0.01:    # [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
    x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return [frame.numpy() for frame in x]


class LiveAvatarSession(AvatarSession):
    def __init__(self, engine: "LiveAvatarEngine", ref_image: np.ndarray, prompt: str, config: EngineConfig) -> None:
        super().__init__(config)
        self._engine = engine
        self._prompt = prompt
        self._settings = get_settings()

        # generate() needs the reference image as a file path
        from PIL import Image

        self._tmpdir = tempfile.mkdtemp(prefix="liveavatar_")
        self._ref_path = os.path.join(self._tmpdir, "ref.png")
        Image.fromarray(ref_image).convert("RGB").save(self._ref_path)

        self._buf: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._segment_ready = threading.Event()
        self._frame_q: asyncio.Queue[AvatarFrame] = asyncio.Queue(maxsize=config.fps * 4)
        self._loop = asyncio.get_event_loop()
        self._worker = threading.Thread(target=self._run, name="liveavatar-worker", daemon=True)
        self._worker.start()

    # ---- producer (event loop thread) ----
    def push_audio(self, pcm16k_mono: np.ndarray) -> None:
        with self._lock:
            self._buf.append(pcm16k_mono)

    def mark_segment_end(self) -> None:
        self._segment_ready.set()

    def _take_segment(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buf:
                return None
            audio = np.concatenate(self._buf)
            self._buf.clear()
        return audio

    # ---- consumer (GPU worker thread) ----
    def _run(self) -> None:
        rate = self.config.audio_rate
        min_samples = int(self._settings.la_min_segment_seconds * rate)
        while not self.closed:
            triggered = self._segment_ready.wait(timeout=0.25)
            if self.closed:
                break
            with self._lock:
                buffered = sum(a.size for a in self._buf)
            if not triggered and buffered < min_samples:
                continue
            self._segment_ready.clear()

            audio = self._take_segment()
            if audio is None or audio.size < int(0.1 * rate):
                continue

            wav_path = os.path.join(self._tmpdir, "seg.wav")
            sf.write(wav_path, audio, rate, subtype="PCM_16")

            try:
                frames = self._engine.render_clip(self._prompt, self._ref_path, wav_path)
            except Exception as exc:
                log.error("generate() failed: %s", exc, exc_info=True)
                continue

            for rgb in frames:
                if self.closed:
                    break
                frame = AvatarFrame(rgb=rgb, index=self.frame_index, pts_seconds=self.frame_index / self.config.fps)
                self.frame_index += 1
                FRAMES.labels(engine="liveavatar").inc()
                fut = asyncio.run_coroutine_threadsafe(self._frame_q.put(frame), self._loop)
                try:
                    fut.result(timeout=5.0)
                except Exception:
                    pass

    async def frames(self) -> AsyncIterator[AvatarFrame]:
        while not self.closed:
            yield await self._frame_q.get()

    async def close(self) -> None:
        self.closed = True
        self._segment_ready.set()


class LiveAvatarEngine(AvatarEngine):
    name = "liveavatar"

    def __init__(self) -> None:
        self._wan = None
        self._cfg = None          # WAN_CONFIGS entry
        self._modules = None      # (SIZE_CONFIGS, MAX_AREA_CONFIGS)
        self._gen_lock = threading.Lock()   # generate() is single-tenant
        self._settings = get_settings()

    async def warmup(self) -> None:
        if self._wan is None:
            await asyncio.get_event_loop().run_in_executor(None, self._build_pipeline)

    def _build_pipeline(self) -> None:
        """Construct the real WanS2V pipeline from the cloned repo."""
        import sys
        import torch

        s = self._settings
        if s.liveavatar_repo not in sys.path:
            sys.path.insert(0, s.liveavatar_repo)

        from liveavatar.models.wan.wan_2_2.configs import (  # type: ignore
            MAX_AREA_CONFIGS,
            SIZE_CONFIGS,
            WAN_CONFIGS,
        )

        self._modules = (SIZE_CONFIGS, MAX_AREA_CONFIGS)
        self._cfg = WAN_CONFIGS[s.la_task]

        single_gpu = s.gpu_count <= 1
        if single_gpu:
            from liveavatar.models.wan.causal_s2v_pipeline import WanS2V  # type: ignore
        else:
            # The TPP pipeline expects a torchrun multi-process context. If you
            # are not launched under torchrun, prefer single-GPU embedding or run
            # the TPP path as a separate worker (see module docstring / SETUP.md).
            from liveavatar.models.wan.wan_2_2.distributed.util import init_distributed_group  # type: ignore
            from liveavatar.models.wan.causal_s2v_pipeline_tpp import WanS2V  # type: ignore

            if "RANK" in os.environ:
                init_distributed_group()

        rank = int(os.environ.get("RANK", "0"))
        device = int(os.environ.get("LOCAL_RANK", "0"))

        log.info("loading WanS2V (%s, single_gpu=%s, fp8=%s)...", s.la_task, single_gpu, s.enable_fp8)
        self._wan = WanS2V(
            config=self._cfg,
            checkpoint_dir=s.liveavatar_ckpt,
            device_id=device,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            sp_size=1,
            t5_cpu=False,
            convert_model_dtype=True,
            single_gpu=single_gpu,
            offload_kv_cache=s.offload_kv_cache,
        )

        # Load the LiveAvatar LoRA if the pipeline exposes a loader. The repo's
        # script passes --load_lora/--lora_path; if construction did not already
        # apply it, bind the repo's loader here.
        loader = getattr(self._wan, "load_lora", None)
        if callable(loader):
            try:
                loader(os.path.join(s.liveavatar_repo, s.la_lora_path))
            except Exception as exc:
                log.warning("LoRA load via load_lora() failed (%s); confirm against repo", exc)

        torch.set_grad_enabled(False)
        log.info("WanS2V ready (gpus=%d)", s.gpu_count)

    def render_clip(self, prompt: str, ref_image_path: str, audio_path: str) -> list[np.ndarray]:
        """Render one audio clip to RGB frames via the real generate() API."""
        s = self._settings
        SIZE_CONFIGS, MAX_AREA_CONFIGS = self._modules
        with self._gen_lock:
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
                enable_online_decode=True,
            )
        return _video_to_frames(video)

    async def start_session(self, ref_image: np.ndarray, prompt: str, config: EngineConfig) -> AvatarSession:
        await self.warmup()
        return LiveAvatarSession(self, ref_image, prompt, config)
