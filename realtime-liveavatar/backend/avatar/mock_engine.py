"""MockAvatarEngine — a real, runnable renderer with NO GPU and NO 14B model.

It animates the uploaded photo: mouth openness tracks audio RMS, plus periodic
eye-blinks and a subtle head bob. It exists so the *entire* platform (OpenAI ->
TTS -> audio features -> frames -> WebRTC -> browser) runs end-to-end on a laptop,
and so CI / smoke tests don't need 5x H800 GPUs. Production swaps in
LiveAvatarEngine via `AVATAR_ENGINE=liveavatar`; nothing else changes.
"""
from __future__ import annotations

import asyncio
import math
from collections import deque
from typing import AsyncIterator

import numpy as np
from PIL import Image, ImageDraw

from backend.avatar.base import AvatarEngine, AvatarFrame, AvatarSession, EngineConfig
from backend.avatar.audio_features import rms_envelope
from backend.utils.logging import get_logger
from backend.utils.metrics import FRAMES

log = get_logger("avatar.mock")


class MockSession(AvatarSession):
    def __init__(self, base_rgb: np.ndarray, config: EngineConfig) -> None:
        super().__init__(config)
        self._base = base_rgb
        self._audio = deque()  # type: deque[float]  # RMS values, one per frame
        self._lock = asyncio.Lock()
        self._t0 = None

    def push_audio(self, pcm16k_mono: np.ndarray) -> None:
        # one RMS value per output frame: hop = audio_rate / fps
        hop = max(1, self.config.audio_rate // self.config.fps)
        env = rms_envelope(pcm16k_mono, hop)
        # normalise into a 0..1 mouth-open factor
        for v in env:
            self._audio.append(min(1.0, v * 6.0))

    def _render(self, openness: float, frame_idx: int) -> np.ndarray:
        h, w = self.config.height, self.config.width
        img = Image.fromarray(self._base).convert("RGB").resize((w, h))

        # subtle head bob
        bob = int(2 * math.sin(frame_idx / 6.0))
        img = img.transform((w, h), Image.AFFINE, (1, 0, 0, 0, 1, -bob), resample=Image.BILINEAR)
        draw = ImageDraw.Draw(img)

        # mouth: dark ellipse in the lower-center, height tracks audio
        cx, cy = w // 2, int(h * 0.72)
        mw = int(w * 0.16)
        mh = int(2 + openness * h * 0.10)
        draw.ellipse([cx - mw, cy - mh, cx + mw, cy + mh], fill=(60, 30, 30))

        # eye blink every ~3s for ~4 frames
        blink = (frame_idx % (self.config.fps * 3)) < 4
        if blink:
            skin = tuple(int(c) for c in self._base.reshape(-1, 3).mean(axis=0))
            ey = int(h * 0.42)
            for ex in (int(w * 0.37), int(w * 0.63)):
                draw.ellipse([ex - 24, ey - 6, ex + 24, ey + 6], fill=skin)

        return np.asarray(img, dtype=np.uint8)

    async def frames(self) -> AsyncIterator[AvatarFrame]:
        loop = asyncio.get_event_loop()
        self._t0 = loop.time()
        period = 1.0 / self.config.fps
        next_t = self._t0
        while not self.closed:
            openness = self._audio.popleft() if self._audio else 0.0
            rgb = self._render(openness, self.frame_index)
            pts = self.frame_index * period
            yield AvatarFrame(rgb=rgb, index=self.frame_index, pts_seconds=pts)
            FRAMES.labels(engine="mock").inc()
            self.frame_index += 1
            next_t += period
            sleep = next_t - loop.time()
            if sleep > 0:
                await asyncio.sleep(sleep)

    async def close(self) -> None:
        self.closed = True


class MockAvatarEngine(AvatarEngine):
    name = "mock"

    async def start_session(
        self, ref_image: np.ndarray, prompt: str, config: EngineConfig
    ) -> AvatarSession:
        log.info("mock session start: prompt=%r size=%dx%d", prompt[:40], config.width, config.height)
        return MockSession(ref_image, config)
