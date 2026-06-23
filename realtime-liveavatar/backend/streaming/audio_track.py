"""WebRTC audio track fed by OpenAI TTS PCM (24kHz, s16le, mono).

Emits fixed 20ms frames on a steady clock, padding with silence when the TTS
buffer is empty, so the timeline stays continuous and A/V stays aligned.
"""
from __future__ import annotations

import asyncio
import time
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame

SAMPLE_RATE = 24000
FRAME_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000  # 480
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2


class AvatarAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buf = bytearray()
        self._lock = asyncio.Lock()
        self._n = 0
        self._start = None

    async def push_pcm(self, pcm24k_s16le: bytes) -> None:
        async with self._lock:
            self._buf.extend(pcm24k_s16le)

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.time()

        # pace to realtime
        target = self._start + self._n * (FRAME_MS / 1000)
        delay = target - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        async with self._lock:
            if len(self._buf) >= BYTES_PER_FRAME:
                chunk = bytes(self._buf[:BYTES_PER_FRAME])
                del self._buf[:BYTES_PER_FRAME]
            else:
                chunk = bytes(self._buf) + b"\x00" * (BYTES_PER_FRAME - len(self._buf))
                self._buf.clear()

        samples = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._n * SAMPLES_PER_FRAME
        frame.time_base = Fraction(1, SAMPLE_RATE)
        self._n += 1
        return frame
