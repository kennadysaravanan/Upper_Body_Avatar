"""Avatar engine interface.

Everything upstream (OpenAI, WebRTC) depends only on this interface, never on a
concrete engine. Swapping the mock renderer for the real 14B LiveAvatar pipeline
is a one-line config change (`AVATAR_ENGINE=liveavatar`).

Contract:
  * `start_session(ref_image, prompt)`  -> warm, ready AvatarSession
  * `session.push_audio(pcm16k_mono)`   -> non-blocking append of driving audio
  * `async for frame in session.frames()` -> RGB uint8 HxWx3 frames, in order
  * `session.close()`                   -> release resources
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np


@dataclass
class AvatarFrame:
    rgb: np.ndarray            # HxWx3 uint8
    index: int                 # monotonically increasing frame number
    pts_seconds: float         # presentation timestamp on the media clock


@dataclass
class EngineConfig:
    width: int = 480
    height: int = 480
    fps: int = 25
    audio_rate: int = 16000


class AvatarSession(abc.ABC):
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.frame_index = 0
        self.closed = False

    @abc.abstractmethod
    def push_audio(self, pcm16k_mono: np.ndarray) -> None:
        """Append float32 mono audio @ config.audio_rate. Must not block."""

    def mark_segment_end(self) -> None:
        """Signal that a spoken clause finished. Streaming engines (mock, TPP)
        may ignore this; the generate()-based LiveAvatar engine uses it to flush
        the buffered audio of one clause into a render call."""
        return None

    @abc.abstractmethod
    def frames(self) -> AsyncIterator[AvatarFrame]:
        """Async generator of rendered frames until close()."""

    @abc.abstractmethod
    async def close(self) -> None:
        ...


class AvatarEngine(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def start_session(
        self, ref_image: np.ndarray, prompt: str, config: EngineConfig
    ) -> AvatarSession:
        ...

    async def warmup(self) -> None:
        """Optionally preload weights at server startup."""
        return None
