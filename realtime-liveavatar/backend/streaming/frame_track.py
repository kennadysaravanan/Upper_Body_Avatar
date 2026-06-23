"""WebRTC video track fed by the avatar engine's frame queue."""
from __future__ import annotations

import asyncio
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import VideoFrame

from backend.avatar.base import AvatarFrame
from backend.utils.logging import get_logger

log = get_logger("streaming.video")
VIDEO_CLOCK = 90000


class AvatarVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, fps: int, width: int, height: int) -> None:
        super().__init__()
        self._fps = fps
        self._queue: asyncio.Queue[AvatarFrame] = asyncio.Queue(maxsize=fps * 2)
        self._n = 0
        self._last: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

    async def put(self, frame: AvatarFrame) -> None:
        """Bounded; drop oldest on overflow so the GPU loop never blocks."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(frame)

    async def recv(self) -> VideoFrame:
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=1.0 / self._fps * 4)
            self._last = frame.rgb
        except asyncio.TimeoutError:
            pass  # repeat last frame to keep the stream alive during gaps

        vf = VideoFrame.from_ndarray(np.ascontiguousarray(self._last), format="rgb24")
        vf.pts = int(self._n * (VIDEO_CLOCK / self._fps))
        vf.time_base = Fraction(1, VIDEO_CLOCK)
        self._n += 1
        return vf
