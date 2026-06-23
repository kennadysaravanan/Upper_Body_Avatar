"""OpenAI streaming Text-To-Speech producing raw PCM (24 kHz, 16-bit, mono).

PCM is chosen deliberately: it needs no decoder, so the path TTS -> avatar audio
features -> WebRTC has zero codec hops, which minimises latency.
"""
from __future__ import annotations

import time
from typing import AsyncIterator

from openai import AsyncOpenAI

from backend.utils.logging import get_logger
from backend.utils.metrics import TTS_LATENCY

log = get_logger("openai.tts")


class TTSClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-tts", voice: str = "alloy") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.voice = voice

    async def stream_pcm(self, text: str, chunk_size: int = 4800) -> AsyncIterator[bytes]:
        """Stream 24kHz s16le mono PCM for `text`. ~100ms per 4800-byte chunk."""
        if not text.strip():
            return
        started = time.perf_counter()
        first = True
        async with self._client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="pcm",  # 24kHz, 16-bit, mono, little-endian
        ) as response:
            async for chunk in response.iter_bytes(chunk_size):
                if not chunk:
                    continue
                if first:
                    TTS_LATENCY.observe(time.perf_counter() - started)
                    log.info("tts first chunk in %.3fs", time.perf_counter() - started)
                    first = False
                yield chunk
