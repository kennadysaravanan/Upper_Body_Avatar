"""OpenAI Speech-To-Text (optional voice-in path).

Used when the browser sends mic audio instead of typed text. Accepts a complete
utterance buffer (WAV bytes) and returns the transcript.
"""
from __future__ import annotations

import io

from openai import AsyncOpenAI

from backend.utils.logging import get_logger

log = get_logger("openai.stt")


class STTClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-transcribe") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def transcribe_wav(self, wav_bytes: bytes) -> str:
        buf = io.BytesIO(wav_bytes)
        buf.name = "utterance.wav"
        result = await self._client.audio.transcriptions.create(
            model=self.model,
            file=buf,
        )
        return result.text.strip()
