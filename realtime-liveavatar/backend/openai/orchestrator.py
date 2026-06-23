"""Conversation orchestrator: user text -> GPT -> TTS -> (audio sink + avatar).

This is the heart of the realtime pipeline. It maximises overlap:
  * GPT streams tokens; we flush at clause boundaries (SentenceAggregator).
  * Each clause is sent to TTS immediately and streamed as PCM.
  * Each PCM chunk is (a) pushed to the WebRTC audio track for playback and
    (b) resampled to 16kHz and pushed to the avatar engine to drive motion.
So the avatar can start moving while GPT is still generating later sentences.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from backend.avatar.audio_features import (
    SentenceAggregator,
    pcm16_to_float32,
    resample_linear,
)
from backend.avatar.base import AvatarSession
from backend.config.settings import get_settings
from backend.openai.llm import LLMClient
from backend.openai.tts import TTSClient
from backend.utils.logging import get_logger

log = get_logger("openai.orchestrator")

# callbacks supplied by the conversation session
TextDeltaCb = Callable[[str, bool], Awaitable[None]]   # (delta, done)
AudioSinkCb = Callable[[bytes], Awaitable[None]]        # 24kHz s16le PCM for playback


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient,
        tts: TTSClient,
        avatar: AvatarSession,
        on_text: TextDeltaCb,
        on_audio: AudioSinkCb,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._avatar = avatar
        self._on_text = on_text
        self._on_audio = on_audio
        self._settings = get_settings()
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self._cancel.set()

    async def handle_user_text(self, text: str) -> None:
        self._cancel.clear()
        agg = SentenceAggregator()
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # consumer: turn clauses into speech + motion, in order
        speaker = asyncio.create_task(self._speak_loop(tts_queue))

        try:
            async for delta in self._llm.stream_reply(text):
                if self._cancel.is_set():
                    break
                await self._on_text(delta, False)
                for clause in agg.add(delta):
                    await tts_queue.put(clause)
            tail = agg.flush()
            if tail and not self._cancel.is_set():
                await tts_queue.put(tail)
        finally:
            await tts_queue.put(None)  # sentinel: end of speech
            await speaker
            await self._on_text("", True)

    async def _speak_loop(self, queue: "asyncio.Queue[str | None]") -> None:
        src_rate = self._settings.audio_sample_rate    # 24000 (TTS)
        dst_rate = self._settings.avatar_audio_rate    # 16000 (avatar)
        while True:
            clause = await queue.get()
            if clause is None:
                return
            if self._cancel.is_set():
                continue
            try:
                async for pcm in self._tts.stream_pcm(clause):
                    if self._cancel.is_set():
                        break
                    # 1) playback (24kHz)
                    await self._on_audio(pcm)
                    # 2) drive avatar motion (resample 24k -> 16k float32)
                    f32 = pcm16_to_float32(pcm)
                    self._avatar.push_audio(resample_linear(f32, src_rate, dst_rate))
                # clause finished speaking -> let generate()-based engines render it
                self._avatar.mark_segment_end()
            except Exception as exc:  # one clause failing must not kill the turn
                log.error("tts/clause failed: %s", exc)
