"""OpenAI LLM streaming (GPT-4o / GPT-4.1 / GPT-5 family)."""
from __future__ import annotations

import time
from typing import AsyncIterator

from openai import AsyncOpenAI

from backend.utils.logging import get_logger
from backend.utils.metrics import LLM_LATENCY

log = get_logger("openai.llm")

SYSTEM_PROMPT = (
    "You are a friendly realtime talking avatar. Reply in short, natural spoken "
    "sentences (1-3 sentences). Avoid markdown, lists, emojis, or code blocks "
    "because your reply will be spoken aloud and lip-synced."
)


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        """Yield assistant text deltas as they arrive."""
        self.history.append({"role": "user", "content": user_text})
        started = time.perf_counter()
        first = True
        collected: list[str] = []

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            temperature=0.7,
            max_tokens=300,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            if first:
                LLM_LATENCY.observe(time.perf_counter() - started)
                log.info("llm first token in %.3fs", time.perf_counter() - started)
                first = False
            collected.append(delta)
            yield delta

        self.history.append({"role": "assistant", "content": "".join(collected)})
        # keep history bounded
        if len(self.history) > 21:
            self.history = [self.history[0]] + self.history[-20:]
