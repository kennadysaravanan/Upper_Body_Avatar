"""OpenAI layer tests with a stubbed client (no network, no API key needed)."""
from __future__ import annotations

import types

import pytest

from backend.openai.llm import LLMClient


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


async def _fake_stream(*_a, **_k):
    async def gen():
        for tok in ["Hello", ", ", "world", "."]:
            yield _Chunk(tok)

    return gen()


@pytest.mark.asyncio
async def test_llm_stream_collects(monkeypatch):
    client = LLMClient(api_key="sk-test-xxxxxxxx", model="gpt-4o")
    client._client.chat.completions.create = _fake_stream  # type: ignore

    out = []
    async for delta in client.stream_reply("hi"):
        out.append(delta)
    assert "".join(out) == "Hello, world."
    # history updated with assistant turn
    assert client.history[-1]["role"] == "assistant"
    assert client.history[-1]["content"] == "Hello, world."
