"""Unit tests for the engine-agnostic pipeline pieces. Run: pytest -q"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from backend.avatar.audio_features import (
    SentenceAggregator,
    pcm16_to_float32,
    resample_linear,
    rms_envelope,
)
from backend.avatar.base import EngineConfig
from backend.avatar.mock_engine import MockAvatarEngine


def test_pcm16_to_float32_range():
    pcm = np.array([32767, -32768, 0], dtype=np.int16).tobytes()
    out = pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert -1.0 <= out.min() and out.max() <= 1.0


def test_resample_changes_length():
    sig = np.ones(24000, dtype=np.float32)
    out = resample_linear(sig, 24000, 16000)
    assert abs(out.size - 16000) <= 1


def test_rms_envelope_hops():
    sig = np.ones(1000, dtype=np.float32)
    env = rms_envelope(sig, hop=100)
    assert env.size == 10
    assert np.allclose(env, 1.0, atol=1e-3)


def test_sentence_aggregator_flushes_clauses():
    agg = SentenceAggregator(min_chars=5)
    out = agg.add("Hello there. How are")
    assert out == ["Hello there."]
    out2 = agg.add(" you doing?")
    assert out2 == ["How are you doing?"]


@pytest.mark.asyncio
async def test_mock_engine_emits_frames():
    engine = MockAvatarEngine()
    cfg = EngineConfig(width=128, height=128, fps=25, audio_rate=16000)
    ref = np.full((128, 128, 3), 200, dtype=np.uint8)
    session = await engine.start_session(ref, "test", cfg)
    session.push_audio(np.random.randn(16000).astype(np.float32) * 0.1)

    frames = []
    async for f in session.frames():
        frames.append(f)
        if len(f.rgb.shape) == 3 and len(frames) >= 3:
            break
    await session.close()
    assert len(frames) >= 3
    assert frames[0].rgb.shape == (128, 128, 3)
    assert frames[1].index == frames[0].index + 1
