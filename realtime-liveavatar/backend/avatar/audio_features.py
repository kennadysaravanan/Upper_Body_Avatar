"""Audio helpers shared by every engine: PCM decode, resample, sentence flush."""
from __future__ import annotations

import re

import numpy as np

_SENTENCE_END = re.compile(r"([.!?;:,]|\n)")


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert s16le PCM bytes to float32 mono in [-1, 1]."""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Lightweight linear resampler (no scipy dependency)."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.size / src_rate
    dst_len = int(round(duration * dst_rate))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, audio.size - 1, num=dst_len)
    return np.interp(src_idx, np.arange(audio.size), audio).astype(np.float32)


def rms_envelope(audio: np.ndarray, hop: int) -> np.ndarray:
    """Per-hop RMS energy — drives the mock engine's mouth opening."""
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    n = max(1, audio.size // hop)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = audio[i * hop : (i + 1) * hop]
        out[i] = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
    return out


class SentenceAggregator:
    """Buffers LLM token deltas and flushes at clause boundaries so TTS can start
    speaking the first clause before the full reply is generated."""

    def __init__(self, min_chars: int = 12) -> None:
        self._buf = ""
        self._min_chars = min_chars

    def add(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            end = match.end()
            candidate = self._buf[:end].strip()
            if len(candidate) >= self._min_chars:
                out.append(candidate)
                self._buf = self._buf[end:]
            else:
                break
        return out

    def flush(self) -> str:
        rest = self._buf.strip()
        self._buf = ""
        return rest
