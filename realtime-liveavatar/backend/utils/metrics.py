"""Prometheus metrics. No-ops gracefully if prometheus_client is missing."""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

    SESSIONS = Gauge("avatar_active_sessions", "Active avatar sessions")
    FRAMES = Counter("avatar_frames_total", "Frames generated", ["engine"])
    LLM_LATENCY = Histogram("avatar_llm_first_token_seconds", "GPT first-token latency")
    TTS_LATENCY = Histogram("avatar_tts_first_chunk_seconds", "TTS first-chunk latency")
    FRAME_LATENCY = Histogram("avatar_first_frame_seconds", "Time-to-first-frame latency")
    _ENABLED = True
except Exception:  # pragma: no cover
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"

    class _Noop:
        def labels(self, *_a, **_k):
            return self

        def inc(self, *_a, **_k):
            pass

        def dec(self, *_a, **_k):
            pass

        def set(self, *_a, **_k):
            pass

        def observe(self, *_a, **_k):
            pass

        def time(self):
            class _Ctx:
                def __enter__(self_):
                    return self_

                def __exit__(self_, *a):
                    return False

            return _Ctx()

    SESSIONS = FRAMES = LLM_LATENCY = TTS_LATENCY = FRAME_LATENCY = _Noop()

    def generate_latest() -> bytes:  # type: ignore
        return b"# metrics disabled (prometheus_client not installed)\n"


def metrics_enabled() -> bool:
    return _ENABLED
