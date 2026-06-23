"""Avatar engine factory (singleton, so the torchrun launcher and the web server
share one engine instance)."""
from __future__ import annotations

from backend.avatar.base import AvatarEngine
from backend.config.settings import get_settings

_ENGINE: AvatarEngine | None = None


def build_engine() -> AvatarEngine:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    settings = get_settings()
    if settings.avatar_engine == "liveavatar":
        from backend.avatar.liveavatar_engine import LiveAvatarEngine

        _ENGINE = LiveAvatarEngine()
    else:
        from backend.avatar.mock_engine import MockAvatarEngine

        _ENGINE = MockAvatarEngine()
    return _ENGINE
