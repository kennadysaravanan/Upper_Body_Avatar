"""Avatar engine factory."""
from __future__ import annotations

from backend.avatar.base import AvatarEngine
from backend.config.settings import get_settings


def build_engine() -> AvatarEngine:
    settings = get_settings()
    if settings.avatar_engine == "liveavatar":
        from backend.avatar.liveavatar_engine import LiveAvatarEngine

        return LiveAvatarEngine()
    from backend.avatar.mock_engine import MockAvatarEngine

    return MockAvatarEngine()
