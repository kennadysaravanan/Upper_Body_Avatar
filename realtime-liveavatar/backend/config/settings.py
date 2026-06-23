"""Central configuration. All values overridable via environment / .env.

Note: the OpenAI API key and model are supplied *per session* by the user from
the browser (per the product spec) and are NOT read from here, except for an
optional server-side fallback key used only for health checks.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # ---- server ----
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ---- avatar engine ----
    # "mock"       -> CPU/GPU-free animated placeholder; whole stack runs anywhere.
    # "liveavatar" -> real WanS2V-14B TPP pipeline (needs the GPU pod + weights).
    avatar_engine: Literal["mock", "liveavatar"] = "mock"
    liveavatar_repo: str = "/workspace/LiveAvatar"
    liveavatar_ckpt: str = "/workspace/LiveAvatar/ckpt/Wan2.2-S2V-14B"
    gpu_count: int = 1                       # 5 for the TPP realtime path
    enable_fp8: bool = False
    enable_compile: bool = False
    offload_model: bool = True               # offload weights between steps (VRAM)
    offload_kv_cache: bool = False

    # ---- liveavatar generate() params (match the repo's single-gpu script) ----
    la_task: str = "s2v-14B"
    la_size: str = "704*384"                 # WAN_CONFIGS size key
    la_infer_frames: int = 48                # frames per generated clip
    la_sample_steps: int = 4                 # distilled
    la_sample_solver: str = "euler"
    la_sample_shift: float = 5.0
    la_guide_scale: float = 1.0
    la_native_fps: int = 16                  # WanS2V output fps; set TARGET_FPS to match
    la_min_segment_seconds: float = 1.0      # min audio buffered before a generate() call
    la_lora_path: str = "ckpt/LiveAvatar"
    la_num_gpus_dit: int = 4                  # DiT GPUs in the 5-GPU TPP layout (4 DiT + 1 VAE)

    # ---- video / media ----
    frame_width: int = 480
    frame_height: int = 480
    target_fps: int = 25                     # WebRTC pacing target
    audio_sample_rate: int = 24000           # OpenAI TTS pcm output rate
    avatar_audio_rate: int = 16000           # wav2vec2 input rate

    # ---- openai defaults (user can override per session) ----
    default_llm_model: str = "gpt-4o-mini"
    default_tts_model: str = "gpt-4o-mini-tts"
    default_tts_voice: str = "alloy"
    default_stt_model: str = "gpt-4o-mini-transcribe"
    openai_health_key: str | None = None     # optional, server-side only

    # ---- infra ----
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://avatar:avatar@localhost:5432/avatar"
    enable_redis: bool = False
    enable_postgres: bool = False

    # ---- webrtc ----
    stun_urls: list[str] = Field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    turn_url: str | None = None              # single TURN url (back-compat)
    turn_urls: list[str] = Field(default_factory=list)  # multiple (udp/tcp/tls)
    turn_user: str | None = None
    turn_password: str | None = None
    max_sessions: int = 8

    def ice_servers(self) -> list[dict]:
        servers: list[dict] = []
        if self.stun_urls:
            servers.append({"urls": self.stun_urls})
        turns = list(self.turn_urls)
        if self.turn_url and self.turn_url not in turns:
            turns.append(self.turn_url)
        if turns:
            servers.append(
                {"urls": turns, "username": self.turn_user, "credential": self.turn_password}
            )
        return servers


@lru_cache
def get_settings() -> Settings:
    return Settings()
