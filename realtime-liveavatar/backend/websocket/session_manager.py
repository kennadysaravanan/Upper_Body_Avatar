"""Per-peer conversation session + global session registry."""
from __future__ import annotations

import asyncio
import base64
import io
import uuid
from typing import Awaitable, Callable

import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from PIL import Image

from backend.avatar.base import AvatarEngine, EngineConfig
from backend.config.settings import get_settings
from backend.openai.llm import LLMClient
from backend.openai.orchestrator import Orchestrator
from backend.openai.tts import TTSClient
from backend.streaming.audio_track import AvatarAudioTrack
from backend.streaming.frame_track import AvatarVideoTrack
from backend.utils.logging import get_logger
from backend.utils.metrics import SESSIONS

log = get_logger("ws.session")

SendCb = Callable[[dict], Awaitable[None]]


def _decode_image(b64: str, size: tuple[int, int]) -> np.ndarray:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.uint8)


class ConversationSession:
    def __init__(self, engine: AvatarEngine, send: SendCb) -> None:
        self.id = uuid.uuid4().hex[:12]
        self._engine = engine
        self._send = send
        self._settings = get_settings()
        self.pc: RTCPeerConnection | None = None
        self.video = AvatarVideoTrack(
            self._settings.target_fps, self._settings.frame_width, self._settings.frame_height
        )
        self.audio = AvatarAudioTrack()
        self._avatar_session = None
        self._orch: Orchestrator | None = None
        self._tasks: list[asyncio.Task] = []

    async def init_from_hello(self, msg) -> None:
        cfg = EngineConfig(
            width=self._settings.frame_width,
            height=self._settings.frame_height,
            fps=self._settings.target_fps,
            audio_rate=self._settings.avatar_audio_rate,
        )
        ref = _decode_image(msg.avatar_image_b64, (cfg.width, cfg.height))
        self._avatar_session = await self._engine.start_session(ref, msg.prompt, cfg)

        llm = LLMClient(msg.openai_api_key, msg.llm_model)
        tts = TTSClient(msg.openai_api_key, msg.tts_model, msg.tts_voice)
        self._orch = Orchestrator(
            llm, tts, self._avatar_session, self._on_text, self.audio.push_pcm
        )

        # build peer connection
        ice = [RTCIceServer(**s) for s in self._settings.ice_servers()]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice))
        self.pc.addTrack(self.video)
        self.pc.addTrack(self.audio)

        @self.pc.on("connectionstatechange")
        async def _on_state():
            log.info("pc[%s] state=%s", self.id, self.pc.connectionState)
            if self.pc.connectionState in ("failed", "closed"):
                await self.close()

        # drain engine frames -> video track
        self._tasks.append(asyncio.create_task(self._drain_frames()))
        SESSIONS.inc()

    async def _drain_frames(self) -> None:
        try:
            async for frame in self._avatar_session.frames():
                await self.video.put(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("frame drain stopped: %s", exc)

    async def handle_offer(self, sdp: str, sdp_type: str) -> dict:
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {"sdp": self.pc.localDescription.sdp, "sdp_type": self.pc.localDescription.type}

    async def handle_user_text(self, text: str) -> None:
        if self._orch:
            self._tasks.append(asyncio.create_task(self._orch.handle_user_text(text)))

    def interrupt(self) -> None:
        if self._orch:
            self._orch.cancel()

    async def _on_text(self, delta: str, done: bool) -> None:
        await self._send({"type": "assistant_text", "delta": delta, "done": done})

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._avatar_session:
            await self._avatar_session.close()
        if self.pc:
            await self.pc.close()
        SESSIONS.dec()
        log.info("session %s closed", self.id)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}

    def add(self, s: ConversationSession) -> None:
        self._sessions[s.id] = s

    def remove(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    @property
    def count(self) -> int:
        return len(self._sessions)

    async def shutdown(self) -> None:
        await asyncio.gather(*(s.close() for s in list(self._sessions.values())), return_exceptions=True)
        self._sessions.clear()
