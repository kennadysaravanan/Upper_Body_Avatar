"""WebSocket signaling + control channel."""
from __future__ import annotations

import json

from aiortc import RTCIceCandidate
from aiortc.sdp import candidate_from_sdp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.avatar import build_engine
from backend.config.settings import get_settings
from backend.models.schemas import HelloMsg
from backend.utils.logging import get_logger
from backend.websocket.session_manager import ConversationSession, SessionManager

log = get_logger("ws.signaling")
router = APIRouter()
manager = SessionManager()
_engine = build_engine()


def _ice_from_client(data: dict) -> RTCIceCandidate:
    raw = data.get("candidate", "")
    if raw.startswith("candidate:"):
        raw = raw[len("candidate:"):]
    cand = candidate_from_sdp(raw)
    cand.sdpMid = data.get("sdpMid")
    cand.sdpMLineIndex = data.get("sdpMLineIndex")
    return cand


@router.websocket("/ws")
async def signaling(ws: WebSocket) -> None:
    settings = get_settings()
    await ws.accept()
    if manager.count >= settings.max_sessions:
        await ws.send_text(json.dumps({"type": "error", "message": "server at capacity"}))
        await ws.close()
        return

    async def send(payload: dict) -> None:
        await ws.send_text(json.dumps(payload))

    session: ConversationSession | None = None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            mtype = msg.get("type")

            if mtype == "hello":
                hello = HelloMsg(**msg)
                session = ConversationSession(_engine, send)
                await session.init_from_hello(hello)
                manager.add(session)
                await send(
                    {"type": "ready", "session_id": session.id, "ice_servers": settings.ice_servers()}
                )

            elif mtype == "offer" and session:
                answer = await session.handle_offer(msg["sdp"], msg.get("sdp_type", "offer"))
                await send({"type": "answer", **answer})

            elif mtype == "ice" and session:
                cand = msg.get("candidate")
                if cand and cand.get("candidate"):
                    await session.pc.addIceCandidate(_ice_from_client(cand))

            elif mtype == "user_text" and session:
                await session.handle_user_text(msg["text"])

            elif mtype == "interrupt" and session:
                session.interrupt()

            else:
                await send({"type": "error", "message": f"unknown or out-of-order: {mtype}"})

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as exc:
        log.error("signaling error: %s", exc)
        try:
            await send({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if session:
            manager.remove(session.id)
            await session.close()
