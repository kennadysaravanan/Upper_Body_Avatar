"""Pydantic models for the WebSocket control protocol and REST payloads."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---- client -> server ----
class HelloMsg(BaseModel):
    type: Literal["hello"] = "hello"
    openai_api_key: str = Field(min_length=10)
    llm_model: str = "gpt-4o"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    avatar_image_b64: str                     # data URL or raw base64 PNG/JPG
    prompt: str = "A friendly person speaking to the camera."


class OfferMsg(BaseModel):
    type: Literal["offer"] = "offer"
    sdp: str
    sdp_type: str = "offer"


class IceMsg(BaseModel):
    type: Literal["ice"] = "ice"
    candidate: dict[str, Any]


class UserTextMsg(BaseModel):
    type: Literal["user_text"] = "user_text"
    text: str = Field(min_length=1)


class InterruptMsg(BaseModel):
    type: Literal["interrupt"] = "interrupt"


ClientMsg = HelloMsg | OfferMsg | IceMsg | UserTextMsg | InterruptMsg


# ---- server -> client ----
class ReadyMsg(BaseModel):
    type: Literal["ready"] = "ready"
    session_id: str
    ice_servers: list[dict]


class AnswerMsg(BaseModel):
    type: Literal["answer"] = "answer"
    sdp: str
    sdp_type: str = "answer"


class AssistantTextMsg(BaseModel):
    type: Literal["assistant_text"] = "assistant_text"
    delta: str = ""
    done: bool = False


class StatusMsg(BaseModel):
    type: Literal["status"] = "status"
    state: str
    detail: Optional[str] = None


class ErrorMsg(BaseModel):
    type: Literal["error"] = "error"
    message: str
