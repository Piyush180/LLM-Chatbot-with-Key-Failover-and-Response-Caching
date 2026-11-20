"""Request / response models matching the API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    system_prompt: str | None = None

    @field_validator("session_id")
    @classmethod
    def _session_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value

    # mode="before" so trimming happens before the 1-4000 length check.
    @field_validator("message", mode="before")
    @classmethod
    def _trim_message(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("system_prompt")
    @classmethod
    def _empty_prompt_is_none(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    cache_enabled: bool
