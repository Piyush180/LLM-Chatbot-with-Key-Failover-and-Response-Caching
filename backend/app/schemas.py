"""Request / response models matching the API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    # Capped like `message`, so an oversized prompt is a 422 instead of a
    # provider rejection that would burn every key and end in a 503.
    system_prompt: str | None = Field(None, max_length=4000)

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

    # mode="before": trim first, so the 4000 limit applies to the trimmed text;
    # blank means "no override".
    @field_validator("system_prompt", mode="before")
    @classmethod
    def _trim_prompt(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value


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
