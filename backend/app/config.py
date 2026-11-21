"""All configuration comes from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def sqlite_path_from_url(url: str) -> str:
    """Turn a SQLAlchemy-style SQLite URL into a filesystem path.

    sqlite:////data/app.db  -> /data/app.db   (absolute)
    sqlite:///./data/app.db -> ./data/app.db  (relative)
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError(f"DATABASE_URL must start with {prefix!r}")
    path = url[len(prefix):]
    if not path:
        raise ValueError("DATABASE_URL has no file path")
    return path


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=list)
    model: str = "gemini/gemini-2.5-flash"
    database_path: str = "./data/app.db"
    allowed_origins: list[str] = field(default_factory=list)
    enable_failure_simulation: bool = False
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        database_path = sqlite_path_from_url(
            os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
        )
        return cls(
            api_keys=_split_csv(os.getenv("LLM_API_KEYS")),
            model=os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash").strip(),
            database_path=database_path,
            allowed_origins=_split_csv(
                os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
            ),
            enable_failure_simulation=_bool(os.getenv("ENABLE_FAILURE_SIMULATION")),
            default_system_prompt=os.getenv("DEFAULT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
            request_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        )
