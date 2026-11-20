"""Test fixtures.

The real LiteLLM pipeline (including its in-memory cache) runs in every test; only the
network call is replaced: a wrapper around litellm.acompletion injects LiteLLM's
own `mock_response`, or raises a LiteLLM exception for keys marked as failing.
"""

from __future__ import annotations

import litellm
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

KEYS = ["sk-test-key-one", "sk-test-key-two"]


class FakeProvider:
    def __init__(self) -> None:
        self.failing: dict[str, Exception] = {}  # api_key -> exception to raise
        self.calls: list[dict] = []  # provider calls that actually ran (cache misses)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = litellm.acompletion

        async def fake_acompletion(**kwargs):
            api_key = kwargs["api_key"]
            if api_key in self.failing:
                raise self.failing[api_key]
            last_user = kwargs["messages"][-1]["content"]
            response = await original(**kwargs, mock_response=f"echo: {last_user}")
            if not (response._hidden_params or {}).get("cache_hit"):
                self.calls.append(kwargs)
            return response

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)


def rate_limit() -> Exception:
    return litellm.RateLimitError(message="quota exceeded", llm_provider="gemini", model="gemini-2.0-flash")


def server_error() -> Exception:
    return litellm.InternalServerError(message="boom", llm_provider="gemini", model="gemini-2.0-flash")


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> FakeProvider:
    fake = FakeProvider()
    fake.install(monkeypatch)
    return fake


def make_settings(tmp_path, **overrides) -> Settings:
    values = dict(
        api_keys=KEYS,
        model="gemini/gemini-2.0-flash",
        database_path=str(tmp_path / "app.db"),
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=True,
        default_system_prompt="You are a test assistant.",
        request_timeout_seconds=5,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client(tmp_path, provider):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


@pytest.fixture
def make_client(tmp_path, provider):
    clients = []

    def factory(**overrides):
        c = TestClient(create_app(make_settings(tmp_path, **overrides)))
        c.__enter__()
        clients.append(c)
        return c

    yield factory
    for c in clients:
        c.__exit__(None, None, None)
