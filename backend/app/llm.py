"""LiteLLM gateway: ordered key pool with automatic failover + LiteLLM response cache.

Flow for one request:
  1. Try key_1, key_2, ... in order via litellm.acompletion(api_key=...).
  2. LiteLLM's built-in cache is checked first on every call. Its key is
     model + messages (the api_key is not part of it), so a reply produced by
     any key is a cache hit for later identical requests.
  3. A key that raises (429 / quota / 5xx / timeout / connection error, or a
     simulated status from X-Simulate-Failure) is skipped and the next key is
     tried. If every key fails, AllProvidersUnavailable -> HTTP 503.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.caching.caching import Cache

logger = logging.getLogger("chatbot.llm")

# LiteLLM's own retries are disabled per call (num_retries=0, max_retries=0),
# so every failure surfaces here and the pool fails over itself.
litellm.suppress_debug_info = True
litellm.drop_params = True

SIMULATABLE_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""


class SimulatedProviderError(Exception):
    """Raised instead of calling the provider when X-Simulate-Failure targets a key."""

    def __init__(self, key_id: str, status_code: int) -> None:
        super().__init__(f"simulated HTTP {status_code} for {key_id}")
        self.status_code = status_code


@dataclass
class LLMResult:
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    key_id: str | None
    fallback_attempts: int


def enable_cache() -> None:
    """Turn on LiteLLM's built-in in-memory response cache."""
    litellm.cache = Cache(type="local")


def parse_simulation_header(value: str | None) -> dict[str, int]:
    """Parse 'key_1=429,key_2=500' or '*=429' into {key_id: status}.

    Malformed pairs and unsupported statuses are ignored.
    """
    result: dict[str, int] = {}
    if not value:
        return result
    for pair in value.split(","):
        key_id, sep, status = pair.strip().partition("=")
        key_id, status = key_id.strip(), status.strip()
        if not sep or not key_id or not status.isdigit():
            continue
        code = int(status)
        if code in SIMULATABLE_STATUSES:
            result[key_id] = code
    return result


def _status_of(exc: BaseException) -> str:
    code = getattr(exc, "status_code", None)
    return str(code) if code is not None else type(exc).__name__


class KeyPool:
    def __init__(self, api_keys: list[str], model: str, timeout_seconds: float) -> None:
        self._keys = list(api_keys)
        self.model = model
        self.timeout_seconds = timeout_seconds

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def key_ids(self) -> list[str]:
        return [f"key_{i}" for i in range(1, len(self._keys) + 1)]

    @property
    def cache_enabled(self) -> bool:
        return litellm.cache is not None

    async def complete(
        self, messages: list[dict[str, str]], simulate: dict[str, int] | None = None
    ) -> LLMResult:
        simulate = simulate or {}
        # While a failure is being simulated, skip the cache lookup so the
        # request really walks the key pool (the answer is still stored).
        cache_control: dict[str, Any] = {"no-cache": True} if simulate else {}

        failures = 0
        for key_id, api_key in zip(self.key_ids, self._keys):
            simulated = simulate.get(key_id, simulate.get("*"))
            try:
                if simulated is not None:
                    raise SimulatedProviderError(key_id, simulated)
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    api_key=api_key,
                    timeout=self.timeout_seconds,
                    num_retries=0,
                    max_retries=0,
                    caching=True,
                    cache=cache_control,
                )
                reply = (response.choices[0].message.content or "").strip()
                if not reply:
                    raise ValueError("provider returned an empty reply")
            except Exception as exc:  # any provider failure -> next key
                failures += 1
                # Log only the status/class: provider messages can echo the key.
                logger.warning("%s failed (%s); failing over", key_id, _status_of(exc))
                continue

            cache_hit = bool((getattr(response, "_hidden_params", None) or {}).get("cache_hit"))
            usage = getattr(response, "usage", None)
            return LLMResult(
                reply=reply,
                model=self.model,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                cache_hit=cache_hit,
                key_id=None if cache_hit else key_id,
                fallback_attempts=0 if cache_hit else failures,
            )

        logger.error("all %d keys failed", len(self._keys))
        raise AllProvidersUnavailable()
