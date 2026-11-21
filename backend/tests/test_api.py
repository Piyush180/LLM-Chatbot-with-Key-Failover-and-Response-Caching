from __future__ import annotations

import pytest

from app.llm import parse_simulation_header

from .conftest import KEYS, rate_limit, server_error

META_FIELDS = {
    "resolution_source", "model", "latency_ms", "input_tokens",
    "output_tokens", "key_id", "fallback_attempts",
}


def chat(client, session_id, message, headers=None, **extra):
    return client.post("/chat", json={"session_id": session_id, "message": message, **extra},
                       headers=headers or {})


# ---------------------------------------------------------------- health

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_health_without_keys_is_503(make_client):
    res = make_client(api_keys=[]).get("/health")
    assert res.status_code == 503
    assert "detail" in res.json()


# ---------------------------------------------------------------- chat contract

def test_live_inference_response_shape(client):
    res = chat(client, "s-1", "What is the capital of France?")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == "s-1"
    assert body["reply"] == "echo: What is the capital of France?"
    meta = body["metadata"]
    assert META_FIELDS <= meta.keys()
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "groq/openai/gpt-oss-120b"
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    for field in ("latency_ms", "input_tokens", "output_tokens"):
        assert isinstance(meta[field], int)
    assert meta["input_tokens"] > 0 and meta["output_tokens"] > 0


def test_api_key_never_leaks(client):
    text = chat(client, "s-1", "hello").text
    for key in KEYS:
        assert key not in text


# ---------------------------------------------------------------- caching

def test_same_first_message_in_new_session_is_cache_hit(client, provider):
    first = chat(client, "s-a", "What is the capital of France?").json()
    second = chat(client, "s-b", "What is the capital of France?").json()

    assert first["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second["reply"] == first["reply"]
    assert second["metadata"]["key_id"] is None
    assert second["metadata"]["fallback_attempts"] == 0
    # tokens are the usage stored with the cached response
    assert second["metadata"]["input_tokens"] == first["metadata"]["input_tokens"]
    assert second["metadata"]["output_tokens"] == first["metadata"]["output_tokens"]
    assert len(provider.calls) == 1


def test_different_message_is_never_cached(client):
    chat(client, "s-a", "What is the capital of France?")
    other = chat(client, "s-b", "What is the capital of Spain?").json()
    assert other["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_history(client):
    chat(client, "s-a", "hi")
    chat(client, "s-a", "second")
    # same new message but different history -> not a cache hit
    res = chat(client, "s-b", "second").json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_key_includes_system_prompt(client):
    chat(client, "s-a", "hi")
    res = chat(client, "s-b", "hi", system_prompt="Talk like a pirate.").json()
    assert res["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_cache_hit_after_failover_answer(client):
    # answered by key_2 -> still a cache hit later (api key is not in the cache key)
    chat(client, "s-a", "hello", headers={"X-Simulate-Failure": "key_1=429"})
    res = chat(client, "s-b", "hello").json()
    assert res["metadata"]["resolution_source"] == "LLM_CACHE"


# ---------------------------------------------------------------- tokens

def test_tokens_are_per_request_not_running_total(client):
    a = chat(client, "s-a", "aaaa").json()["metadata"]
    b = chat(client, "s-b", "bbbb").json()["metadata"]
    assert abs(a["input_tokens"] - b["input_tokens"]) <= 2


# ---------------------------------------------------------------- multi-turn

def test_multi_turn_history_is_sent_to_llm(client, provider):
    chat(client, "s-1", "My name is Ada.")
    chat(client, "s-1", "What is my name?")
    sent = provider.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "My name is Ada."
    assert sent[2]["content"] == "echo: My name is Ada."
    assert sent[3]["content"] == "What is my name?"


def test_sessions_are_separate(client, provider):
    chat(client, "s-1", "first in one")
    chat(client, "s-2", "first in two")
    sent = provider.calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]


def test_history_endpoint(client):
    chat(client, "s-1", "What is the capital of France?")
    res = client.get("/sessions/s-1/messages")
    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == "s-1"
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0] == {"role": "user", "content": "What is the capital of France?"}
    assert msgs[1]["content"] == "echo: What is the capital of France?"
    assert msgs[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert isinstance(msgs[1]["metadata"]["input_tokens"], int)
    assert isinstance(msgs[1]["metadata"]["output_tokens"], int)
    assert all(m["role"] != "system" for m in msgs)


def test_history_survives_restart(make_client):
    chat(make_client(), "s-1", "remember me")
    res = make_client().get("/sessions/s-1/messages")
    assert res.status_code == 200
    assert len(res.json()["messages"]) == 2


def test_unknown_session_is_404(client):
    res = client.get("/sessions/nope/messages")
    assert res.status_code == 404
    assert res.json() == {"detail": "session not found"}


def test_system_prompt_is_stored_for_the_session(client, provider):
    chat(client, "s-1", "hi", system_prompt="Talk like a pirate.")
    chat(client, "s-1", "again")  # no system_prompt -> stored one is used
    assert provider.calls[-1]["messages"][0] == {"role": "system", "content": "Talk like a pirate."}
    chat(client, "s-2", "hi there")
    assert provider.calls[-1]["messages"][0] == {"role": "system", "content": "You are a test assistant."}


# ---------------------------------------------------------------- validation (422)

@pytest.mark.parametrize("payload", [
    {"session_id": "s-1"},
    {"message": "hi"},
    {"session_id": "s-1", "message": ""},
    {"session_id": "s-1", "message": "    \n\t "},
    {"session_id": "s-1", "message": "x" * 4001},
    {"session_id": "", "message": "hi"},
    {"session_id": "   ", "message": "hi"},
    {"session_id": "s" * 129, "message": "hi"},
    {"session_id": 123, "message": "hi"},
])
def test_invalid_requests_are_422(client, payload):
    res = client.post("/chat", json=payload)
    assert res.status_code == 422
    assert "detail" in res.json()


def test_system_prompt_over_4000_chars_is_422_and_spends_no_key(client, provider):
    res = chat(client, "s-1", "hi", system_prompt="p" * 4001)
    assert res.status_code == 422
    assert provider.calls == []
    assert client.get("/sessions/s-1/messages").status_code == 404


def test_system_prompt_trimmed_before_length_check(client, provider):
    res = chat(client, "s-1", "hi", system_prompt="  " + "p" * 4000 + "  ")
    assert res.status_code == 200
    assert provider.calls[-1]["messages"][0]["content"] == "p" * 4000


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_only_contract_endpoints_are_exposed(client, path):
    res = client.get(path)
    assert res.status_code == 404
    assert res.json() == {"detail": "Not Found"}


def test_message_trimmed_before_length_check(client):
    res = chat(client, "s-1", "  " + "x" * 4000 + "  ")
    assert res.status_code == 200


def test_malformed_json_is_422(client):
    res = client.post("/chat", content=b"{not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_failed_validation_is_not_stored(client):
    client.post("/chat", json={"session_id": "s-bad", "message": ""})
    assert client.get("/sessions/s-bad/messages").status_code == 404


# ---------------------------------------------------------------- failover (simulated)

@pytest.mark.parametrize("header", ["key_1=429", "key_1=500", "key_1=502", "key_1=503"])
def test_simulated_key_1_failure_falls_over_to_key_2(client, header):
    res = chat(client, "s-1", "hello", headers={"X-Simulate-Failure": header})
    assert res.status_code == 200
    meta = res.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert res.json()["reply"] == "echo: hello"  # a real answer, not an error in a 200


@pytest.mark.parametrize("header", ["*=429", "key_1=429,key_2=500"])
def test_all_keys_failing_is_503_and_not_stored(client, header):
    res = chat(client, "s-1", "hello", headers={"X-Simulate-Failure": header})
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-1/messages").status_code == 404


def test_simulation_bypasses_cache(client):
    chat(client, "s-a", "hello")
    res = chat(client, "s-b", "hello", headers={"X-Simulate-Failure": "*=429"})
    assert res.status_code == 503


def test_simulation_is_per_request(client):
    chat(client, "s-1", "one", headers={"X-Simulate-Failure": "key_1=429"})
    res = chat(client, "s-1", "two").json()
    assert res["metadata"]["key_id"] == "key_1"


def test_simulation_ignored_when_disabled(make_client):
    c = make_client(enable_failure_simulation=False)
    res = chat(c, "s-1", "hello", headers={"X-Simulate-Failure": "*=429"})
    assert res.status_code == 200
    assert res.json()["metadata"]["key_id"] == "key_1"


# ---------------------------------------------------------------- failover (real provider errors)

def test_real_rate_limit_falls_over(client, provider):
    provider.failing[KEYS[0]] = rate_limit()
    res = chat(client, "s-1", "hello")
    assert res.status_code == 200
    assert res.json()["metadata"]["key_id"] == "key_2"
    assert res.json()["metadata"]["fallback_attempts"] == 1


def test_real_errors_on_every_key_is_503(client, provider):
    provider.failing[KEYS[0]] = rate_limit()
    provider.failing[KEYS[1]] = server_error()
    res = chat(client, "s-1", "hello")
    assert res.status_code == 503
    assert res.json() == {"detail": "all providers unavailable"}


# ---------------------------------------------------------------- header parsing

def test_parse_simulation_header():
    assert parse_simulation_header("key_1=429") == {"key_1": 429}
    assert parse_simulation_header("key_1=429, key_2=500") == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("*=503") == {"*": 503}
    assert parse_simulation_header("key_1=404,garbage,=500,key_2=abc") == {}
    assert parse_simulation_header(None) == {}
