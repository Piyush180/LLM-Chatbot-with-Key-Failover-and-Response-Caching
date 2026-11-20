# LLM Chatbot with Key Failover and Response Caching

React (Vite) frontend · FastAPI backend · LiteLLM gateway (Gemini) · SQLite storage.

```
/backend             FastAPI app (app/), pytest suite (tests/), multi-stage Dockerfile
/frontend            React + Vite app, multi-stage Dockerfile (nginx runtime)
docker-compose.yml   backend :8000 + frontend :3000, named volume app-data -> /data
Makefile             make up / make down (+ make test)
.env.example         every variable with placeholder values
render.yaml          Render blueprint for the backend
```

## 1. Live URLs

| Tier | URL |
|---|---|
| Frontend (Vercel) | `https://<your-app>.vercel.app` |
| Backend (Render) | `https://<your-service>.onrender.com` |
| Health check | `https://<your-service>.onrender.com/health` |

> The backend runs on Render's free plan: the first request after ~15 minutes idle wakes the service (~30–60 s).

## 2. Run locally

### With Docker

```bash
cp .env.example .env        # then put two real Gemini keys in LLM_API_KEYS
make up                     # backend http://localhost:8000, frontend http://localhost:3000
make down                   # stop (SQLite history in the app-data volume is kept)
make test                   # optional: run the backend tests in Docker
```

### Without Docker

Backend (Python 3.12):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
export LLM_API_KEYS="key-one,key-two" DATABASE_URL="sqlite:///./data/app.db" \
       ALLOWED_ORIGINS="http://localhost:3000" ENABLE_FAILURE_SIMULATION=true
uvicorn app.main:app --port 8000
pytest -q                                               # tests (no real keys needed)
```

Frontend (Node 22):

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev          # http://localhost:3000
```

### Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `LLM_API_KEYS` | Comma-separated key pool (≥ 2). Ids are `key_1`, `key_2`, … in order | — |
| `LLM_MODEL` | LiteLLM model name for every key | `gemini/gemini-2.0-flash` |
| `DATABASE_URL` | SQLite location | `sqlite:///./data/app.db` (Docker: `sqlite:////data/app.db`) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `http://localhost:3000` |
| `ENABLE_FAILURE_SIMULATION` | `true` enables `X-Simulate-Failure` | `false` |
| `DEFAULT_SYSTEM_PROMPT` | Used when a session has no system prompt of its own | `You are a helpful, concise assistant.` |
| `LLM_TIMEOUT_SECONDS` | Per-key provider timeout before failing over | `30` |
| `VITE_API_URL` | Frontend build time: backend base URL | `http://localhost:8000` |

### Deploying

* **Backend → Render**: New → Blueprint → pick this repo (`render.yaml`). Set the secret values `LLM_API_KEYS` and `ALLOWED_ORIGINS` (the Vercel URL) in the dashboard. `ENABLE_FAILURE_SIMULATION=true` is already set.
* **Frontend → Vercel**: import the repo with **Root Directory** `frontend` (Vite preset). Set `VITE_API_URL` to the Render URL, then deploy.

## 3. Design decisions

### Key fallback
* `backend/app/llm.py` → `KeyPool.complete()` walks the pool **in order** (`key_1`, `key_2`, …). Each attempt is one `litellm.acompletion(model, messages, api_key=<key>, num_retries=0, max_retries=0, timeout=…)` call.
* LiteLLM's own retries are switched off, so every failure comes back to the pool straight away. Any LiteLLM exception moves on to the next key. That covers `RateLimitError` (HTTP 429, including Gemini's `RESOURCE_EXHAUSTED` quota errors), 5xx (`InternalServerError`, `ServiceUnavailableError`, …), timeouts, connection errors and also auth errors. A single bad key never takes the chat down.
* `fallback_attempts` is the number of keys that failed before the one that answered. `key_id` names the key that answered; the key itself is never returned. Logs record only the key id and status code, never the provider's error message, because that message can include the key.
* When every key fails, the backend returns `503 {"detail": "all providers unavailable"}` and writes nothing to SQLite.
* **`X-Simulate-Failure`** (honoured only when `ENABLE_FAILURE_SIMULATION=true`): the header is parsed into `{key_id: status}`; `*` means every key, and a named key overrides `*`. Malformed pairs and statuses other than 429/500/502/503 are ignored. A targeted key raises a `SimulatedProviderError` instead of calling the provider, inside the same `try/except` path as a real error. That makes the simulation exercise exactly the real failover code, for that one request only.
* **Simulation bypasses the cache lookup** (LiteLLM `cache={"no-cache": True}`). Without that, a repeated message would be answered from cache and `*=429` could never produce the required 503. The reply is still written to the cache.
* An ordered loop (rather than LiteLLM's `Router`) was chosen because the contract needs a deterministic `key_id` / `fallback_attempts` and per-key failure injection, which the Router's randomised deployment selection doesn't give.

### Caching
* LiteLLM's built-in in-memory cache: `litellm.cache = Cache(type="local")`, and every call passes `caching=True`. The pinned LiteLLM release (1.110.x) no longer ships its old `disk` cache type, and the PDF allows "in-memory or disk", so in-memory it is. Trade-off: the cache starts empty after a restart, while the conversation history in SQLite persists.
* **Cache key** = LiteLLM's hash of the model plus the full message list (system prompt + history + new message). The API key is *not* part of it, so an answer produced by `key_2` after a failover is a cache hit later too. Consequences:
  * the same first message in a new session, with the same system prompt → `LLM_CACHE`;
  * any different message, history or system prompt → a new key → `LIVE_INFERENCE`.
* Gotcha found while testing: in LiteLLM 1.110.2 every cache hit imports proxy modules that need `orjson` and `soundfile`, and neither is in LiteLLM's base dependencies. Without them every hit raised `ModuleNotFoundError`, so both are pinned explicitly in `requirements.txt`, and the cache-hit tests guard against this coming back.
* A hit is detected from `response._hidden_params["cache_hit"]`. On a hit the reply text is identical, the tokens are the usage stored with the cached response, `key_id = null` and `fallback_attempts = 0`.
* `latency_ms` is measured on the server with `time.perf_counter()` from the moment the request arrives. A hit is an in-process memory lookup (a few ms) against a network round trip of hundreds of ms.

### SQLite session storage
* Stdlib `sqlite3`, one short-lived connection per operation, WAL mode. The schema is created on startup (`CREATE TABLE IF NOT EXISTS`), at the path given by `DATABASE_URL` (`sqlite:////data/app.db` in Docker, on the named volume `app-data`).
* The `messages` table is exactly the one in the spec, plus an index on `(session_id, id)`. The extra `sessions` table holds each session's `system_prompt` override.
* **Multi-turn:** the client sends only the new message. The backend loads the session's history (oldest first), adds the system prompt in front and the new user message at the end, and sends that list to the LLM.
* **Atomic writes:** a turn (user + assistant messages + system-prompt upsert) is written in one transaction, and only after the LLM succeeds. A 422 or 503 therefore never touches history.
* **System prompt:** a `system_prompt` sent with `/chat` is saved for that session and used for its later turns until a new one is sent. Without one, `DEFAULT_SYSTEM_PROMPT` applies. `GET /sessions/{id}/messages` never returns it.
* A session "exists" once it has at least one stored message; otherwise `GET /sessions/{id}/messages` → `404 {"detail": "session not found"}`.

### Errors
Every error is JSON with `detail`: 422 from FastAPI validation (missing, empty or whitespace-only fields, `message` over 4000 characters after trimming, `session_id` over 128 characters, malformed JSON), 503 when every key failed, 404 for an unknown session or route, and a generic `500 {"detail": "internal server error"}` catch-all, never a stack trace. The UI turns each of these into a readable red message in the chat.

## 4. Verification

Set `B=http://localhost:8000` (or the Render URL).

```bash
# health
curl $B/health
# -> {"status":"ok","keys_configured":2,"cache_enabled":true}

# live inference
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-1","message":"What is the capital of France?"}'
# -> resolution_source LIVE_INFERENCE, key_id key_1, fallback_attempts 0

# cache hit: same first message, new session
curl -s -X POST $B/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-2","message":"What is the capital of France?"}'
# -> LLM_CACHE, identical reply, key_id null, much lower latency_ms

# failover
curl -s -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: key_1=429' \
  -d '{"session_id":"s-3","message":"Tell me a joke"}'
# -> 200, key_id key_2, fallback_attempts 1   (same with key_1=500)

# every key failing
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -H 'X-Simulate-Failure: *=429' \
  -d '{"session_id":"s-4","message":"hello"}'
# -> 503 {"detail":"all providers unavailable"}; then GET /sessions/s-4/messages -> 404

# multi-turn + history
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-5","message":"My name is Ada."}'
curl -s -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-5","message":"What is my name?"}'
curl -s $B/sessions/s-5/messages

# validation
curl -s -i -X POST $B/chat -H 'Content-Type: application/json' -d '{"session_id":"s-6","message":"   "}'   # 422
```

**In the UI:** each assistant reply shows its own `resolution_source` (green `LIVE_INFERENCE` / amber `LLM_CACHE`), model, latency in ms, and input/output tokens. These are the numbers for that request only, never a running total. Send a message, reload the page (a new session starts) and send the same message again to see `LLM_CACHE` with a lower latency. Ask a follow-up question to see multi-turn memory; the input tokens grow as the history grows.

**Automated:** `make test` (or `pytest -q` in `backend/`) runs 40+ tests. They cover the contract shapes, cache hit/miss rules (history, system prompt, answers from a fallback key), per-request tokens, multi-turn, session separation, history and 404, every 422 case, simulated and real failover, and 503 with nothing stored. The real LiteLLM pipeline and cache run in the tests; only the network call is replaced by LiteLLM's `mock_response`.

## 5. Setup Used

* Editor: Visual Studio Code
* AI tools: Claude Code (Claude Opus 5.5) in the VS Code extension

## 6. Bonus features (if any)

* **"New chat" button**: starts a new session (a new client-generated `session_id`) without reloading the page. This makes it quick to show that sessions stay separate and that the same first message in a new session is a cache hit.
