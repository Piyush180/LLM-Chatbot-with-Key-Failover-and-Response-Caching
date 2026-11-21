"""FastAPI app: GET /health, POST /chat, GET /sessions/{session_id}/messages."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .db import Database
from .llm import (AllProvidersUnavailable, KeyPool, enable_cache, parse_simulation_header,
                  warm_up_cache)
from .schemas import ChatMetadata, ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatbot")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = Database(settings.database_path)
    pool = KeyPool(settings.api_keys, settings.model, settings.request_timeout_seconds)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        db.init()
        enable_cache()
        await warm_up_cache(settings.model)
        logger.info(
            "started: model=%s keys=%d simulation=%s",
            settings.model, len(pool), settings.enable_failure_simulation,
        )
        yield

    # The contract allows exactly three endpoints, so FastAPI's built-in
    # /docs, /redoc and /openapi.json are switched off.
    app = FastAPI(title="LLM Chatbot", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.db = db
    app.state.pool = pool

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # ---- error handling: always JSON with a `detail` field, never a stack trace
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                            headers=getattr(exc, "headers", None))

    @app.exception_handler(AllProvidersUnavailable)
    async def _all_failed(_: Request, __: AllProvidersUnavailable) -> JSONResponse:
        return JSONResponse({"detail": "all providers unavailable"}, status_code=503)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error: %s", type(exc).__name__)
        return JSONResponse({"detail": "internal server error"}, status_code=500)

    # ---- routes
    @app.get("/health", response_model=HealthResponse)
    async def health() -> JSONResponse | HealthResponse:
        db_ok = await run_in_threadpool(db.ping)
        if not db_ok or len(pool) == 0:
            reason = "database unreachable" if not db_ok else "no API keys configured"
            return JSONResponse({"detail": reason}, status_code=503)
        return HealthResponse(status="ok", keys_configured=len(pool), cache_enabled=pool.cache_enabled)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        x_simulate_failure: Annotated[str | None, Header()] = None,
    ) -> ChatResponse:
        started = time.perf_counter()

        simulate = (
            parse_simulation_header(x_simulate_failure)
            if settings.enable_failure_simulation
            else {}
        )

        stored_prompt = await run_in_threadpool(db.get_system_prompt, body.session_id)
        system_prompt = body.system_prompt or stored_prompt or settings.default_system_prompt
        history = await run_in_threadpool(db.get_history, body.session_id)

        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": body.message})

        result = await pool.complete(messages, simulate)  # raises AllProvidersUnavailable

        metadata = ChatMetadata(
            resolution_source="LLM_CACHE" if result.cache_hit else "LIVE_INFERENCE",
            model=result.model,
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )
        await run_in_threadpool(
            db.save_turn, body.session_id, body.message, result.reply,
            metadata.model_dump(), body.system_prompt,
        )
        return ChatResponse(session_id=body.session_id, reply=result.reply, metadata=metadata)

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str) -> JSONResponse:
        history = await run_in_threadpool(db.get_history, session_id)
        if not history:
            return JSONResponse({"detail": "session not found"}, status_code=404)
        return JSONResponse({"session_id": session_id, "messages": history})

    return app


app = create_app()
