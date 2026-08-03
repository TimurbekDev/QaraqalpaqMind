"""FastAPI application factory.

`create_app(config)` builds a fully wired server for any configuration, which
is what lets the test suite spin up a real app against the echo backend in
milliseconds - no GPU, no network, no model.

Middleware order is significant and runs outermost first:

    CORS -> request logging -> auth -> rate limit -> routes

Auth before rate limiting so limits are keyed by verified caller rather than by
whatever the client claimed. Logging outside auth so rejected requests are
still recorded - a burst of 401s is exactly what you want to see.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..common.logging import get_logger
from .backends.base import ModelBackend
from .config import BackendKind, ServeConfig
from .middleware.auth import build_auth_middleware, client_id
from .middleware.limits import ConcurrencyGuard, build_rate_limit_middleware
from .routes import chat, health
from .schemas.chat import ErrorResponse, as_dict

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def build_backend(config: ServeConfig) -> ModelBackend:
    """Construct the configured backend. The only place engines are named."""
    kind = config.backend.kind

    if kind is BackendKind.ECHO:
        from .backends.echo import EchoBackend

        return EchoBackend()

    if kind is BackendKind.TRANSFORMERS:
        from .backends.local import TransformersBackend

        return TransformersBackend(
            config.backend.model,
            device=config.backend.device,
            dtype=config.backend.dtype,
            max_new_tokens=config.backend.max_new_tokens,
        )

    from .backends.vllm import VLLMBackend

    return VLLMBackend(
        base_url=config.backend.vllm_url,
        model=config.backend.model,
        api_key=config.backend.vllm_api_key,
        timeout=config.backend.request_timeout,
    )


def create_app(config: ServeConfig | None = None) -> FastAPI:
    """Build the API server."""
    settings = config or ServeConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = settings
        app.state.backend = build_backend(settings)
        app.state.concurrency = ConcurrencyGuard(settings.rate_limit.max_concurrent_per_key)
        logger.info(
            "API ready",
            extra={
                "backend": settings.backend.kind.value,
                "model": settings.backend.model,
                "auth": settings.auth.enabled,
                "rate_limit": settings.rate_limit.enabled,
            },
        )
        try:
            yield
        finally:
            await app.state.backend.aclose()
            logger.info("API stopped")

    app = FastAPI(
        title="QaraqalpaqMind API",
        description="OpenAI-compatible chat completions for the Karakalpak language model.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Innermost first: FastAPI runs middleware in reverse registration order.
    app.middleware("http")(build_rate_limit_middleware(settings.rate_limit))
    app.middleware("http")(build_auth_middleware(settings.auth))
    app.middleware("http")(_request_logging_middleware(settings))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback to a client: it can disclose paths, versions
        # and prompt content. The detail goes to the log, the client gets an id.
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled error", extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content=as_dict(
                ErrorResponse.of(
                    f"internal error (request {request_id})", "server_error", "internal_error"
                )
            ),
        )

    return app


def _request_logging_middleware(
    settings: ServeConfig,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Assign a request id, time the request, and record the outcome."""

    async def middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.monotonic()

        response = await call_next(request)

        elapsed = time.monotonic() - started
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "Request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_s": round(elapsed, 3),
                "client": client_id(request),
            },
        )
        return response

    return middleware
