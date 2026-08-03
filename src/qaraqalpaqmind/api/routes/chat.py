"""Chat completions: `POST /v1/chat/completions`.

Streaming and non-streaming share one code path through `ModelBackend.stream`,
so the streaming behaviour is exercised by every request rather than only by
streaming clients.

The concurrency guard is held for the whole generation, including the streaming
tail. Releasing it when the response object is returned would defeat it
entirely: an SSE response returns immediately and generates for minutes.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...common.logging import get_logger
from ...preprocessing.orthography import to_latin2016
from ..backends.base import BackendError, ModelBackend
from ..config import ServeConfig
from ..middleware.auth import client_id
from ..middleware.limits import ConcurrencyGuard
from ..schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    DeltaMessage,
    Message,
    ModelCard,
    ModelList,
    StreamChoice,
    Usage,
    _new_id,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])

_SSE_DONE = "data: [DONE]\n\n"


def _backend(request: Request) -> ModelBackend:
    return request.app.state.backend  # type: ignore[no-any-return]


def _config(request: Request) -> ServeConfig:
    return request.app.state.config  # type: ignore[no-any-return]


def _guard(request: Request) -> ConcurrencyGuard:
    return request.app.state.concurrency  # type: ignore[no-any-return]


@router.get("/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    """The models this server will answer for."""
    config = _config(request)
    return ModelList(data=[ModelCard(id=config.served_model_name)])


@router.post(
    "/chat/completions",
    # The route returns either an SSE stream or a JSON body depending on
    # `stream`, so FastAPI cannot derive one response model from the return
    # annotation. The union is documented here instead.
    response_model=None,
    responses={
        200: {
            "description": "A completion, or a text/event-stream of chunks when stream=true.",
            "content": {
                "application/json": {"schema": ChatCompletionResponse.model_json_schema()},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        }
    },
)
async def create_chat_completion(
    request: Request, body: ChatCompletionRequest
) -> StreamingResponse | ChatCompletionResponse:
    config = _config(request)
    backend = _backend(request)
    guard = _guard(request)
    caller = client_id(request)

    # Held until generation finishes, not until the response is returned - a
    # streaming response returns instantly and then runs for minutes.
    if not await guard.acquire(caller):
        raise HTTPException(
            status_code=429,
            detail=(
                f"too many concurrent requests (limit "
                f"{config.rate_limit.max_concurrent_per_key}); wait for one to finish"
            ),
        )

    started = time.monotonic()
    try:
        if body.stream:
            return StreamingResponse(
                _stream_response(backend, body, config, guard, caller, started),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    # Nginx buffers proxied responses by default, which holds
                    # SSE chunks until the buffer fills and makes streaming
                    # look broken. This disables it per-response.
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        result = await backend.generate(body)
        text = to_latin2016(result.text) if body.normalize_orthography else result.text

        _log_completion(config, caller, body, text, time.monotonic() - started)
        return ChatCompletionResponse(
            model=config.served_model_name,
            choices=[
                Choice(
                    message=Message(role="assistant", content=text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    except BackendError as exc:
        logger.warning("Backend error", extra={"client": caller, "error": str(exc)})
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        # Streaming releases inside the generator; this covers the rest.
        if not body.stream:
            await guard.release(caller)


async def _stream_response(
    backend: ModelBackend,
    body: ChatCompletionRequest,
    config: ServeConfig,
    guard: ConcurrencyGuard,
    caller: str,
    started: float,
) -> AsyncIterator[str]:
    """Server-sent events in the OpenAI streaming format."""
    completion_id = _new_id("chatcmpl")
    collected: list[str] = []

    def chunk(delta: DeltaMessage, finish: str | None = None) -> str:
        payload = ChatCompletionChunk(
            id=completion_id,
            model=config.served_model_name,
            choices=[StreamChoice(delta=delta, finish_reason=finish)],
        )
        return f"data: {json.dumps(payload.model_dump(mode='json', exclude_none=True))}\n\n"

    try:
        # The first chunk carries the role, as OpenAI clients expect.
        yield chunk(DeltaMessage(role="assistant"))

        async for piece in backend.stream(body):
            collected.append(piece)
            # Orthography normalisation is skipped while streaming: it needs
            # whole words, and applying it per-fragment would corrupt text
            # split mid-word.
            yield chunk(DeltaMessage(content=piece))

        yield chunk(DeltaMessage(), finish="stop")
        yield _SSE_DONE

        _log_completion(config, caller, body, "".join(collected), time.monotonic() - started)

    except BackendError as exc:
        logger.warning("Backend error mid-stream", extra={"client": caller, "error": str(exc)})
        # The response already has status 200, so the error must travel in the
        # stream itself; a client cannot be told 502 at this point.
        error = {"error": {"message": str(exc), "type": "server_error"}}
        yield f"data: {json.dumps(error)}\n\n"
        yield _SSE_DONE
    except Exception as exc:
        logger.exception("Unexpected error mid-stream", extra={"client": caller})
        error = {"error": {"message": "internal error", "type": "server_error"}}
        yield f"data: {json.dumps(error)}\n\n"
        yield _SSE_DONE
        raise exc from None
    finally:
        await guard.release(caller)


def _log_completion(
    config: ServeConfig,
    caller: str,
    body: ChatCompletionRequest,
    text: str,
    elapsed: float,
) -> None:
    """Record the request. Prompt and completion text only if configured.

    Off by default: prompts are the user's private content and often contain
    personal data, so logging them is a decision to be taken deliberately.
    """
    fields: dict[str, object] = {
        "client": caller,
        "messages": len(body.messages),
        "stream": body.stream,
        "elapsed_s": round(elapsed, 3),
        "response_chars": len(text),
    }
    if config.observability.log_prompts:
        fields["prompt"] = body.prompt_text()[:2000]
    if config.observability.log_completions:
        fields["completion"] = text[:2000]

    if elapsed > config.observability.slow_request_seconds:
        logger.warning("Slow completion", extra=fields)
    else:
        logger.info("Completion served", extra=fields)
