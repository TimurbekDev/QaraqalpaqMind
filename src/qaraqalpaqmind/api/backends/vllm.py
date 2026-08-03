"""vLLM backend: proxies to a running vLLM OpenAI-compatible server.

vLLM is not imported here and is not a dependency of the API. It runs as a
separate process - usually its own container - and this talks to it over HTTP.

That separation is deliberate. The API gateway restarts in a second; a vLLM
process holding a 16 GB model in VRAM takes a minute. Coupling them would mean
every configuration change to auth or rate limiting costs a model reload, and a
crash in either takes down both.

    vllm serve models/merged/qwen3-8b-kaa --port 8001
    qm serve --backend vllm --vllm-url http://localhost:8001/v1
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ...common.logging import get_logger
from ..schemas.chat import ChatCompletionRequest
from .base import BackendError, GenerationResult, ModelBackend

logger = get_logger(__name__)

_STREAM_DONE = "[DONE]"


class VLLMBackend(ModelBackend):
    """Talks to a vLLM server over its OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "qaraqalpaqmind",
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        import httpx

        self.name = model
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Long timeout: a 2,000-token generation on a busy server legitimately
        # takes minutes, and cutting it off mid-stream looks like a model bug.
        self._client = httpx.AsyncClient(
            base_url=self._base_url, headers=headers, timeout=timeout
        )

    def _payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        import httpx

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=self._payload(request, stream=True)
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise BackendError(
                        f"vLLM returned {response.status_code}: {body[:400]}",
                        status_code=502 if response.status_code >= 500 else 400,
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == _STREAM_DONE:
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Unparseable SSE chunk from vLLM", extra={"line": line[:200]})
                        continue
                    for choice in chunk.get("choices", []):
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            yield content

        except httpx.HTTPError as exc:
            raise BackendError(
                f"cannot reach vLLM at {self._base_url}: {type(exc).__name__}: {exc}",
                status_code=503,
            ) from exc

    async def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        """Non-streaming request, so vLLM's own token counts are used."""
        import httpx

        try:
            response = await self._client.post(
                "/chat/completions", json=self._payload(request, stream=False)
            )
        except httpx.HTTPError as exc:
            raise BackendError(
                f"cannot reach vLLM at {self._base_url}: {type(exc).__name__}: {exc}",
                status_code=503,
            ) from exc

        if response.status_code >= 400:
            raise BackendError(
                f"vLLM returned {response.status_code}: {response.text[:400]}",
                status_code=502 if response.status_code >= 500 else 400,
            )

        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise BackendError("vLLM returned no choices")

        usage = payload.get("usage") or {}
        return GenerationResult(
            text=(choices[0].get("message") or {}).get("content", ""),
            finish_reason=choices[0].get("finish_reason") or "stop",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def health(self) -> bool:
        import httpx

        try:
            response = await self._client.get("/models", timeout=5.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    async def aclose(self) -> None:
        await self._client.aclose()
