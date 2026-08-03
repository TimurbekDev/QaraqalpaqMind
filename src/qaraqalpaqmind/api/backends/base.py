"""The model backend contract.

Everything above this line - routes, auth, streaming, the whole frontend - is
written against `ModelBackend` and never against a specific inference engine.
That is what lets the API be developed and tested today, months before the
fine-tuned weights exist:

    echo         no model at all. Deterministic. Used by the test suite.
    transformers a small model on any machine. Used for development.
    vllm         a running vLLM server. Used in production.

Swapping between them is a config change. If the serving code called vLLM
directly, developing the API would require a GPU and a 16 GB download, and the
tests would need both.

Streaming is the primary interface and non-streaming is derived from it, rather
than the other way round. Doing it the other way makes it easy to ship a
`/v1/chat/completions` that works and a streaming path that was never really
exercised.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..schemas.chat import ChatCompletionRequest


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """A completed generation."""

    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BackendError(RuntimeError):
    """The backend could not serve the request.

    Carries an HTTP status so routes can translate upstream failures faithfully
    instead of reporting every one as a 500.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelBackend(ABC):
    """Generates text from chat messages."""

    #: Reported by /v1/models and echoed in responses.
    name: str = "unknown"

    @abstractmethod
    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        """Yield response text incrementally.

        Declared as a plain method returning an AsyncIterator rather than as an
        `async def ... yield`, so subclasses are free to implement it either as
        an async generator or by returning one.
        """

    async def generate(self, request: ChatCompletionRequest) -> GenerationResult:
        """Collect a full response.

        Defined in terms of `stream`, so the streaming path is exercised by
        every request rather than only by streaming clients.
        """
        parts: list[str] = []
        async for piece in self.stream(request):
            parts.append(piece)
        text = "".join(parts)
        return GenerationResult(
            text=text,
            prompt_tokens=_rough_tokens(request.prompt_text()),
            completion_tokens=_rough_tokens(text),
        )

    async def health(self) -> bool:
        """Whether the backend can serve right now."""
        return True

    async def aclose(self) -> None:
        """Release resources. Called on application shutdown.

        A no-op by default: most backends hold nothing that needs closing.
        """
        return None


def _rough_tokens(text: str) -> int:
    """Approximate token count for usage reporting.

    2.3 characters per token, measured for Karakalpak against the Qwen3
    tokenizer (docs/TOKENIZER.md). Backends that know the real count override
    this by returning their own figures.
    """
    return max(1, round(len(text) / 2.3)) if text else 0
