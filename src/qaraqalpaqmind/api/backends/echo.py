"""A backend that needs no model at all.

Every test in the API suite runs against this. It makes the whole HTTP surface -
routing, auth, rate limiting, SSE framing, error envelopes, usage accounting -
testable in milliseconds with no GPU, no download and no network.

It is not a mock in the usual sense: it implements the real `ModelBackend`
contract and streams token by token, so the streaming path under test is the
same code path production uses.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..schemas.chat import ChatCompletionRequest
from .base import BackendError, ModelBackend

# Deterministic Karakalpak, so tests can assert on output and a developer
# hitting the endpoint sees something in the right language and script.
_DEFAULT_REPLY = (
    "Salawmatsız! Men QaraqalpaqMind sınaq rejiminde islep turıppan. "
    "Bul juwap haqıyqıy modelden emes, sınaq baytınnan berilmekte."
)


class EchoBackend(ModelBackend):
    """Streams a canned reply, or echoes the last user message.

    Args:
        reply: Fixed text to return. When None, the last user message is echoed
            back, which makes request plumbing visible in the response.
        delay: Seconds between chunks, for exercising client-side streaming.
        fail: Raise `BackendError` on every call, to test error handling.
    """

    name = "echo"

    def __init__(
        self,
        reply: str | None = _DEFAULT_REPLY,
        *,
        delay: float = 0.0,
        fail: bool = False,
    ) -> None:
        self._reply = reply
        self._delay = delay
        self._fail = fail

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        if self._fail:
            raise BackendError("echo backend configured to fail", status_code=503)

        text = self._reply
        if text is None:
            last_user = next(
                (m.content for m in reversed(request.messages) if m.role == "user"), ""
            )
            text = f"Echo: {last_user}"

        # Word by word, keeping the separator, so reassembling the stream
        # reproduces the text exactly - a joiner bug would otherwise hide here.
        words = text.split(" ")
        for index, word in enumerate(words):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield word if index == 0 else f" {word}"

    async def health(self) -> bool:
        return not self._fail
