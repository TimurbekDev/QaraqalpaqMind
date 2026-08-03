"""OpenAI-compatible chat completion schemas.

The API speaks the OpenAI wire format deliberately. It is what every client
library, every UI framework and every evaluation harness already targets, so
choosing it means the Next.js frontend, `curl`, the OpenAI Python SDK and
`llm`-style CLIs all work without an adapter.

Only the parts we actually serve are modelled. Fields that exist in the OpenAI
spec but are not implemented are rejected rather than silently ignored: a client
that sends `logit_bias` and has it dropped gets wrong results and no warning.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant"]


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


class Message(BaseModel):
    """One turn in a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions."""

    # `extra="forbid"` on purpose: an unsupported sampling parameter that is
    # silently dropped produces wrong output with no way to notice.
    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="Model id. Ignored unless the server serves several.")
    messages: list[Message] = Field(min_length=1)

    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stop: list[str] | None = Field(default=None, max_length=8)
    seed: int | None = None

    # Karakalpak-specific, and not part of the OpenAI spec. Off by default so a
    # standard client is unaffected.
    normalize_orthography: bool = Field(
        default=False,
        description=(
            "Normalise the response to Latin 2016 before returning it. The model is "
            "trained to produce it, but a low-temperature slip into the 2009 apostrophe "
            "convention is cheap to correct deterministically."
        ),
    )

    @model_validator(mode="after")
    def _must_end_with_a_user_or_system_turn(self) -> ChatCompletionRequest:
        if self.messages[-1].role == "assistant":
            raise ValueError(
                "the final message is from the assistant, so there is nothing to respond to"
            )
        return self

    def prompt_text(self) -> str:
        """Flattened text, for token accounting and contamination checks."""
        return "\n".join(m.content for m in self.messages)


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = 0
    message: Message
    finish_reason: Literal["stop", "length", "content_filter"] | None = "stop"


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# --- streaming ------------------------------------------------------------


class DeltaMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = 0
    delta: DeltaMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    """One server-sent event in a streaming response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]


# --- models endpoint ------------------------------------------------------


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "qaraqalpaqmind"


class ModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"] = "list"
    data: list[ModelCard]


# --- errors ---------------------------------------------------------------


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    code: str | None = None
    param: str | None = None


class ErrorResponse(BaseModel):
    """OpenAI's error envelope, so existing clients parse our failures too."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail

    @classmethod
    def of(cls, message: str, error_type: str, code: str | None = None) -> ErrorResponse:
        return cls(error=ErrorDetail(message=message, type=error_type, code=code))


def as_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)
