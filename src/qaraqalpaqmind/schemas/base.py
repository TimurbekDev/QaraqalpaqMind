"""Shared envelope for every task-specific dataset record.

Phases 6 to 8 all consume JSONL, and each was free to invent its own field
names until this module existed. Fixing the contract here means the SFT
trainer, the DPO builder and the evaluation harness read the same shape, and a
malformed record is caught by a validator rather than by a confusing loss curve
forty minutes into training.

Every record carries provenance for the same reason `Document` does: a training
set whose rows cannot be traced to a source and a licence is one that cannot be
published or partially withdrawn.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common.io import text_sha1
from ..common.records import Script


class TaskType(StrEnum):
    """The task a record teaches. One JSONL file holds one task type."""

    PRETRAIN = "pretrain"
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    TRANSLATION = "translation"
    GRAMMAR = "grammar"
    QA = "qa"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    BENCHMARK = "benchmark"
    PREFERENCE = "preference"


class Provenance(BaseModel):
    """Where a record came from and what may be done with it."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Registry id, or a generator name for synthetic data.")
    source_url: str | None = None
    license: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    # Synthetic data must be labelled. Training on model output without knowing
    # it is model output is how a corpus quietly degrades across generations.
    synthetic: bool = False
    generator: str | None = Field(
        default=None, description="Model or script that produced a synthetic record."
    )
    human_reviewed: bool = False

    @field_validator("generator")
    @classmethod
    def _generator_requires_synthetic(cls, value: str | None, info: Any) -> str | None:
        if value and not info.data.get("synthetic", False):
            raise ValueError("generator is set, so synthetic must be true")
        return value


class TaskRecord(BaseModel):
    """Base for all task records.

    Subclasses add their task-specific fields and implement `to_messages`, which
    is what makes every task type trainable through one chat-formatted SFT path.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    task: TaskType
    lang: str = "kaa"
    script: Script = Script.LATIN
    provenance: Provenance
    meta: dict[str, Any] = Field(default_factory=dict)
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)

    def model_post_init(self, _context: Any) -> None:
        if not self.id:
            # Content-derived so the same item written twice collapses to one id.
            object.__setattr__(self, "id", self.fingerprint())

    def fingerprint(self) -> str:
        """Stable id derived from the record's content."""
        payload = self.model_dump(mode="json", exclude={"id", "provenance", "meta"})
        return text_sha1(f"{self.task.value}\x00{sorted(payload.items())}")

    def to_messages(self) -> list[dict[str, str]]:
        """Render the record as chat messages for supervised fine-tuning."""
        raise NotImplementedError

    def training_text(self) -> str:
        """All text in the record, for length statistics and contamination checks."""
        return "\n".join(message["content"] for message in self.to_messages())


def system_prompt(text: str | None = None) -> dict[str, str]:
    """Default Karakalpak system prompt, or a custom one."""
    return {
        "role": "system",
        "content": text
        or "Sen qaraqalpaq tilinde járdem beretuǵın paydalı járdemshisen.",
    }
