"""Typed JSONL schemas for every dataset a training phase consumes.

One contract, fixed here, so the SFT trainer, the DPO builder and the
evaluation harness agree on field names instead of each inventing their own.
Every task type renders to chat messages through `to_messages`, which is what
lets Phase 6 train on all of them through a single code path.
"""

from __future__ import annotations

from .base import Provenance, TaskRecord, TaskType, system_prompt
from .tasks import (
    RECORD_TYPES,
    BenchmarkRecord,
    CodingRecord,
    ConversationRecord,
    GrammarRecord,
    InstructionRecord,
    MathRecord,
    PretrainRecord,
    QARecord,
    ReasoningRecord,
    SummarizationRecord,
    TranslationRecord,
    Turn,
    parse_record,
)

__all__ = [
    "RECORD_TYPES",
    "BenchmarkRecord",
    "CodingRecord",
    "ConversationRecord",
    "GrammarRecord",
    "InstructionRecord",
    "MathRecord",
    "PretrainRecord",
    "Provenance",
    "QARecord",
    "ReasoningRecord",
    "SummarizationRecord",
    "TaskRecord",
    "TaskType",
    "TranslationRecord",
    "Turn",
    "parse_record",
    "system_prompt",
]
