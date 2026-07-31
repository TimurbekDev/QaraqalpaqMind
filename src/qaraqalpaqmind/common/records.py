"""The universal record envelope.

Every document in `interim/`, `processed/` and `datasets/pretrain/` is one of
these, whatever produced it - a crawler, a Wikipedia dump, or a Hugging Face
dataset. Downstream stages therefore never need to know where text came from,
only that it arrived in this shape.

Provenance fields are mandatory, not decorative. A corpus whose documents
cannot be traced back to a URL and a licence is a corpus that cannot be
published, audited, or partially withdrawn on request.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .io import text_sha1


class Script(StrEnum):
    """Writing system, tracked separately from language.

    Karakalpak is written in both Latin (official since 1994) and Cyrillic
    (most pre-1994 books and a large share of the live web). Collapsing them
    into one field would make script-balance decisions impossible.

    This is the single definition used across the project; `preprocessing.script`
    imports it rather than declaring its own.
    """

    LATIN = "latin"
    CYRILLIC = "cyrillic"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"


class Quality(BaseModel):
    """Cleaning verdict. Populated in Phase 3; empty until then."""

    model_config = ConfigDict(extra="forbid")

    score: float | None = Field(default=None, ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)


class Document(BaseModel):
    """One unit of text plus everything needed to justify keeping it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable content-derived identifier.")
    text: str

    # --- provenance (mandatory) ---
    source_id: str = Field(description="Registry id, e.g. 'wiki_kaa'.")
    source_url: str | None = None
    fetched_at: datetime
    license: str

    # --- language ---
    lang: str = "kaa"
    lang_conf: float | None = Field(default=None, ge=0.0, le=1.0)
    script: Script = Script.UNKNOWN

    # --- free-form, source-specific ---
    meta: dict[str, Any] = Field(default_factory=dict)
    quality: Quality = Field(default_factory=Quality)

    @field_validator("text")
    @classmethod
    def _text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Document.text must not be blank")
        return value

    @classmethod
    def create(
        cls,
        *,
        text: str,
        source_id: str,
        license: str,
        source_url: str | None = None,
        fetched_at: datetime | None = None,
        script: Script = Script.UNKNOWN,
        meta: dict[str, Any] | None = None,
    ) -> Self:
        """Build a document, deriving the id from its content and origin.

        Hashing `source_id + source_url + text` means the same article ingested
        twice collapses to one id, while identical boilerplate on two different
        pages stays distinct - which is what dedup in Phase 3 needs to reason
        about separately.
        """
        return cls(
            id=text_sha1(f"{source_id}\x00{source_url or ''}\x00{text}"),
            text=text,
            source_id=source_id,
            source_url=source_url,
            fetched_at=fetched_at or datetime.now(tz=UTC),
            license=license,
            script=script,
            meta=meta or {},
        )

    def char_count(self) -> int:
        return len(self.text)

    def word_count(self) -> int:
        return len(self.text.split())

    def estimated_tokens(self) -> int:
        """Rough Qwen3 token estimate for Karakalpak.

        Measured against the Qwen3 tokenizer in Phase 4; until then we use
        ~3.1 characters per token, which is typical for an agglutinative Turkic
        language in a tokenizer with no dedicated Karakalpak vocabulary. Treat
        it as an order-of-magnitude figure, not a budget.
        """
        return round(len(self.text) / 3.1)
