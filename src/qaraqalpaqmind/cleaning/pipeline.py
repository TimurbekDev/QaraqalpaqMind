"""The cleaning pass: `data/interim/` -> `data/processed/`.

For each document, in this order:

1. **Normalise** text - mojibake, invisible characters, unicode form, quotes,
   whitespace, emails.
2. **Unify orthography** - Cyrillic and the two older Latin conventions all
   become Latin 2016. The original script is recorded first, so nothing about
   provenance is lost.
3. **Assess** quality - score and flags, computed on the *cleaned* text
   because that is what training would see.

Order is not arbitrary. Normalisation before orthography means the transliterator
sees canonical characters rather than look-alikes and decomposed accents.
Assessment last means thresholds are applied to the final text.

Nothing is deleted. Rejected documents are written to a separate file so that a
threshold can be revisited without re-running the whole pipeline - and so that
"what did we throw away?" is a question with an answer.
"""

from __future__ import annotations

import collections
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..common.io import read_jsonl, write_jsonl
from ..common.logging import get_logger
from ..common.paths import INTERIM_DIR, PROCESSED_DIR, ensure_dir
from ..common.records import Document, Script
from ..preprocessing.orthography import to_latin2016
from ..preprocessing.script import detect_script
from .filters import Assessment, FilterConfig, assess, should_keep
from .normalize import NormalizeConfig, normalize_text

logger = get_logger(__name__)

_PROGRESS_EVERY = 25_000


@dataclass(slots=True)
class CleanStats:
    """What the cleaning pass did to one source."""

    source_id: str
    read: int = 0
    kept: int = 0
    rejected: int = 0
    below_threshold: int = 0
    transliterated: int = 0
    chars_in: int = 0
    chars_out: int = 0
    flags: collections.Counter[str] = field(default_factory=collections.Counter)
    scripts_in: collections.Counter[str] = field(default_factory=collections.Counter)

    @property
    def keep_rate(self) -> float:
        return self.kept / self.read if self.read else 0.0

    def summary(self) -> str:
        return (
            f"{self.source_id}: read={self.read:,} kept={self.kept:,} "
            f"({self.keep_rate:.1%}) rejected={self.rejected:,} "
            f"low_score={self.below_threshold:,} translit={self.transliterated:,} "
            f"chars {self.chars_in / 1e6:.1f}M -> {self.chars_out / 1e6:.1f}M"
        )


def interim_path(source_id: str) -> Path:
    return INTERIM_DIR / f"{source_id}.jsonl.zst"


def processed_path(source_id: str) -> Path:
    return ensure_dir(PROCESSED_DIR) / f"{source_id}.jsonl.zst"


def rejected_path(source_id: str) -> Path:
    return ensure_dir(PROCESSED_DIR / "rejected") / f"{source_id}.jsonl.zst"


def clean_document(
    document: Document,
    *,
    normalize_config: NormalizeConfig | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[Document, Assessment]:
    """Normalise, transliterate and assess one document.

    Returns the cleaned document and its assessment. The caller decides what to
    do with a low score, which keeps policy out of this function.
    """
    original_script = document.script if document.script is not Script.UNKNOWN else detect_script(
        document.text
    )

    text = normalize_text(document.text, normalize_config)
    text = to_latin2016(text)

    cleaned = document.model_copy(
        update={
            "text": text,
            "script": Script.LATIN if text else document.script,
            "meta": document.meta | {"original_script": original_script.value},
        }
    )

    assessment = assess(
        text, filter_config, extractor=document.meta.get("extractor")
    )
    return cleaned.model_copy(update={"quality": assessment.to_quality()}), assessment


def clean_source(
    source_id: str,
    *,
    normalize_config: NormalizeConfig | None = None,
    filter_config: FilterConfig | None = None,
    keep_rejected: bool = True,
    limit: int | None = None,
) -> CleanStats:
    """Run the cleaning pass over one source."""
    source_path = interim_path(source_id)
    if not source_path.exists():
        raise FileNotFoundError(f"No interim data for '{source_id}': {source_path}")

    cfg = filter_config or FilterConfig()
    stats = CleanStats(source_id=source_id)
    rejected: list[dict[str, object]] = []

    def stream() -> Iterator[dict[str, object]]:
        for index, row in enumerate(read_jsonl(source_path)):
            if limit is not None and index >= limit:
                return

            document = Document.model_validate(row)
            stats.read += 1
            stats.chars_in += len(document.text)
            stats.scripts_in[document.script.value] += 1

            cleaned, assessment = clean_document(
                document, normalize_config=normalize_config, filter_config=cfg
            )
            for flag in assessment.flags:
                stats.flags[flag.value] += 1
            if document.script is Script.CYRILLIC:
                stats.transliterated += 1

            if should_keep(assessment, cfg):
                stats.kept += 1
                stats.chars_out += len(cleaned.text)
                yield cleaned.model_dump(mode="json")
            else:
                if assessment.rejected:
                    stats.rejected += 1
                else:
                    stats.below_threshold += 1
                if keep_rejected:
                    rejected.append(cleaned.model_dump(mode="json"))

            if stats.read % _PROGRESS_EVERY == 0:
                logger.info("Cleaning progress", extra={"summary": stats.summary()})

    write_jsonl(processed_path(source_id), stream())

    if keep_rejected and rejected:
        # Kept so a threshold can be revisited without re-running everything,
        # and so "what did we throw away?" has an answer.
        write_jsonl(rejected_path(source_id), iter(rejected))

    logger.info("Cleaning finished", extra={"summary": stats.summary()})
    return stats


def available_sources() -> list[str]:
    """Source ids with interim data on disk."""
    return sorted(p.name.split(".")[0] for p in INTERIM_DIR.glob("*.jsonl.zst"))
