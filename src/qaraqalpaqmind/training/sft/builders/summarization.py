"""Summarisation data from Wikipedia lead sections.

A Wikipedia article's opening paragraph is a human-written summary of the
article, produced by an editor who read it. That makes lead-versus-body the
only source of genuine Karakalpak summarisation supervision available without
paying annotators.

The technique is standard and its weakness is worth stating: a lead section is
an *introduction*, not an abstract. It over-represents definitions and dates and
under-represents whatever the article spends most of its length on. Records are
marked `human_reviewed=False` accordingly, and quality filters below discard the
cases where the heuristic clearly fails.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....dedup.pipeline import output_path
from ....schemas import Provenance, SummarizationRecord

logger = get_logger(__name__)

# The lead must be a real summary, not a stub, and the body must be long enough
# that summarising it is a task rather than a copy.
_MIN_SUMMARY_CHARS = 120
_MAX_SUMMARY_CHARS = 1_200
_MIN_BODY_CHARS = 800
_MAX_BODY_CHARS = 12_000
# A "summary" that is most of the article teaches copying.
_MAX_COMPRESSION = 0.5
_MIN_COMPRESSION = 0.02

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(slots=True)
class SummarizationStats:
    read: int = 0
    emitted: int = 0
    skipped_short_body: int = 0
    skipped_lead: int = 0
    skipped_compression: int = 0

    def summary(self) -> str:
        return (
            f"summarization: {self.emitted:,} pairs from {self.read:,} articles "
            f"(dropped {self.skipped_short_body:,} short bodies, "
            f"{self.skipped_lead:,} unusable leads, "
            f"{self.skipped_compression:,} bad compression ratios)"
        )


def split_lead_and_body(text: str) -> tuple[str, str] | None:
    """Separate an article's lead section from the rest.

    Wikipedia ingestion prepends the title, so the first paragraph is dropped
    when it is just that title.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if len(paragraphs) < 3:
        return None

    # The ingester writes "Title\n\nbody"; a very short first paragraph with no
    # sentence-ending punctuation is that title, not prose.
    start = 1 if len(paragraphs[0]) < 80 and not paragraphs[0].endswith(".") else 0
    if len(paragraphs) - start < 3:
        return None

    lead = paragraphs[start]
    body = "\n\n".join(paragraphs[start:])
    return lead, body


def build(
    *,
    limit: int | None = None,
    dataset: str = "pretrain_v1",
    source_id: str = "wiki_kaa",
) -> Iterator[SummarizationRecord]:
    """Yield lead-section summarisation pairs from encyclopedia articles."""
    stats = SummarizationStats()

    provenance = Provenance(
        source_id="summarization_from_wiki_lead",
        source_url="https://kaa.wikipedia.org",
        license="CC-BY-SA-4.0",
        # The summary is human-written, but the pairing is a heuristic.
        synthetic=False,
        human_reviewed=False,
    )

    for row in read_jsonl(output_path(dataset)):
        if limit is not None and stats.emitted >= limit:
            break
        if row.get("source_id") != source_id:
            continue

        text = str(row.get("text", ""))
        stats.read += 1

        if not (_MIN_BODY_CHARS <= len(text) <= _MAX_BODY_CHARS):
            stats.skipped_short_body += 1
            continue

        split = split_lead_and_body(text)
        if split is None:
            stats.skipped_lead += 1
            continue

        lead, body = split
        if not (_MIN_SUMMARY_CHARS <= len(lead) <= _MAX_SUMMARY_CHARS):
            stats.skipped_lead += 1
            continue

        compression = len(lead) / len(body)
        if not (_MIN_COMPRESSION <= compression <= _MAX_COMPRESSION):
            stats.skipped_compression += 1
            continue

        yield SummarizationRecord(
            document=body,
            summary=lead,
            provenance=provenance,
            meta={"compression": round(compression, 3), "title": row.get("meta", {}).get("title")},
        )
        stats.emitted += 1

    logger.info("Summarization builder finished", extra={"summary": stats.summary()})
