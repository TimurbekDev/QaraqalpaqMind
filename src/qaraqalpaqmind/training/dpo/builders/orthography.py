"""Orthography preference pairs.

The cleanest preference signal this project has. Phase 6's grammar builder
already produces (incorrect, correct) pairs by reversing normalisation; the
same machinery gives DPO pairs where `chosen` and `rejected` differ in *exactly*
one dimension - which spelling convention was used - and in nothing else.

That precision matters more in DPO than in SFT. DPO raises the likelihood of
`chosen` relative to `rejected`, so any dimension the two differ in is taught,
including the ones nobody intended. Pairs generated from the same sentence
cannot differ in content, length, register or topic, only in orthography.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....dedup.pipeline import output_path
from ....schemas import PreferenceRecord, Provenance
from ...sft.builders.grammar import ErrorType, corrupt

logger = get_logger(__name__)

CRITERION = "orthography"

_MIN_CHARS = 40
_MAX_CHARS = 400
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MARKERS = "áóúǵńıÁÓÚǴŃ"

# Prompts that invite free generation rather than correction. The model should
# prefer correct spelling when *writing*, not only when asked to fix something -
# training it only on correction prompts teaches a narrower behaviour.
_PROMPTS = (
    "Tómendegi gápti durıs jazıw qaǵıydaları boyınsha jaz:",
    "Bul gápti házirgi álipbe standartında jaz:",
    "Tómendegi tekstti durıs orfografiya menen jaz:",
)


@dataclass(slots=True)
class OrthographyStats:
    read: int = 0
    emitted: int = 0
    skipped_no_markers: int = 0
    by_error: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"orthography preferences: {self.emitted:,} pairs from {self.read:,} sentences "
            f"({self.skipped_no_markers:,} had nothing to prefer)"
        )


def build(
    *,
    limit: int | None = None,
    dataset: str = "pretrain_v1",
    seed: int = 20260731,
    errors: tuple[ErrorType, ...] = tuple(ErrorType),
) -> Iterator[PreferenceRecord]:
    """Yield preference pairs that differ only in spelling convention."""
    rng = random.Random(seed)
    stats = OrthographyStats()

    provenance = Provenance(
        source_id="dpo_orthography",
        license="derived from pretrain_v1; see per-source manifests",
        # The rejected side is generated. The chosen side is real corpus text.
        synthetic=True,
        generator="qaraqalpaqmind.training.dpo.builders.orthography",
        human_reviewed=False,
    )

    for row in read_jsonl(output_path(dataset)):
        if limit is not None and stats.emitted >= limit:
            break

        for sentence in _SENTENCE_SPLIT.split(str(row.get("text", ""))):
            if limit is not None and stats.emitted >= limit:
                break
            cleaned = sentence.strip()
            if not (_MIN_CHARS <= len(cleaned) <= _MAX_CHARS):
                continue
            stats.read += 1

            if not any(marker in cleaned for marker in _MARKERS):
                stats.skipped_no_markers += 1
                continue

            error = errors[rng.randrange(len(errors))]
            rejected = corrupt(cleaned, error, rng)
            if rejected.strip() == cleaned.strip():
                stats.skipped_no_markers += 1
                continue

            prompt = _PROMPTS[rng.randrange(len(_PROMPTS))]
            yield PreferenceRecord(
                prompt=f"{prompt}\n\n{rejected}",
                chosen=cleaned,
                rejected=rejected,
                criterion=CRITERION,
                provenance=provenance,
                meta={"error_type": error.value},
            )
            stats.emitted += 1
            stats.by_error[error.value] = stats.by_error.get(error.value, 0) + 1

    logger.info("Orthography preference builder finished", extra={"summary": stats.summary()})
    if stats.by_error:
        logger.info("Orthography preference errors", extra=dict(stats.by_error))
