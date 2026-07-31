"""Translation SFT data, built from the dilmash parallel corpus.

This is the one task where Karakalpak has genuinely abundant supervised data:
215,417 human-translated pairs across kaa-en, kaa-ru and kaa-uz, already
ingested with each Karakalpak sentence carrying its partner in
`meta.parallel_text`.

Both directions are generated. A model trained only kaa->X learns to translate
*out of* Karakalpak and not into it, and translating into the low-resource
language is the direction that matters most for a Karakalpak assistant.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....common.paths import PROCESSED_DIR
from ....schemas import Provenance, TranslationRecord

logger = get_logger(__name__)

# PROCESSED, not interim. Two reasons, both found by reading built output:
#
# 1. Orthography. `data/interim/` is pre-normalisation, so its Karakalpak still
#    carries 2009 apostrophe and stripped-diacritic spellings - "senin birak
#    ózin" where the standard is "seniń biraq óziń". Those would become
#    training *targets*, teaching the model to produce inconsistent spelling.
# 2. Quality. Phase 3 already dropped the rows whose "Karakalpak" column is
#    actually English, along with the length and symbol outliers.
#
# `meta.parallel_text` survives cleaning, so the partner side is still here -
# untouched, correctly, since Karakalpak orthography rules must not be applied
# to English or Russian.
DILMASH = PROCESSED_DIR / "hf_dilmash_parallel.jsonl.zst"

_MIN_CHARS = 15
_MAX_CHARS = 2_000
# A translation whose sides differ wildly in length is usually misaligned.
_MAX_LENGTH_RATIO = 3.0


@dataclass(slots=True)
class TranslationStats:
    read: int = 0
    emitted: int = 0
    skipped_length: int = 0
    skipped_ratio: int = 0
    skipped_identical: int = 0
    by_direction: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"translation: {self.emitted:,} records from {self.read:,} pairs "
            f"(dropped {self.skipped_length:,} by length, {self.skipped_ratio:,} by ratio, "
            f"{self.skipped_identical:,} identical)"
        )


def _acceptable(source: str, target: str, stats: TranslationStats) -> bool:
    if not (_MIN_CHARS <= len(source) <= _MAX_CHARS):
        stats.skipped_length += 1
        return False
    if not (_MIN_CHARS <= len(target) <= _MAX_CHARS):
        stats.skipped_length += 1
        return False
    if source.strip() == target.strip():
        # Numbers, names and dates translate to themselves; they teach copying,
        # not translation.
        stats.skipped_identical += 1
        return False
    ratio = max(len(source), len(target)) / min(len(source), len(target))
    if ratio > _MAX_LENGTH_RATIO:
        stats.skipped_ratio += 1
        return False
    return True


def build(
    *,
    limit: int | None = None,
    both_directions: bool = True,
    seed: int = 20260731,
    source_path: str | None = None,
) -> Iterator[TranslationRecord]:
    """Yield translation records from the ingested dilmash corpus."""
    path = source_path or DILMASH
    stats = TranslationStats()
    rng = random.Random(seed)

    provenance = Provenance(
        source_id="hf_dilmash_parallel",
        source_url="https://huggingface.co/datasets/tahrirchi/dilmash",
        license="MIT",
        human_reviewed=False,
    )

    for row in read_jsonl(path):
        if limit is not None and stats.emitted >= limit:
            break

        meta = row.get("meta", {})
        partner_lang = meta.get("parallel_lang")
        partner_text = meta.get("parallel_text")
        kaa_text = str(row.get("text", ""))

        if not isinstance(partner_text, str) or not isinstance(partner_lang, str):
            continue
        stats.read += 1

        if not _acceptable(kaa_text, partner_text, stats):
            continue

        other = partner_lang.split("_")[0].lower()

        # Out of Karakalpak.
        yield TranslationRecord(
            source_lang="kaa",
            target_lang=other,
            source_text=kaa_text,
            target_text=partner_text,
            provenance=provenance,
        )
        stats.emitted += 1
        stats.by_direction[f"kaa->{other}"] = stats.by_direction.get(f"kaa->{other}", 0) + 1

        # Into Karakalpak. This is the direction a Karakalpak assistant needs
        # most, and a one-directional dataset does not teach it.
        if both_directions and (limit is None or stats.emitted < limit):
            yield TranslationRecord(
                source_lang=other,
                target_lang="kaa",
                source_text=partner_text,
                target_text=kaa_text,
                provenance=provenance,
            )
            stats.emitted += 1
            key = f"{other}->kaa"
            stats.by_direction[key] = stats.by_direction.get(key, 0) + 1

    logger.info("Translation builder finished", extra={"summary": stats.summary()})
    if stats.by_direction:
        logger.info("Translation directions", extra=dict(stats.by_direction))
    _ = rng  # reserved for future sampling; kept so the seed is part of the API
