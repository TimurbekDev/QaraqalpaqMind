"""Response-quality preference pairs, from the Phase 3 quality scores.

Every cleaned document carries a score and flags. Documents that were *rejected*
during cleaning still exist in `data/processed/rejected/`, and pairing a kept
document against a rejected one from the same source teaches the difference
between usable prose and boilerplate, truncation or symbol soup.

This is the weakest of the three real-signal builders and the reason is worth
stating: the two sides are different documents, so they differ in content as
well as quality. Pairs are restricted to the same source and a similar length to
narrow that, but the signal is still less clean than the orthography and
language builders, where both sides say the same thing.

It is included because the failure it targets - a model that emits navigation
menus and truncated fragments - is real and visible, and because the alternative
is having no quality signal at all until a model exists to sample from.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ....cleaning.filters import compute_stats
from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....common.paths import PROCESSED_DIR
from ....schemas import PreferenceRecord, Provenance

logger = get_logger(__name__)

CRITERION = "response_quality"

_MIN_CHARS = 150
_MAX_CHARS = 1_200
# Sides must be within this factor of each other, so the pair does not simply
# teach "longer is better".
_MAX_LENGTH_RATIO = 1.6

# The chosen side must be good prose, not merely text that survived cleaning.
# Those are different bars, and conflating them produced a pair whose "chosen"
# side was an arithmetic table - "10 + 6 = 16 / 10 + 7 = 17 / ..." - which had
# passed the cleaner but would teach the model to emit multiplication tables.
_MIN_CHOSEN_SCORE = 0.7
_MAX_TOP_WORD_FRACTION = 0.12
_MIN_MEAN_WORD_LENGTH = 4.0
_MIN_ALPHA_WORD_FRACTION = 0.85
_MAX_DIGIT_RATIO = 0.05

_PROMPTS = (
    "Tómendegi tema boyınsha maǵlıwmat jazıp ber:",
    "Usı mániste tekst jaz:",
)


@dataclass(slots=True)
class QualityStats:
    kept_read: int = 0
    rejected_read: int = 0
    emitted: int = 0
    unmatched: int = 0
    not_prose: int = 0
    by_flag: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"quality preferences: {self.emitted:,} pairs from {self.kept_read:,} kept and "
            f"{self.rejected_read:,} rejected documents ({self.unmatched:,} unmatched, "
            f"{self.not_prose:,} chosen candidates rejected as not prose)"
        )


def is_good_prose(text: str) -> bool:
    """Is this text worth teaching the model to imitate?

    Surviving the cleaner is a lower bar than being a good response. Tables,
    lists of numbers and heavily repetitive text all pass cleaning and would be
    actively harmful as the `chosen` side of a preference pair.
    """
    stats = compute_stats(text)
    return (
        stats.words >= 20
        and stats.mean_word_length >= _MIN_MEAN_WORD_LENGTH
        and stats.alpha_word_fraction >= _MIN_ALPHA_WORD_FRACTION
        and stats.digit_ratio <= _MAX_DIGIT_RATIO
        and stats.top_word_fraction <= _MAX_TOP_WORD_FRACTION
        and stats.ends_with_sentence
    )


def _load_rejected(source_id: str) -> list[dict[str, object]]:
    path = PROCESSED_DIR / "rejected" / f"{source_id}.jsonl.zst"
    if not path.exists():
        return []
    return [
        row
        for row in read_jsonl(path)
        if _MIN_CHARS <= len(str(row.get("text", ""))) <= _MAX_CHARS
    ]


def build(
    *,
    limit: int | None = None,
    sources: tuple[str, ...] = ("gov_jokargikenes", "edu_ndpi", "gov_qrdsm", "wiki_kaa"),
    seed: int = 20260731,
    processed_dir: Path | None = None,
) -> Iterator[PreferenceRecord]:
    """Yield pairs preferring a kept document over a rejected one."""
    rng = random.Random(seed)
    stats = QualityStats()
    directory = processed_dir or PROCESSED_DIR

    provenance = Provenance(
        source_id="dpo_response_quality",
        license="derived from pretrain_v1; see per-source manifests",
        synthetic=False,
        human_reviewed=False,
    )

    for source_id in sources:
        kept_path = directory / f"{source_id}.jsonl.zst"
        if not kept_path.exists():
            continue

        rejected = _load_rejected(source_id)
        stats.rejected_read += len(rejected)
        if not rejected:
            stats.unmatched += 1
            continue

        for row in read_jsonl(kept_path):
            if limit is not None and stats.emitted >= limit:
                return

            text = str(row.get("text", ""))
            if not (_MIN_CHARS <= len(text) <= _MAX_CHARS):
                continue
            stats.kept_read += 1

            quality_block = row.get("quality")
            score = (
                float(quality_block.get("score") or 0.0)
                if isinstance(quality_block, dict)
                else 0.0
            )
            if score < _MIN_CHOSEN_SCORE:
                continue
            if not is_good_prose(text):
                stats.not_prose += 1
                continue

            # Pick a rejected document of comparable length, so the pair does
            # not accidentally teach that longer answers are better.
            candidates = [
                candidate
                for candidate in rejected
                if 1 / _MAX_LENGTH_RATIO
                <= len(str(candidate.get("text", ""))) / len(text)
                <= _MAX_LENGTH_RATIO
            ]
            if not candidates:
                continue

            worse = candidates[rng.randrange(len(candidates))]
            worse_text = str(worse.get("text", ""))
            if worse_text.strip() == text.strip():
                continue

            worse_quality = worse.get("quality")
            flags = (
                list(worse_quality.get("flags") or [])
                if isinstance(worse_quality, dict)
                else []
            )
            prompt = _PROMPTS[rng.randrange(len(_PROMPTS))]
            topic = " ".join(text.split()[:8])

            yield PreferenceRecord(
                prompt=f"{prompt}\n\n{topic}",
                chosen=text,
                rejected=worse_text,
                criterion=CRITERION,
                provenance=provenance,
                meta={"source_id": source_id, "rejected_flags": flags, "chosen_score": score},
            )
            stats.emitted += 1
            for flag in flags:
                stats.by_flag[flag] = stats.by_flag.get(flag, 0) + 1

    logger.info("Quality preference builder finished", extra={"summary": stats.summary()})
    if stats.by_flag:
        logger.info("Rejected-side flags", extra=dict(stats.by_flag))
