"""Assemble the preference dataset.

Same shape as the SFT mixture and for the same reasons: cap per criterion,
deduplicate, check contamination, split per criterion before shuffling.

One rule is specific to preference data. A pair whose two sides are nearly
identical teaches noise - DPO will happily learn a preference between two
equally good answers if it is given one - so pairs below a minimum edit
distance are dropped rather than trained on.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ...common.io import write_jsonl
from ...common.logging import get_logger
from ...common.paths import DPO_DIR, ensure_dir
from ...dedup.blocklist import is_contaminated, load_blocklist
from ...dedup.exact import content_hash
from ...schemas import PreferenceRecord

logger = get_logger(__name__)

DEFAULT_MIXTURE: dict[str, float] = {
    "language_consistency": 0.40,
    "orthography": 0.40,
    "response_quality": 0.20,
}

# Below this, `chosen` and `rejected` are too alike for the pair to carry a
# signal worth training on.
_MIN_DIFFERENCE_RATIO = 0.02


@dataclass(slots=True)
class PreferenceStats:
    offered: int = 0
    kept: int = 0
    duplicates: int = 0
    contaminated: int = 0
    over_cap: int = 0
    too_similar: int = 0
    train: int = 0
    validation: int = 0
    by_criterion: dict[str, int] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"preferences: {self.kept:,} of {self.offered:,} offered "
            f"(train {self.train:,} / val {self.validation:,}; dropped "
            f"{self.duplicates:,} duplicates, {self.over_cap:,} over cap, "
            f"{self.too_similar:,} too similar, {self.contaminated:,} contaminated)"
        )


def difference_ratio(chosen: str, rejected: str) -> float:
    """Rough share of characters that differ between the two sides."""
    longer = max(len(chosen), len(rejected))
    if longer == 0:
        return 0.0
    shared = sum(1 for a, b in zip(chosen, rejected, strict=False) if a == b)
    return 1.0 - shared / longer


def resolve_caps(target_size: int, mixture: dict[str, float]) -> dict[str, int]:
    total = sum(mixture.values())
    if total <= 0:
        raise ValueError("mixture proportions sum to zero")
    return {name: max(1, int(target_size * share / total)) for name, share in mixture.items()}


def assemble(
    sources: Iterable[PreferenceRecord],
    *,
    target_size: int = 20_000,
    mixture: dict[str, float] | None = None,
    validation_split: float = 0.02,
    seed: int = 20260731,
    check_contamination: bool = True,
) -> tuple[list[PreferenceRecord], list[PreferenceRecord], PreferenceStats]:
    """Build a balanced, deduplicated preference dataset."""
    caps = resolve_caps(target_size, mixture or DEFAULT_MIXTURE)
    blocked = load_blocklist() if check_contamination else set()
    stats = PreferenceStats()

    seen: set[str] = set()
    collected: dict[str, list[PreferenceRecord]] = {name: [] for name in caps}

    for record in sources:
        stats.offered += 1
        cap = caps.get(record.criterion)
        if cap is None:
            continue
        bucket = collected[record.criterion]
        if len(bucket) >= cap:
            stats.over_cap += 1
            continue

        if difference_ratio(record.chosen, record.rejected) < _MIN_DIFFERENCE_RATIO:
            stats.too_similar += 1
            continue

        digest = content_hash(f"{record.prompt}\x00{record.chosen}\x00{record.rejected}")
        if digest in seen:
            stats.duplicates += 1
            continue

        if blocked and is_contaminated(record.chosen, blocked):
            stats.contaminated += 1
            continue

        seen.add(digest)
        bucket.append(record)
        stats.kept += 1

    rng = random.Random(seed)
    train: list[PreferenceRecord] = []
    validation: list[PreferenceRecord] = []

    for criterion, records in collected.items():
        stats.by_criterion[criterion] = len(records)
        if len(records) < caps[criterion]:
            stats.shortfall[criterion] = caps[criterion] - len(records)

        rng.shuffle(records)
        cut = int(len(records) * validation_split)
        validation.extend(records[:cut])
        train.extend(records[cut:])

    rng.shuffle(train)
    rng.shuffle(validation)
    stats.train, stats.validation = len(train), len(validation)

    logger.info("Preference mixture assembled", extra={"summary": stats.summary()})
    if stats.shortfall:
        logger.warning("Criteria short of their share", extra=dict(stats.shortfall))
    return train, validation, stats


def write_mixture(
    train: list[PreferenceRecord],
    validation: list[PreferenceRecord],
    name: str = "dpo_v1",
) -> tuple[Path, Path]:
    """Write train and validation splits to `data/datasets/dpo/`."""
    directory = ensure_dir(DPO_DIR)
    train_path = directory / f"{name}_train.jsonl.zst"
    validation_path = directory / f"{name}_val.jsonl.zst"

    write_jsonl(train_path, (r.model_dump(mode="json") for r in train))
    write_jsonl(validation_path, (r.model_dump(mode="json") for r in validation))
    return train_path, validation_path


def chain_builders(*builders: Iterator[PreferenceRecord]) -> Iterator[PreferenceRecord]:
    """Interleave builders so no single one fills the caps first."""
    active = list(builders)
    while active:
        for builder in list(active):
            try:
                yield next(builder)
            except StopIteration:
                active.remove(builder)
