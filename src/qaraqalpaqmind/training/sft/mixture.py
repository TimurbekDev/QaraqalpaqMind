"""Assemble the SFT training mixture.

Running the builders produces wildly unbalanced output: dilmash yields ~430k
translation records while the seed sets yield 45. Training on that directly
would produce a translation engine that occasionally answers questions.

So the mixture is *capped per task*, not merely concatenated. The caps in
`DEFAULT_MIXTURE` are proportions of the final dataset, chosen so that no task
can dominate simply because its data was cheap to obtain.

The assembler also does three things that must not be left to the trainer:

* deduplicates across tasks, since the grammar and summarisation builders both
  draw on the same corpus
* checks every record against the benchmark contamination blocklist
* splits train and validation *before* shuffling task order, so the validation
  set covers every task rather than whichever one happened to land last
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ...common.io import write_jsonl
from ...common.logging import get_logger
from ...common.paths import SFT_DIR, ensure_dir
from ...dedup.blocklist import is_contaminated, load_blocklist
from ...dedup.exact import content_hash
from ...schemas import TaskRecord, TaskType

logger = get_logger(__name__)

# Share of the final mixture, per task. Deliberately not proportional to how
# much data each builder can produce.
#
# Translation is capped at 30% despite being able to supply 90%: it is the one
# task with abundant data, and letting abundance decide the mixture would train
# a translator that occasionally answers questions.
DEFAULT_MIXTURE: dict[TaskType, float] = {
    TaskType.TRANSLATION: 0.30,
    TaskType.GRAMMAR: 0.20,
    TaskType.SUMMARIZATION: 0.15,
    TaskType.INSTRUCTION: 0.12,
    TaskType.QA: 0.10,
    TaskType.CONVERSATION: 0.05,
    TaskType.REASONING: 0.04,
    TaskType.MATH: 0.02,
    TaskType.CODING: 0.02,
}


@dataclass(slots=True)
class MixtureStats:
    """What assembly produced, and what it discarded."""

    offered: int = 0
    kept: int = 0
    duplicates: int = 0
    contaminated: int = 0
    over_cap: int = 0
    train: int = 0
    validation: int = 0
    by_task: dict[str, int] = field(default_factory=dict)
    shortfall: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"mixture: {self.kept:,} of {self.offered:,} offered "
            f"(train {self.train:,} / val {self.validation:,}; "
            f"dropped {self.duplicates:,} duplicates, {self.over_cap:,} over cap, "
            f"{self.contaminated:,} contaminated)"
        )


def resolve_caps(target_size: int, mixture: dict[TaskType, float]) -> dict[TaskType, int]:
    """Turn proportions into absolute per-task record caps."""
    total = sum(mixture.values())
    if total <= 0:
        raise ValueError("mixture proportions sum to zero")
    return {task: max(1, int(target_size * share / total)) for task, share in mixture.items()}


def achievable_size(
    available: dict[TaskType, int], mixture: dict[TaskType, float]
) -> tuple[int, TaskType | None]:
    """Largest mixture that respects the proportions, given what exists.

    Answers the question the shortfall report raises: "so how big a *balanced*
    dataset can I actually build?" The limit is set by whichever task has the
    least data relative to its intended share.
    """
    total = sum(mixture.values())
    if total <= 0:
        return 0, None

    best: int | None = None
    binding: TaskType | None = None
    for task, share in mixture.items():
        if share <= 0:
            continue
        limit = int(available.get(task, 0) * total / share)
        if best is None or limit < best:
            best, binding = limit, task
    return (best or 0), binding


def assemble(
    sources: Iterable[TaskRecord],
    *,
    target_size: int = 50_000,
    mixture: dict[TaskType, float] | None = None,
    validation_split: float = 0.02,
    seed: int = 20260731,
    check_contamination: bool = True,
) -> tuple[list[TaskRecord], list[TaskRecord], MixtureStats]:
    """Build a balanced, deduplicated, contamination-checked SFT mixture."""
    caps = resolve_caps(target_size, mixture or DEFAULT_MIXTURE)
    blocked = load_blocklist() if check_contamination else set()
    stats = MixtureStats()

    seen: set[str] = set()
    collected: dict[TaskType, list[TaskRecord]] = {task: [] for task in caps}

    for record in sources:
        stats.offered += 1
        cap = caps.get(record.task)
        if cap is None:
            continue
        bucket = collected[record.task]
        if len(bucket) >= cap:
            stats.over_cap += 1
            continue

        text = record.training_text()
        digest = content_hash(text)
        if digest in seen:
            # The grammar and summarisation builders both draw on pretrain_v1,
            # so cross-task duplication is expected rather than exceptional.
            stats.duplicates += 1
            continue

        if blocked and is_contaminated(text, blocked):
            stats.contaminated += 1
            continue

        seen.add(digest)
        bucket.append(record)
        stats.kept += 1

    rng = random.Random(seed)
    train: list[TaskRecord] = []
    validation: list[TaskRecord] = []

    for task, records in collected.items():
        stats.by_task[task.value] = len(records)
        cap = caps[task]
        if len(records) < cap:
            stats.shortfall[task.value] = cap - len(records)

        # Split per task, so validation covers every task rather than whichever
        # one happened to be last in the stream.
        rng.shuffle(records)
        cut = int(len(records) * validation_split)
        validation.extend(records[:cut])
        train.extend(records[cut:])

    rng.shuffle(train)
    rng.shuffle(validation)
    stats.train, stats.validation = len(train), len(validation)

    logger.info("Mixture assembled", extra={"summary": stats.summary()})
    if stats.shortfall:
        logger.warning(
            "Some tasks could not fill their share of the mixture",
            extra=dict(stats.shortfall),
        )
    return train, validation, stats


def write_mixture(
    train: list[TaskRecord],
    validation: list[TaskRecord],
    name: str = "sft_v1",
) -> tuple[Path, Path]:
    """Write train and validation splits to `data/datasets/sft/`."""
    directory = ensure_dir(SFT_DIR)
    train_path = directory / f"{name}_train.jsonl.zst"
    validation_path = directory / f"{name}_val.jsonl.zst"

    write_jsonl(train_path, (r.model_dump(mode="json") for r in train))
    write_jsonl(validation_path, (r.model_dump(mode="json") for r in validation))
    return train_path, validation_path


def chain_builders(*builders: Iterator[TaskRecord]) -> Iterator[TaskRecord]:
    """Interleave builder outputs so no single source fills the caps first.

    Concatenating instead would let translation - which yields hundreds of
    thousands of records - exhaust the deduplication budget before the seed
    sets are read at all.
    """
    active = list(builders)
    while active:
        for builder in list(active):
            try:
                yield next(builder)
            except StopIteration:
                active.remove(builder)
