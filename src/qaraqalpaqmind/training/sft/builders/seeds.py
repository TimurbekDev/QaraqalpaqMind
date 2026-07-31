"""Hand-authored seed data for tasks with no existing Karakalpak corpus.

Translation, grammar and summarisation can be derived from real data. Nothing
in Karakalpak exists for instruction following, dialogue, reasoning, coding or
mathematics - so those start from written examples.

Seeds live in `seeds/*.jsonl` at the repository root, committed and reviewable,
for the same reason `benchmarks/` is: they are small, hand-checked, and must be
diffable. The format is deliberately minimal - task plus its fields - and this
loader supplies provenance, so contributing an example does not require knowing
the record schema.

A seed set is a starting point, not a dataset. Scaling it means either paying
for human authoring or generating from a model, and the second path must set
`synthetic=True` so the distinction survives into the training mixture.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....common.paths import PROJECT_ROOT
from ....schemas import Provenance, TaskRecord, TaskType, parse_record

logger = get_logger(__name__)

SEEDS_DIR = PROJECT_ROOT / "seeds"


@dataclass(slots=True)
class SeedStats:
    files: int = 0
    read: int = 0
    emitted: int = 0
    invalid: int = 0
    by_task: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"seeds: {self.emitted:,} records from {self.files} files "
            f"({self.invalid} invalid)"
        )


def _default_provenance(path: Path) -> Provenance:
    return Provenance(
        source_id=f"seed_{path.stem}",
        license="CC0-1.0",
        synthetic=False,
        human_reviewed=True,
    )


def load_file(path: Path, stats: SeedStats | None = None) -> Iterator[TaskRecord]:
    """Load and validate one seed file.

    Invalid rows are reported with their line number and skipped, rather than
    aborting the load. A malformed contribution should not stop the other
    hundred from being usable.
    """
    tracker = stats if stats is not None else SeedStats()
    provenance = _default_provenance(path)

    for line_number, row in enumerate(read_jsonl(path), start=1):
        tracker.read += 1
        payload = dict(row)
        payload.setdefault("provenance", provenance.model_dump(mode="json"))

        try:
            record = parse_record(payload)
        except (ValueError, TypeError) as exc:
            tracker.invalid += 1
            logger.warning(
                "Invalid seed record",
                extra={"file": path.name, "line": line_number, "error": str(exc).split("\n")[0]},
            )
            continue

        tracker.by_task[record.task.value] = tracker.by_task.get(record.task.value, 0) + 1
        tracker.emitted += 1
        yield record


def build(
    *,
    tasks: tuple[TaskType, ...] | None = None,
    seeds_dir: Path | None = None,
    limit: int | None = None,
) -> Iterator[TaskRecord]:
    """Load every seed file, optionally restricted to certain task types."""
    directory = seeds_dir or SEEDS_DIR
    stats = SeedStats()

    if not directory.is_dir():
        logger.warning("No seeds directory", extra={"path": str(directory)})
        return

    wanted = set(tasks) if tasks else None
    emitted = 0

    for path in sorted(directory.glob("*.jsonl")):
        stats.files += 1
        for record in load_file(path, stats):
            if wanted is not None and record.task not in wanted:
                continue
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                logger.info("Seed builder finished", extra={"summary": stats.summary()})
                return

    logger.info("Seed builder finished", extra={"summary": stats.summary()})
    if stats.by_task:
        logger.info("Seed task counts", extra=dict(stats.by_task))


def available_tasks(seeds_dir: Path | None = None) -> dict[str, int]:
    """Count seed records per task, for reporting what exists."""
    counts: dict[str, int] = {}
    for record in build(seeds_dir=seeds_dir):
        counts[record.task.value] = counts.get(record.task.value, 0) + 1
    return counts
