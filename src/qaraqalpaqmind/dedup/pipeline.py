"""Cross-source deduplication: `data/processed/` -> `data/datasets/pretrain/`.

Deduplication must run **across** sources, not within them. GlotCC is largely
scraped `kaa.wikipedia`; dilmash draws 23% of its pairs from news sites we also
crawl; the same court ruling appears in both sud.uz locales. Deduplicating each
source in isolation would leave every one of those pairs in the corpus.

When two documents are duplicates, the surviving copy is the one from the
higher-quality source. `dilmash` is human-translated and `glotcc` is Common
Crawl, so the Common Crawl copy is the one that goes - and the corpus keeps the
better-edited text rather than whichever happened to be read first.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..common.io import read_jsonl, write_jsonl
from ..common.logging import get_logger
from ..common.paths import PRETRAIN_DIR, PROCESSED_DIR, ensure_dir
from ..common.records import Document
from ..crawlers.core.registry import SourceRegistry, load_registry
from .blocklist import is_contaminated, load_blocklist
from .exact import content_hash
from .minhash import MinHashConfig, duplicate_ids, find_near_duplicates

logger = get_logger(__name__)

_DEFAULT_OUTPUT = "pretrain_v1"


@dataclass(slots=True)
class DedupStats:
    """What deduplication removed, and from where."""

    read: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    contaminated: int = 0
    kept: int = 0
    chars_in: int = 0
    chars_out: int = 0
    removed_by_source: dict[str, int] = field(default_factory=dict)
    kept_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def unique_rate(self) -> float:
        return self.kept / self.read if self.read else 0.0

    def summary(self) -> str:
        return (
            f"read={self.read:,} kept={self.kept:,} ({self.unique_rate:.1%}) "
            f"exact={self.exact_duplicates:,} near={self.near_duplicates:,} "
            f"contaminated={self.contaminated:,} "
            f"chars {self.chars_in / 1e6:.1f}M -> {self.chars_out / 1e6:.1f}M"
        )


def source_priority(registry: SourceRegistry) -> dict[str, tuple[int, int, str]]:
    """Rank sources so the best copy of a duplicate survives.

    Higher registry `quality` wins; ties break on crawl `priority`, then id so
    the result is deterministic across runs.
    """
    return {
        spec.id: (-spec.quality, spec.priority, spec.id) for spec in registry.sources
    }


def _load_all(processed_dir: Path, registry: SourceRegistry) -> list[Document]:
    """Read every processed document, ordered so better sources come first."""
    ranking = source_priority(registry)
    held_out = registry.held_out_ids()
    documents: list[Document] = []

    for path in sorted(processed_dir.glob("*.jsonl.zst")):
        source_id = path.name.split(".")[0]
        if source_id in held_out:
            # Belt and braces: cleaning already refuses to write held-out data
            # here, but a stale file from before that rule would otherwise be
            # loaded straight into the training set.
            logger.warning(
                "Refusing to read held-out source into a training set",
                extra={"source": source_id, "path": str(path)},
            )
            continue
        count = 0
        for row in read_jsonl(path):
            documents.append(Document.model_validate(row))
            count += 1
        logger.info("Loaded source", extra={"source": source_id, "documents": count})

    documents.sort(key=lambda d: ranking.get(d.source_id, (0, 99, d.source_id)))
    return documents


def deduplicate(
    *,
    output_name: str = _DEFAULT_OUTPUT,
    minhash_config: MinHashConfig | None = None,
    skip_near: bool = False,
    processed_dir: Path | None = None,
) -> DedupStats:
    """Deduplicate the processed corpus into a training-ready dataset."""
    registry = load_registry()
    source_dir = processed_dir or PROCESSED_DIR
    documents = _load_all(source_dir, registry)

    stats = DedupStats(read=len(documents))
    stats.chars_in = sum(len(d.text) for d in documents)
    blocked = load_blocklist()

    # --- exact duplicates -------------------------------------------------
    seen_hashes: set[str] = set()
    survivors: list[Document] = []
    for document in documents:
        digest = content_hash(document.text)
        if digest in seen_hashes:
            stats.exact_duplicates += 1
            stats.removed_by_source[document.source_id] = (
                stats.removed_by_source.get(document.source_id, 0) + 1
            )
            continue
        seen_hashes.add(digest)
        survivors.append(document)

    logger.info(
        "Exact deduplication done",
        extra={"removed": stats.exact_duplicates, "remaining": len(survivors)},
    )

    # --- near duplicates --------------------------------------------------
    if not skip_near and survivors:
        clusters = find_near_duplicates(
            ((d.id, d.text) for d in survivors), minhash_config
        )
        dropped = duplicate_ids(clusters)
        stats.near_duplicates = len(dropped)
        for document in survivors:
            if document.id in dropped:
                stats.removed_by_source[document.source_id] = (
                    stats.removed_by_source.get(document.source_id, 0) + 1
                )
        survivors = [d for d in survivors if d.id not in dropped]

    # --- benchmark contamination -----------------------------------------
    def emit() -> Iterator[dict[str, object]]:
        for document in survivors:
            if is_contaminated(document.text, blocked):
                stats.contaminated += 1
                continue
            stats.kept += 1
            stats.chars_out += len(document.text)
            stats.kept_by_source[document.source_id] = (
                stats.kept_by_source.get(document.source_id, 0) + 1
            )
            yield document.model_dump(mode="json")

    target = ensure_dir(PRETRAIN_DIR) / f"{output_name}.jsonl.zst"
    write_jsonl(target, emit())

    logger.info("Deduplication finished", extra={"summary": stats.summary()})
    return stats


def output_path(output_name: str = _DEFAULT_OUTPUT) -> Path:
    return PRETRAIN_DIR / f"{output_name}.jsonl.zst"
