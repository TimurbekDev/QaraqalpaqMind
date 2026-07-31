"""Near-duplicate detection with MinHash and LSH.

Exact hashing misses the cases that matter most here: the same article with a
different headline, a wiki page scraped twice a year apart, a court ruling
republished with an added paragraph. MinHash estimates Jaccard similarity over
shingles, and LSH makes finding the similar pairs sub-quadratic - which it has
to be, because 346k documents is 60 billion pairs.

**Short documents need different shingles from long ones.** Two thirds of this
corpus is sentence-level data, and a 5-word shingle over an 8-word sentence
yields four shingles - far too few for a stable similarity estimate. So short
texts are shingled over characters instead, where an 8-word sentence still
produces ~40 shingles.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from ..common.config import StrictModel
from ..common.logging import get_logger
from .exact import canonical_form

logger = get_logger(__name__)

# Fixed so a rerun reproduces the same clusters.
_SEED = 20260731


class MinHashConfig(StrictModel):
    """Shingling and LSH parameters."""

    num_perm: int = Field(default=128, ge=16, le=512)
    threshold: float = Field(
        default=0.80,
        ge=0.1,
        le=1.0,
        description="Jaccard similarity above which two documents are duplicates.",
    )
    word_shingle: int = Field(default=5, ge=1, le=10)
    char_shingle: int = Field(default=12, ge=4, le=40)
    # Below this word count a document is shingled over characters instead.
    short_document_words: int = Field(default=30, ge=1)


@dataclass(frozen=True, slots=True)
class DuplicateCluster:
    """One group of near-identical documents."""

    winner: str
    duplicates: tuple[str, ...]

    @property
    def size(self) -> int:
        return 1 + len(self.duplicates)


def shingles(text: str, config: MinHashConfig | None = None) -> set[str]:
    """Break text into overlapping pieces for similarity comparison.

    Word shingles for prose; character shingles for anything short, where word
    shingles would be too few to estimate similarity from.
    """
    cfg = config or MinHashConfig()
    canonical = canonical_form(text)
    words = canonical.split()

    if len(words) >= cfg.short_document_words:
        size = cfg.word_shingle
        if len(words) < size:
            return {" ".join(words)}
        return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}

    size = cfg.char_shingle
    if len(canonical) < size:
        return {canonical} if canonical else set()
    return {canonical[i : i + size] for i in range(len(canonical) - size + 1)}


def _build_minhash(text: str, config: MinHashConfig) -> Any:
    """Build a MinHash signature. Returns `datasketch.MinHash`, which is untyped."""
    from datasketch import MinHash

    signature = MinHash(num_perm=config.num_perm, seed=_SEED)
    for shingle in shingles(text, config):
        signature.update(shingle.encode("utf-8"))
    return signature


def find_near_duplicates(
    documents: Iterable[tuple[str, str]],
    config: MinHashConfig | None = None,
) -> list[DuplicateCluster]:
    """Group near-identical documents.

    Args:
        documents: `(document_id, text)` pairs in priority order - the first
            member of a cluster becomes its winner, so the caller sorts by
            source quality before calling.

    Returns:
        One cluster per group of duplicates. Documents with no duplicate do not
        appear.
    """
    from datasketch import MinHashLSH

    cfg = config or MinHashConfig()
    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)

    winner_of: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    processed = 0

    for document_id, text in documents:
        processed += 1
        if processed % 50_000 == 0:
            logger.info(
                "Near-duplicate scan progress",
                extra={"processed": processed, "clusters": len(members)},
            )

        signature = _build_minhash(text, cfg)
        matches = lsh.query(signature)

        if matches:
            # Attach to the cluster of the first match, following it to its
            # root so chains of similar documents collapse into one group.
            winner = winner_of.get(str(matches[0]), str(matches[0]))
            winner_of[document_id] = winner
            members.setdefault(winner, []).append(document_id)
        else:
            winner_of[document_id] = document_id
            lsh.insert(document_id, signature)

    clusters = [
        DuplicateCluster(winner=winner, duplicates=tuple(duplicates))
        for winner, duplicates in members.items()
    ]
    logger.info(
        "Near-duplicate scan complete",
        extra={
            "documents": processed,
            "clusters": len(clusters),
            "removed": sum(len(c.duplicates) for c in clusters),
        },
    )
    return clusters


def duplicate_ids(clusters: Iterable[DuplicateCluster]) -> set[str]:
    """Every id that should be dropped, given a set of clusters."""
    return {document_id for cluster in clusters for document_id in cluster.duplicates}


def iter_survivors(
    documents: Iterable[tuple[str, str]], config: MinHashConfig | None = None
) -> Iterator[str]:
    """Ids of documents that are not near-duplicates of an earlier one."""
    dropped = duplicate_ids(find_near_duplicates(documents, config))
    for document_id, _ in documents:
        if document_id not in dropped:
            yield document_id
