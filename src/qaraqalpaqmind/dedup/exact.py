"""Exact-duplicate detection.

Cheap, certain, and worth doing before anything expensive: a hash comparison
settles identical documents in one pass, and on this corpus a large share of
the duplication is exact rather than approximate. GlotCC is largely scraped
`kaa.wikipedia`, and the crawlers picked up the same article under several URLs.

The hash is taken over a *canonical* form of the text - case-folded, whitespace
collapsed, punctuation dropped - so that two documents differing only in
formatting collapse together. Normalisation in Phase 3.2 already removed the
invisible differences; this handles the visible-but-meaningless ones.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator

from ..common.logging import get_logger

logger = get_logger(__name__)

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

def canonical_form(text: str) -> str:
    """Reduce text to what matters for identity comparison.

    Case, punctuation and whitespace layout are discarded. Letters - including
    the Karakalpak acute-accented ones - are preserved exactly, so `sóz` and
    `soz` remain distinct documents. They are different words, and merging them
    would collapse documents that genuinely differ.

    KNOWN LIMIT: an ALL-CAPS copy of a document may not match its original.
    Karakalpak distinguishes dotted `i` from dotless `ı`, but `.upper()` maps
    *both* to `I`, so uppercase text is genuinely ambiguous and no folding
    recovers it. Mapping `I` back to `ı` was tried and is worse: it fixes rare
    all-caps documents while breaking the common case, where a sentence-initial
    `Islam` would stop matching a mid-sentence `islam`. Plain casefold is
    correct for mixed-case text, which is nearly all of the corpus; all-caps
    duplicates are left to the near-duplicate pass.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Stable 16-byte hash of a document's canonical form.

    Truncated SHA-256 rather than the full digest: 128 bits makes an accidental
    collision impossible at this corpus size and halves the memory held while
    several hundred thousand hashes are in a dict.
    """
    return hashlib.sha256(canonical_form(text).encode("utf-8")).hexdigest()[:32]


def find_exact_duplicates(
    documents: Iterable[tuple[str, str]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map every document id to the id of its canonical copy.

    Args:
        documents: `(document_id, text)` pairs, already in the order that
            decides which copy wins - the first one seen for a given hash is
            kept, so the caller sorts by source priority before calling.

    Returns:
        `(winner_by_id, duplicates_by_winner)`. Every id appears in the first
        mapping; ids that are their own winner are the survivors.
    """
    winner_for_hash: dict[str, str] = {}
    winner_by_id: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}

    for document_id, text in documents:
        digest = content_hash(text)
        winner = winner_for_hash.get(digest)
        if winner is None:
            winner_for_hash[digest] = document_id
            winner_by_id[document_id] = document_id
        else:
            winner_by_id[document_id] = winner
            duplicates.setdefault(winner, []).append(document_id)

    logger.info(
        "Exact duplicate scan complete",
        extra={
            "documents": len(winner_by_id),
            "unique": len(winner_for_hash),
            "duplicates": len(winner_by_id) - len(winner_for_hash),
        },
    )
    return winner_by_id, duplicates


def iter_unique(documents: Iterable[tuple[str, str]]) -> Iterator[str]:
    """Stream the ids of exact-unique documents, holding only hashes in memory."""
    seen: set[str] = set()
    for document_id, text in documents:
        digest = content_hash(text)
        if digest not in seen:
            seen.add(digest)
            yield document_id
