"""Benchmark contamination blocking.

FLORES+ is the standard translation benchmark for low-resource languages, which
means it is also *on the web*, and therefore inside the web-derived parts of
this corpus. GlotCC scraped pages that quote it. If those sentences reach the
training set, Phase 8 translation scores measure memorisation and nothing else.

So every held-out sentence is hashed and any training document containing one
is dropped. This is not optional and it is not a heuristic: a benchmark that
leaked is a benchmark that lies.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson

from ..common.io import read_jsonl
from ..common.logging import get_logger
from ..common.paths import DATASETS_DIR, ensure_dir
from .exact import content_hash

logger = get_logger(__name__)

_MIN_BLOCKED_WORDS = 6


def blocklist_path() -> Path:
    return ensure_dir(DATASETS_DIR / "eval") / "contamination_blocklist.json"


def build_blocklist(sources: Iterable[Path]) -> set[str]:
    """Hash every sentence of every held-out file.

    Very short sentences are skipped: they recur naturally in any corpus, so
    blocking them would delete legitimate text without protecting anything.
    """
    hashes: set[str] = set()
    skipped = 0

    for path in sources:
        if not path.exists():
            logger.warning("Held-out file missing", extra={"path": str(path)})
            continue
        for row in read_jsonl(path):
            text = str(row.get("text", ""))
            for sentence in _sentences(text):
                if len(sentence.split()) < _MIN_BLOCKED_WORDS:
                    skipped += 1
                    continue
                hashes.add(content_hash(sentence))

    logger.info(
        "Blocklist built", extra={"hashes": len(hashes), "skipped_short": skipped}
    )
    return hashes


def _sentences(text: str) -> Iterator[str]:
    for chunk in text.replace("!", ".").replace("?", ".").split("."):
        cleaned = chunk.strip()
        if cleaned:
            yield cleaned


def save_blocklist(hashes: set[str]) -> Path:
    target = blocklist_path()
    target.write_bytes(orjson.dumps({"count": len(hashes), "hashes": sorted(hashes)}))
    logger.info("Blocklist saved", extra={"path": str(target), "count": len(hashes)})
    return target


def load_blocklist() -> set[str]:
    """Load the blocklist, or an empty set if none has been built yet."""
    target = blocklist_path()
    if not target.exists():
        logger.warning(
            "NO CONTAMINATION BLOCKLIST. Benchmark sentences may be in the training "
            "corpus, which would make Phase 8 scores meaningless. Build it with "
            "`qm dedup blocklist` once FLORES+ has been ingested."
        )
        return set()
    payload = orjson.loads(target.read_bytes())
    return set(payload["hashes"])


def is_contaminated(text: str, blocked: set[str]) -> bool:
    """True if any sentence of `text` appears in the held-out set.

    Sentences are split from the ORIGINAL text, not a canonicalised copy:
    canonicalisation strips the punctuation that sentence splitting relies on,
    so splitting afterwards yields one undifferentiated chunk that never
    matches anything. `content_hash` canonicalises each sentence itself.
    """
    if not blocked:
        return False
    return any(
        len(sentence.split()) >= _MIN_BLOCKED_WORDS and content_hash(sentence) in blocked
        for sentence in _sentences(text)
    )
