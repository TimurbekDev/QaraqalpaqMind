"""Turn crawled raw HTML into interim `Document`s.

This is the bridge between Phase 2's crawler and the same interim format the
bulk ingesters produce. It reads the crawl state database for provenance and
the blob store for bytes, so extraction can be re-run any number of times
without touching anybody's server - which is the entire point of keeping
`data/raw/` immutable.

Filtering is deliberately minimal here: drop what is not text, and drop pages
with no extractable content. Quality judgement belongs to Phase 3, which can
see the whole corpus and therefore knows what "boilerplate" looks like across
a site rather than within one page.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from ..common.logging import get_logger
from ..common.paths import DATA_DIR
from ..common.records import Document
from ..crawlers.core.registry import SourceSpec
from ..crawlers.core.state import CrawlState
from ..crawlers.core.storage import RawStore
from ..preprocessing.html import extract_main_text
from ..preprocessing.script import detect_script
from .base import Ingester

logger = get_logger(__name__)

_TEXTUAL_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class CrawledIngester(Ingester):
    """Extracts text from `data/raw/<source_id>/` into interim documents."""

    def __init__(
        self,
        spec: SourceSpec,
        *,
        state: CrawlState | None = None,
        store: RawStore | None = None,
    ) -> None:
        super().__init__(spec)
        self._own_state = state is None
        self._state = state or CrawlState(DATA_DIR / "state" / "crawl.db")
        self._store = store or RawStore(spec.id)

    def documents(self, limit: int | None = None) -> Iterator[Document]:
        emitted = skipped_binary = skipped_empty = missing = 0
        seen_hashes: set[str] = set()

        try:
            for row in self._state.iter_fetched(self.spec.id):
                content_type = (row["content_type"] or "").lower()
                if not any(kind in content_type for kind in _TEXTUAL_TYPES):
                    skipped_binary += 1
                    continue

                # Two URLs serving identical bytes share one blob; extracting it
                # twice would only create work for the deduplicator.
                content_hash = row["content_hash"]
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                try:
                    raw = self._store.read(row["content_path"])
                except OSError:
                    missing += 1
                    continue

                page = extract_main_text(_decode(raw), url=row["url"])
                if not page.is_usable:
                    skipped_empty += 1
                    continue

                yield Document.create(
                    text=page.text,
                    source_id=self.spec.id,
                    license=self.spec.license,
                    source_url=row["url"],
                    fetched_at=_parse_time(row["fetched_at"]),
                    script=detect_script(page.text),
                    meta={
                        "title": page.title,
                        "published_at": page.published_at,
                        "author": page.author,
                        "extractor": page.extractor,
                        "content_hash": content_hash,
                    },
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
        finally:
            logger.info(
                "Extraction finished",
                extra={
                    "source": self.spec.id,
                    "emitted": emitted,
                    "duplicate_blobs": len(seen_hashes),
                    "non_text": skipped_binary,
                    "no_content": skipped_empty,
                    "missing_blobs": missing,
                },
            )
            if self._own_state:
                self._state.close()


def _decode(raw: bytes) -> str:
    """Decode page bytes, preferring UTF-8 for the reasons given in fetcher.py."""
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(tz=UTC)


def state_db_path() -> Path:
    return DATA_DIR / "state" / "crawl.db"
