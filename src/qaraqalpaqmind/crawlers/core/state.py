"""Resumable crawl state, backed by SQLite.

A crawl of two thousand articles at three seconds per request runs for over an
hour. It *will* be interrupted - by a dropped connection, a laptop lid, or a
Ctrl-C. Restarting from zero each time is not acceptable, and neither is
re-fetching pages we already have.

Every URL therefore lives in a durable table with a status. Restarting a crawl
simply resumes handing out `pending` rows. SQLite is the right tool here: one
file, no server, transactional, and trivially inspectable with `sqlite3` when
something looks wrong.

Deliberately synchronous. Local SQLite writes take microseconds while every
request takes seconds, so the event-loop blocking is irrelevant and avoiding an
async driver keeps the code obvious.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType

from ...common.io import text_sha1
from ...common.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url_hash      TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL,
    url           TEXT NOT NULL,
    status        TEXT NOT NULL,
    depth         INTEGER NOT NULL DEFAULT 0,
    attempts      INTEGER NOT NULL DEFAULT 0,
    http_status   INTEGER,
    content_hash  TEXT,
    content_path  TEXT,
    content_type  TEXT,
    error         TEXT,
    discovered_at TEXT NOT NULL,
    fetched_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_urls_source_status ON urls (source_id, status);
CREATE INDEX IF NOT EXISTS idx_urls_content_hash  ON urls (content_hash);
"""


class UrlStatus(StrEnum):
    PENDING = "pending"
    FETCHED = "fetched"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class UrlRecord:
    url_hash: str
    source_id: str
    url: str
    status: UrlStatus
    depth: int
    attempts: int


def url_hash(url: str) -> str:
    """Stable primary key for a URL. Normalise before calling."""
    return text_sha1(url)


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class CrawlState:
    """Durable frontier and result log for one or more sources."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL survives a hard kill without corrupting the frontier.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def __enter__(self) -> CrawlState:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # --- frontier ---------------------------------------------------------

    def add_urls(self, source_id: str, urls: Iterable[str], depth: int = 0) -> int:
        """Register URLs as pending. Already-known URLs are left untouched.

        Returns the number of genuinely new URLs, which is what makes a crawl
        loop terminate: when discovery stops adding rows, the site is exhausted.
        """
        rows = [(url_hash(u), source_id, u, UrlStatus.PENDING.value, depth, _now()) for u in urls]
        if not rows:
            return 0

        with closing(self._conn.cursor()) as cursor:
            cursor.execute("BEGIN")
            cursor.executemany(
                """
                INSERT OR IGNORE INTO urls
                    (url_hash, source_id, url, status, depth, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            added = cursor.rowcount
            cursor.execute("COMMIT")

        if added:
            logger.debug("Frontier grew", extra={"source": source_id, "added": added})
        return added

    def next_pending(self, source_id: str, limit: int = 50) -> list[UrlRecord]:
        """Claim up to `limit` pending URLs, shallowest first."""
        cursor = self._conn.execute(
            """
            SELECT url_hash, source_id, url, status, depth, attempts
            FROM urls
            WHERE source_id = ? AND status = ?
            ORDER BY depth ASC, rowid ASC
            LIMIT ?
            """,
            (source_id, UrlStatus.PENDING.value, limit),
        )
        return [
            UrlRecord(
                url_hash=row["url_hash"],
                source_id=row["source_id"],
                url=row["url"],
                status=UrlStatus(row["status"]),
                depth=row["depth"],
                attempts=row["attempts"],
            )
            for row in cursor.fetchall()
        ]

    def known(self, urls: Iterable[str]) -> set[str]:
        """Subset of `urls` already present in the table."""
        candidates = list(urls)
        found: set[str] = set()
        for start in range(0, len(candidates), 500):
            chunk = candidates[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            cursor = self._conn.execute(
                f"SELECT url FROM urls WHERE url_hash IN ({placeholders})",
                [url_hash(u) for u in chunk],
            )
            found.update(row["url"] for row in cursor.fetchall())
        return found

    # --- results ----------------------------------------------------------

    def mark_fetched(
        self,
        url_hash_value: str,
        *,
        http_status: int,
        content_hash: str,
        content_path: str,
        content_type: str,
    ) -> None:
        self._conn.execute(
            """
            UPDATE urls
            SET status = ?, http_status = ?, content_hash = ?, content_path = ?,
                content_type = ?, fetched_at = ?, error = NULL,
                attempts = attempts + 1
            WHERE url_hash = ?
            """,
            (
                UrlStatus.FETCHED.value,
                http_status,
                content_hash,
                content_path,
                content_type,
                _now(),
                url_hash_value,
            ),
        )

    def mark_failed(self, url_hash_value: str, *, error: str, http_status: int | None = None) -> None:
        self._conn.execute(
            """
            UPDATE urls
            SET status = ?, error = ?, http_status = ?, fetched_at = ?,
                attempts = attempts + 1
            WHERE url_hash = ?
            """,
            (UrlStatus.FAILED.value, error[:500], http_status, _now(), url_hash_value),
        )

    def mark_skipped(self, url_hash_value: str, *, reason: str) -> None:
        self._conn.execute(
            "UPDATE urls SET status = ?, error = ?, fetched_at = ? WHERE url_hash = ?",
            (UrlStatus.SKIPPED.value, reason[:500], _now(), url_hash_value),
        )

    def retry_failed(self, source_id: str, max_attempts: int = 3) -> int:
        """Return failed URLs under the attempt ceiling to the frontier."""
        cursor = self._conn.execute(
            "UPDATE urls SET status = ? WHERE source_id = ? AND status = ? AND attempts < ?",
            (UrlStatus.PENDING.value, source_id, UrlStatus.FAILED.value, max_attempts),
        )
        return cursor.rowcount

    # --- reporting --------------------------------------------------------

    def stats(self, source_id: str | None = None) -> dict[str, int]:
        """Counts per status, for the whole DB or one source."""
        if source_id is None:
            cursor = self._conn.execute("SELECT status, COUNT(*) AS n FROM urls GROUP BY status")
        else:
            cursor = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM urls WHERE source_id = ? GROUP BY status",
                (source_id,),
            )
        return {row["status"]: row["n"] for row in cursor.fetchall()}

    def iter_fetched(self, source_id: str) -> Iterator[sqlite3.Row]:
        """Every successfully fetched row for a source, for the extraction stage."""
        cursor = self._conn.execute(
            """
            SELECT url, content_path, content_type, content_hash, fetched_at
            FROM urls
            WHERE source_id = ? AND status = ? AND content_path IS NOT NULL
            ORDER BY rowid
            """,
            (source_id, UrlStatus.FETCHED.value),
        )
        yield from cursor
