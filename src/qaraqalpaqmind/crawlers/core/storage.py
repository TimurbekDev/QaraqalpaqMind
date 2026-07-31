"""Immutable, content-addressed store for fetched bytes.

`data/raw/` is written once and never edited. When the HTML extractor improves
in Phase 3, we re-run extraction over these bytes instead of re-crawling
somebody's server - which is both faster and the polite thing to do.

Files are addressed by the SHA-256 of their content, not by URL, which gives
exact-duplicate collapsing for free: a site that serves the same article under
three URLs costs one file. The URL to file mapping lives in the crawl state DB.

Paths are sharded by the first two hex characters of the hash, because a flat
directory with 30,000 entries is slow to list on every filesystem we care about.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ...common.logging import get_logger
from ...common.paths import RAW_DIR, ensure_dir

logger = get_logger(__name__)

_EXTENSION_BY_TYPE: dict[str, str] = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "application/xml": ".xml",
    "text/xml": ".xml",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


@dataclass(frozen=True, slots=True)
class StoredBlob:
    """Where a fetched payload landed."""

    content_hash: str
    relative_path: str
    absolute_path: Path
    size_bytes: int
    was_new: bool


def _extension_for(content_type: str) -> str:
    mime = content_type.split(";", 1)[0].strip().lower()
    return _EXTENSION_BY_TYPE.get(mime, ".bin")


class RawStore:
    """Content-addressed blob store rooted at `data/raw/<source_id>/`."""

    def __init__(self, source_id: str, root: Path | None = None) -> None:
        self._source_id = source_id
        self._root = ensure_dir((root or RAW_DIR) / source_id)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, content_hash: str, content_type: str) -> Path:
        return self._root / content_hash[:2] / f"{content_hash}{_extension_for(content_type)}"

    def store(self, content: bytes, content_type: str) -> StoredBlob:
        """Write `content` under its own hash. Re-storing the same bytes is a no-op."""
        content_hash = hashlib.sha256(content).hexdigest()
        target = self.path_for(content_hash, content_type)

        was_new = not target.exists()
        if was_new:
            ensure_dir(target.parent)
            # Write to a temp file then rename, so an interrupted crawl can
            # never leave a half-written blob that later looks complete.
            temp = target.with_suffix(target.suffix + ".part")
            temp.write_bytes(content)
            temp.replace(target)

        return StoredBlob(
            content_hash=content_hash,
            relative_path=target.relative_to(self._root.parent).as_posix(),
            absolute_path=target,
            size_bytes=len(content),
            was_new=was_new,
        )

    def read(self, relative_path: str) -> bytes:
        """Read back a blob using the path recorded in the crawl state DB."""
        return (self._root.parent / relative_path).read_bytes()

    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self._root.rglob("*") if f.is_file())
