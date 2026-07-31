"""Streaming JSONL I/O with transparent compression.

The whole pipeline speaks one format: JSON Lines, one record per line, UTF-8.
Files may be plain (`.jsonl`), gzip (`.jsonl.gz`) or zstandard (`.jsonl.zst`).
Zstd is the default for corpora: ~3x smaller than gzip at ~5x the read speed,
which matters when a cleaning pass touches 20 GB.

Everything here is a generator. No function ever loads a corpus into RAM.
"""

from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import orjson

from .logging import get_logger

logger = get_logger(__name__)

_ZSTD_LEVEL = 10
_HASH_CHUNK = 1 << 20


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[IO[bytes]]:
    """Open a possibly-compressed file in binary mode based on its suffix."""
    suffixes = "".join(path.suffixes[-2:])
    binary_mode = "rb" if mode == "r" else "wb"

    if suffixes.endswith(".zst"):
        import zstandard

        raw = path.open(binary_mode)
        try:
            if mode == "r":
                reader = zstandard.ZstdDecompressor().stream_reader(raw)
                yield reader
            else:
                writer = zstandard.ZstdCompressor(level=_ZSTD_LEVEL).stream_writer(raw)
                try:
                    yield writer
                finally:
                    writer.close()
        finally:
            raw.close()
    elif suffixes.endswith(".gz"):
        with gzip.open(path, binary_mode) as handle:
            yield handle  # type: ignore[misc]
    else:
        with path.open(binary_mode) as handle:
            yield handle


def read_jsonl(path: str | Path, *, skip_errors: bool = True) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file. Malformed lines are logged and skipped."""
    file_path = Path(path)
    bad = 0
    with _open_text(file_path, "r") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield orjson.loads(stripped)
            except orjson.JSONDecodeError:
                bad += 1
                if not skip_errors:
                    raise
                if bad <= 5:
                    logger.warning("Malformed JSONL line", extra={"file": str(file_path), "line": lineno})
    if bad:
        logger.warning("Skipped malformed lines", extra={"file": str(file_path), "count": bad})


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to a JSONL file, creating parent dirs. Returns the count."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with _open_text(file_path, "w") as handle:
        for record in records:
            handle.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            written += 1
    logger.info("Wrote JSONL", extra={"file": str(file_path), "records": written})
    return written


def count_lines(path: str | Path) -> int:
    """Count non-empty lines without parsing JSON."""
    with _open_text(Path(path), "r") as handle:
        return sum(1 for line in handle if line.strip())


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes - used for dataset manifests and reproducibility."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha1(text: str) -> str:
    """Stable content hash for exact-duplicate detection (Phase 3)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
