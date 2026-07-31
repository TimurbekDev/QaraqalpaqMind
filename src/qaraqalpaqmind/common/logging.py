"""Project-wide logging setup.

Every entrypoint calls `setup_logging()` exactly once; every module then does
`logger = get_logger(__name__)`. Two output modes:

* human  - colourised console via `rich`, for interactive runs
* json   - one JSON object per line, for long crawls / training jobs that are
           later grepped or shipped to Loki

Set `QM_LOG_JSON=true` to switch. A rotating file handler is always attached
so an unattended 12-hour crawl leaves a trace on disk.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import LOGS_DIR, ensure_dir

_CONFIGURED = False

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"asctime", "message", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """Render a record as a single JSON line, keeping any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Structured context passed as logger.info("...", extra={"url": ...}).
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        return json.dumps(payload, ensure_ascii=False, default=str)


def _console_handler(use_json: bool) -> logging.Handler:
    if use_json:
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        return handler

    try:
        from rich.logging import RichHandler
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        return handler

    return RichHandler(rich_tracebacks=True, show_path=False, log_time_format="%H:%M:%S")


def setup_logging(
    level: str | int | None = None,
    *,
    log_file: Path | None = None,
    use_json: bool | None = None,
) -> None:
    """Configure the root logger. Idempotent - safe to call from any entrypoint.

    Args:
        level: Log level; defaults to `$QM_LOG_LEVEL` or INFO.
        log_file: Destination file; defaults to `logs/qaraqalpaqmind.log`.
        use_json: Force JSON console output; defaults to `$QM_LOG_JSON`.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level: str | int = level if level is not None else os.getenv("QM_LOG_LEVEL", "INFO")
    resolved_json = (
        use_json
        if use_json is not None
        else os.getenv("QM_LOG_JSON", "false").lower() in {"1", "true", "yes"}
    )
    target_file = log_file or (ensure_dir(LOGS_DIR) / "qaraqalpaqmind.log")

    file_handler = RotatingFileHandler(
        target_file, maxBytes=64 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers.clear()
    root.addHandler(_console_handler(resolved_json))
    root.addHandler(file_handler)

    # Third-party libraries are chatty at DEBUG; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets", "trafilatura"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
