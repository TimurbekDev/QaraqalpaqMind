"""Shared primitives: paths, logging, config loading, JSONL I/O.

Import from here rather than from submodules, so internal reorganisation stays
invisible to the rest of the codebase.
"""

from __future__ import annotations

from .config import StrictModel, load_config, load_raw
from .io import count_lines, file_sha256, read_jsonl, text_sha1, write_jsonl
from .logging import get_logger, setup_logging
from .paths import (
    CONFIGS_DIR,
    DATA_DIR,
    DATASETS_DIR,
    INTERIM_DIR,
    LOGS_DIR,
    MANIFESTS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    ensure_dir,
)

__all__ = [
    "CONFIGS_DIR",
    "DATASETS_DIR",
    "DATA_DIR",
    "INTERIM_DIR",
    "LOGS_DIR",
    "MANIFESTS_DIR",
    "MODELS_DIR",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "RAW_DIR",
    "StrictModel",
    "count_lines",
    "ensure_dir",
    "file_sha256",
    "get_logger",
    "load_config",
    "load_raw",
    "read_jsonl",
    "setup_logging",
    "text_sha1",
    "write_jsonl",
]
