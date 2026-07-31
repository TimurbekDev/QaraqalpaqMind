"""Load `.env` before anything reads the environment.

`paths.py` resolves `QM_DATA_DIR` and friends at import time, and the source
registry expands `${QM_USER_AGENT}` when it is parsed. Both happen early, so
`.env` has to be loaded earlier still - which is why this is invoked from the
package `__init__`, not from a CLI entrypoint.

Without it, `.env` is a file that documents settings nobody reads: every
variable in `.env.example` was silently ignored, including `HF_TOKEN`.

Real environment variables always win over `.env`, so CI and container
deployments override the file rather than fighting it.
"""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def project_env_file() -> Path:
    """`.env` at the repository root, next to `pyproject.toml`."""
    return Path(__file__).resolve().parents[3] / ".env"


def load_env(path: Path | None = None, *, override: bool = False) -> bool:
    """Load environment variables from `.env`. Idempotent.

    Args:
        path: File to read; defaults to the repository root `.env`.
        override: Let the file win over already-set variables. Off by default
            so a real environment variable always beats the file.

    Returns:
        True if a file was found and read.
    """
    global _LOADED
    if _LOADED and path is None:
        return True

    target = path or project_env_file()
    if not target.is_file():
        _LOADED = True
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_manually(target, override=override)
    else:
        load_dotenv(target, override=override)

    if path is None:
        _LOADED = True
    return True


def _load_manually(path: Path, *, override: bool) -> None:
    """Minimal fallback parser, so the package works without python-dotenv."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if value and (override or key not in os.environ):
            os.environ[key] = value


def has_secret(name: str) -> bool:
    """Whether a secret is configured, without ever revealing its value."""
    return bool(os.getenv(name, "").strip())
