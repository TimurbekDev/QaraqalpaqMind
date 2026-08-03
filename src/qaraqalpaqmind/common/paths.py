"""Single source of truth for every filesystem location in the project.

Nothing anywhere else in the codebase may hardcode a path. Import from here
instead, so that moving `data/` to another disk is a one-variable change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_THIS_FILE: Final[Path] = Path(__file__).resolve()

# Files that identify a checkout rather than an installed package.
_ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", "configs")


def _looks_like_checkout(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in _ROOT_MARKERS)


def _resolve_project_root() -> Path:
    """Locate the repository root.

    Deriving it from `__file__` alone assumes an editable install. Under a
    regular `pip install`, `src/qaraqalpaqmind/common/paths.py` becomes
    `site-packages/qaraqalpaqmind/common/paths.py`, and four levels up lands
    somewhere in the interpreter's lib directory - so `configs/` and `data/`
    silently resolve to paths that do not exist.

    Checked in order: an explicit override, the working directory, then the
    derived path.
    """
    override = os.getenv("QM_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    # src/qaraqalpaqmind/common/paths.py -> repo root is 4 levels up.
    derived = _THIS_FILE.parents[3]
    if _looks_like_checkout(derived):
        return derived

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if _looks_like_checkout(candidate):
            return candidate

    # Nothing matched: fall back to the derived path so behaviour is unchanged
    # for anyone who was relying on it.
    return derived


PROJECT_ROOT: Final[Path] = _resolve_project_root()


def _dir_from_env(env_var: str, default: Path) -> Path:
    """Return an override from the environment, or the default, as an absolute path."""
    raw = os.getenv(env_var, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


# --- Top-level areas -------------------------------------------------------
DATA_DIR: Final[Path] = _dir_from_env("QM_DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR: Final[Path] = _dir_from_env("QM_MODELS_DIR", PROJECT_ROOT / "models")
LOGS_DIR: Final[Path] = _dir_from_env("QM_LOGS_DIR", PROJECT_ROOT / "logs")
CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"
BENCHMARKS_DIR: Final[Path] = PROJECT_ROOT / "benchmarks"

# --- Data lifecycle stages -------------------------------------------------
# raw       : exactly what the crawler received, never edited
# interim   : extracted text, still dirty (one file per source)
# processed : cleaned + deduplicated shards, ready to be assembled
# datasets  : final train-ready JSONL, versioned and hash-manifested
RAW_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
DATASETS_DIR: Final[Path] = DATA_DIR / "datasets"
MANIFESTS_DIR: Final[Path] = DATA_DIR / "manifests"

PRETRAIN_DIR: Final[Path] = DATASETS_DIR / "pretrain"
SFT_DIR: Final[Path] = DATASETS_DIR / "sft"
DPO_DIR: Final[Path] = DATASETS_DIR / "dpo"
EVAL_DIR: Final[Path] = DATASETS_DIR / "eval"

# --- Model checkpoints -----------------------------------------------------
BASE_MODEL_DIR: Final[Path] = MODELS_DIR / "base"
CPT_MODEL_DIR: Final[Path] = MODELS_DIR / "cpt"
SFT_MODEL_DIR: Final[Path] = MODELS_DIR / "sft"
DPO_MODEL_DIR: Final[Path] = MODELS_DIR / "dpo"
MERGED_MODEL_DIR: Final[Path] = MODELS_DIR / "merged"


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_source_dir(source_id: str) -> Path:
    """Raw landing zone for one crawl source, e.g. `data/raw/wikipedia_kaa`."""
    return ensure_dir(RAW_DIR / source_id)


def interim_source_file(source_id: str) -> Path:
    """Extracted-text shard for one source, e.g. `data/interim/news_qmuz.jsonl.zst`."""
    ensure_dir(INTERIM_DIR)
    return INTERIM_DIR / f"{source_id}.jsonl.zst"
