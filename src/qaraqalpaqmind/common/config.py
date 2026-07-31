"""Typed YAML configuration loading.

Design rules:

1. Every stage (crawl, clean, cpt, sft, dpo, eval, rag, serve) is driven by a
   YAML file under `configs/`, never by CLI flag soup.
2. YAML files may inherit via a top-level `_base_: path/to/other.yaml` key, so
   `configs/cpt/qwen3_8b_lora.yaml` only states its deltas from a shared base.
3. `${ENV_VAR}` and `${ENV_VAR:default}` are expanded from the environment,
   so secrets never live in the repo.
4. The merged dict is validated into a pydantic model. An unknown or misspelled
   key fails loudly *before* a 40-hour training run starts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict

from .paths import CONFIGS_DIR, PROJECT_ROOT

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_BASE_KEY = "_base_"


class StrictModel(BaseModel):
    """Base for all config models: unknown keys are an error, not a warning."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


def _expand_env(value: Any) -> Any:
    """Recursively substitute `${VAR}` / `${VAR:default}` inside strings."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            resolved = os.getenv(var)
            if resolved is None:
                if default is None:
                    raise KeyError(f"Environment variable {var!r} referenced in config is unset")
                resolved = default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` onto `base`. Nested dicts merge; lists/scalars replace."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(path: str | Path) -> Path:
    """Accept an absolute path, a repo-relative path, or a `configs/`-relative name."""
    candidate = Path(path)
    for option in (candidate, PROJECT_ROOT / candidate, CONFIGS_DIR / candidate):
        if option.is_file():
            return option.resolve()
    raise FileNotFoundError(f"Config not found: {path}")


def load_raw(path: str | Path, *, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML file, resolving `_base_` inheritance and env interpolation."""
    resolved = _resolve(path)
    if resolved in _seen:
        raise ValueError(f"Circular _base_ inheritance at {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{resolved} must contain a YAML mapping at the top level")

    base_ref = data.pop(_BASE_KEY, None)
    if base_ref is not None:
        base_path = (resolved.parent / str(base_ref)).resolve()
        base_data = load_raw(base_path, _seen=_seen | {resolved})
        data = _deep_merge(base_data, data)

    return cast("dict[str, Any]", _expand_env(data))


def load_config[T: StrictModel](path: str | Path, model: type[T]) -> T:
    """Load `path` and validate it into `model`.

    Example:
        cfg = load_config("cpt/qwen3_8b_lora.yaml", CPTConfig)
    """
    return model.model_validate(load_raw(path))
