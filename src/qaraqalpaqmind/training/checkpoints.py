"""Checkpoint discovery and resume handling.

Cloud GPUs are interruptible. A pod can be reclaimed mid-run, and a restart that
silently began again from step 0 would waste every hour already spent - while
looking exactly like a normal start in the logs.

So resuming is the default, and it is loud: the resolved decision is logged
before training begins, with the step number being resumed from.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..common.logging import get_logger

logger = get_logger(__name__)

# Written by transformers' Trainer as `checkpoint-<global_step>`.
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def find_checkpoints(output_dir: Path) -> list[Path]:
    """Every checkpoint in `output_dir`, oldest first."""
    if not output_dir.is_dir():
        return []
    numbered: list[tuple[int, Path]] = []
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.match(entry.name)
        if match:
            numbered.append((int(match.group(1)), entry))
    return [path for _, path in sorted(numbered)]


def latest_checkpoint(output_dir: Path) -> Path | None:
    """The newest checkpoint, or None if there is none."""
    checkpoints = find_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def checkpoint_step(checkpoint: Path) -> int | None:
    match = _CHECKPOINT_PATTERN.match(checkpoint.name)
    return int(match.group(1)) if match else None


def resolve_resume(setting: str, output_dir: Path) -> str | bool | None:
    """Turn the config value into what `Trainer.train()` expects.

    Returns a path string to resume from, or None to start fresh. The decision
    is logged either way, because "did it resume?" must never be a question
    answered by watching the loss curve.
    """
    value = (setting or "auto").strip()

    if value.lower() in {"never", "false", "no", "off"}:
        logger.info("Resume disabled; starting from step 0")
        return None

    if value.lower() != "auto":
        explicit = Path(value)
        if not explicit.is_absolute():
            explicit = output_dir / value
        if not explicit.is_dir():
            raise FileNotFoundError(
                f"resume_from_checkpoint points at {explicit}, which does not exist. "
                f'Use "auto" to continue from the newest checkpoint, or "never" to '
                "start fresh."
            )
        logger.info("Resuming from explicit checkpoint", extra={"path": str(explicit)})
        return str(explicit)

    newest = latest_checkpoint(output_dir)
    if newest is None:
        logger.info(
            "No checkpoint found; starting from step 0", extra={"output_dir": str(output_dir)}
        )
        return None

    step = checkpoint_step(newest)
    logger.warning(
        "RESUMING an interrupted run - not starting from scratch",
        extra={"checkpoint": str(newest), "step": step},
    )
    return str(newest)


def describe(output_dir: Path) -> str:
    """Human-readable summary of what is on disk, for the CLI."""
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return f"no checkpoints in {output_dir}"
    steps = [checkpoint_step(c) for c in checkpoints]
    return (
        f"{len(checkpoints)} checkpoint(s) in {output_dir}: "
        f"steps {', '.join(str(s) for s in steps)}"
    )
