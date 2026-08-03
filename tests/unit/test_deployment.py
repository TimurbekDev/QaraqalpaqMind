"""Tests for deployment-critical behaviour: resume, paths, preflight.

These guard failures that only appear on a cloud GPU, where they are expensive:
a run that silently restarts from step 0 after a pod reclaim, or a config path
that resolves to nothing because the package was installed non-editable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qaraqalpaqmind.common import runtime
from qaraqalpaqmind.common.paths import CONFIGS_DIR, PROJECT_ROOT
from qaraqalpaqmind.training.checkpoints import (
    checkpoint_step,
    describe,
    find_checkpoints,
    latest_checkpoint,
    resolve_resume,
)


def _make_checkpoints(root: Path, steps: list[int]) -> None:
    for step in steps:
        (root / f"checkpoint-{step}").mkdir(parents=True)


# --- checkpoint discovery -------------------------------------------------


def test_checkpoints_are_ordered_numerically(tmp_path: Path) -> None:
    # Lexical sorting would put checkpoint-1000 before checkpoint-200.
    _make_checkpoints(tmp_path, [200, 1000, 40])
    assert [c.name for c in find_checkpoints(tmp_path)] == [
        "checkpoint-40",
        "checkpoint-200",
        "checkpoint-1000",
    ]
    assert latest_checkpoint(tmp_path).name == "checkpoint-1000"  # type: ignore[union-attr]


def test_unrelated_directories_are_ignored(tmp_path: Path) -> None:
    _make_checkpoints(tmp_path, [10])
    (tmp_path / "runs").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    assert [c.name for c in find_checkpoints(tmp_path)] == ["checkpoint-10"]


def test_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    assert find_checkpoints(tmp_path / "absent") == []
    assert latest_checkpoint(tmp_path / "absent") is None


def test_checkpoint_step_is_parsed() -> None:
    assert checkpoint_step(Path("checkpoint-1750")) == 1750
    assert checkpoint_step(Path("something-else")) is None


# --- resume resolution ----------------------------------------------------


def test_auto_resumes_from_the_newest_checkpoint(tmp_path: Path) -> None:
    # The failure this prevents: a reclaimed pod restarts, training silently
    # begins at step 0, and every hour already spent is wasted.
    _make_checkpoints(tmp_path, [100, 500])
    resolved = resolve_resume("auto", tmp_path)
    assert resolved is not None
    assert Path(str(resolved)).name == "checkpoint-500"


def test_auto_starts_fresh_when_there_is_nothing_to_resume(tmp_path: Path) -> None:
    assert resolve_resume("auto", tmp_path) is None


@pytest.mark.parametrize("setting", ["never", "NEVER", "false", "no", "off"])
def test_resume_can_be_disabled(setting: str, tmp_path: Path) -> None:
    _make_checkpoints(tmp_path, [100])
    assert resolve_resume(setting, tmp_path) is None


def test_explicit_checkpoint_is_honoured(tmp_path: Path) -> None:
    _make_checkpoints(tmp_path, [100, 500])
    resolved = resolve_resume("checkpoint-100", tmp_path)
    assert Path(str(resolved)).name == "checkpoint-100"


def test_explicit_missing_checkpoint_fails_loudly(tmp_path: Path) -> None:
    # Silently starting fresh here would be worse than an error: the operator
    # asked for a specific checkpoint.
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_resume("checkpoint-9999", tmp_path)


def test_describe_reports_what_is_on_disk(tmp_path: Path) -> None:
    assert "no checkpoints" in describe(tmp_path)
    _make_checkpoints(tmp_path, [50, 100])
    summary = describe(tmp_path)
    assert "2 checkpoint" in summary
    assert "50, 100" in summary


def test_every_trainer_wires_resume() -> None:
    # A trainer that forgets this restarts from zero on every pod reclaim.
    for module in ("cpt", "sft", "dpo"):
        source = (
            PROJECT_ROOT / "src" / "qaraqalpaqmind" / "training" / module / "train.py"
        ).read_text(encoding="utf-8")
        assert "resume_from_checkpoint=resume" in source, module
        assert "resolve_resume" in source, module


# --- path resolution ------------------------------------------------------


def test_project_root_finds_the_checkout() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert CONFIGS_DIR.is_dir()


def test_project_root_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Needed when the package is installed non-editable, where deriving the
    # root from __file__ lands inside site-packages.
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "configs").mkdir()
    monkeypatch.setenv("QM_PROJECT_ROOT", str(tmp_path))

    from qaraqalpaqmind.common.paths import _resolve_project_root

    assert _resolve_project_root() == tmp_path.resolve()


def test_shipped_configs_resolve_from_any_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    from qaraqalpaqmind.common.config import load_config
    from qaraqalpaqmind.training.config import CPTConfig

    monkeypatch.chdir(Path(os.environ.get("TEMP", "/tmp")))
    config = load_config("cpt/qwen3_8b_qlora_24gb.yaml", CPTConfig)
    assert config.model.name == "Qwen/Qwen3-8B"


# --- runtime preflight ----------------------------------------------------


def test_gpu_info_is_safe_without_a_gpu() -> None:
    info = runtime.gpu_info()
    assert isinstance(info.available, bool)
    assert "no CUDA device" in info.summary() or info.name


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 9), True), ((8, 0), True), ((7, 5), False), ((0, 0), False)],
)
def test_bf16_support_follows_compute_capability(
    capability: tuple[int, int], expected: bool
) -> None:
    # The RTX 4090 is sm_89, so bf16 is available; a T4 (sm_75) is not.
    info = runtime.GpuInfo(available=True, capability=capability)
    assert info.supports_bf16 is expected
    assert info.supports_flash_attention_2 is expected


def test_hf_cache_follows_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert runtime.hf_cache_dir() == tmp_path / "hub"

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "direct"))
    assert runtime.hf_cache_dir() == tmp_path / "direct"


def test_default_cache_is_flagged_as_ephemeral(monkeypatch: pytest.MonkeyPatch) -> None:
    # On a container this is the overlay filesystem: a 16 GB model download is
    # lost on restart and repeated every run.
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert runtime.cache_is_ephemeral()


def test_configured_cache_is_not_flagged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert not runtime.cache_is_ephemeral()


def test_free_disk_walks_up_to_an_existing_parent(tmp_path: Path) -> None:
    assert runtime.free_disk_gb(tmp_path / "does" / "not" / "exist") > 0


def test_preflight_returns_findings() -> None:
    findings = runtime.preflight()
    assert findings
    assert all(f.level in {"ok", "warn", "error"} for f in findings)
