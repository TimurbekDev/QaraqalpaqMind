"""Guards against the OOM that actually happened on a 4090.

Training ran cleanly at batch 1 for 200 steps, then the first evaluation
allocated 9.27 GiB in one go and died. The cause was not the train batch size:
`TrainingArguments.per_device_eval_batch_size` defaults to **8**, and the
configs only ever set the train batch.

The logits tensor is what makes that fatal. It is
(batch x sequence x vocabulary) and the causal-LM loss upcasts it to fp32:

    1 x 2048 x 151,936 x 4 bytes = 1.16 GiB
    8 x 2048 x 151,936 x 4 bytes = 9.27 GiB   <- the failed allocation

These tests fail if any shipped config regresses to the default.
"""

from __future__ import annotations

import pytest

from qaraqalpaqmind.common.config import load_config
from qaraqalpaqmind.common.runtime import Finding, GpuInfo
from qaraqalpaqmind.training.cli import _check_logits_fit, _logits_gb
from qaraqalpaqmind.training.config import CPTConfig, RuntimeConfig
from qaraqalpaqmind.training.dpo.config import DPOConfig
from qaraqalpaqmind.training.sft.config import SFTConfig

#: What a 24 GB card can spare for a single logits allocation.
_MAX_LOGITS_GB = 4.0

#: Config path, model, and where that stage keeps its sequence length. The
#: three stages genuinely name it differently: CPT packs to a fixed
#: `data.sequence_length`, SFT truncates conversations at
#: `max_sequence_length`, and DPO's `max_length` bounds prompt + completion.
_24GB_CONFIGS = [
    ("cpt/qwen3_8b_qlora_24gb.yaml", CPTConfig, "data.sequence_length"),
    ("cpt/base.yaml", CPTConfig, "data.sequence_length"),
    ("sft/qwen3_8b_qlora_24gb.yaml", SFTConfig, "max_sequence_length"),
    ("dpo/qwen3_8b_qlora_24gb.yaml", DPOConfig, "max_length"),
]


def _resolve(config: object, dotted: str) -> int:
    for part in dotted.split("."):
        config = getattr(config, part)
    assert isinstance(config, int)
    return config


def test_the_default_is_not_the_transformers_default() -> None:
    # Ours must be 1. Inheriting 8 is exactly what caused the OOM.
    assert RuntimeConfig().per_device_eval_batch_size == 1


def test_prediction_loss_only_is_on_by_default() -> None:
    # Otherwise the Trainer gathers logits from every eval batch to pass to a
    # metrics function that does not exist here.
    assert RuntimeConfig().prediction_loss_only


@pytest.mark.parametrize(
    ("name", "model", "seq_field"), _24GB_CONFIGS, ids=[c[0] for c in _24GB_CONFIGS]
)
def test_shipped_configs_set_eval_batch_explicitly(
    name: str, model: type, seq_field: str
) -> None:
    config = load_config(name, model)
    assert config.runtime.per_device_eval_batch_size <= 2, name
    assert config.runtime.prediction_loss_only, name


def test_logits_memory_matches_the_observed_failure() -> None:
    # The real allocation that failed was 9.27 GiB. If this drifts, the memory
    # model in `qm train plan` has stopped describing reality.
    assert _logits_gb(batch=8, sequence_length=2048) == pytest.approx(13.9, abs=0.2)
    # bf16 + fp32 copy; the fp32 part alone is the 9.27 GiB in the traceback.
    fp32_only = 8 * 2048 * 151_936 * 4 / 1024**3
    assert fp32_only == pytest.approx(9.27, abs=0.05)


@pytest.mark.parametrize(
    ("name", "model", "seq_field"), _24GB_CONFIGS, ids=[c[0] for c in _24GB_CONFIGS]
)
def test_logits_fit_on_a_24gb_card(name: str, model: type, seq_field: str) -> None:
    config = load_config(name, model)
    sequence_length = _resolve(config, seq_field)

    for label, batch in (
        ("train", config.runtime.per_device_batch_size),
        ("eval", config.runtime.per_device_eval_batch_size),
    ):
        used = _logits_gb(batch, sequence_length)
        assert used <= _MAX_LOGITS_GB, f"{name} {label} logits need {used:.2f} GB"


def test_logits_scale_linearly_with_batch_and_sequence() -> None:
    assert _logits_gb(2, 2048) == pytest.approx(2 * _logits_gb(1, 2048))
    assert _logits_gb(1, 4096) == pytest.approx(2 * _logits_gb(1, 2048))


# --- preflight gate -------------------------------------------------------
#
# The run that OOMed passed preflight. These cover the check that would have
# stopped it.

_RTX_4090 = GpuInfo(available=True, name="NVIDIA GeForce RTX 4090", total_memory_gb=24.0,
                    capability=(8, 9), count=1)


def _with_runtime(**overrides: object) -> CPTConfig:
    """The shipped config with runtime fields swapped. Configs are frozen."""
    config = load_config("cpt/qwen3_8b_qlora_24gb.yaml", CPTConfig)
    return config.model_copy(update={"runtime": config.runtime.model_copy(update=overrides)})


def _findings_for(gpu: GpuInfo = _RTX_4090, **runtime: object) -> list[Finding]:
    findings: list[Finding] = []
    _check_logits_fit(_with_runtime(**runtime), gpu, findings)
    return findings


def test_preflight_passes_the_shipped_config() -> None:
    assert _findings_for() == []


def test_preflight_rejects_the_transformers_default_eval_batch() -> None:
    findings = _findings_for(per_device_eval_batch_size=8)
    assert [f.level for f in findings] == ["error"]
    # The message has to name the setting, or it is not actionable at 3am.
    assert "per_device_eval_batch_size" in findings[0].message


def test_preflight_rejects_accumulating_eval_logits() -> None:
    # Batch size is fine here; the leak is the Trainer keeping every batch.
    findings = _findings_for(prediction_loss_only=False)
    assert [f.level for f in findings] == ["error"]
    assert "prediction_loss_only" in findings[0].message


def test_preflight_assumes_a_24gb_card_when_no_gpu_is_present() -> None:
    # Running preflight on a laptop must still catch a bad config, otherwise
    # the check only fires once you are already paying for the pod.
    findings = _findings_for(gpu=GpuInfo(available=False), per_device_eval_batch_size=8)
    assert [f.level for f in findings] == ["error"]


def test_preflight_budgets_against_the_actual_card() -> None:
    # The gate is a fit test against real free memory, not a fixed batch cap.
    # Batch 8 needs 13.91 GB: over the ~11 GB a 4090 has left after weights and
    # activations, comfortably inside an 80 GB A100.
    a100 = GpuInfo(available=True, name="A100-SXM4-80GB", total_memory_gb=80.0,
                   capability=(8, 0), count=1)
    assert _findings_for(gpu=a100, per_device_eval_batch_size=8) == []
    assert len(_findings_for(per_device_eval_batch_size=8)) == 1


def test_preflight_does_not_block_a_merely_lopsided_eval_batch() -> None:
    # Batch 4 fits on a 4090 (6.96 GB of ~11 GB), so preflight lets it through.
    # `qm train plan` still warns that evaluation becomes the memory peak -
    # that is advice, and this is a gate; conflating them would make the gate
    # too noisy to trust.
    assert _findings_for(per_device_eval_batch_size=4) == []
