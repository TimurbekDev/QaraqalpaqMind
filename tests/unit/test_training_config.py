"""Tests for training configuration and sequence packing.

No GPU, no model download. These check the arithmetic and the guardrails - the
things that decide whether a run is even worth launching.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qaraqalpaqmind.common.config import load_config
from qaraqalpaqmind.training.config import CPTConfig, DataConfig, RuntimeConfig, TuningMethod
from qaraqalpaqmind.training.cpt.packing import (
    PackingStats,
    count_sequences,
    describe_schedule,
    pack_documents,
)

CORPUS_TOKENS = 28_683_631


class StubTokenizer:
    """Splits on whitespace and assigns one id per word."""

    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(word) for word in text.split()]


# --- shipped configurations ----------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["qwen3_8b_qlora_24gb.yaml", "qwen3_8b_lora_a100.yaml", "qwen3_8b_full_multigpu.yaml"],
)
def test_shipped_configs_are_valid(name: str) -> None:
    config = load_config(f"cpt/{name}", CPTConfig)
    assert config.model.name == "Qwen/Qwen3-8B"
    assert config.data.sequence_length > 0


def test_default_config_is_qlora_for_a_24gb_card() -> None:
    config = load_config("cpt/qwen3_8b_qlora_24gb.yaml", CPTConfig)
    assert config.method is TuningMethod.QLORA
    assert config.runtime.gradient_checkpointing
    assert config.runtime.per_device_batch_size == 1


def test_full_finetuning_config_uses_a_safe_learning_rate() -> None:
    config = load_config("cpt/qwen3_8b_full_multigpu.yaml", CPTConfig)
    assert config.method is TuningMethod.FULL
    assert config.optim.learning_rate <= 5e-5
    assert config.runtime.num_epochs == 1.0


# --- guardrails -----------------------------------------------------------


def test_full_finetuning_rejects_the_lora_learning_rate() -> None:
    # 1e-4 is right for LoRA and would wreck the base model's multilingual
    # ability if applied to all 8B parameters.
    with pytest.raises(ValidationError, match="too high for full"):
        CPTConfig.model_validate(
            {"method": "full", "optim": {"learning_rate": 1e-4}}
        )


def test_gradient_checkpointing_forbids_the_kv_cache() -> None:
    with pytest.raises(ValidationError, match="use_cache"):
        CPTConfig.model_validate(
            {"runtime": {"gradient_checkpointing": True}, "model": {"use_cache": True}}
        )


def test_only_one_precision_mode() -> None:
    with pytest.raises(ValidationError, match="bf16 or fp16"):
        RuntimeConfig(bf16=True, fp16=True)


def test_replay_ratio_requires_a_replay_dataset() -> None:
    with pytest.raises(ValidationError, match="replay_dataset"):
        DataConfig(replay_ratio=0.1)


def test_unknown_keys_are_rejected() -> None:
    # A typo must fail in the first second, not after a night of training.
    with pytest.raises(ValidationError):
        CPTConfig.model_validate({"optim": {"learing_rate": 1e-4}})


# --- arithmetic -----------------------------------------------------------


def test_effective_batch_and_tokens_per_step() -> None:
    runtime = RuntimeConfig(per_device_batch_size=2, gradient_accumulation_steps=8)
    assert runtime.effective_batch_size() == 16
    assert runtime.effective_batch_size(world_size=4) == 64
    assert runtime.tokens_per_step(2048) == 32_768


def test_estimated_steps_matches_the_plan() -> None:
    config = load_config("cpt/qwen3_8b_qlora_24gb.yaml", CPTConfig)
    steps = config.estimated_steps(CORPUS_TOKENS)
    # 28.7M tokens / 32,768 per step x 2 epochs
    assert 1_600 <= steps <= 1_900, steps


def test_describe_schedule() -> None:
    schedule = describe_schedule(
        corpus_tokens=CORPUS_TOKENS,
        sequence_length=2048,
        effective_batch_size=16,
        num_epochs=2.0,
    )
    assert schedule["sequences"] == CORPUS_TOKENS // 2048
    assert schedule["tokens_per_step"] == 32_768
    assert schedule["total_steps"] == schedule["steps_per_epoch"] * 2
    assert schedule["tokens_seen"] == CORPUS_TOKENS * 2


def test_count_sequences() -> None:
    assert count_sequences(4096, 2048) == 2
    assert count_sequences(4095, 2048) == 1


# --- packing --------------------------------------------------------------


def test_packing_yields_fixed_length_sequences() -> None:
    tokenizer = StubTokenizer()
    documents = ["bir eki úsh tórt bes altı jeti segiz toǵız on"] * 10
    packed = list(pack_documents(documents, tokenizer, sequence_length=8))

    assert packed
    for sequence in packed:
        assert len(sequence["input_ids"]) == 8
        assert len(sequence["attention_mask"]) == 8
        # Every token is a prediction target in causal pretraining.
        assert sequence["labels"] == sequence["input_ids"]


def test_documents_are_separated_by_eos() -> None:
    # Without a separator the model never learns where a document ends, which
    # shows up later as generations that will not stop.
    packed = list(pack_documents(["bir eki", "úsh tórt"], StubTokenizer(), sequence_length=3))
    assert StubTokenizer.eos_token_id in packed[0]["input_ids"]


def test_eos_can_be_disabled() -> None:
    packed = list(
        pack_documents(["bir eki úsh"] * 4, StubTokenizer(), sequence_length=3, add_eos=False)
    )
    assert all(StubTokenizer.eos_token_id not in s["input_ids"] for s in packed)


def test_packing_tracks_utilisation() -> None:
    stats = PackingStats()
    list(pack_documents(["bir eki úsh"] * 10, StubTokenizer(), 8, stats=stats))
    assert stats.documents == 10
    assert stats.sequences > 0
    assert 0 < stats.utilisation <= 1.0
    assert "documents" in stats.summary()


def test_blank_documents_are_skipped() -> None:
    stats = PackingStats()
    list(pack_documents(["", "   ", "bir eki úsh tórt"], StubTokenizer(), 2, stats=stats))
    assert stats.documents == 1


def test_partial_tail_is_dropped_not_padded() -> None:
    # Padding it would introduce the only padded example in the entire run.
    stats = PackingStats()
    packed = list(pack_documents(["bir eki úsh"], StubTokenizer(), sequence_length=100, stats=stats))
    assert packed == []
    assert stats.dropped_tail_tokens > 0


def test_packing_requires_an_eos_token_when_separating() -> None:
    class NoEos(StubTokenizer):
        eos_token_id = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="eos_token_id"):
        list(pack_documents(["bir eki"], NoEos(), sequence_length=4))
