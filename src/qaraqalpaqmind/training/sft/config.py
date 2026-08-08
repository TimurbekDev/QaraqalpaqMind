"""Supervised fine-tuning configuration.

Reuses the model, LoRA, optimiser, runtime and logging blocks from continued
pretraining, and adds what is specific to SFT: which CPT adapter to build on,
which mixture to train, and how loss is masked.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from ...common.config import StrictModel
from ..config import (
    LoggingConfig,
    LoraConfig,
    ModelConfig,
    OptimConfig,
    QuantizationConfig,
    RuntimeConfig,
    TuningMethod,
)


class SFTConfig(StrictModel):
    """A complete supervised fine-tuning run."""

    method: TuningMethod = TuningMethod.QLORA

    # The Phase 5 adapter. SFT builds on continued pretraining rather than the
    # raw base model: CPT teaches the language, SFT teaches usefulness in it.
    # Setting this to null trains on the base model directly, which produces a
    # much weaker Karakalpak model and is warned about at load time.
    cpt_adapter: Path | None = Path("models/cpt/qlora_24gb")

    dataset: str = "sft_v1"

    # Shorter than CPT's 2048. SFT examples are conversations, not documents,
    # and a longer window would be mostly padding - packing is off here.
    max_sequence_length: int = Field(default=1024, ge=128, le=32768)

    # Compute loss on assistant turns only. With this off the model learns to
    # generate user questions as readily as answers, which at inference looks
    # like continuing the user's turn instead of replying to it.
    completion_only_loss: bool = True

    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _full_finetuning_needs_a_lower_rate(self) -> SFTConfig:
        if self.method is TuningMethod.FULL and self.optim.learning_rate > 5e-5:
            raise ValueError(
                f"learning_rate {self.optim.learning_rate:g} is far too high for full "
                "fine-tuning; use 1e-5 to 5e-5."
            )
        return self

    @model_validator(mode="after")
    def _warn_on_many_epochs(self) -> SFTConfig:
        # SFT overfits far faster than continued pretraining: the dataset is
        # smaller and every example is seen as an exact target.
        if self.runtime.num_epochs > 5:
            raise ValueError(
                f"{self.runtime.num_epochs:g} SFT epochs will overfit. Two to three is "
                "the usual range; raise this deliberately if you have measured that it "
                "helps."
            )
        return self


class MixtureConfig(StrictModel):
    """How the SFT dataset is assembled."""

    name: str = "sft_v1"
    # Set by how much instruction-following data exists, not by how much data
    # exists. At 50,000 the caps for translation and grammar are larger than
    # everything else put together, and the first SFT run came out a translation
    # engine that answers questions in Azerbaijani.
    target_size: int = Field(default=20_000, ge=100)
    validation_split: float = Field(default=0.02, ge=0.0, le=0.3)
    seed: int = 20260731
    check_contamination: bool = True

    # Per-task share of the final mixture. Not proportional to how much data
    # each builder can produce - translation could supply 90% and is capped.
    proportions: dict[str, float] = Field(
        default_factory=lambda: {
            "instruction": 0.28,
            "translation": 0.25,
            "grammar": 0.15,
            "summarization": 0.13,
            "qa": 0.10,
            "conversation": 0.04,
            "reasoning": 0.03,
            "math": 0.01,
            "coding": 0.01,
        }
    )

    # Caps on how much each builder is asked for, to bound build time.
    max_translation: int = Field(default=40_000, ge=0)
    max_grammar: int = Field(default=30_000, ge=0)
    max_summarization: int = Field(default=15_000, ge=0)
    # Fills the instruction and qa slots, which the seed sets cannot: 45
    # hand-authored examples against a 0.22 combined share is what produced an
    # SFT model that answers in Azerbaijani.
    max_grounded_qa: int = Field(default=15_000, ge=0)

    both_translation_directions: bool = True
    grammar_explanations: bool = True

    @model_validator(mode="after")
    def _proportions_must_be_positive(self) -> MixtureConfig:
        if sum(self.proportions.values()) <= 0:
            raise ValueError("mixture proportions sum to zero")
        if any(share < 0 for share in self.proportions.values()):
            raise ValueError("mixture proportions must not be negative")
        return self
