"""Direct preference optimisation configuration."""

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


class DPOConfig(StrictModel):
    """A complete DPO run.

    DPO trains a policy against a frozen reference. With LoRA the reference is
    the adapter-disabled model, so no second copy of the weights is needed -
    which is what makes this fit on one card at all.
    """

    method: TuningMethod = TuningMethod.QLORA

    # The Phase 6 output. DPO refines an instruction-following model; running it
    # on a base or CPT-only checkpoint optimises preferences over a model that
    # cannot follow instructions yet, which teaches very little.
    sft_adapter: Path | None = Path("models/sft/qlora_24gb")

    dataset: str = "dpo_v1"

    # beta controls how far the policy may drift from the reference. Low values
    # allow large drift and, on a small preference set, reward hacking - the
    # model finds a degenerate output that scores well and collapses onto it.
    # 0.1 is the standard starting point; raise it if outputs degrade.
    beta: float = Field(default=0.1, gt=0.0, le=1.0)

    loss_type: str = Field(
        default="sigmoid",
        description="sigmoid (standard DPO), ipo, hinge, or another TRL loss.",
    )

    max_length: int = Field(default=1024, ge=128)
    max_prompt_length: int = Field(default=512, ge=64)

    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)

    # DPO-specific defaults, not the shared ones. `OptimConfig` defaults to
    # 1e-4, which suits continued pretraining and which this class's own
    # validator rejects - so inheriting it made `DPOConfig()` impossible to
    # construct without overrides.
    optim: OptimConfig = Field(
        default_factory=lambda: OptimConfig(learning_rate=5e-6, warmup_ratio=0.1, weight_decay=0.0)
    )
    runtime: RuntimeConfig = Field(
        default_factory=lambda: RuntimeConfig(num_epochs=1.0, gradient_accumulation_steps=16)
    )
    logging: LoggingConfig = Field(
        default_factory=lambda: LoggingConfig(
            output_dir=Path("models/dpo"), run_name="qwen3-8b-kaa-dpo"
        )
    )

    @model_validator(mode="after")
    def _prompt_must_fit_inside_the_window(self) -> DPOConfig:
        if self.max_prompt_length >= self.max_length:
            raise ValueError(
                f"max_prompt_length ({self.max_prompt_length}) must be less than "
                f"max_length ({self.max_length}), or responses are truncated to nothing"
            )
        return self

    @model_validator(mode="after")
    def _learning_rate_must_be_small(self) -> DPOConfig:
        # DPO on top of SFT needs a rate well below SFT's. It is a refinement
        # pass over a model that already works; a large rate erases the
        # instruction-following that SFT just installed.
        if self.optim.learning_rate > 1e-5:
            raise ValueError(
                f"learning_rate {self.optim.learning_rate:g} is too high for DPO. Use "
                "1e-6 to 5e-6: this refines an SFT model rather than training one, and "
                "a large rate erases the instruction-following SFT installed."
            )
        return self

    @model_validator(mode="after")
    def _few_epochs(self) -> DPOConfig:
        if self.runtime.num_epochs > 3:
            raise ValueError(
                f"{self.runtime.num_epochs:g} DPO epochs will over-optimise. One to two "
                "is standard; preference training degrades quickly past that."
            )
        return self


class PreferenceMixtureConfig(StrictModel):
    """How the preference dataset is assembled."""

    name: str = "dpo_v1"
    target_size: int = Field(default=20_000, ge=100)
    validation_split: float = Field(default=0.02, ge=0.0, le=0.3)
    seed: int = 20260731
    check_contamination: bool = True

    # Share of the final mixture per criterion. Orthography and language
    # consistency dominate because their pairs are the cleanest: both sides say
    # the same thing and differ in exactly one dimension.
    proportions: dict[str, float] = Field(
        default_factory=lambda: {
            "language_consistency": 0.40,
            "orthography": 0.40,
            "response_quality": 0.20,
        }
    )

    max_orthography: int = Field(default=15_000, ge=0)
    max_language: int = Field(default=15_000, ge=0)
    max_quality: int = Field(default=5_000, ge=0)

    @model_validator(mode="after")
    def _proportions_must_be_positive(self) -> PreferenceMixtureConfig:
        if sum(self.proportions.values()) <= 0:
            raise ValueError("mixture proportions sum to zero")
        if any(share < 0 for share in self.proportions.values()):
            raise ValueError("mixture proportions must not be negative")
        return self
