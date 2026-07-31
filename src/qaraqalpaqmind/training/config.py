"""Typed training configuration.

Every hyperparameter lives in a YAML file under `configs/`, validated into
these models before anything loads a GPU. A misspelled key fails in the first
second rather than after a night of training, and a finished run is fully
described by one config plus one dataset manifest.

The defaults here are chosen for *this* corpus - 28.7M Karakalpak tokens
against an 8B model - and the reasoning for each is in `docs/TRAINING.md`.
Copying them to a larger corpus without rereading that is a mistake.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..common.config import StrictModel


class TuningMethod(StrEnum):
    """How much of the model is updated."""

    QLORA = "qlora"  # 4-bit base, LoRA adapters. Fits 8B on 24GB.
    LORA = "lora"  # bf16 base, LoRA adapters. Needs ~40GB for 8B.
    FULL = "full"  # every parameter. Needs multiple 80GB cards for 8B.


class ModelConfig(StrictModel):
    """Which model, and how it is loaded."""

    name: str = "Qwen/Qwen3-8B"
    revision: str | None = Field(
        default=None,
        description="Pin a commit hash for reproducibility. None tracks the branch.",
    )
    trust_remote_code: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    use_cache: bool = False  # incompatible with gradient checkpointing


class QuantizationConfig(StrictModel):
    """4-bit loading. Only read when `method` is qlora."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    # Double quantisation saves ~0.4 bits per parameter at no measurable cost.
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"


class LoraConfig(StrictModel):
    """Adapter shape.

    `r=64` is deliberately generous for a corpus this small. Continued
    pretraining has to move the model's representation of an entire language,
    which is a larger change than the instruction-following tweaks LoRA is
    usually used for, and a rank that is too low simply cannot express it.
    """

    r: int = Field(default=64, ge=1, le=512)
    alpha: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    bias: Literal["none", "all", "lora_only"] = "none"
    # Every linear layer, not just attention. Restricting LoRA to q/v
    # projections is a instruction-tuning convention; for language adaptation
    # the MLP matrices are where most of the lexical knowledge sits.
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Train the embedding and output layers too when the vocabulary changes.
    modules_to_save: list[str] = Field(default_factory=list)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class DataConfig(StrictModel):
    """What is trained on, and how sequences are built."""

    dataset: str = "pretrain_v1"
    text_field: str = "text"

    # 2048 Karakalpak tokens is roughly 4,700 characters. Qwen3 supports far
    # longer contexts, but the median document in this corpus is short and
    # longer sequences would be mostly padding or unrelated concatenation.
    sequence_length: int = Field(default=2048, ge=128, le=32768)

    # Concatenate documents and chunk, rather than pad each one. Padding a
    # corpus whose median document is 30 words would waste most of the compute.
    packing: bool = True
    # Insert EOS between packed documents so the model still learns boundaries.
    add_eos_between_documents: bool = True

    validation_split: float = Field(default=0.01, ge=0.0, le=0.5)
    shuffle_seed: int = 20260731
    num_workers: int = Field(default=2, ge=0, le=32)

    # Multilingual replay. Continued pretraining on one language pulls the
    # model towards it and away from everything else; mixing in a fraction of
    # the model's original distribution is the standard mitigation.
    replay_dataset: str | None = None
    replay_ratio: float = Field(default=0.0, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def _replay_needs_a_dataset(self) -> DataConfig:
        if self.replay_ratio > 0 and not self.replay_dataset:
            raise ValueError("replay_ratio is set but replay_dataset is not")
        return self


class OptimConfig(StrictModel):
    """Optimiser and schedule.

    The learning rate is the single most consequential number here. LoRA
    tolerates - and needs - a rate one to two orders of magnitude above full
    fine-tuning, because only a low-rank projection is being updated.
    """

    learning_rate: float = Field(default=1e-4, gt=0, le=1e-2)
    optimizer: str = "adamw_torch_fused"
    weight_decay: float = Field(default=0.01, ge=0.0)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # 0.95 rather than 0.999: standard for LLM pretraining
    adam_epsilon: float = 1e-8
    max_grad_norm: float = Field(default=1.0, gt=0)

    lr_scheduler: Literal["cosine", "linear", "constant", "constant_with_warmup"] = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    # Do not decay to exactly zero: the last steps then contribute nothing.
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class RuntimeConfig(StrictModel):
    """Batch shape, precision and memory."""

    num_epochs: float = Field(default=2.0, gt=0, le=100)
    max_steps: int | None = Field(default=None, ge=1)

    per_device_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)

    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    deepspeed: Path | None = None
    fsdp: str = ""

    seed: int = 20260731
    dataloader_pin_memory: bool = True

    @model_validator(mode="after")
    def _one_precision_mode(self) -> RuntimeConfig:
        if self.bf16 and self.fp16:
            raise ValueError("choose bf16 or fp16, not both")
        return self

    def effective_batch_size(self, world_size: int = 1) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps * world_size

    def tokens_per_step(self, sequence_length: int, world_size: int = 1) -> int:
        return self.effective_batch_size(world_size) * sequence_length


class LoggingConfig(StrictModel):
    """Checkpointing and experiment tracking."""

    output_dir: Path = Path("models/cpt")
    run_name: str = "qwen3-8b-kaa-cpt"

    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=200, ge=1)
    eval_steps: int = Field(default=200, ge=1)
    save_total_limit: int = Field(default=3, ge=1)

    report_to: list[str] = Field(default_factory=lambda: ["wandb"])
    wandb_project: str = "qaraqalpaqmind"
    # Keep the best checkpoint by validation loss, not merely the last one.
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class CPTConfig(StrictModel):
    """A complete continued-pretraining run."""

    method: TuningMethod = TuningMethod.QLORA
    model: ModelConfig = Field(default_factory=ModelConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _full_finetuning_needs_a_lower_rate(self) -> CPTConfig:
        # 1e-4 is right for LoRA and catastrophic for full fine-tuning, where
        # it will destroy the multilingual ability this project depends on.
        if self.method is TuningMethod.FULL and self.optim.learning_rate > 5e-5:
            raise ValueError(
                f"learning_rate {self.optim.learning_rate:g} is far too high for full "
                "fine-tuning; use 1e-5 to 5e-5. The LoRA default of 1e-4 will wreck "
                "the base model's multilingual ability."
            )
        return self

    @model_validator(mode="after")
    def _gradient_checkpointing_disables_cache(self) -> CPTConfig:
        if self.runtime.gradient_checkpointing and self.model.use_cache:
            raise ValueError("use_cache must be false when gradient_checkpointing is on")
        return self

    def estimated_steps(self, corpus_tokens: int, world_size: int = 1) -> int:
        """Optimiser steps for one full pass, given a corpus token count."""
        tokens_per_step = self.runtime.tokens_per_step(self.data.sequence_length, world_size)
        return max(1, int(corpus_tokens * self.runtime.num_epochs / tokens_per_step))
