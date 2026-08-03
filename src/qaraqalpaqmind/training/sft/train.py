"""Supervised fine-tuning with TRL's SFTTrainer.

Runs on top of the Phase 5 continued-pretraining adapter, not on the raw base
model: CPT teaches the language, SFT teaches the model to be useful in it.
Skipping straight to SFT would ask an 8B model to learn Karakalpak from 50k
instruction pairs, which it cannot.

Two details do most of the work here:

* **The chat template.** Records are rendered with the tokenizer's own template
  so training and inference format messages identically. Hand-assembling a
  prompt string is how a model ends up unable to follow the format it was
  served with.
* **Completion-only loss.** Loss is computed on assistant turns only. Training
  on the prompt tokens as well teaches the model to generate questions, which
  shows up at inference as a model that continues the user's turn instead of
  answering it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.io import read_jsonl
from ...common.logging import get_logger
from ...common.paths import PROJECT_ROOT, SFT_DIR
from ...schemas import parse_record
from ..checkpoints import resolve_resume
from ..config import TuningMethod
from .config import SFTConfig

logger = get_logger(__name__)


def load_tokenizer(config: SFTConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(
            f"{config.model.name} has no chat template. SFT needs one so training and "
            "inference format messages identically."
        )
    return tokenizer


def load_model(config: SFTConfig) -> Any:
    """Load the base model, then the CPT adapter, then a fresh SFT adapter.

    The CPT adapter is *merged* rather than stacked. Two live adapters would
    have to be kept together for every later step - serving, DPO, evaluation -
    and merging makes the language knowledge part of the weights the SFT
    adapter trains against.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, config.model.torch_dtype)
    kwargs: dict[str, Any] = {
        "revision": config.model.revision,
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": config.model.attn_implementation,
        "dtype": dtype,
    }

    if config.method is TuningMethod.QLORA:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)
    model.config.use_cache = False

    if config.cpt_adapter is not None:
        adapter_path = config.cpt_adapter
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"No CPT adapter at {adapter_path}. Run `qm train cpt` first, or set "
                "cpt_adapter to null to fine-tune the base model directly - which "
                "will produce a much weaker Karakalpak model."
            )
        from peft import PeftModel

        logger.info("Merging CPT adapter", extra={"path": str(adapter_path)})
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    else:
        logger.warning(
            "No CPT adapter configured. Fine-tuning the base model directly means "
            "asking it to learn Karakalpak from instruction pairs alone."
        )

    if config.method is TuningMethod.FULL:
        return model

    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, prepare_model_for_kbit_training

    if config.method is TuningMethod.QLORA:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.runtime.gradient_checkpointing
        )

    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias=config.lora.bias,
            target_modules=config.lora.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def load_split(name: str, split: str) -> Any:
    """Load one SFT split and render it as chat messages."""
    from datasets import Dataset

    path = SFT_DIR / f"{name}_{split}.jsonl.zst"
    if not path.exists():
        raise FileNotFoundError(f"No SFT data at {path}. Build it with `qm sft build`.")

    rows = []
    for raw in read_jsonl(path):
        record = parse_record(raw)
        rows.append({"messages": record.to_messages(), "task": record.task.value})

    if not rows:
        raise ValueError(f"{path} is empty")
    logger.info("Loaded SFT split", extra={"split": split, "records": len(rows)})
    return Dataset.from_list(rows)


def build_trainer(config: SFTConfig, model: Any, tokenizer: Any, train: Any, eval_: Any) -> Any:
    from trl import SFTConfig as TRLConfig
    from trl import SFTTrainer

    output_dir = config.logging.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    arguments = TRLConfig(
        output_dir=str(output_dir),
        run_name=config.logging.run_name,
        num_train_epochs=config.runtime.num_epochs,
        max_steps=config.runtime.max_steps or -1,
        per_device_train_batch_size=config.runtime.per_device_batch_size,
        # Explicit: the transformers default is 8, which OOMs at the first
        # evaluation because the logits tensor is batch x seq x vocab.
        per_device_eval_batch_size=config.runtime.per_device_eval_batch_size,
        prediction_loss_only=config.runtime.prediction_loss_only,
        gradient_accumulation_steps=config.runtime.gradient_accumulation_steps,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=config.optim.learning_rate,
        optim=config.optim.optimizer,
        weight_decay=config.optim.weight_decay,
        max_grad_norm=config.optim.max_grad_norm,
        lr_scheduler_type=config.optim.lr_scheduler,
        warmup_ratio=config.optim.warmup_ratio,
        bf16=config.runtime.bf16,
        fp16=config.runtime.fp16,
        logging_steps=config.logging.logging_steps,
        save_steps=config.logging.save_steps,
        eval_steps=config.logging.eval_steps,
        eval_strategy="steps" if eval_ is not None else "no",
        save_total_limit=config.logging.save_total_limit,
        load_best_model_at_end=config.logging.load_best_model_at_end and eval_ is not None,
        metric_for_best_model=config.logging.metric_for_best_model,
        greater_is_better=config.logging.greater_is_better,
        report_to=config.logging.report_to,
        seed=config.runtime.seed,
        deepspeed=str(config.runtime.deepspeed) if config.runtime.deepspeed else None,
        max_length=config.max_sequence_length,
        packing=False,  # never pack SFT: it would blend one example's answer into the next
        # Loss on assistant turns only. Without this the model learns to
        # generate user questions as readily as answers.
        assistant_only_loss=config.completion_only_loss,
    )

    return SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=eval_,
        processing_class=tokenizer,
    )


def run(config: SFTConfig) -> Path:
    """Execute a supervised fine-tuning run and return the output directory."""
    import os

    if config.logging.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", config.logging.wandb_project)

    logger.info(
        "Starting supervised fine-tuning",
        extra={
            "model": config.model.name,
            "method": config.method.value,
            "cpt_adapter": str(config.cpt_adapter) if config.cpt_adapter else None,
            "dataset": config.dataset,
        },
    )

    tokenizer = load_tokenizer(config)
    train_set = load_split(config.dataset, "train")
    try:
        eval_set = load_split(config.dataset, "val")
    except FileNotFoundError:
        eval_set = None
        logger.warning("No validation split; training blind to overfitting")

    model = load_model(config)
    trainer = build_trainer(config, model, tokenizer, train_set, eval_set)
    resume = resolve_resume(config.runtime.resume_from_checkpoint, Path(trainer.args.output_dir))
    trainer.train(resume_from_checkpoint=resume)

    output_dir = Path(trainer.args.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("SFT complete", extra={"output_dir": str(output_dir)})
    return output_dir


def main() -> None:
    """Entrypoint for `accelerate launch -m qaraqalpaqmind.training.sft.train`."""
    import argparse

    from ...common.config import load_config

    parser = argparse.ArgumentParser(description="Supervised fine-tuning.")
    parser.add_argument("--config", required=True, help="Path to an SFT YAML config.")
    arguments = parser.parse_args()

    run(load_config(arguments.config, SFTConfig))


if __name__ == "__main__":
    main()
