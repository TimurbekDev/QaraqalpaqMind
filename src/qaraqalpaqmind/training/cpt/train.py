"""Continued pretraining for Qwen3-8B on Karakalpak.

Run:
    qm train cpt --config configs/cpt/qwen3_8b_qlora_24gb.yaml

or, for multi-GPU:
    accelerate launch -m qaraqalpaqmind.training.cpt.train --config <path>

Heavy imports (torch, transformers, peft) happen inside functions so that
`qm --help` and `qm train plan` work on a machine with no CUDA stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.io import read_jsonl
from ...common.logging import get_logger
from ...common.paths import PROJECT_ROOT
from ...dedup.pipeline import output_path
from ..config import CPTConfig, TuningMethod
from .packing import PackingStats, pack_documents

logger = get_logger(__name__)


def load_tokenizer(config: CPTConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        # Qwen ships no pad token. Padding with EOS is standard, and harmless
        # here because packing means nothing is actually padded.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: CPTConfig) -> Any:
    """Load the base model, quantised and adapter-wrapped as configured."""
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
            load_in_4bit=config.quantization.load_in_4bit,
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(config.model.name, **kwargs)
    model.config.use_cache = config.model.use_cache

    if config.method is TuningMethod.FULL:
        logger.warning(
            "Full-parameter training on a 28.7M-token corpus. This is very likely "
            "to cause catastrophic forgetting of the multilingual ability that "
            "makes cross-lingual transfer work. LoRA is the recommended method."
        )
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
            modules_to_save=config.lora.modules_to_save or None,
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def build_dataset(config: CPTConfig, tokenizer: Any) -> tuple[Any, Any, PackingStats]:
    """Pack the corpus into fixed-length sequences and split off a validation set."""
    from datasets import Dataset

    source = output_path(config.data.dataset)
    if not source.exists():
        raise FileNotFoundError(
            f"No dataset at {source}. Build it with `qm dedup run` first."
        )

    stats = PackingStats()
    texts = (str(row[config.data.text_field]) for row in read_jsonl(source))
    sequences = list(
        pack_documents(
            texts,
            tokenizer,
            config.data.sequence_length,
            add_eos=config.data.add_eos_between_documents,
            stats=stats,
        )
    )
    if not sequences:
        raise ValueError("packing produced no sequences; is the dataset empty?")

    dataset = Dataset.from_list(sequences).shuffle(seed=config.data.shuffle_seed)

    if config.data.validation_split <= 0:
        return dataset, None, stats

    split = dataset.train_test_split(
        test_size=config.data.validation_split, seed=config.data.shuffle_seed
    )
    return split["train"], split["test"], stats


def build_trainer(config: CPTConfig, model: Any, tokenizer: Any, train: Any, eval_: Any) -> Any:
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    output_dir = config.logging.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    arguments = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config.logging.run_name,
        num_train_epochs=config.runtime.num_epochs,
        max_steps=config.runtime.max_steps or -1,
        per_device_train_batch_size=config.runtime.per_device_batch_size,
        gradient_accumulation_steps=config.runtime.gradient_accumulation_steps,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=config.optim.learning_rate,
        optim=config.optim.optimizer,
        weight_decay=config.optim.weight_decay,
        adam_beta1=config.optim.adam_beta1,
        adam_beta2=config.optim.adam_beta2,
        adam_epsilon=config.optim.adam_epsilon,
        max_grad_norm=config.optim.max_grad_norm,
        lr_scheduler_type=config.optim.lr_scheduler,
        warmup_ratio=config.optim.warmup_ratio,
        bf16=config.runtime.bf16,
        fp16=config.runtime.fp16,
        tf32=config.runtime.tf32,
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
        dataloader_num_workers=config.data.num_workers,
        dataloader_pin_memory=config.runtime.dataloader_pin_memory,
        deepspeed=str(config.runtime.deepspeed) if config.runtime.deepspeed else None,
    )

    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=eval_,
        # mlm=False: causal language modelling, not masked.
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )


def run(config: CPTConfig) -> Path:
    """Execute a continued-pretraining run and return the output directory."""
    import os

    if config.logging.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", config.logging.wandb_project)

    logger.info(
        "Starting continued pretraining",
        extra={
            "model": config.model.name,
            "method": config.method.value,
            "sequence_length": config.data.sequence_length,
            "tokens_per_step": config.runtime.tokens_per_step(config.data.sequence_length),
        },
    )

    tokenizer = load_tokenizer(config)
    train_set, eval_set, packing = build_dataset(config, tokenizer)
    logger.info("Dataset ready", extra={"summary": packing.summary()})

    model = load_model(config)
    trainer = build_trainer(config, model, tokenizer, train_set, eval_set)

    trainer.train()

    output_dir = Path(trainer.args.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Training complete", extra={"output_dir": str(output_dir)})
    return output_dir


def main() -> None:
    """Entrypoint for `accelerate launch -m qaraqalpaqmind.training.cpt.train`."""
    import argparse

    from ...common.config import load_config

    parser = argparse.ArgumentParser(description="Continued pretraining.")
    parser.add_argument("--config", required=True, help="Path to a CPT YAML config.")
    arguments = parser.parse_args()

    run(load_config(arguments.config, CPTConfig))


if __name__ == "__main__":
    main()
