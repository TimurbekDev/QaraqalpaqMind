"""Direct preference optimisation with TRL's DPOTrainer.

DPO skips reward-model training entirely: it optimises the policy directly
against pairwise preferences, using the frozen pre-DPO model as an implicit
reference. `beta` sets how far the policy may drift from that reference.

Runs on top of the Phase 6 SFT adapter. Running it on a base or CPT-only
checkpoint would optimise preferences over a model that cannot follow
instructions yet.

With LoRA the reference model is the *same* model with adapters disabled, so no
second copy of the weights is loaded. That is what makes this fit on one card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.io import read_jsonl
from ...common.logging import get_logger
from ...common.paths import DPO_DIR, PROJECT_ROOT
from ...schemas import PreferenceRecord
from ..checkpoints import resolve_resume
from ..config import TuningMethod
from .config import DPOConfig

logger = get_logger(__name__)


def load_tokenizer(config: DPOConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(f"{config.model.name} has no chat template")
    return tokenizer


def load_model(config: DPOConfig) -> Any:
    """Base model with the SFT adapter merged, then a fresh DPO adapter."""
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

    if config.sft_adapter is not None:
        adapter_path = config.sft_adapter
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"No SFT adapter at {adapter_path}. Run `qm train sft` first. DPO on a "
                "model that cannot follow instructions optimises preferences over "
                "behaviour that does not exist yet."
            )
        from peft import PeftModel

        logger.info("Merging SFT adapter", extra={"path": str(adapter_path)})
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
    else:
        logger.warning("No SFT adapter configured; DPO will run on the base model")

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
    """Load a preference split in the conversational format TRL expects."""
    from datasets import Dataset

    path = DPO_DIR / f"{name}_{split}.jsonl.zst"
    if not path.exists():
        raise FileNotFoundError(f"No DPO data at {path}. Build it with `qm dpo build`.")

    rows = [PreferenceRecord.model_validate(raw).to_dpo_row() for raw in read_jsonl(path)]
    if not rows:
        raise ValueError(f"{path} is empty")

    logger.info("Loaded preference split", extra={"split": split, "pairs": len(rows)})
    return Dataset.from_list(rows)


def build_trainer(config: DPOConfig, model: Any, tokenizer: Any, train: Any, eval_: Any) -> Any:
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    output_dir = config.logging.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    arguments = TRLDPOConfig(
        output_dir=str(output_dir),
        run_name=config.logging.run_name,
        beta=config.beta,
        loss_type=config.loss_type,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        num_train_epochs=config.runtime.num_epochs,
        max_steps=config.runtime.max_steps or -1,
        per_device_train_batch_size=config.runtime.per_device_batch_size,
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
    )

    return DPOTrainer(
        model=model,
        # ref_model=None with a PEFT model: TRL uses the adapter-disabled model
        # as the reference, so the weights are not loaded twice.
        ref_model=None,
        args=arguments,
        train_dataset=train,
        eval_dataset=eval_,
        processing_class=tokenizer,
    )


def run(config: DPOConfig) -> Path:
    """Execute a DPO run and return the output directory."""
    import os

    if config.logging.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", config.logging.wandb_project)

    logger.info(
        "Starting preference optimisation",
        extra={
            "model": config.model.name,
            "sft_adapter": str(config.sft_adapter) if config.sft_adapter else None,
            "beta": config.beta,
            "loss_type": config.loss_type,
        },
    )

    tokenizer = load_tokenizer(config)
    train_set = load_split(config.dataset, "train")
    try:
        eval_set = load_split(config.dataset, "val")
    except FileNotFoundError:
        eval_set = None
        logger.warning("No validation split; training blind to over-optimisation")

    model = load_model(config)
    trainer = build_trainer(config, model, tokenizer, train_set, eval_set)
    resume = resolve_resume(config.runtime.resume_from_checkpoint, Path(trainer.args.output_dir))
    trainer.train(resume_from_checkpoint=resume)

    output_dir = Path(trainer.args.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("DPO complete", extra={"output_dir": str(output_dir)})
    return output_dir


def main() -> None:
    """Entrypoint for `accelerate launch -m qaraqalpaqmind.training.dpo.train`."""
    import argparse

    from ...common.config import load_config

    parser = argparse.ArgumentParser(description="Direct preference optimisation.")
    parser.add_argument("--config", required=True, help="Path to a DPO YAML config.")
    arguments = parser.parse_args()

    run(load_config(arguments.config, DPOConfig))


if __name__ == "__main__":
    main()
