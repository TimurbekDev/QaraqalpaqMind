"""`qm train` - continued pretraining.

    qm train plan --config <yaml>     what the run would do, without a GPU
    qm train cpt  --config <yaml>     run it
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from ..common.config import load_config

if TYPE_CHECKING:
    from ..common.runtime import Finding, GpuInfo
from ..common.logging import get_logger
from ..ingest.manifest import read_manifest
from .config import CPTConfig, TuningMethod
from .cpt.packing import describe_schedule

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Train the model.", no_args_is_help=True)

_DEFAULT_CONFIG = "cpt/qwen3_8b_qlora_24gb.yaml"

# Rough VRAM required to hold weights, adapters and optimizer state for 8B,
# before activations. Activations add ~4-6GB at sequence length 2048.
_WEIGHT_MEMORY_GB = {TuningMethod.QLORA: 6.3, TuningMethod.LORA: 17.5, TuningMethod.FULL: 128.0}


@app.command()
def plan(
    config_path: str = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    world_size: int = typer.Option(1, "--gpus", min=1),
) -> None:
    """Describe the run before committing a GPU to it.

    Answers "how many steps, how many tokens, will it fit?" from the config
    alone. Launching a run that would take a week nobody has is a mistake worth
    catching in one second rather than one night.
    """
    config = load_config(config_path, CPTConfig)

    try:
        manifest = read_manifest(config.data.dataset)
    except (OSError, ValueError):
        corpus_tokens, source = 28_683_631, "measured default"
    else:
        corpus_tokens = manifest.tokens
        # The chars/3.1 estimate understated this corpus by 35%, so a plan built
        # on it would size the run wrongly.
        source = (
            f"measured, {manifest.tokenizer}"
            if manifest.measured_tokens is not None
            else "ESTIMATE - run `qm tokenizer count` for the real figure"
        )

    schedule = describe_schedule(
        corpus_tokens,
        config.data.sequence_length,
        config.runtime.effective_batch_size(world_size),
        config.runtime.num_epochs,
    )

    table = Table(title=f"Run plan: {config.logging.run_name}")
    table.add_column("setting")
    table.add_column("value", justify="right")

    rows = [
        ("model", config.model.name),
        ("method", config.method.value),
        ("GPUs", str(world_size)),
        ("sequence length", f"{config.data.sequence_length:,} tokens"),
        ("corpus", f"{corpus_tokens:,} tokens ({source})"),
        ("epochs", f"{config.runtime.num_epochs:g}"),
        ("", ""),
        ("per-device batch", str(config.runtime.per_device_batch_size)),
        ("gradient accumulation", str(config.runtime.gradient_accumulation_steps)),
        ("effective batch", f"{config.runtime.effective_batch_size(world_size)} sequences"),
        ("tokens per step", f"{schedule['tokens_per_step']:,}"),
        ("", ""),
        ("training sequences", f"{schedule['sequences']:,}"),
        ("steps per epoch", f"{schedule['steps_per_epoch']:,}"),
        ("total steps", f"[bold]{schedule['total_steps']:,}[/]"),
        ("tokens seen", f"{schedule['tokens_seen']:,}"),
        ("", ""),
        ("learning rate", f"{config.optim.learning_rate:g}"),
        ("schedule", f"{config.optim.lr_scheduler}, warmup {config.optim.warmup_ratio:.0%}"),
        ("checkpoint every", f"{config.logging.save_steps} steps"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)

    if config.method is not TuningMethod.FULL:
        trainable = _lora_parameter_estimate(config)
        console.print(
            f"Trainable parameters: ~[cyan]{trainable / 1e6:.0f}M[/] "
            f"of 8,000M ([cyan]{trainable / 8e9:.2%}[/])"
        )

    memory = _WEIGHT_MEMORY_GB[config.method] / max(
        world_size if config.runtime.deepspeed else 1, 1
    )
    console.print(
        f"Weights + optimizer: ~[cyan]{memory:.1f} GB[/] per GPU, "
        f"plus ~2.5-4 GB activations at this sequence length"
    )
    _report_logits_memory(config)
    _warn(config, schedule)


# Qwen3-8B vocabulary. The logits tensor is (batch x seq x vocab), and the
# causal-LM loss upcasts it to fp32 - which is the largest single allocation in
# the whole run and was missing from the original budget.
_QWEN3_VOCAB = 151_936


def _logits_gb(batch: int, sequence_length: int) -> float:
    """Peak logits memory: the bf16 tensor plus its fp32 copy in the loss."""
    elements = batch * sequence_length * _QWEN3_VOCAB
    return (elements * 2 + elements * 4) / 1024**3


def _report_logits_memory(config: CPTConfig) -> None:
    """Show the logits cost for train and eval batches separately.

    This exists because a config that set only the train batch size trained
    fine for 200 steps and then OOMed at the first evaluation: transformers
    defaults `per_device_eval_batch_size` to 8, which asks for 9.27 GiB in one
    allocation at sequence length 2048.
    """
    train = _logits_gb(config.runtime.per_device_batch_size, config.data.sequence_length)
    evaluate = _logits_gb(
        config.runtime.per_device_eval_batch_size, config.data.sequence_length
    )

    console.print(
        f"Logits (batch x seq x {_QWEN3_VOCAB:,} vocab, bf16 + fp32 loss copy): "
        f"train [cyan]{train:.2f} GB[/], eval [cyan]{evaluate:.2f} GB[/]"
    )

    if evaluate > train * 1.5:
        console.print(
            f"  [yellow]Eval batch {config.runtime.per_device_eval_batch_size} costs "
            f"{evaluate / max(train, 0.01):.1f}x the train batch's logits memory.[/] "
            "Evaluation will be the peak, and it only happens every eval_steps."
        )
    if evaluate > 6.0:
        console.print(
            f"  [red]Eval logits alone need {evaluate:.1f} GB.[/] Lower "
            "runtime.per_device_eval_batch_size."
        )


def _lora_parameter_estimate(config: CPTConfig) -> float:
    """Rough trainable-parameter count for Qwen3-8B with these LoRA settings."""
    # Qwen3-8B: 36 layers, hidden 4096, intermediate 12288.
    layers, hidden, intermediate = 36, 4096, 12288
    per_layer = 0.0
    for module in config.lora.target_modules:
        out = intermediate if module in {"gate_proj", "up_proj"} else hidden
        in_features = intermediate if module == "down_proj" else hidden
        per_layer += config.lora.r * (in_features + out)
    return per_layer * layers


def _warn(config: CPTConfig, schedule: dict[str, int | float]) -> None:
    if config.method is TuningMethod.FULL:
        console.print(
            "\n[red]Full-parameter training on a 28.7M-token corpus.[/] That is roughly "
            "280 parameters per training token and is very likely to cause catastrophic "
            "forgetting of the multilingual ability cross-lingual transfer depends on. "
            "LoRA is the recommended method for this project."
        )
    if config.runtime.num_epochs > 3:
        console.print(
            f"\n[yellow]{config.runtime.num_epochs:g} epochs on a small corpus invites "
            "memorisation.[/] Watch validation loss and stop when it turns."
        )
    if int(schedule["total_steps"]) < 100:
        console.print(
            f"\n[yellow]Only {schedule['total_steps']} optimiser steps.[/] The learning-rate "
            "schedule will barely leave warmup; lower the effective batch size or raise "
            "the epoch count."
        )
    if config.data.replay_ratio == 0:
        console.print(
            "\n[bright_black]No multilingual replay configured. If evaluation shows the "
            "model losing English or Russian, set data.replay_dataset and replay_ratio.[/]"
        )


@app.command()
def preflight(
    config_path: str = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Check the machine can run this config, before downloading 16 GB."""
    from ..common.runtime import Finding, gpu_info
    from ..common.runtime import preflight as run_preflight

    config = load_config(config_path, CPTConfig)
    findings = run_preflight()

    colours = {"ok": "green", "warn": "yellow", "error": "red"}
    for finding in findings:
        console.print(f"[{colours[finding.level]}]{finding.level.upper():<5}[/] {finding.message}")

    gpu = gpu_info()
    if gpu.available and config.model.attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            console.print(
                "[red]ERROR[/] config asks for flash_attention_2 but flash-attn is not "
                'installed. Either `pip install -e ".[flash]" --no-build-isolation` or '
                "set model.attn_implementation to sdpa."
            )
            findings.append(Finding("error", "flash-attn missing"))

    _check_logits_fit(config, gpu, findings)
    _checkpoint_state(config)

    if any(f.level == "error" for f in findings):
        console.print("\n[red]Not ready.[/] Fix the errors above before training.")
        raise typer.Exit(code=1)
    console.print("\n[green]Ready to train.[/]")


#: Everything that is resident while the logits tensor exists: 4-bit base
#: weights, adapters, gradients, optimizer states, checkpointed activations and
#: the CUDA context. Measured for Qwen3-8B QLoRA at r=64.
_RESIDENT_GB = 13.0


def _check_logits_fit(config: CPTConfig, gpu: GpuInfo, findings: list[Finding]) -> None:
    """Fail preflight if the eval logits tensor cannot fit on this card.

    A real run passed every other check, trained for 200 steps, and then died
    asking for 9.27 GiB at the first evaluation. Nothing in preflight looked at
    `per_device_eval_batch_size`, whose transformers default is 8.
    """
    from ..common.runtime import Finding

    train = _logits_gb(config.runtime.per_device_batch_size, config.data.sequence_length)
    evaluate = _logits_gb(
        config.runtime.per_device_eval_batch_size, config.data.sequence_length
    )
    peak = max(train, evaluate)

    # Assume the 24GB target card when running preflight on a laptop, so the
    # check still means something before the pod exists.
    total = gpu.total_memory_gb if gpu.available and gpu.total_memory_gb else 24.0
    budget = total - _RESIDENT_GB

    if peak > budget:
        problem = (
            f"logits need {peak:.2f} GB but only ~{budget:.1f} GB is free on "
            f"{total:.0f} GB after weights and activations. Eval batch "
            f"{config.runtime.per_device_eval_batch_size} x seq "
            f"{config.data.sequence_length:,} x {_QWEN3_VOCAB:,} vocab. Set "
            "runtime.per_device_eval_batch_size to 1."
        )
    elif not config.runtime.prediction_loss_only:
        # Per-batch size is fine, but the Trainer will accumulate logits across
        # the entire eval set to hand to a metrics function we do not define.
        problem = (
            f"runtime.prediction_loss_only is false, so eval logits accumulate across "
            f"the whole eval set at {peak:.2f} GB per batch. Set it to true."
        )
    else:
        console.print(
            f"[green]OK   [/] logits {peak:.2f} GB peak "
            f"(train batch {config.runtime.per_device_batch_size}, "
            f"eval batch {config.runtime.per_device_eval_batch_size}, "
            f"seq {config.data.sequence_length:,})"
        )
        return

    console.print(f"[red]ERROR[/] {problem}")
    findings.append(Finding("error", problem))


def _checkpoint_state(config: CPTConfig) -> None:
    from ..common.paths import PROJECT_ROOT
    from .checkpoints import describe

    output_dir = config.logging.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    console.print(f"[bright_black]{describe(output_dir)}[/]")
    console.print(
        f"[bright_black]resume policy: {config.runtime.resume_from_checkpoint}[/]"
    )


@app.command()
def cpt(
    config_path: str = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate the config and stop."),
) -> None:
    """Run continued pretraining."""
    config = load_config(config_path, CPTConfig)

    if dry_run:
        console.print(f"[green]Config valid[/]: {config_path}")
        console.print(f"  method={config.method.value} model={config.model.name}")
        _checkpoint_state(config)
        return

    from .cpt.train import run

    output = run(config)
    console.print(f"[green]Done.[/] Adapter written to {output}")


@app.command()
def sft(
    config_path: str = typer.Option("sft/qwen3_8b_qlora_24gb.yaml", "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate the config and stop."),
) -> None:
    """Run supervised fine-tuning on top of the CPT adapter."""
    from .sft.config import SFTConfig

    config = load_config(config_path, SFTConfig)

    if dry_run:
        console.print(f"[green]Config valid[/]: {config_path}")
        console.print(
            f"  method={config.method.value} dataset={config.dataset} "
            f"cpt_adapter={config.cpt_adapter}"
        )
        return

    from .sft.train import run as run_sft

    output = run_sft(config)
    console.print(f"[green]Done.[/] Adapter written to {output}")


@app.command()
def dpo(
    config_path: str = typer.Option("dpo/qwen3_8b_qlora_24gb.yaml", "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate the config and stop."),
) -> None:
    """Run direct preference optimisation on top of the SFT adapter."""
    from .dpo.config import DPOConfig

    config = load_config(config_path, DPOConfig)

    if dry_run:
        console.print(f"[green]Config valid[/]: {config_path}")
        console.print(
            f"  beta={config.beta} lr={config.optim.learning_rate:g} "
            f"sft_adapter={config.sft_adapter}"
        )
        return

    from .dpo.train import run as run_dpo

    output = run_dpo(config)
    console.print(f"[green]Done.[/] Adapter written to {output}")


@app.command("configs")
def list_configs() -> None:
    """Show the available training configurations."""
    from ..common.paths import CONFIGS_DIR

    directory = CONFIGS_DIR / "cpt"
    for path in sorted(directory.glob("*.yaml")):
        if path.name == "base.yaml":
            continue
        try:
            config = load_config(f"cpt/{path.name}", CPTConfig)
        except Exception as exc:
            console.print(f"  [red]{path.name}[/]: {type(exc).__name__}: {exc}")
            continue
        console.print(
            f"  [cyan]{path.name:<34}[/] {config.method.value:<6} "
            f"batch={config.runtime.effective_batch_size()} "
            f"lr={config.optim.learning_rate:g} epochs={config.runtime.num_epochs:g}"
        )


def _config_path(name: str) -> Path:
    return Path(name)
