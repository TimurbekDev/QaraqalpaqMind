"""`qm train` - continued pretraining.

    qm train plan --config <yaml>     what the run would do, without a GPU
    qm train cpt  --config <yaml>     run it
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..common.config import load_config
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
        f"plus ~4-6 GB activations at this sequence length"
    )
    _warn(config, schedule)


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
def cpt(
    config_path: str = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate the config and stop."),
) -> None:
    """Run continued pretraining."""
    config = load_config(config_path, CPTConfig)

    if dry_run:
        console.print(f"[green]Config valid[/]: {config_path}")
        console.print(f"  method={config.method.value} model={config.model.name}")
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
