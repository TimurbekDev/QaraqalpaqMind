"""`qm dpo` - build and inspect preference data.

    qm dpo build      assemble pairs into data/datasets/dpo/
    qm dpo inspect    show chosen/rejected pairs from the built set
"""

from __future__ import annotations

import itertools

import typer
from rich.console import Console
from rich.table import Table

from ...common.config import load_config
from ...common.io import read_jsonl
from ...common.logging import get_logger
from ...common.paths import DPO_DIR
from ...schemas import PreferenceRecord
from .builders import language, orthography, quality
from .config import PreferenceMixtureConfig
from .mixture import PreferenceStats, assemble, chain_builders, write_mixture

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Build preference data for DPO.", no_args_is_help=True)


@app.command()
def build(
    config_path: str = typer.Option("dpo/mixture_v1.yaml", "--config", "-c"),
    target_size: int | None = typer.Option(None, "--size", "-n", min=100),
) -> None:
    """Assemble the preference dataset from every builder."""
    config = load_config(config_path, PreferenceMixtureConfig)
    size = target_size or config.target_size

    console.print(f"Building [cyan]{config.name}[/], target {size:,} pairs")

    builders = chain_builders(
        language.build(limit=config.max_language, seed=config.seed),
        orthography.build(limit=config.max_orthography, seed=config.seed),
        quality.build(limit=config.max_quality, seed=config.seed),
    )

    train, validation, stats = assemble(
        builders,
        target_size=size,
        mixture=config.proportions,
        validation_split=config.validation_split,
        seed=config.seed,
        check_contamination=config.check_contamination,
    )

    train_path, validation_path = write_mixture(train, validation, config.name)
    _report(stats, str(train_path), str(validation_path))


def _report(stats: PreferenceStats, train_path: str, validation_path: str) -> None:
    table = Table(title="Preference mixture")
    for column in ("criterion", "pairs", "share"):
        table.add_column(column, justify="right")

    total = sum(stats.by_criterion.values()) or 1
    for criterion, count in sorted(stats.by_criterion.items(), key=lambda kv: -kv[1]):
        table.add_row(criterion, f"{count:,}", f"{count / total:.1%}")
    console.print(table)

    console.print(
        f"train [green]{stats.train:,}[/] / val [green]{stats.validation:,}[/]  "
        f"[bright_black](dropped {stats.duplicates:,} duplicates, "
        f"{stats.over_cap:,} over cap, {stats.too_similar:,} too similar, "
        f"{stats.contaminated:,} contaminated)[/]"
    )
    console.print(f"  {train_path}\n  {validation_path}")

    if stats.shortfall:
        console.print("\n[yellow]Criteria short of their share:[/]")
        for criterion, missing in sorted(stats.shortfall.items(), key=lambda kv: -kv[1]):
            console.print(f"  {criterion:<24} short by {missing:,}")


@app.command()
def inspect(
    dataset: str = typer.Option("dpo_v1", "--dataset", "-d"),
    criterion: str | None = typer.Option(None, "--criterion", "-t"),
    count: int = typer.Option(3, "--count", "-n", min=1, max=20),
) -> None:
    """Show chosen/rejected pairs, to check what is actually being taught."""
    path = DPO_DIR / f"{dataset}_train.jsonl.zst"
    if not path.exists():
        raise typer.BadParameter(f"No dataset at {path}. Run `qm dpo build` first.")

    shown = 0
    for raw in itertools.islice(read_jsonl(path), 20_000):
        if shown >= count:
            break
        if criterion and raw.get("criterion") != criterion:
            continue
        record = PreferenceRecord.model_validate(raw)
        console.print(f"\n[bold cyan]--- {record.criterion}[/]  {record.id[:12]}")
        console.print(f"[bright_black]prompt:[/] {record.prompt[:200]}")
        console.print(f"[green]chosen  :[/] {record.chosen[:200]}")
        console.print(f"[red]rejected:[/] {record.rejected[:200]}")
        shown += 1

    if shown == 0:
        console.print("[yellow]No pairs found.[/]")
