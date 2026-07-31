"""`qm sft` - build and inspect the supervised fine-tuning mixture.

    qm sft seeds                 what hand-authored seed data exists
    qm sft build                 assemble the mixture into data/datasets/sft/
    qm sft inspect               show rendered examples from the built mixture
"""

from __future__ import annotations

import itertools

import typer
from rich.console import Console
from rich.table import Table

from ...common.config import load_config
from ...common.io import read_jsonl
from ...common.logging import get_logger
from ...common.paths import SFT_DIR
from ...schemas import TaskType, parse_record
from .builders import grammar, seeds, summarization, translation
from .config import MixtureConfig
from .mixture import achievable_size, assemble, chain_builders, write_mixture

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Build supervised fine-tuning data.", no_args_is_help=True)


@app.command("seeds")
def show_seeds() -> None:
    """Count the hand-authored seed records, by task."""
    counts = seeds.available_tasks()
    if not counts:
        console.print("[yellow]No seed data found in seeds/.[/]")
        return

    table = Table(title="Hand-authored seeds")
    table.add_column("task")
    table.add_column("records", justify="right")
    for task, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        table.add_row(task, f"{count:,}")
    console.print(table)
    console.print(
        f"[bright_black]{sum(counts.values())} records. This is a seed set, not a "
        "dataset - see seeds/README.md for how to scale it.[/]"
    )


@app.command()
def plan(
    config_path: str = typer.Option("sft/mixture_v1.yaml", "--config", "-c"),
) -> None:
    """How large a *balanced* mixture the available data supports.

    `build` reports a shortfall after the fact. This answers the question that
    raises: given the seed sets that exist, how big can a mixture be before the
    proportions stop being honest?
    """
    config = load_config(config_path, MixtureConfig)
    proportions = {TaskType(task): share for task, share in config.proportions.items()}

    seed_counts = seeds.available_tasks()
    available: dict[TaskType, int] = {
        TaskType.TRANSLATION: config.max_translation,
        TaskType.GRAMMAR: config.max_grammar,
        TaskType.SUMMARIZATION: config.max_summarization,
    }
    for task, count in seed_counts.items():
        available[TaskType(task)] = count

    size, binding = achievable_size(available, proportions)

    table = Table(title="What the available data supports")
    for column in ("task", "share", "available", "needed at target", "balanced max"):
        table.add_column(column, justify="right")

    total_share = sum(proportions.values())
    for task, share in sorted(proportions.items(), key=lambda kv: -kv[1]):
        have = available.get(task, 0)
        needed = int(config.target_size * share / total_share)
        style = "red" if have < needed else "green"
        table.add_row(
            task.value,
            f"{share:.0%}",
            f"[{style}]{have:,}[/]",
            f"{needed:,}",
            f"{int(size * share / total_share):,}",
        )
    console.print(table)

    console.print(
        f"\nTarget size in config: [cyan]{config.target_size:,}[/]. "
        f"Proportion-respecting maximum: [bold]{size:,}[/]"
        + (f", limited by [yellow]{binding.value}[/]." if binding else ".")
    )
    if size < config.target_size:
        console.print(
            "[yellow]Building at the configured target will over-weight the tasks that "
            "have data.[/] Either add seed examples, or build at the balanced maximum:\n"
            f"  qm sft build --size {size:,}".replace(",", "")
        )


@app.command()
def build(
    config_path: str = typer.Option("sft/mixture_v1.yaml", "--config", "-c"),
    target_size: int | None = typer.Option(None, "--size", "-n", min=100),
) -> None:
    """Assemble the SFT mixture from every builder."""
    config = load_config(config_path, MixtureConfig)
    size = target_size or config.target_size

    console.print(f"Building [cyan]{config.name}[/], target {size:,} records")

    builders = chain_builders(
        translation.build(
            limit=config.max_translation,
            both_directions=config.both_translation_directions,
            seed=config.seed,
        ),
        grammar.build(
            limit=config.max_grammar,
            seed=config.seed,
            include_explanation=config.grammar_explanations,
        ),
        summarization.build(limit=config.max_summarization),
        seeds.build(),
    )

    proportions = {TaskType(task): share for task, share in config.proportions.items()}
    train, validation, stats = assemble(
        builders,
        target_size=size,
        mixture=proportions,
        validation_split=config.validation_split,
        seed=config.seed,
        check_contamination=config.check_contamination,
    )

    train_path, validation_path = write_mixture(train, validation, config.name)
    _report(stats, str(train_path), str(validation_path))


def _report(stats: object, train_path: str, validation_path: str) -> None:
    from .mixture import MixtureStats

    assert isinstance(stats, MixtureStats)

    table = Table(title="SFT mixture")
    for column in ("task", "records", "share"):
        table.add_column(column, justify="right")

    total = sum(stats.by_task.values()) or 1
    for task, count in sorted(stats.by_task.items(), key=lambda kv: -kv[1]):
        table.add_row(task, f"{count:,}", f"{count / total:.1%}")
    console.print(table)

    console.print(
        f"train [green]{stats.train:,}[/] / val [green]{stats.validation:,}[/]  "
        f"[bright_black](dropped {stats.duplicates:,} duplicates, "
        f"{stats.over_cap:,} over cap, {stats.contaminated:,} contaminated)[/]"
    )
    console.print(f"  {train_path}\n  {validation_path}")

    if stats.shortfall:
        console.print(
            "\n[yellow]These tasks could not fill their share of the mixture:[/]"
        )
        for task, missing in sorted(stats.shortfall.items(), key=lambda kv: -kv[1]):
            console.print(f"  {task:<16} short by {missing:,}")
        console.print(
            "[bright_black]Expected for the seed-backed tasks. Add examples to "
            "seeds/ to close the gap.[/]"
        )


@app.command()
def inspect(
    dataset: str = typer.Option("sft_v1", "--dataset", "-d"),
    task: str | None = typer.Option(None, "--task", "-t"),
    count: int = typer.Option(2, "--count", "-n", min=1, max=20),
) -> None:
    """Show how built records render as chat messages."""
    path = SFT_DIR / f"{dataset}_train.jsonl.zst"
    if not path.exists():
        raise typer.BadParameter(f"No dataset at {path}. Run `qm sft build` first.")

    shown = 0
    for raw in itertools.islice(read_jsonl(path), 20_000):
        if shown >= count:
            break
        if task and raw.get("task") != task:
            continue
        record = parse_record(raw)
        console.print(f"\n[bold cyan]--- {record.task.value}[/]  {record.id[:12]}")
        for message in record.to_messages():
            colour = {"system": "bright_black", "user": "cyan", "assistant": "green"}
            console.print(f"[{colour.get(message['role'], 'white')}]{message['role']}:[/]")
            body = message["content"]
            console.print(f"  {body[:400]}{'...' if len(body) > 400 else ''}")
        shown += 1

    if shown == 0:
        console.print(f"[yellow]No records found{f' for task {task}' if task else ''}.[/]")
