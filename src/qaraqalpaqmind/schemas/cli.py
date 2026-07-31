"""`qm schema` - inspect and validate task datasets.

    qm schema list                    the eleven task types
    qm schema example <task>          a worked Karakalpak example, as JSONL
    qm schema chat <task>             how that example renders as chat messages
    qm schema validate <file>         check a JSONL file against its schema
"""

from __future__ import annotations

import pathlib
from collections import Counter

import orjson
import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..common.io import read_jsonl
from ..common.logging import get_logger
from .base import TaskType
from .examples import EXAMPLES
from .tasks import RECORD_TYPES, parse_record

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Dataset schemas for every task type.", no_args_is_help=True)


@app.command("list")
def list_tasks() -> None:
    """Show every task type and its required fields."""
    table = Table(title="Task record types")
    for column in ("task", "required fields", "purpose"):
        table.add_column(column, overflow="fold")

    purposes = {
        TaskType.PRETRAIN: "raw text for continued pretraining",
        TaskType.INSTRUCTION: "single-turn instruction following",
        TaskType.CONVERSATION: "multi-turn dialogue",
        TaskType.TRANSLATION: "kaa <-> uz/ru/en translation",
        TaskType.GRAMMAR: "orthography and grammar correction",
        TaskType.QA: "question answering, optionally grounded",
        TaskType.SUMMARIZATION: "condensing a document",
        TaskType.REASONING: "questions needing stated steps",
        TaskType.CODING: "programming, prompted in Karakalpak",
        TaskType.MATH: "mathematics with a worked solution",
        TaskType.BENCHMARK: "evaluation items - never trained on",
    }

    for task, model in RECORD_TYPES.items():
        required = [
            name
            for name, field in model.model_fields.items()
            if field.is_required() and name not in {"task", "provenance"}
        ]
        table.add_row(task.value, ", ".join(required) or "-", purposes.get(task, ""))
    console.print(table)


@app.command()
def example(task: str) -> None:
    """Print a worked Karakalpak example for a task, as it appears in JSONL."""
    record = EXAMPLES.get(_parse_task(task))
    if record is None:
        raise typer.BadParameter(f"no example for '{task}'")
    payload = orjson.dumps(record.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    console.print(Syntax(payload.decode(), "json", theme="ansi_dark", word_wrap=True))


@app.command()
def chat(task: str) -> None:
    """Show how a task renders as chat messages for supervised fine-tuning."""
    record = EXAMPLES.get(_parse_task(task))
    if record is None:
        raise typer.BadParameter(f"no example for '{task}'")
    for message in record.to_messages():
        colour = {"system": "bright_black", "user": "cyan", "assistant": "green"}
        console.print(f"[{colour.get(message['role'], 'white')}]{message['role']}[/]")
        console.print(f"  {message['content']}\n")


@app.command()
def validate(
    path: pathlib.Path,
    show: int = typer.Option(5, "--show", help="How many errors to print."),
) -> None:
    """Validate a JSONL dataset, reporting every row that fails."""
    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")

    total = 0
    failures: list[tuple[int, str]] = []
    tasks: Counter[str] = Counter()

    for line_number, row in enumerate(read_jsonl(path), start=1):
        total += 1
        try:
            record = parse_record(row)
        except (ValueError, TypeError) as exc:
            failures.append((line_number, str(exc).split("\n")[0]))
        else:
            tasks[record.task.value] += 1

    console.print(f"{total:,} rows, [green]{total - len(failures):,} valid[/]", end="")
    console.print(f", [red]{len(failures):,} invalid[/]" if failures else "")

    if tasks:
        console.print("  " + ", ".join(f"{task}={count:,}" for task, count in tasks.most_common()))

    for line_number, message in failures[:show]:
        console.print(f"  [red]line {line_number}[/]: {message}")
    if len(failures) > show:
        console.print(f"  [bright_black]... and {len(failures) - show:,} more[/]")

    if failures:
        raise typer.Exit(code=1)


def _parse_task(value: str) -> TaskType:
    try:
        return TaskType(value)
    except ValueError as exc:
        known = ", ".join(t.value for t in TaskType)
        raise typer.BadParameter(f"unknown task '{value}'. Known: {known}") from exc
