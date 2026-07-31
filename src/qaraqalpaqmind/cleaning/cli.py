"""`qm clean` - normalise, transliterate and quality-filter the corpus.

    qm clean sources              what has interim data waiting
    qm clean run <source_id>      clean one source into data/processed/
    qm clean all                  clean everything
    qm clean sample <source_id>   show what cleaning does, without writing
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..common.io import read_jsonl
from ..common.logging import get_logger
from ..common.records import Document
from .filters import FilterConfig
from .pipeline import (
    CleanStats,
    available_sources,
    clean_document,
    clean_source,
    interim_path,
)

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Clean and filter the corpus.", no_args_is_help=True)


def _filter_config(min_score: float, no_language_check: bool) -> FilterConfig:
    return FilterConfig(min_quality_score=min_score, check_language=not no_language_check)


@app.command("sources")
def list_sources() -> None:
    """Show sources with interim data ready to clean."""
    sources = available_sources()
    if not sources:
        console.print("[yellow]No interim data. Run `qm ingest` first.[/]")
        return
    for source in sources:
        size = interim_path(source).stat().st_size / 1_048_576
        console.print(f"  {source:<28} {size:>7.1f} MB")


@app.command()
def sample(
    source: str,
    count: int = typer.Option(3, "--count", "-n", min=1, max=20),
) -> None:
    """Show before/after for a few documents, without writing anything."""
    for index, row in enumerate(read_jsonl(interim_path(source))):
        if index >= count:
            break
        document = Document.model_validate(row)
        cleaned, assessment = clean_document(document)

        console.print(f"\n[bold cyan]--- {document.id[:12]}[/]  {document.source_url or ''}")
        console.print(f"[bright_black]script {document.script.value} -> {cleaned.script.value}, "
                      f"score {assessment.score:.2f}, flags {[f.value for f in assessment.flags]}[/]")
        console.print(f"[yellow]before[/] {document.text[:220]!r}")
        console.print(f"[green]after [/] {cleaned.text[:220]!r}")


@app.command()
def run(
    source: str,
    min_score: float = typer.Option(0.40, "--min-score", min=0.0, max=1.0),
    no_language_check: bool = typer.Option(False, "--no-language-check"),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
) -> None:
    """Clean one source into data/processed/."""
    stats = clean_source(
        source, filter_config=_filter_config(min_score, no_language_check), limit=limit
    )
    _report([stats])


@app.command("all")
def run_all(
    min_score: float = typer.Option(0.40, "--min-score", min=0.0, max=1.0),
    no_language_check: bool = typer.Option(False, "--no-language-check"),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
) -> None:
    """Clean every source with interim data."""
    config = _filter_config(min_score, no_language_check)
    results: list[CleanStats] = []
    for source in available_sources():
        console.print(f"[bold cyan]{source}[/]")
        try:
            results.append(clean_source(source, filter_config=config, limit=limit))
        except Exception as exc:
            console.print(f"[red]failed:[/] {type(exc).__name__}: {exc}")
            logger.exception("Cleaning failed", extra={"source": source})
    _report(results)


def _report(results: list[CleanStats]) -> None:
    table = Table(title="Cleaning results")
    for column in ("source", "read", "kept", "keep %", "rejected", "low score", "translit", "M chars"):
        table.add_column(column, justify="right")

    for stats in sorted(results, key=lambda s: -s.read):
        style = "green" if stats.keep_rate >= 0.8 else "yellow" if stats.keep_rate >= 0.5 else "red"
        table.add_row(
            stats.source_id,
            f"{stats.read:,}",
            f"{stats.kept:,}",
            f"[{style}]{stats.keep_rate:.1%}[/]",
            f"{stats.rejected:,}",
            f"{stats.below_threshold:,}",
            f"{stats.transliterated:,}",
            f"{stats.chars_out / 1e6:.2f}",
        )
    console.print(table)

    read = sum(s.read for s in results)
    kept = sum(s.kept for s in results)
    chars = sum(s.chars_out for s in results)
    console.print(
        f"Total: [green]{kept:,}[/] of {read:,} documents kept "
        f"({kept / read if read else 0:.1%}), {chars / 1e6:.1f}M characters, "
        f"~{round(chars / 3.1):,} estimated tokens"
    )

    flags: dict[str, int] = {}
    for stats in results:
        for flag, count in stats.flags.items():
            flags[flag] = flags.get(flag, 0) + count
    if flags:
        console.print("\n[bold]Why documents lost points[/]")
        for flag, count in sorted(flags.items(), key=lambda kv: -kv[1]):
            console.print(f"  {flag:<24}{count:>10,}")
