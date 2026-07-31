"""`qm ingest` - pull bulk sources into data/interim/.

    qm ingest list                      what can be ingested, and what already was
    qm ingest inspect <source_id>       print raw rows, to verify a schema mapping
    qm ingest run     <source_id>       ingest into data/interim/ + write a manifest
    qm ingest all                       every enabled dump/hf source, in priority order
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..common.logging import get_logger
from ..crawlers.core.registry import AccessMethod, SourceRegistry, SourceSpec, load_registry
from .base import Ingester
from .manifest import Manifest, manifest_path, read_manifest

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Ingest bulk datasets and dumps.", no_args_is_help=True)

_BULK_ACCESS = {AccessMethod.DUMP, AccessMethod.HF}


def _resolve(registry: SourceRegistry, source_id: str) -> SourceSpec:
    try:
        return registry.by_id(source_id)
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unknown source '{source_id}'. Run `qm ingest list`."
        ) from exc


def build_ingester(spec: SourceSpec) -> Ingester:
    """Pick the ingester for a source, by access method."""
    if spec.access is AccessMethod.DUMP:
        from .wikipedia import WikipediaIngester

        return WikipediaIngester(spec)
    if spec.access is AccessMethod.HF:
        from .huggingface import HuggingFaceIngester

        return HuggingFaceIngester(spec)
    raise typer.BadParameter(
        f"'{spec.id}' has access='{spec.access.value}'. "
        "Only 'dump' and 'hf' sources are ingested; use `qm crawl` for the rest."
    )


@app.command("list")
def list_sources() -> None:
    """Show bulk sources and whether they have been ingested."""
    registry = load_registry()
    specs = [s for s in registry.enabled_sources() if s.access in _BULK_ACCESS]

    table = Table(title=f"Bulk sources ({len(specs)})")
    for column in ("id", "access", "licence", "est MB", "ingested", "docs", "~tokens"):
        table.add_column(column, overflow="fold")

    for spec in specs:
        done = manifest_path(spec.id).exists()
        manifest = read_manifest(spec.id) if done else None
        table.add_row(
            spec.id,
            spec.access.value,
            spec.license,
            f"{spec.est_size_mb:.0f}",
            "[green]yes[/]" if done else "[bright_black]no[/]",
            f"{manifest.documents:,}" if manifest else "-",
            f"{manifest.estimated_tokens:,}" if manifest else "-",
        )
    console.print(table)


@app.command()
def inspect(source: str, rows: int = typer.Option(3, "--rows", "-n", min=1, max=20)) -> None:
    """Print raw dataset rows so a schema mapping can be checked, not guessed."""
    registry = load_registry()
    spec = _resolve(registry, source)
    if spec.access is not AccessMethod.HF:
        raise typer.BadParameter("inspect only applies to Hugging Face sources.")

    from .huggingface import inspect_schema, require_mapping

    mapping = require_mapping(spec.id)
    console.print(f"[cyan]{mapping.repo}[/] config={mapping.config} splits={mapping.splits}")
    if mapping.notes:
        console.print(f"[bright_black]{mapping.notes}[/]")

    for index, row in enumerate(inspect_schema(spec.id, rows)):
        console.print(f"\n[bold]row {index}[/] columns={sorted(row)}")
        for key, value in row.items():
            preview = str(value)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            console.print(f"  [green]{key}[/]: {preview}")


@app.command()
def run(
    source: str,
    limit: int | None = typer.Option(None, "--limit", "-n", min=1, help="Stop after N documents."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count and measure without writing."),
) -> None:
    """Ingest one source into data/interim/."""
    registry = load_registry()
    spec = _resolve(registry, source)
    ingester = build_ingester(spec)

    console.print(f"Ingesting [cyan]{spec.id}[/]{' (dry run)' if dry_run else ''}")
    manifest = ingester.run(limit=limit, dry_run=dry_run)
    _report(manifest)


@app.command("all")
def run_all(
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
    skip_existing: bool = typer.Option(True, "--skip-existing/--redo"),
) -> None:
    """Ingest every enabled bulk source, in registry priority order."""
    registry = load_registry()
    specs = [s for s in registry.enabled_sources() if s.access in _BULK_ACCESS]

    results: list[Manifest] = []
    for spec in specs:
        if skip_existing and manifest_path(spec.id).exists():
            console.print(f"[bright_black]skip {spec.id} (already ingested)[/]")
            continue
        console.print(f"\n[bold cyan]{spec.id}[/]")
        try:
            results.append(build_ingester(spec).run(limit=limit))
        except Exception as exc:
            # One unavailable dataset must not abort the whole batch.
            console.print(f"[red]failed:[/] {type(exc).__name__}: {exc}")
            logger.exception("Ingest failed", extra={"source": spec.id})

    console.print("\n[bold]Summary[/]")
    for manifest in results:
        _report(manifest)

    total = sum(m.estimated_tokens for m in results)
    console.print(f"\nTotal this run: [green]~{total:,}[/] estimated tokens")


def _report(manifest: Manifest) -> None:
    console.print(f"  {manifest.summary()}")
    if manifest.sha256:
        console.print(f"  [bright_black]sha256 {manifest.sha256[:16]}...  {manifest.path}[/]")
