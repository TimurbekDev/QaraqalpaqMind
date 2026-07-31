"""`qm dedup` - collapse duplicates into a training-ready dataset.

    qm dedup blocklist        build the benchmark-contamination blocklist
    qm dedup run              deduplicate data/processed/ -> data/datasets/pretrain/
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ..common.logging import get_logger
from ..common.paths import INTERIM_DIR
from ..ingest.manifest import Manifest, write_manifest
from .blocklist import blocklist_path, build_blocklist, save_blocklist
from .minhash import MinHashConfig
from .pipeline import DedupStats, deduplicate, output_path

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Deduplicate the corpus.", no_args_is_help=True)

# Sources whose sentences must never appear in training data.
HELD_OUT_SOURCES = ("flores_plus_kaa",)


@app.command()
def blocklist() -> None:
    """Build the contamination blocklist from held-out benchmark data."""
    paths = [INTERIM_DIR / f"{source}.jsonl.zst" for source in HELD_OUT_SOURCES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        console.print(
            "[yellow]Held-out data not ingested yet:[/] "
            + ", ".join(p.name for p in missing)
        )
        console.print(
            "FLORES+ is gated on the Hugging Face Hub. Accept the terms at\n"
            "  https://huggingface.co/datasets/openlanguagedata/flores_plus\n"
            "then set HF_TOKEN in .env and run `qm ingest run flores_plus_kaa`."
        )
        raise typer.Exit(code=1)

    hashes = build_blocklist(paths)
    save_blocklist(hashes)
    console.print(f"Blocked [green]{len(hashes):,}[/] benchmark sentences -> {blocklist_path()}")


@app.command()
def run(
    output: str = typer.Option("pretrain_v1", "--output", "-o"),
    threshold: float = typer.Option(0.80, "--threshold", min=0.1, max=1.0),
    num_perm: int = typer.Option(128, "--num-perm", min=16, max=512),
    skip_near: bool = typer.Option(False, "--skip-near", help="Exact duplicates only."),
) -> None:
    """Deduplicate the processed corpus into a training-ready dataset."""
    config = MinHashConfig(threshold=threshold, num_perm=num_perm)
    console.print(
        f"Deduplicating -> [cyan]{output}[/] "
        f"(jaccard >= {threshold}, {num_perm} permutations"
        f"{', exact only' if skip_near else ''})"
    )

    stats = deduplicate(output_name=output, minhash_config=config, skip_near=skip_near)
    _report(stats, output)


def _report(stats: DedupStats, output_name: str) -> None:
    table = Table(title="Deduplication")
    table.add_column("stage")
    table.add_column("documents", justify="right")
    table.add_row("read", f"{stats.read:,}")
    table.add_row("exact duplicates", f"-{stats.exact_duplicates:,}")
    table.add_row("near duplicates", f"-{stats.near_duplicates:,}")
    table.add_row("benchmark contamination", f"-{stats.contaminated:,}")
    table.add_row("[bold]kept[/]", f"[bold green]{stats.kept:,}[/]")
    console.print(table)

    if stats.removed_by_source:
        removal = Table(title="Removed by source")
        for column in ("source", "removed", "kept", "removal %"):
            removal.add_column(column, justify="right")
        sources = set(stats.removed_by_source) | set(stats.kept_by_source)
        for source in sorted(sources, key=lambda s: -stats.removed_by_source.get(s, 0)):
            removed = stats.removed_by_source.get(source, 0)
            kept = stats.kept_by_source.get(source, 0)
            total = removed + kept
            removal.add_row(
                source, f"{removed:,}", f"{kept:,}", f"{removed / total:.1%}" if total else "-"
            )
        console.print(removal)

    tokens = round(stats.chars_out / 3.1)
    console.print(
        f"\n[bold]{stats.kept:,} unique documents, {stats.chars_out / 1e6:.1f}M characters, "
        f"~{tokens:,} estimated tokens[/]"
    )

    target = output_path(output_name)
    if target.exists():
        manifest = Manifest.build(
            source_id=output_name,
            path=target,
            documents=stats.kept,
            characters=stats.chars_out,
            words=0,
            estimated_tokens=tokens,
            license="mixed - see per-source manifests",
            source_url="local://data/processed",
        )
        write_manifest(manifest)
        digest = manifest.sha256 or "unavailable"
        console.print(f"[bright_black]manifest sha256 {digest[:16]}...[/]")
