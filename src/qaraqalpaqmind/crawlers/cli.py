"""`qm crawl` - inspect the registry and run crawls.

    qm crawl list                      what is registered, and why it is on or off
    qm crawl status                    frontier and fetch counts per source
    qm crawl seed   <source_id>        discover URLs without fetching any pages
    qm crawl run    <source_id>        fetch pages into data/raw/
    qm crawl retry  <source_id>        return failed URLs to the frontier
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from ..common.logging import get_logger
from ..common.paths import DATA_DIR, ensure_dir
from .core.crawler import Crawler, CrawlStats
from .core.fetcher import Fetcher
from .core.registry import AccessMethod, SourceRegistry, SourceSpec, load_registry
from .core.state import CrawlState
from .core.storage import RawStore

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Collect raw Karakalpak text from registered sources.", no_args_is_help=True)

# A single database for every source keeps `qm crawl status` a one-file question.
_STATE_DB = "crawl.db"

# Deliberately small. Crawling somebody's server is not a thing to do by accident:
# an explicit --max-pages is required to go beyond a sample.
_DEFAULT_MAX_PAGES = 25


def _state() -> CrawlState:
    return CrawlState(ensure_dir(DATA_DIR / "state") / _STATE_DB)


def _resolve(registry: SourceRegistry, source_id: str) -> SourceSpec:
    try:
        return registry.by_id(source_id)
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unknown source '{source_id}'. Run `qm crawl list` to see registered ids."
        ) from exc


def _require_crawlable(spec: SourceSpec) -> None:
    if spec.access is not AccessMethod.CRAWL:
        raise typer.BadParameter(
            f"'{spec.id}' has access='{spec.access.value}', not 'crawl'. "
            "Bulk datasets and dumps are ingested by `qm dataset`, not crawled."
        )
    if not spec.enabled:
        reason = spec.notes.strip().split("\n")[0] or "no reason recorded"
        raise typer.BadParameter(
            f"'{spec.id}' is disabled in the registry. Reason: {reason}\n"
            "Enable it in configs/crawl/sources.yaml only after resolving that."
        )


@app.command("list")
def list_sources(
    all_sources: bool = typer.Option(False, "--all", "-a", help="Include disabled sources."),
) -> None:
    """Show registered sources."""
    registry = load_registry()
    specs = registry.sources if all_sources else registry.enabled_sources()

    table = Table(title=f"Registered sources ({len(specs)})")
    for column in ("id", "kind", "access", "pri", "MB", "q", "scripts", "legal", "on"):
        table.add_column(column, overflow="fold")

    for spec in sorted(specs, key=lambda s: (not s.enabled, s.priority, s.id)):
        table.add_row(
            spec.id,
            spec.kind.value,
            spec.access.value,
            str(spec.priority),
            f"{spec.est_size_mb:.0f}",
            str(spec.quality),
            "+".join(spec.scripts),
            spec.legal.value,
            "[green]yes[/]" if spec.enabled else "[bright_black]no[/]",
        )
    console.print(table)
    console.print(f"Estimated enabled total: [cyan]{registry.total_estimated_mb()} MB[/]")


@app.command()
def status(source: str | None = typer.Argument(None, help="Limit to one source id.")) -> None:
    """Show crawl progress."""
    with _state() as state:
        if source is not None:
            counts = state.stats(source)
            console.print(f"[bold]{source}[/]: {counts or 'nothing crawled yet'}")
            return

        table = Table(title="Crawl status")
        for column in ("source", "pending", "fetched", "failed", "skipped"):
            table.add_column(column)

        registry = load_registry()
        any_rows = False
        for spec in registry.sources:
            counts = state.stats(spec.id)
            if not counts:
                continue
            any_rows = True
            table.add_row(
                spec.id,
                str(counts.get("pending", 0)),
                str(counts.get("fetched", 0)),
                str(counts.get("failed", 0)),
                str(counts.get("skipped", 0)),
            )

        console.print(table if any_rows else "No crawl has been run yet.")


@app.command()
def seed(source: str) -> None:
    """Discover URLs for a source without fetching any pages."""
    registry = load_registry()
    spec = _resolve(registry, source)
    _require_crawlable(spec)

    async def go() -> int:
        # CrawlState is a synchronous context manager on purpose (see state.py),
        # so it cannot share an `async with` clause with the fetcher.
        async with Fetcher(
            user_agent=registry.user_agent, default_delay=spec.delay_seconds
        ) as fetcher:
            with _state() as state:
                return await Crawler(spec, fetcher, state).seed()

    added = asyncio.run(go())
    console.print(f"Seeded [green]{added}[/] new URLs for [cyan]{spec.id}[/]")


@app.command()
def run(
    source: str,
    max_pages: int = typer.Option(_DEFAULT_MAX_PAGES, "--max-pages", "-n", min=1),
    max_depth: int = typer.Option(3, "--max-depth", min=0),
    unlimited: bool = typer.Option(False, "--unlimited", help="Ignore --max-pages. Be sure."),
) -> None:
    """Crawl a source into data/raw/."""
    registry = load_registry()
    spec = _resolve(registry, source)
    _require_crawlable(spec)

    limit = None if unlimited else max_pages
    console.print(
        f"Crawling [cyan]{spec.id}[/] at {spec.delay_seconds}s/request, "
        f"limit={'none' if limit is None else limit}, depth={max_depth}"
    )

    async def go() -> CrawlStats:
        async with Fetcher(
            user_agent=registry.user_agent, default_delay=spec.delay_seconds
        ) as fetcher:
            with _state() as state:
                crawler = Crawler(spec, fetcher, state, RawStore(spec.id), max_depth=max_depth)
                await crawler.seed()
                return await crawler.run(max_pages=limit)

    stats = asyncio.run(go())
    console.print(f"\n[bold]{stats.summary()}[/]")

    if stats.scored_pages and stats.kaa_ratio < 0.5:
        console.print(
            f"[yellow]Warning:[/] only {stats.kaa_ratio:.0%} of pages look Karakalpak. "
            "Check allowed_paths for this source - the crawl may be in the wrong locale."
        )
    for error in stats.errors[:5]:
        console.print(f"[bright_black]{error}[/]")


@app.command("all")
def run_all(
    max_pages: int = typer.Option(500, "--max-pages", "-n", min=1, help="Per source."),
    max_depth: int = typer.Option(4, "--max-depth", min=0),
    skip: list[str] = typer.Option([], "--skip", help="Source ids to leave out."),
    only: list[str] = typer.Option([], "--only", help="Restrict to these source ids."),
) -> None:
    """Crawl every enabled source, concurrently across hosts.

    All sources share ONE `Fetcher`, which matters: the rate limiter is keyed by
    host, so two sources on the same host (sud.uz has a Latin and a Cyrillic
    locale) are still served one request at a time. Give each source its own
    fetcher and that guarantee silently disappears.
    """
    registry = load_registry()
    specs = [s for s in registry.enabled_sources() if s.access is AccessMethod.CRAWL]
    if only:
        specs = [s for s in specs if s.id in set(only)]
    if skip:
        specs = [s for s in specs if s.id not in set(skip)]

    if not specs:
        console.print("[yellow]No matching enabled crawl sources.[/]")
        return

    console.print(
        f"Crawling [cyan]{len(specs)}[/] sources, up to {max_pages} pages each, depth {max_depth}"
    )
    for spec in specs:
        console.print(f"  [bright_black]{spec.id:<20} {spec.url}  {spec.delay_seconds}s[/]")

    async def go() -> list[CrawlStats]:
        async with Fetcher(user_agent=registry.user_agent, default_delay=2.0) as fetcher:
            with _state() as state:

                async def one(spec: SourceSpec) -> CrawlStats:
                    crawler = Crawler(
                        spec, fetcher, state, RawStore(spec.id), max_depth=max_depth
                    )
                    try:
                        await crawler.seed()
                        return await crawler.run(max_pages=max_pages)
                    except Exception as exc:
                        # One dead host must not abort the batch.
                        logger.exception("Crawl failed", extra={"source": spec.id})
                        stats = CrawlStats(source_id=spec.id)
                        stats.errors.append(f"{type(exc).__name__}: {exc}")
                        return stats

                return await asyncio.gather(*(one(spec) for spec in specs))

    results = asyncio.run(go())
    _summarise(results)


def _summarise(results: list[CrawlStats]) -> None:
    table = Table(title="Crawl results")
    for column in ("source", "fetched", "failed", "dupes", "new urls", "kaa %", "MB"):
        table.add_column(column, justify="right")

    for stats in sorted(results, key=lambda s: -s.fetched):
        ratio = f"{stats.kaa_ratio:.0%}" if stats.scored_pages else "-"
        style = "red" if stats.scored_pages and stats.kaa_ratio < 0.5 else "green"
        table.add_row(
            stats.source_id,
            str(stats.fetched),
            str(stats.failed),
            str(stats.duplicates),
            str(stats.discovered),
            f"[{style}]{ratio}[/]",
            f"{stats.bytes_stored / 1_048_576:.1f}",
        )
    console.print(table)

    total_mb = sum(s.bytes_stored for s in results) / 1_048_576
    console.print(f"Fetched [green]{sum(s.fetched for s in results)}[/] pages, {total_mb:.1f} MB raw")

    for stats in results:
        if stats.scored_pages and stats.kaa_ratio < 0.5:
            console.print(
                f"[yellow]{stats.source_id}: only {stats.kaa_ratio:.0%} Karakalpak - "
                "check allowed_paths, the crawl may be in the wrong locale.[/]"
            )
        for error in stats.errors[:3]:
            console.print(f"[bright_black]{stats.source_id}: {error}[/]")


@app.command()
def retry(
    source: str,
    max_attempts: int = typer.Option(3, "--max-attempts", min=1),
) -> None:
    """Return failed URLs to the frontier so the next run picks them up."""
    with _state() as state:
        requeued = state.retry_failed(source, max_attempts=max_attempts)
    console.print(f"Requeued [green]{requeued}[/] failed URLs for [cyan]{source}[/]")
