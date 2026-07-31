"""The crawl orchestrator.

Ties the frontier, fetcher and blob store together into a loop that is:

* **resumable** - all state is in SQLite, so Ctrl-C then re-run continues
* **bounded** - by page count and link depth, never "until the internet ends"
* **scoped** - only URLs the registry declared in-scope are ever queued
* **observable** - a running Karakalpak-content ratio is logged, so an operator
  notices within seconds if a crawl has wandered into the Russian locale

Requests within one source are issued sequentially. That is not an oversight:
the per-host delay serialises them anyway, and pretending otherwise would only
add complexity. Throughput comes from crawling several *sources* concurrently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common.logging import get_logger
from ...preprocessing.script import analyse
from .fetcher import Fetcher, FetchOutcome
from .registry import SourceSpec
from .state import CrawlState
from .storage import RawStore
from .urls import extract_links, in_scope, normalise_url, parse_sitemap

logger = get_logger(__name__)

_SITEMAP_CHILD_LIMIT = 50
_BATCH_SIZE = 25
_PROGRESS_EVERY = 25


@dataclass(slots=True)
class CrawlStats:
    """Outcome of one `Crawler.run()` call."""

    source_id: str
    seeded: int = 0
    fetched: int = 0
    failed: int = 0
    skipped: int = 0
    duplicates: int = 0
    discovered: int = 0
    bytes_stored: int = 0
    kaa_pages: int = 0
    scored_pages: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def kaa_ratio(self) -> float:
        """Share of fetched HTML pages that look like Karakalpak."""
        return self.kaa_pages / self.scored_pages if self.scored_pages else 0.0

    def summary(self) -> str:
        return (
            f"{self.source_id}: fetched={self.fetched} failed={self.failed} "
            f"skipped={self.skipped} dupes={self.duplicates} "
            f"new_urls={self.discovered} kaa={self.kaa_ratio:.0%} "
            f"stored={self.bytes_stored / 1_048_576:.1f}MB"
        )


class Crawler:
    """Crawls one registered source into `data/raw/<source_id>/`."""

    def __init__(
        self,
        spec: SourceSpec,
        fetcher: Fetcher,
        state: CrawlState,
        store: RawStore | None = None,
        *,
        max_depth: int = 3,
    ) -> None:
        self._spec = spec
        self._fetcher = fetcher
        self._state = state
        self._store = store or RawStore(spec.id)
        self._max_depth = max_depth
        self._fetcher.set_host_delay(str(spec.url), spec.delay_seconds)

    async def seed(self) -> int:
        """Populate the frontier from robots.txt sitemaps, then the landing page.

        Sitemaps are strongly preferred: they are cheap, complete, and mean we
        do not have to walk a site's navigation to find its articles.
        """
        seeds: list[str] = []
        start_url = str(self._spec.url)

        if self._spec.use_sitemap:
            for sitemap_url in await self._fetcher.sitemaps_for(start_url):
                seeds.extend(await self._expand_sitemap(sitemap_url))
            if not seeds:
                logger.info(
                    "No usable sitemap, falling back to link crawl",
                    extra={"source": self._spec.id},
                )
        else:
            logger.info("Sitemap disabled for this source", extra={"source": self._spec.id})

        normalised_start = normalise_url(start_url)
        if normalised_start:
            seeds.append(normalised_start)

        in_scope_seeds = [u for u in dict.fromkeys(seeds) if in_scope(u, self._spec)]
        added = self._state.add_urls(self._spec.id, in_scope_seeds)
        logger.info(
            "Seeded frontier",
            extra={"source": self._spec.id, "candidates": len(in_scope_seeds), "new": added},
        )
        return added

    async def _expand_sitemap(self, sitemap_url: str, depth: int = 0) -> list[str]:
        """Return page URLs from a sitemap, recursing once into sitemap indexes."""
        result = await self._fetcher.fetch(sitemap_url)
        if not result.ok:
            # Broken sitemap children are common (kknews.uz returns HTTP 500 on
            # two of twelve). Log and continue; never abort the crawl for this.
            logger.info(
                "Sitemap unavailable",
                extra={"url": sitemap_url, "error": result.error},
            )
            return []

        locations, is_index = parse_sitemap(result.text())
        if not is_index or depth >= 1:
            return [u for u in (normalise_url(loc) for loc in locations) if u]

        children: list[str] = []
        for child in locations[:_SITEMAP_CHILD_LIMIT]:
            children.extend(await self._expand_sitemap(child, depth + 1))
        return children

    async def run(self, max_pages: int | None = None) -> CrawlStats:
        """Drain the frontier until it is empty or `max_pages` have been fetched."""
        limit = max_pages if max_pages is not None else self._spec.max_pages
        stats = CrawlStats(source_id=self._spec.id)

        while limit is None or stats.fetched < limit:
            remaining = None if limit is None else limit - stats.fetched
            batch_size = _BATCH_SIZE if remaining is None else min(_BATCH_SIZE, remaining)
            batch = self._state.next_pending(self._spec.id, limit=batch_size)
            if not batch:
                break

            for record in batch:
                await self._process(record.url, record.url_hash, record.depth, stats)

                total = stats.fetched + stats.failed + stats.skipped
                if total % _PROGRESS_EVERY == 0:
                    logger.info("Crawl progress", extra={"summary": stats.summary()})

        logger.info("Crawl finished", extra={"summary": stats.summary()})
        return stats

    async def _process(self, url: str, url_hash_value: str, depth: int, stats: CrawlStats) -> None:
        result = await self._fetcher.fetch(url)

        if result.outcome is FetchOutcome.ROBOTS_DENIED:
            self._state.mark_skipped(url_hash_value, reason="robots.txt denied")
            stats.skipped += 1
            return

        if not result.ok or result.content is None:
            error = result.error or "unknown error"
            self._state.mark_failed(url_hash_value, error=error, http_status=result.status_code)
            stats.failed += 1
            if len(stats.errors) < 20:
                stats.errors.append(f"{url}: {error}")
            return

        blob = self._store.store(result.content, result.content_type)
        self._state.mark_fetched(
            url_hash_value,
            http_status=result.status_code or 0,
            content_hash=blob.content_hash,
            content_path=blob.relative_path,
            content_type=result.content_type,
        )
        stats.fetched += 1
        if blob.was_new:
            stats.bytes_stored += blob.size_bytes
        else:
            stats.duplicates += 1

        if result.is_html:
            await self._follow_links(result.text(), result.final_url or url, depth, stats)

    async def _follow_links(self, html: str, base_url: str, depth: int, stats: CrawlStats) -> None:
        """Score the page for Karakalpak content and queue its in-scope links."""
        from selectolax.parser import HTMLParser

        tree = HTMLParser(html)
        for tag in tree.css("script, style, noscript"):
            tag.decompose()
        text = tree.body.text(separator=" ", strip=True) if tree.body else ""

        # Language is measured, not enforced: raw storage is Phase 2's job and
        # filtering is Phase 3's. But a crawl whose ratio collapses is a bug
        # (usually a wrong locale prefix) and must be visible immediately.
        if len(text) >= 200:
            stats.scored_pages += 1
            if analyse(text).is_probably_karakalpak:
                stats.kaa_pages += 1

        if depth >= self._max_depth:
            return

        candidates = [u for u in extract_links(html, base_url) if in_scope(u, self._spec)]
        stats.discovered += self._state.add_urls(self._spec.id, candidates, depth=depth + 1)
