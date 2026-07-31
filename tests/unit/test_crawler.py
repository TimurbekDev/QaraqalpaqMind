"""End-to-end crawl of a fake multilingual site, entirely offline.

The fixture site mirrors the shape of the real Tier 2 sources: a Karakalpak
locale under /qq/, a Russian locale that must never be touched, a sitemap index
with one broken child, and a duplicate article served under two URLs.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from qaraqalpaqmind.crawlers.core.crawler import Crawler
from qaraqalpaqmind.crawlers.core.fetcher import Fetcher
from qaraqalpaqmind.crawlers.core.registry import SourceSpec
from qaraqalpaqmind.crawlers.core.state import CrawlState, UrlStatus
from qaraqalpaqmind.crawlers.core.storage import RawStore

KAA_BODY = (
    "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp ótti. "
    "Sessiyada mámleketlik hám jámiyetlik ómirdiń áhmiyetli máseleleri boyınsha "
    "qararlar qabıl etildi. Bul jıldıń juwmaqları boyınsha esabat tıńlandı."
)

ROBOTS = """User-agent: *
Disallow: /ru/
Sitemap: https://example.uz/sitemap.xml
"""

SITEMAP_INDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.uz/sm-posts.xml</loc></sitemap>
  <sitemap><loc>https://example.uz/sm-broken.xml</loc></sitemap>
</sitemapindex>"""

SITEMAP_POSTS = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.uz/qq/news/1</loc></url>
  <url><loc>https://example.uz/qq/news/2</loc></url>
  <url><loc>https://example.uz/ru/news/1</loc></url>
</urlset>"""


def _article(body: str, links: str = "") -> httpx.Response:
    return httpx.Response(
        200,
        text=f"<html><body><article>{body}</article>{links}</body></html>",
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _site(log: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if log is not None:
            log.append(path)

        match path:
            case "/robots.txt":
                return httpx.Response(200, text=ROBOTS, headers={"content-type": "text/plain"})
            case "/sitemap.xml":
                return httpx.Response(
                    200, text=SITEMAP_INDEX, headers={"content-type": "application/xml"}
                )
            case "/sm-posts.xml":
                return httpx.Response(
                    200, text=SITEMAP_POSTS, headers={"content-type": "application/xml"}
                )
            case "/sm-broken.xml":
                # Real sites do this: kknews.uz returns 500 on 2 of 12 children.
                return httpx.Response(500, text="boom")
            case "/qq/":
                return _article(
                    KAA_BODY,
                    links=(
                        '<a href="/qq/news/1">bir</a>'
                        '<a href="/qq/news/3">úsh</a>'
                        '<a href="/ru/news/1">rus</a>'
                        '<a href="/qq/logo.png">logo</a>'
                    ),
                )
            case "/qq/news/1" | "/qq/news/2":
                return _article(KAA_BODY)
            case "/qq/news/3":
                # Byte-identical to /qq/news/1: must be stored once.
                return _article(KAA_BODY)
            case "/ru/news/1":
                return _article("Состоялось очередное заседание Сената.")
            case _:
                return httpx.Response(404, text="nope")

    return httpx.MockTransport(handler)


def _spec(**overrides: object) -> SourceSpec:
    base: dict[str, object] = {
        "id": "news_example",
        "name": "Example",
        "kind": "news",
        "access": "crawl",
        "url": "https://example.uz/qq/",
        "legal": "government_work",
        "license": "unknown",
        "update_frequency": "daily",
        "est_size_mb": 1.0,
        "quality": 3,
        "allowed_paths": ["/qq/"],
        "denied_paths": ["/ru/"],
        "delay_seconds": 0.0,
    }
    return SourceSpec.model_validate(base | overrides)


def _build(tmp_path: Path, log: list[str] | None = None, **spec_overrides: object) -> tuple[
    Crawler, CrawlState, RawStore, Fetcher
]:
    spec = _spec(**spec_overrides)
    fetcher = Fetcher(user_agent="test-bot", default_delay=0.0, transport=_site(log))
    state = CrawlState(tmp_path / "crawl.db")
    store = RawStore(spec.id, root=tmp_path / "raw")
    return Crawler(spec, fetcher, state, store), state, store, fetcher


async def test_seed_uses_the_sitemap_and_drops_other_locales(tmp_path: Path) -> None:
    crawler, state, _, fetcher = _build(tmp_path)
    try:
        added = await crawler.seed()
        queued = {r.url for r in state.next_pending("news_example", limit=100)}
    finally:
        await fetcher.aclose()
        state.close()

    assert added == 3  # two sitemap articles plus the landing page
    assert queued == {
        "https://example.uz/qq/news/1",
        "https://example.uz/qq/news/2",
        "https://example.uz/qq/",
    }
    # The Russian article was in the sitemap and must have been filtered out.
    assert "https://example.uz/ru/news/1" not in queued


async def test_broken_sitemap_child_does_not_abort_the_crawl(tmp_path: Path) -> None:
    log: list[str] = []
    crawler, state, _, fetcher = _build(tmp_path, log)
    try:
        assert await crawler.seed() > 0
    finally:
        await fetcher.aclose()
        state.close()
    assert "/sm-broken.xml" in log


async def test_full_crawl_stores_pages_and_follows_links(tmp_path: Path) -> None:
    crawler, state, store, fetcher = _build(tmp_path)
    try:
        await crawler.seed()
        stats = await crawler.run()
        counts = state.stats("news_example")
    finally:
        await fetcher.aclose()
        state.close()

    # /qq/, news/1, news/2, plus news/3 discovered by following links.
    assert stats.fetched == 4
    assert stats.failed == 0
    assert stats.discovered == 1
    assert counts[UrlStatus.FETCHED.value] == 4
    assert store.total_bytes() > 0


async def test_duplicate_content_is_stored_once(tmp_path: Path) -> None:
    crawler, state, store, fetcher = _build(tmp_path)
    try:
        await crawler.seed()
        stats = await crawler.run()
    finally:
        await fetcher.aclose()
        state.close()

    # news/1, news/2 and news/3 are byte-identical; /qq/ differs.
    assert stats.duplicates == 2
    assert len(list(store.root.rglob("*.html"))) == 2


async def test_russian_locale_is_never_fetched(tmp_path: Path) -> None:
    log: list[str] = []
    crawler, state, _, fetcher = _build(tmp_path, log)
    try:
        await crawler.seed()
        await crawler.run()
    finally:
        await fetcher.aclose()
        state.close()

    assert not any(path.startswith("/ru/") for path in log), log


async def test_assets_are_never_fetched(tmp_path: Path) -> None:
    log: list[str] = []
    crawler, state, _, fetcher = _build(tmp_path, log)
    try:
        await crawler.seed()
        await crawler.run()
    finally:
        await fetcher.aclose()
        state.close()

    assert "/qq/logo.png" not in log


async def test_language_ratio_is_measured(tmp_path: Path) -> None:
    crawler, state, _, fetcher = _build(tmp_path)
    try:
        await crawler.seed()
        stats = await crawler.run()
    finally:
        await fetcher.aclose()
        state.close()

    # Every in-scope page is Karakalpak; a collapse here means a wrong locale.
    assert stats.scored_pages == 4
    assert stats.kaa_ratio == 1.0


async def test_crawl_resumes_without_refetching(tmp_path: Path) -> None:
    crawler, state, _, fetcher = _build(tmp_path)
    try:
        await crawler.seed()
        first = await crawler.run(max_pages=2)
    finally:
        await fetcher.aclose()
        state.close()

    log: list[str] = []
    crawler2, state2, _, fetcher2 = _build(tmp_path, log)
    try:
        second = await crawler2.run()
        counts = state2.stats("news_example")
    finally:
        await fetcher2.aclose()
        state2.close()

    assert first.fetched == 2
    assert second.fetched >= 1
    assert counts.get(UrlStatus.PENDING.value, 0) == 0
    # Nothing fetched in the first run may be requested again.
    assert len(log) == len(set(log)), log


async def test_max_pages_is_respected(tmp_path: Path) -> None:
    crawler, state, _, fetcher = _build(tmp_path)
    try:
        await crawler.seed()
        stats = await crawler.run(max_pages=1)
    finally:
        await fetcher.aclose()
        state.close()
    assert stats.fetched == 1


async def test_depth_limit_stops_link_following(tmp_path: Path) -> None:
    spec = _spec()
    fetcher = Fetcher(user_agent="test-bot", default_delay=0.0, transport=_site())
    state = CrawlState(tmp_path / "crawl.db")
    store = RawStore(spec.id, root=tmp_path / "raw")
    crawler = Crawler(spec, fetcher, state, store, max_depth=0)
    try:
        await crawler.seed()
        stats = await crawler.run()
    finally:
        await fetcher.aclose()
        state.close()

    # news/3 is only reachable by following a link, so it must not appear.
    assert stats.discovered == 0
    assert stats.fetched == 3
