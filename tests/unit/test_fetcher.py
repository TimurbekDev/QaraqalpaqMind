"""Tests for the fetcher, robots enforcement and the rate limiter.

Everything runs against an in-process `httpx.MockTransport`, so the suite stays
offline, deterministic and fast.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from qaraqalpaqmind.crawlers.core.fetcher import Fetcher, FetchOutcome
from qaraqalpaqmind.crawlers.core.rate_limit import HostRateLimiter

UA = "QaraqalpaqMindBot/test"

ROBOTS = """User-agent: *
Disallow: /ru/
Crawl-delay: 5
Sitemap: https://example.uz/sitemap.xml
"""


def _fake_site(call_log: list[str] | None = None) -> httpx.MockTransport:
    attempts: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if call_log is not None:
            call_log.append(path)

        if path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS, headers={"content-type": "text/plain"})
        if path == "/qq/ok":
            return httpx.Response(
                200, text="<html><body>hám menen</body></html>",
                headers={"content-type": "text/html; charset=utf-8"},
            )
        if path == "/qq/flaky":
            attempts[path] = attempts.get(path, 0) + 1
            if attempts[path] < 3:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, text="<html>ok</html>", headers={"content-type": "text/html"})
        if path == "/qq/always-500":
            return httpx.Response(500, text="boom")
        if path == "/qq/missing":
            return httpx.Response(404, text="nope")
        if path == "/qq/huge":
            return httpx.Response(
                200, content=b"x" * (17 * 1024 * 1024), headers={"content-type": "text/html"}
            )
        if path == "/qq/cp1251":
            # Genuine cp1251 bytes. Note the text is RUSSIAN: cp1251 has no
            # code points for Karakalpak's Қ ә ң ө ү ғ ҳ ў.
            return httpx.Response(
                200,
                content="Узбекистан".encode("cp1251"),
                headers={"content-type": "text/html; charset=windows-1251"},
            )
        if path == "/qq/mislabelled":
            # UTF-8 Karakalpak served with a lying cp1251 header - the common
            # real-world case, and the one that silently corrupts a corpus.
            return httpx.Response(
                200,
                content="Қарақалпақстан Республикасы".encode(),
                headers={"content-type": "text/html; charset=windows-1251"},
            )
        if path == "/ru/blocked":
            return httpx.Response(200, text="<html>russian</html>")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _fetcher(**overrides: object) -> Fetcher:
    kwargs: dict[str, object] = {
        "user_agent": UA,
        "default_delay": 0.0,
        "transport": _fake_site(),
        "max_retries": 3,
    }
    return Fetcher(**(kwargs | overrides))  # type: ignore[arg-type]


async def test_successful_fetch() -> None:
    async with _fetcher() as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/ok")
        assert result.ok
        assert result.outcome is FetchOutcome.OK
        assert result.status_code == 200
        assert result.is_html
        assert "hám menen" in result.text()


async def test_robots_disallow_is_enforced() -> None:
    async with _fetcher() as fetcher:
        result = await fetcher.fetch("https://example.uz/ru/blocked")
        assert result.outcome is FetchOutcome.ROBOTS_DENIED
        assert result.content is None


async def test_robots_can_be_bypassed_only_explicitly() -> None:
    # Exists for offline fixtures and self-owned hosts; the registry never sets it.
    async with _fetcher(respect_robots=False) as fetcher:
        assert (await fetcher.fetch("https://example.uz/ru/blocked")).ok


async def test_robots_crawl_delay_raises_the_host_delay() -> None:
    async with _fetcher() as fetcher:
        await fetcher.fetch("https://example.uz/qq/ok")
        # robots.txt asks for 5s; our configured 0s must not win.
        assert fetcher.limiter.delay_for("example.uz") == 5.0


async def test_sitemaps_come_from_robots() -> None:
    async with _fetcher() as fetcher:
        assert await fetcher.sitemaps_for("https://example.uz/qq/ok") == (
            "https://example.uz/sitemap.xml",
        )


async def test_transient_failures_are_retried() -> None:
    async with _fetcher() as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/flaky")
        assert result.ok, result.error


async def test_client_errors_are_not_retried() -> None:
    log: list[str] = []
    async with _fetcher(transport=_fake_site(log)) as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/missing")
        assert result.outcome is FetchOutcome.HTTP_ERROR
        assert result.status_code == 404
        assert log.count("/qq/missing") == 1


async def test_persistent_server_errors_give_up() -> None:
    log: list[str] = []
    async with _fetcher(transport=_fake_site(log), max_retries=2) as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/always-500")
        assert result.outcome is FetchOutcome.TRANSPORT_ERROR
        assert log.count("/qq/always-500") == 2


async def test_oversized_responses_are_rejected() -> None:
    async with _fetcher() as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/huge")
        assert result.outcome is FetchOutcome.TOO_LARGE
        assert result.content is None


async def test_genuine_cp1251_is_decoded() -> None:
    # Older .uz pages still ship windows-1251 for Russian content.
    async with _fetcher() as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/cp1251")
        assert result.text() == "Узбекистан"


async def test_mislabelled_utf8_is_not_mangled() -> None:
    # A lying `charset=windows-1251` header must not win over valid UTF-8:
    # cp1251 maps nearly every byte, so trusting it yields silent mojibake
    # instead of an error. Karakalpak Cyrillic cannot be cp1251 at all.
    async with _fetcher() as fetcher:
        result = await fetcher.fetch("https://example.uz/qq/mislabelled")
        assert result.text() == "Қарақалпақстан Республикасы"


async def test_unreachable_host_is_reported_not_raised() -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("getaddrinfo failed", request=request)

    async with _fetcher(transport=httpx.MockTransport(explode), max_retries=1) as fetcher:
        result = await fetcher.fetch("https://dead.uz/qq/")
        assert result.outcome is FetchOutcome.TRANSPORT_ERROR
        assert "ConnectError" in (result.error or "")


# --- rate limiter ---------------------------------------------------------


async def test_rate_limiter_spaces_requests_to_one_host() -> None:
    limiter = HostRateLimiter(default_delay=0.05)
    start = time.monotonic()
    for _ in range(3):
        async with limiter.acquire("example.uz"):
            pass
    # First call is free; the two after it each wait one delay.
    assert time.monotonic() - start >= 0.10


async def test_rate_limiter_does_not_couple_different_hosts() -> None:
    limiter = HostRateLimiter(default_delay=0.2)

    async def hit(host: str) -> None:
        async with limiter.acquire(host):
            pass

    start = time.monotonic()
    await asyncio.gather(hit("a.uz"), hit("b.uz"), hit("c.uz"))
    assert time.monotonic() - start < 0.2


def test_rate_limiter_delay_only_increases() -> None:
    limiter = HostRateLimiter(default_delay=1.0)
    limiter.set_delay("a.uz", 5.0)
    limiter.set_delay("a.uz", 2.0)
    assert limiter.delay_for("a.uz") == 5.0


# --- robots.txt edge cases ------------------------------------------------


@pytest.mark.parametrize(
    ("robots_status", "robots_body"),
    [(404, "not found"), (200, ""), (500, "server error")],
)
async def test_absent_robots_means_allowed(robots_status: int, robots_body: str) -> None:
    # RFC 9309: an unavailable or empty robots.txt does not imply a blanket ban.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(robots_status, text=robots_body)
        return httpx.Response(200, text="<html>ok</html>", headers={"content-type": "text/html"})

    async with _fetcher(transport=httpx.MockTransport(handler)) as fetcher:
        assert (await fetcher.fetch("https://example.uz/qq/x")).ok


async def test_unreachable_robots_means_allowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            raise httpx.ConnectTimeout("timeout", request=request)
        return httpx.Response(200, text="<html>ok</html>", headers={"content-type": "text/html"})

    async with _fetcher(transport=httpx.MockTransport(handler)) as fetcher:
        assert (await fetcher.fetch("https://example.uz/qq/x")).ok


async def test_robots_is_fetched_once_per_host() -> None:
    log: list[str] = []
    async with _fetcher(transport=_fake_site(log)) as fetcher:
        for path in ("ok", "ok", "missing"):
            await fetcher.fetch(f"https://example.uz/qq/{path}")
    assert log.count("/robots.txt") == 1
