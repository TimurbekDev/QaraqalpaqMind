"""robots.txt fetching, caching and enforcement.

Policy decisions baked in here, so no source module can accidentally opt out:

* One robots.txt per host, fetched once per process and cached.
* A host whose robots.txt is *unreachable* is treated as ALLOWED. This matches
  RFC 9309: an unavailable robots.txt does not imply a blanket ban. A robots.txt
  that returns 4xx is likewise "no rules".
* A host whose robots.txt cannot be parsed is treated as DISALLOWED. If we
  cannot understand the rules, we do not guess.
* `Crawl-delay` always wins when it is *longer* than our configured delay.
  Sites are allowed to ask us to slow down, never to speed us up.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from ...common.logging import get_logger
from .rate_limit import HostRateLimiter

logger = get_logger(__name__)

_ROBOTS_TIMEOUT = 15.0


@dataclass(frozen=True, slots=True)
class RobotsRules:
    """Parsed robots.txt for one host."""

    host: str
    parser: RobotFileParser | None
    crawl_delay: float | None
    sitemaps: tuple[str, ...]
    fetch_failed: bool

    def allows(self, user_agent: str, url: str) -> bool:
        if self.parser is None:
            # No usable rules. Unreachable -> allowed; unparseable -> blocked.
            return self.fetch_failed
        return self.parser.can_fetch(user_agent, url)


class RobotsCache:
    """Fetches and caches robots.txt per host, and applies Crawl-delay."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        limiter: HostRateLimiter | None = None,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._limiter = limiter
        self._cache: dict[str, RobotsRules] = {}

    async def get(self, url: str) -> RobotsRules:
        """Return the rules for `url`'s host, fetching them on first use."""
        host = urlsplit(url).netloc
        cached = self._cache.get(host)
        if cached is not None:
            return cached

        rules = await self._fetch(url, host)
        self._cache[host] = rules

        if self._limiter is not None and rules.crawl_delay is not None:
            # set_delay only ever increases the delay.
            self._limiter.set_delay(host, rules.crawl_delay)

        return rules

    async def allows(self, url: str) -> bool:
        return (await self.get(url)).allows(self._user_agent, url)

    async def _fetch(self, url: str, host: str) -> RobotsRules:
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{host}/robots.txt"

        try:
            response = await self._client.get(robots_url, timeout=_ROBOTS_TIMEOUT)
        except httpx.HTTPError as exc:
            logger.info(
                "robots.txt unreachable, treating host as allowed",
                extra={"host": host, "error": str(exc)},
            )
            return RobotsRules(host, None, None, (), fetch_failed=True)

        if response.status_code >= 400 or not response.text.strip():
            logger.debug("No robots.txt rules", extra={"host": host, "status": response.status_code})
            return RobotsRules(host, None, None, (), fetch_failed=True)

        parser = RobotFileParser()
        try:
            parser.parse(response.text.splitlines())
        except Exception as exc:
            logger.warning(
                "robots.txt unparseable, blocking host to be safe",
                extra={"host": host, "error": str(exc)},
            )
            return RobotsRules(host, None, None, (), fetch_failed=False)

        raw_delay = parser.crawl_delay(self._user_agent)
        crawl_delay = float(raw_delay) if raw_delay is not None else None

        sitemaps = tuple(
            line.split(":", 1)[1].strip()
            for line in response.text.splitlines()
            if line.lower().startswith("sitemap:")
        )

        logger.info(
            "robots.txt loaded",
            extra={"host": host, "crawl_delay": crawl_delay, "sitemaps": len(sitemaps)},
        )
        return RobotsRules(host, parser, crawl_delay, sitemaps, fetch_failed=False)
