"""The single HTTP entry point for every crawler.

No source module opens its own connection. Everything goes through `Fetcher`,
which guarantees, for every request, that:

* robots.txt was consulted and obeyed
* the per-host delay was waited out
* transient failures were retried with exponential backoff, and permanent ones
  were not retried at all
* the response was size-capped, so one misconfigured server cannot exhaust RAM

Centralising this is what makes politeness auditable: there is exactly one place
where a request can be made, and it always applies the rules.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from urllib.parse import urlsplit

import httpx

from ...common.logging import get_logger
from .rate_limit import HostRateLimiter
from .robots import RobotsCache

logger = get_logger(__name__)

_MAX_CONTENT_BYTES = 16 * 1024 * 1024
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 522, 524})
_MAX_RETRY_SLEEP = 60.0


class FetchOutcome(StrEnum):
    OK = "ok"
    HTTP_ERROR = "http_error"
    TRANSPORT_ERROR = "transport_error"
    ROBOTS_DENIED = "robots_denied"
    TOO_LARGE = "too_large"


@dataclass(frozen=True, slots=True)
class FetchResult:
    """Outcome of one fetch attempt."""

    url: str
    outcome: FetchOutcome
    final_url: str | None = None
    status_code: int | None = None
    content: bytes | None = None
    content_type: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome is FetchOutcome.OK and self.content is not None

    @property
    def is_html(self) -> bool:
        mime = self.content_type.split(";", 1)[0].strip().lower()
        return mime in {"text/html", "application/xhtml+xml"}

    def text(self) -> str:
        """Decode the payload, tolerating the mislabelled encodings common on .uz hosts.

        UTF-8 is tried BEFORE the declared charset, which is deliberate. UTF-8
        is self-validating: arbitrary single-byte-encoded text almost never
        forms a valid UTF-8 sequence, so a successful strict decode is near
        proof. The reverse is not true - decoding UTF-8 bytes as cp1251
        *succeeds* and yields mojibake, because cp1251 maps nearly every byte.
        Trusting a wrong `charset=windows-1251` header would therefore silently
        corrupt the text rather than fail loudly.

        This matters specifically for Karakalpak: cp1251 has no code points for
        `Қ ә ң ө ү ғ ҳ ў`, so genuine Karakalpak Cyrillic *cannot* be cp1251.
        Any page carrying those letters is UTF-8, whatever its header claims.
        """
        if self.content is None:
            return ""
        for encoding in ("utf-8", self._declared_encoding(), "cp1251", "latin-1"):
            if not encoding:
                continue
            try:
                return self.content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.content.decode("utf-8", errors="replace")

    def _declared_encoding(self) -> str | None:
        for part in self.content_type.split(";")[1:]:
            key, _, value = part.strip().partition("=")
            if key.strip().lower() == "charset":
                return value.strip().strip("\"'") or None
        return None


class Fetcher:
    """Polite, retrying HTTP client shared by every source module."""

    def __init__(
        self,
        *,
        user_agent: str,
        default_delay: float = 2.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        respect_robots: bool = True,
        verify_tls: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._respect_robots = respect_robots
        self._limiter = HostRateLimiter(default_delay=default_delay)
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "kaa,uz;q=0.8,ru;q=0.6,*;q=0.4",
            },
            timeout=timeout,
            follow_redirects=True,
            verify=verify_tls,
            # Injected in tests so the whole stack can be exercised offline.
            transport=transport,
        )
        self._robots = RobotsCache(self._client, user_agent, self._limiter)

    @property
    def limiter(self) -> HostRateLimiter:
        return self._limiter

    async def __aenter__(self) -> Fetcher:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_host_delay(self, url: str, delay: float) -> None:
        """Raise the delay for a URL's host. Never lowers an existing one."""
        self._limiter.set_delay(urlsplit(url).netloc.lower(), delay)

    async def sitemaps_for(self, url: str) -> tuple[str, ...]:
        """Sitemap URLs declared in the host's robots.txt."""
        return (await self._robots.get(url)).sitemaps

    async def fetch(self, url: str) -> FetchResult:
        """Fetch one URL, applying robots, rate limiting and retries."""
        if self._respect_robots and not await self._robots.allows(url):
            logger.info("robots.txt denied", extra={"url": url})
            return FetchResult(url=url, outcome=FetchOutcome.ROBOTS_DENIED, error="robots.txt denied")

        host = urlsplit(url).netloc.lower()
        last_error = "unknown"
        last_status: int | None = None

        for attempt in range(1, self._max_retries + 1):
            async with self._limiter.acquire(host):
                try:
                    response = await self._client.get(url)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.debug(
                        "Transport error", extra={"url": url, "attempt": attempt, "error": last_error}
                    )
                else:
                    last_status = response.status_code
                    if response.status_code in _RETRY_STATUSES:
                        last_error = f"HTTP {response.status_code}"
                        await self._honour_retry_after(response, attempt)
                        continue
                    if response.status_code >= 400:
                        return FetchResult(
                            url=url,
                            outcome=FetchOutcome.HTTP_ERROR,
                            final_url=str(response.url),
                            status_code=response.status_code,
                            error=f"HTTP {response.status_code}",
                        )
                    return self._build_ok_result(url, response)

            if attempt < self._max_retries:
                await asyncio.sleep(self._backoff(attempt))

        logger.warning("Fetch failed", extra={"url": url, "error": last_error})
        return FetchResult(
            url=url,
            outcome=FetchOutcome.TRANSPORT_ERROR,
            status_code=last_status,
            error=last_error,
        )

    def _build_ok_result(self, url: str, response: httpx.Response) -> FetchResult:
        content = response.content
        if len(content) > _MAX_CONTENT_BYTES:
            logger.warning("Response too large", extra={"url": url, "bytes": len(content)})
            return FetchResult(
                url=url,
                outcome=FetchOutcome.TOO_LARGE,
                final_url=str(response.url),
                status_code=response.status_code,
                error=f"{len(content)} bytes exceeds cap",
            )
        return FetchResult(
            url=url,
            outcome=FetchOutcome.OK,
            final_url=str(response.url),
            status_code=response.status_code,
            content=content,
            content_type=response.headers.get("content-type", ""),
        )

    async def _honour_retry_after(self, response: httpx.Response, attempt: int) -> None:
        """Sleep as the server asked, or back off if it did not say."""
        header = response.headers.get("retry-after", "")
        try:
            requested = float(header)
        except ValueError:
            requested = 0.0
        await asyncio.sleep(min(max(requested, self._backoff(attempt)), _MAX_RETRY_SLEEP))

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so parallel workers do not resynchronise."""
        return min(2.0**attempt, _MAX_RETRY_SLEEP) * (0.5 + random.random() / 2)
