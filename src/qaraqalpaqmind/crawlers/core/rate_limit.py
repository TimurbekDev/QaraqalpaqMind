"""Per-host politeness throttling.

One shared limiter governs the whole crawl. Concurrency is global (we may want
eight requests in flight), but the delay is *per host*, so a fast site never
gets hammered because a slow one is holding the crawler up.

The contract is deliberately strict: a host is served by one request at a time,
and consecutive requests to it are separated by at least `delay`. That is
slower than a token bucket, and intentionally so - we are a guest on servers
run by public institutions with modest hardware.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

from ...common.logging import get_logger

logger = get_logger(__name__)


class HostRateLimiter:
    """Serialise requests per host and enforce a minimum gap between them.

    Example:
        limiter = HostRateLimiter(default_delay=2.0)
        async with limiter.acquire("kknews.uz"):
            ...  # exactly one request to this host at a time
    """

    def __init__(self, default_delay: float = 2.0) -> None:
        self._default_delay = default_delay
        self._delays: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._guard = asyncio.Lock()

    def set_delay(self, host: str, delay: float) -> None:
        """Override the delay for one host, e.g. from a robots.txt Crawl-delay."""
        previous = self._delays.get(host)
        if previous is None or delay > previous:
            self._delays[host] = delay
            logger.debug("Rate limit set", extra={"host": host, "delay": delay})

    def delay_for(self, host: str) -> float:
        return self._delays.get(host, self._default_delay)

    async def _lock_for(self, host: str) -> asyncio.Lock:
        async with self._guard:
            if host not in self._locks:
                self._locks[host] = asyncio.Lock()
            return self._locks[host]

    def acquire(self, host: str) -> _HostSlot:
        """Return an async context manager holding this host's request slot."""
        return _HostSlot(self, host)


class _HostSlot:
    """Async context manager: hold a host's lock and wait out its delay."""

    __slots__ = ("_host", "_limiter", "_lock")

    def __init__(self, limiter: HostRateLimiter, host: str) -> None:
        self._limiter = limiter
        self._host = host
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        self._lock = await self._limiter._lock_for(self._host)
        await self._lock.acquire()

        delay = self._limiter.delay_for(self._host)
        last = self._limiter._last_request.get(self._host)
        if last is not None:
            waited = time.monotonic() - last
            if waited < delay:
                await asyncio.sleep(delay - waited)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Stamp on exit, not entry: the gap is measured between the END of one
        # request and the START of the next, so a slow server is never also
        # punished with a shorter effective delay.
        self._limiter._last_request[self._host] = time.monotonic()
        if self._lock is not None:
            self._lock.release()
