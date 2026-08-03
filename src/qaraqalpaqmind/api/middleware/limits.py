"""Rate limiting and concurrency control.

Two separate protections, because they stop different failures:

* **Rate limit** caps requests per minute. Stops a client hammering the endpoint.
* **Concurrency cap** limits simultaneous *in-flight generations* per key. This
  is the one that matters for an LLM: a single caller opening twenty streaming
  requests occupies the GPU indefinitely while staying well under any per-minute
  limit. Request counting alone does not see it.

State is in process memory. Correct for one instance; with N replicas behind a
load balancer the effective limit is N times the configured one. Move to Redis
before scaling out.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ...common.logging import get_logger
from ..config import RateLimitConfig
from ..schemas.chat import ErrorResponse, as_dict
from .auth import client_id

logger = get_logger(__name__)

_WINDOW_SECONDS = 60.0


class SlidingWindowLimiter:
    """Per-client request counting over a rolling minute.

    A sliding window rather than a fixed one: with fixed windows a client can
    send the full allowance at 59.9s and again at 60.1s, producing twice the
    intended rate at the boundary.
    """

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self._limit = requests_per_minute
        self._burst = burst
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, float]:
        """Record a request. Returns `(allowed, retry_after_seconds)`."""
        async with self._lock:
            now = time.monotonic()
            window = self._events[key]

            while window and now - window[0] > _WINDOW_SECONDS:
                window.popleft()

            if len(window) >= self._limit:
                return False, max(0.0, _WINDOW_SECONDS - (now - window[0]))

            # Burst check: too many requests in the last second.
            recent = sum(1 for stamp in window if now - stamp < 1.0)
            if recent >= self._burst:
                return False, 1.0

            window.append(now)
            return True, 0.0

    async def forget(self, key: str) -> None:
        async with self._lock:
            self._events.pop(key, None)


class ConcurrencyGuard:
    """Caps simultaneous in-flight generations per client."""

    def __init__(self, max_concurrent: int) -> None:
        self._max = max_concurrent
        self._active: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def acquire(self, key: str) -> bool:
        async with self._lock:
            if self._active[key] >= self._max:
                return False
            self._active[key] += 1
            return True

    async def release(self, key: str) -> None:
        async with self._lock:
            self._active[key] = max(0, self._active[key] - 1)
            if self._active[key] == 0:
                self._active.pop(key, None)

    def in_flight(self, key: str) -> int:
        return self._active.get(key, 0)


def build_rate_limit_middleware(
    config: RateLimitConfig,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Create rate-limiting middleware bound to its own state."""
    limiter = SlidingWindowLimiter(config.requests_per_minute, config.burst)

    async def middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not config.enabled or request.url.path in {"/healthz", "/readyz", "/metrics"}:
            return await call_next(request)

        key = client_id(request)
        allowed, retry_after = await limiter.check(key)
        if not allowed:
            logger.info(
                "Rate limit exceeded", extra={"client": key, "retry_after": round(retry_after, 1)}
            )
            return JSONResponse(
                status_code=429,
                content=as_dict(
                    ErrorResponse.of(
                        f"rate limit exceeded; retry in {retry_after:.0f}s",
                        "rate_limit_error",
                        "rate_limit_exceeded",
                    )
                ),
                headers={
                    "Retry-After": str(max(1, int(retry_after))),
                    "X-RateLimit-Limit": str(config.requests_per_minute),
                },
            )

        return await call_next(request)

    return middleware
