"""Health, readiness and metrics.

Liveness and readiness are separate on purpose. An orchestrator that conflates
them restarts a pod whose model is merely still loading - and since loading
takes a minute, that produces a restart loop that never converges.

    /healthz  the process is alive. Never touches the backend.
    /readyz   the backend can actually serve. Fails while a model loads.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request, Response

from ...common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

_STARTED_AT = time.monotonic()


@router.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness. Deliberately does not check the backend."""
    return {"status": "ok", "uptime_seconds": round(time.monotonic() - _STARTED_AT, 1)}


@router.get("/readyz")
async def readyz(request: Request, response: Response) -> dict[str, Any]:
    """Readiness. Returns 503 until the backend can serve."""
    backend = request.app.state.backend
    try:
        ready = await backend.health()
    except Exception as exc:
        logger.warning("Readiness check failed", extra={"error": str(exc)})
        ready = False

    if not ready:
        response.status_code = 503

    return {
        "status": "ready" if ready else "not_ready",
        "backend": type(backend).__name__,
        "model": getattr(backend, "name", "unknown"),
    }


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    """Prometheus exposition."""
    config = request.app.state.config
    if not config.observability.metrics_enabled:
        return Response(status_code=404)

    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except ImportError:
        return Response("prometheus_client is not installed\n", media_type="text/plain")

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
