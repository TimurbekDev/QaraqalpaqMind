"""API-key authentication.

Three properties matter here and each is easy to get wrong:

1. **Constant-time comparison.** `==` on secrets leaks their prefix through
   timing. `secrets.compare_digest` does not.
2. **Keys never appear in logs.** Requests are attributed by a short hash of
   the key, so rate limits and request logs are traceable without a leaked
   secret ending up in a log aggregator.
3. **No keys means no access.** When auth is enabled and no keys are
   configured, every request is refused. Failing open here would silently
   expose a GPU to the internet.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from ...common.logging import get_logger
from ..config import AuthConfig
from ..schemas.chat import ErrorResponse, as_dict

logger = get_logger(__name__)

#: Request attribute holding the caller's identity for downstream middleware.
CLIENT_ID_ATTR = "qm_client_id"

ANONYMOUS = "anonymous"


def key_fingerprint(api_key: str) -> str:
    """Short, stable, non-reversible id for a key.

    Used in logs and rate-limit buckets so a caller can be traced without the
    secret itself being written anywhere.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def extract_key(request: Request) -> str | None:
    """Read the key from `Authorization: Bearer` or `X-API-Key`."""
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        candidate = header[7:].strip()
        if candidate:
            return candidate
    direct = request.headers.get("x-api-key", "").strip()
    return direct or None


def verify(api_key: str, valid_keys: set[str]) -> bool:
    """Constant-time membership test.

    Every candidate is compared even after a match, so the time taken does not
    reveal which key matched or how many were checked.
    """
    matched = False
    for known in valid_keys:
        if secrets.compare_digest(api_key, known):
            matched = True
    return matched


def build_auth_middleware(
    config: AuthConfig,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """Create the authentication middleware for this configuration."""
    public = tuple(config.public_paths)

    async def middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not config.enabled:
            setattr(request.state, CLIENT_ID_ATTR, ANONYMOUS)
            return await call_next(request)

        if request.url.path in public:
            setattr(request.state, CLIENT_ID_ATTR, ANONYMOUS)
            return await call_next(request)

        valid_keys = config.load_keys()
        if not valid_keys:
            # Refuse rather than serve. An open GPU endpoint is worse than a
            # broken one, and this is loud enough to be noticed immediately.
            logger.error(
                "Auth is enabled but no keys are configured; refusing every request",
                extra={"env_var": config.keys_env_var},
            )
            return JSONResponse(
                status_code=503,
                content=as_dict(
                    ErrorResponse.of(
                        f"server misconfigured: no API keys in ${config.keys_env_var}",
                        "server_error",
                        "no_keys_configured",
                    )
                ),
            )

        api_key = extract_key(request)
        if api_key is None:
            return _unauthorized("missing API key; send Authorization: Bearer <key>")

        if not verify(api_key, valid_keys):
            logger.warning(
                "Rejected an invalid API key",
                # The fingerprint, never the key.
                extra={"fingerprint": key_fingerprint(api_key), "path": request.url.path},
            )
            return _unauthorized("invalid API key")

        setattr(request.state, CLIENT_ID_ATTR, key_fingerprint(api_key))
        return await call_next(request)

    return middleware


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content=as_dict(ErrorResponse.of(message, "invalid_request_error", "invalid_api_key")),
        headers={"WWW-Authenticate": "Bearer"},
    )


def client_id(request: Request) -> str:
    """The caller's fingerprint, for rate limiting and logs."""
    return getattr(request.state, CLIENT_ID_ATTR, ANONYMOUS)
