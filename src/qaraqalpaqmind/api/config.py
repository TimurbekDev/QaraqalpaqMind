"""API server configuration."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from ..common.config import StrictModel


class BackendKind(StrEnum):
    """Which inference engine serves requests."""

    ECHO = "echo"  # no model; tests and smoke checks
    TRANSFORMERS = "transformers"  # in-process, small model, development
    VLLM = "vllm"  # separate vLLM server, production


class BackendConfig(StrictModel):
    kind: BackendKind = BackendKind.ECHO

    # For `transformers`: a local path or Hub id. For `vllm`: the name the vLLM
    # server was started with. This is the single value that changes when the
    # fine-tuned model replaces the development one.
    model: str = "Qwen/Qwen3-0.6B"

    vllm_url: str = "http://localhost:8001/v1"
    vllm_api_key: str | None = None
    request_timeout: float = Field(default=300.0, gt=0)

    device: str | None = None
    dtype: str = "auto"
    max_new_tokens: int = Field(default=512, ge=1, le=8192)


class AuthConfig(StrictModel):
    """API-key authentication.

    Keys are read from the environment, never from the config file, so a
    config can be committed and a key cannot be committed with it.
    """

    enabled: bool = True

    # Comma-separated keys in this variable. Absent means no key is valid,
    # which combined with `enabled` refuses every request rather than
    # accidentally serving an open endpoint.
    keys_env_var: str = "QM_API_KEYS"

    # Endpoints reachable without a key. Health must be, or the orchestrator
    # cannot check liveness; metrics is scraped by Prometheus inside the
    # network perimeter.
    public_paths: list[str] = Field(
        default_factory=lambda: ["/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"]
    )

    def load_keys(self) -> set[str]:
        raw = os.getenv(self.keys_env_var, "")
        return {key.strip() for key in raw.split(",") if key.strip()}


class RateLimitConfig(StrictModel):
    """Per-key request limits.

    A token-bucket over a fixed window, held in process memory. That is correct
    for a single instance and wrong for several behind a load balancer - with
    N replicas the effective limit is N times this. Note it before scaling out.
    """

    enabled: bool = True
    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=1, description="Requests allowed in an instant burst.")

    # A generation holding a worker for minutes is how a server becomes
    # unresponsive without any single request looking abusive.
    max_concurrent_per_key: int = Field(default=4, ge=1)


class ServerConfig(StrictModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)

    # Browsers enforce this; a wildcard with credentials is rejected by them
    # anyway, so the default is the local frontend only.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    max_request_bytes: int = Field(default=1_000_000, ge=1024)
    request_timeout: float = Field(default=300.0, gt=0)


class ObservabilityConfig(StrictModel):
    metrics_enabled: bool = True
    # Prompt text can contain personal data and is often the user's private
    # content. Logging it by default would be a privacy decision made silently.
    log_prompts: bool = False
    log_completions: bool = False
    slow_request_seconds: float = Field(default=30.0, gt=0)


class ServeConfig(StrictModel):
    """A complete API server configuration."""

    backend: BackendConfig = Field(default_factory=BackendConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    served_model_name: str = Field(
        default="qaraqalpaqmind",
        description="The id clients send and see, independent of the checkpoint behind it.",
    )

    @model_validator(mode="after")
    def _warn_about_open_endpoints(self) -> ServeConfig:
        if not self.auth.enabled and "0.0.0.0" in self.server.host:
            # Not an error: this is exactly what you want inside a private
            # network or a compose stack. But it must be a deliberate choice.
            import warnings

            warnings.warn(
                "auth is disabled and the server binds 0.0.0.0, so the API is open to "
                "anything that can reach the host. Intended only behind a trusted "
                "network boundary.",
                stacklevel=2,
            )
        return self


def default_config_path() -> Path:
    from ..common.paths import CONFIGS_DIR

    return CONFIGS_DIR / "serve" / "dev.yaml"
