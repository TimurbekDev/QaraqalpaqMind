"""Tests for the API server.

Every one runs against the echo backend: no GPU, no download, no network. That
is the point of the swappable backend - the HTTP surface is fully testable
months before the fine-tuned weights exist, and the streaming path under test is
the same code production uses.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from qaraqalpaqmind.api.app import build_backend, create_app
from qaraqalpaqmind.api.backends.base import BackendError
from qaraqalpaqmind.api.backends.echo import EchoBackend
from qaraqalpaqmind.api.config import (
    AuthConfig,
    BackendConfig,
    BackendKind,
    RateLimitConfig,
    ServeConfig,
    ServerConfig,
)
from qaraqalpaqmind.api.middleware.auth import extract_key, key_fingerprint, verify
from qaraqalpaqmind.api.middleware.limits import ConcurrencyGuard, SlidingWindowLimiter
from qaraqalpaqmind.api.schemas.chat import ChatCompletionRequest, Message
from qaraqalpaqmind.common.config import load_config

CHAT_URL = "/v1/chat/completions"


def _config(**overrides: object) -> ServeConfig:
    base: dict[str, object] = {
        "backend": BackendConfig(kind=BackendKind.ECHO),
        "auth": AuthConfig(enabled=False),
        "rate_limit": RateLimitConfig(enabled=False),
        # Loopback, not 0.0.0.0: with auth off the config warns about binding
        # all interfaces, and that warning is correct - it should not be muted
        # by 23 test fixtures firing it.
        "server": ServerConfig(host="127.0.0.1"),
    }
    return ServeConfig(**(base | overrides))  # type: ignore[arg-type]


def test_open_binding_without_auth_warns() -> None:
    # The warning silenced above must still fire for a real misconfiguration.
    with pytest.warns(UserWarning, match="open to"):
        ServeConfig(auth=AuthConfig(enabled=False), server=ServerConfig(host="0.0.0.0"))


def _client(config: ServeConfig | None = None) -> TestClient:
    return TestClient(create_app(config or _config()))


def _body(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "model": "qaraqalpaqmind",
        "messages": [{"role": "user", "content": "Salawmatsız ba?"}],
    }
    return base | overrides


# --- health ---------------------------------------------------------------


def test_healthz_does_not_touch_the_backend() -> None:
    # Liveness must not fail while a model loads, or an orchestrator restarts
    # the pod forever and never converges.
    config = _config(backend=BackendConfig(kind=BackendKind.ECHO))
    app = create_app(config)
    with TestClient(app) as client:
        app.state.backend = EchoBackend(fail=True)
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503


def test_readyz_reports_the_backend() -> None:
    with _client() as client:
        payload = client.get("/readyz").json()
        assert payload["status"] == "ready"
        assert payload["backend"] == "EchoBackend"


def test_metrics_endpoint() -> None:
    with _client() as client:
        assert client.get("/metrics").status_code == 200


# --- models ---------------------------------------------------------------


def test_models_endpoint_lists_the_served_name() -> None:
    with _client() as client:
        payload = client.get("/v1/models").json()
        assert payload["object"] == "list"
        assert payload["data"][0]["id"] == "qaraqalpaqmind"


# --- non-streaming completions -------------------------------------------


def test_completion_returns_the_openai_shape() -> None:
    with _client() as client:
        payload = client.post(CHAT_URL, json=_body()).json()

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "qaraqalpaqmind"
    choice = payload["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] > 0


def test_response_is_in_karakalpak() -> None:
    with _client() as client:
        content = client.post(CHAT_URL, json=_body()).json()["choices"][0]["message"]["content"]
    assert "Salawmatsız" in content


def test_request_id_header_is_returned() -> None:
    with _client() as client:
        response = client.post(CHAT_URL, json=_body())
    assert response.headers["X-Request-ID"]


def test_supplied_request_id_is_echoed() -> None:
    with _client() as client:
        response = client.post(CHAT_URL, json=_body(), headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"


# --- validation -----------------------------------------------------------


def test_unknown_sampling_parameters_are_rejected() -> None:
    # Silently dropping an unsupported parameter gives wrong output with no
    # way for the caller to notice.
    with _client() as client:
        response = client.post(CHAT_URL, json=_body(logit_bias={"1": 2}))
    assert response.status_code == 422


def test_empty_messages_are_rejected() -> None:
    with _client() as client:
        assert client.post(CHAT_URL, json=_body(messages=[])).status_code == 422


def test_trailing_assistant_message_is_rejected() -> None:
    with _client() as client:
        response = client.post(
            CHAT_URL, json=_body(messages=[{"role": "assistant", "content": "Salawmatsız!"}])
        )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field", [{"temperature": 5.0}, {"top_p": 0.0}, {"max_tokens": 0}, {"max_tokens": 99999}]
)
def test_out_of_range_parameters_are_rejected(field: dict[str, object]) -> None:
    with _client() as client:
        assert client.post(CHAT_URL, json=_body(**field)).status_code == 422


# --- streaming ------------------------------------------------------------


def _parse_sse(text: str) -> list[dict[str, object]]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line.removeprefix("data:").strip()
            if payload != "[DONE]":
                events.append(json.loads(payload))
    return events


def test_streaming_emits_openai_chunks() -> None:
    with _client() as client:
        response = client.post(CHAT_URL, json=_body(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Nginx buffers proxied responses by default, which holds SSE chunks until
    # the buffer fills and makes streaming look broken.
    assert response.headers["X-Accel-Buffering"] == "no"

    events = _parse_sse(response.text)
    assert events
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert response.text.rstrip().endswith("data: [DONE]")


def test_streamed_chunks_reassemble_to_the_full_text() -> None:
    # A joiner bug that drops or duplicates spaces would hide here.
    with _client() as client:
        streamed = client.post(CHAT_URL, json=_body(stream=True))
        whole = client.post(CHAT_URL, json=_body()).json()["choices"][0]["message"]["content"]

    pieces = [
        (e["choices"][0]["delta"].get("content") or "") for e in _parse_sse(streamed.text)
    ]
    assert "".join(pieces) == whole


def test_all_chunks_share_one_completion_id() -> None:
    with _client() as client:
        events = _parse_sse(client.post(CHAT_URL, json=_body(stream=True)).text)
    assert len({e["id"] for e in events}) == 1


def test_backend_failure_mid_stream_is_reported_in_the_stream() -> None:
    # The response already carries status 200 by then, so the error has to
    # travel inside the stream; the client cannot be told 502 at that point.
    app = create_app(_config())
    with TestClient(app) as client:
        app.state.backend = EchoBackend(fail=True)
        response = client.post(CHAT_URL, json=_body(stream=True))

    assert response.status_code == 200
    assert "error" in response.text
    assert response.text.rstrip().endswith("data: [DONE]")


def test_backend_failure_without_streaming_maps_to_a_status_code() -> None:
    app = create_app(_config())
    with TestClient(app) as client:
        app.state.backend = EchoBackend(fail=True)
        response = client.post(CHAT_URL, json=_body())
    assert response.status_code == 503


# --- authentication -------------------------------------------------------


def test_requests_without_a_key_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QM_API_KEYS", "secret-one,secret-two")
    with _client(_config(auth=AuthConfig(enabled=True))) as client:
        response = client.post(CHAT_URL, json=_body())
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_a_valid_bearer_key_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QM_API_KEYS", "secret-one,secret-two")
    with _client(_config(auth=AuthConfig(enabled=True))) as client:
        response = client.post(
            CHAT_URL, json=_body(), headers={"Authorization": "Bearer secret-two"}
        )
    assert response.status_code == 200


def test_x_api_key_header_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QM_API_KEYS", "secret-one")
    with _client(_config(auth=AuthConfig(enabled=True))) as client:
        response = client.post(CHAT_URL, json=_body(), headers={"X-API-Key": "secret-one"})
    assert response.status_code == 200


def test_an_invalid_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QM_API_KEYS", "secret-one")
    with _client(_config(auth=AuthConfig(enabled=True))) as client:
        response = client.post(CHAT_URL, json=_body(), headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_auth_enabled_with_no_keys_refuses_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    # Failing open here would leave a GPU exposed to the internet.
    monkeypatch.delenv("QM_API_KEYS", raising=False)
    with _client(_config(auth=AuthConfig(enabled=True))) as client:
        response = client.post(CHAT_URL, json=_body(), headers={"X-API-Key": "anything"})
    assert response.status_code == 503
    assert "no API keys" in response.json()["error"]["message"]


def test_health_stays_public_under_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # An orchestrator cannot present a key, so liveness must not require one.
    monkeypatch.setenv("QM_API_KEYS", "secret")
    with _client(_config(auth=AuthConfig(enabled=True))) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_key_verification_is_constant_time_shaped() -> None:
    assert verify("abc", {"abc", "def"})
    assert not verify("abd", {"abc", "def"})
    assert not verify("abc", set())


def test_fingerprints_are_stable_and_not_the_key() -> None:
    fingerprint = key_fingerprint("super-secret")
    assert fingerprint == key_fingerprint("super-secret")
    assert "super-secret" not in fingerprint
    assert len(fingerprint) == 12


def test_key_extraction_handles_both_header_styles() -> None:
    from starlette.datastructures import Headers
    from starlette.requests import Request

    def make(headers: dict[str, str]) -> Request:
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": Headers(headers).raw,
        }
        return Request(scope)

    assert extract_key(make({"authorization": "Bearer abc"})) == "abc"
    assert extract_key(make({"x-api-key": "def"})) == "def"
    assert extract_key(make({})) is None
    assert extract_key(make({"authorization": "Basic xyz"})) is None


# --- limits ---------------------------------------------------------------


async def test_sliding_window_allows_then_blocks() -> None:
    limiter = SlidingWindowLimiter(requests_per_minute=3, burst=100)
    for _ in range(3):
        allowed, _ = await limiter.check("client")
        assert allowed
    allowed, retry_after = await limiter.check("client")
    assert not allowed
    assert retry_after > 0


async def test_limits_are_per_client() -> None:
    limiter = SlidingWindowLimiter(requests_per_minute=1, burst=100)
    assert (await limiter.check("a"))[0]
    assert (await limiter.check("b"))[0]
    assert not (await limiter.check("a"))[0]


async def test_burst_is_capped_separately() -> None:
    limiter = SlidingWindowLimiter(requests_per_minute=1000, burst=2)
    assert (await limiter.check("c"))[0]
    assert (await limiter.check("c"))[0]
    allowed, _ = await limiter.check("c")
    assert not allowed


async def test_concurrency_guard_limits_in_flight_requests() -> None:
    # The protection that actually matters for an LLM: one caller opening many
    # streaming requests occupies the GPU while staying under any rate limit.
    guard = ConcurrencyGuard(max_concurrent=2)
    assert await guard.acquire("x")
    assert await guard.acquire("x")
    assert not await guard.acquire("x")

    await guard.release("x")
    assert await guard.acquire("x")


async def test_concurrency_guard_is_per_client() -> None:
    guard = ConcurrencyGuard(max_concurrent=1)
    assert await guard.acquire("a")
    assert await guard.acquire("b")
    assert not await guard.acquire("a")


def test_rate_limited_requests_return_429_with_retry_after() -> None:
    config = _config(rate_limit=RateLimitConfig(enabled=True, requests_per_minute=2, burst=100))
    with _client(config) as client:
        statuses = [client.post(CHAT_URL, json=_body()).status_code for _ in range(4)]

    assert statuses[:2] == [200, 200]
    assert 429 in statuses[2:]


def test_concurrency_limit_is_released_after_each_request() -> None:
    # A guard that leaked would wedge the server after N requests.
    config = _config(rate_limit=RateLimitConfig(enabled=False, max_concurrent_per_key=1))
    with _client(config) as client:
        for _ in range(5):
            assert client.post(CHAT_URL, json=_body()).status_code == 200


# --- backends -------------------------------------------------------------


def test_build_backend_selects_by_kind() -> None:
    from qaraqalpaqmind.api.backends.echo import EchoBackend as Echo

    assert isinstance(build_backend(_config()), Echo)


async def test_echo_backend_generate_matches_its_stream() -> None:
    backend = EchoBackend()
    request = ChatCompletionRequest(
        model="m", messages=[Message(role="user", content="Salawmatsız?")]
    )
    streamed = "".join([piece async for piece in backend.stream(request)])
    assert (await backend.generate(request)).text == streamed


async def test_echo_backend_can_echo_the_user_message() -> None:
    backend = EchoBackend(reply=None)
    request = ChatCompletionRequest(
        model="m", messages=[Message(role="user", content="Nókis")]
    )
    assert "Nókis" in (await backend.generate(request)).text


async def test_backend_error_carries_a_status_code() -> None:
    backend = EchoBackend(fail=True)
    request = ChatCompletionRequest(model="m", messages=[Message(role="user", content="hi")])
    with pytest.raises(BackendError) as caught:
        await backend.generate(request)
    assert caught.value.status_code == 503


# --- configuration --------------------------------------------------------


@pytest.mark.parametrize("name", ["dev.yaml", "local_model.yaml", "production.yaml"])
def test_shipped_serve_configs_are_valid(name: str) -> None:
    config = load_config(f"serve/{name}", ServeConfig)
    assert config.served_model_name == "qaraqalpaqmind"


def test_production_config_is_locked_down() -> None:
    config = load_config("serve/production.yaml", ServeConfig)
    assert config.auth.enabled
    assert config.rate_limit.enabled
    # Prompts are user content; logging them must be a deliberate decision.
    assert not config.observability.log_prompts
    assert not config.observability.log_completions
    assert "*" not in config.server.cors_origins


def test_dev_config_needs_no_model() -> None:
    config = load_config("serve/dev.yaml", ServeConfig)
    assert config.backend.kind is BackendKind.ECHO
    assert config.server.host == "127.0.0.1"


def test_keys_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keys must never live in a config file that gets committed.
    monkeypatch.setenv("QM_API_KEYS", " one , two ,, three ")
    assert AuthConfig().load_keys() == {"one", "two", "three"}
