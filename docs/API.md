# API

OpenAI-compatible chat completions for the Karakalpak model.

```bash
qm serve                                            # echo backend, no model
qm serve --config configs/serve/local_model.yaml    # Qwen3-0.6B, real output
qm serve --config configs/serve/production.yaml     # vLLM, auth, limits
```

---

## 1. Why it is OpenAI-compatible

Every client library, chat UI framework and evaluation harness already targets
that wire format. Adopting it means the Next.js frontend, `curl`, the OpenAI
Python SDK and any LLM CLI work against this server with no adapter:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")
response = client.chat.completions.create(
    model="qaraqalpaqmind",
    messages=[{"role": "user", "content": "Qaraqalpaqstannıń paytaxtı qaysı qala?"}],
    stream=True,
)
```

Only the parts actually served are modelled, and **unsupported parameters are
rejected rather than ignored**. A client sending `logit_bias` and having it
silently dropped gets wrong results with no way to notice; it gets a 422 instead.

## 2. The swappable backend

This is what lets the API be built and tested today, months before the
fine-tuned weights exist. Nothing above `ModelBackend` knows which engine is
behind it.

| Backend | Needs | Use |
|---|---|---|
| `echo` | nothing | Tests, smoke checks. Canned Karakalpak reply |
| `transformers` | a few hundred MB | Development. `Qwen3-0.6B` runs on CPU |
| `vllm` | GPU + running vLLM server | Production |

Switching is a config change:

```yaml
backend:
  kind: vllm
  model: qaraqalpaqmind          # what vLLM was started with
  vllm_url: http://vllm:8001/v1
```

**vLLM runs as a separate process**, not in this one. The gateway restarts in a
second; a process holding 16 GB of weights takes a minute. Coupling them would
make every auth or rate-limit change cost a model reload, and a crash in either
would take down both.

**Streaming is the primary interface.** `generate()` is defined in terms of
`stream()`, so the streaming path is exercised by every request rather than only
by streaming clients — the usual way a `/chat/completions` ships working while
its streaming counterpart was never really tested.

## 3. Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/v1/chat/completions` | yes | `stream: true` for SSE |
| GET | `/v1/models` | yes | |
| GET | `/healthz` | no | Liveness. **Never touches the backend** |
| GET | `/readyz` | no | Readiness. 503 while a model loads |
| GET | `/metrics` | no | Prometheus |
| GET | `/docs` | no | OpenAPI UI |

Liveness and readiness are separate deliberately. An orchestrator that conflates
them restarts a pod whose model is merely still loading — and since loading takes
a minute, that produces a restart loop which never converges.

### Karakalpak-specific extension

```json
{ "normalize_orthography": true }
```

Normalises the response to Latin 2016 before returning it. The model is trained
to produce it, but a low-temperature slip into the 2009 apostrophe convention is
cheap to fix deterministically. Off by default, so a standard OpenAI client is
unaffected.

Not applied while streaming: normalisation needs whole words, and applying it to
a fragment split mid-word would corrupt it.

## 4. Security

**API keys come from the environment, never from the config file**, so a config
can be committed and a key cannot be committed with it:

```bash
export QM_API_KEYS="key-one,key-two"
```

| Property | How |
|---|---|
| Constant-time comparison | `secrets.compare_digest`; `==` leaks the prefix through timing |
| Keys never logged | Requests attributed by a 12-char SHA-256 fingerprint |
| No keys ⇒ no access | Auth enabled with zero keys **refuses every request** |

That last one matters: failing open would leave a GPU exposed to the internet.
It returns 503 with a message naming the missing environment variable.

### Two separate limits

**Rate limit** — sliding window, per key. A fixed window lets a client send the
full allowance at 59.9 s and again at 60.1 s, producing double the intended rate
at the boundary.

**Concurrency cap** — simultaneous *in-flight generations* per key. This is the
one that protects the GPU: a caller opening twenty streaming requests occupies it
indefinitely while staying far under any per-minute limit. Request counting alone
cannot see that.

The guard is held for the whole generation including the streaming tail —
releasing it when the response object is returned would defeat it entirely, since
an SSE response returns instantly and then runs for minutes.

Both live in process memory. Correct for one instance; **with N replicas behind a
load balancer the effective limit is N times the configured one.** Move to Redis
before scaling out.

### Logging

Prompts and completions are **not logged by default**. They are the user's
private content and often contain personal data, so recording them is a decision
with a retention policy attached, not a default. Enable deliberately:

```yaml
observability:
  log_prompts: true
```

Tracebacks never reach a client — they can disclose paths, versions and prompt
content. The client gets a request id; the detail goes to the log.

## 5. Nginx

```nginx
location /v1/ {
    proxy_pass http://api:8000;
    proxy_buffering off;          # or SSE chunks are held until the buffer fills
    proxy_read_timeout 300s;      # a long generation is not a stalled connection
    proxy_set_header Connection '';
    proxy_http_version 1.1;
}
```

The server also sets `X-Accel-Buffering: no` per response, so streaming works
even if `proxy_buffering` is left on.

## 6. Testing

Every API test runs against the echo backend: no GPU, no download, no network,
milliseconds per test. 47 tests cover routing, auth, rate limiting, concurrency,
SSE framing, error envelopes and config validation.

The echo backend is not a mock — it implements the real `ModelBackend` contract
and streams word by word, so the path under test is the path production uses.

```bash
pytest tests/unit/test_api.py -q
```

## 7. Not done yet

- **Docker / Compose / Nginx configs** — next
- **`/v1/rag/chat`** — waits on Phase 9
- **Real model output** — the echo and small-model backends prove the plumbing;
  Karakalpak quality waits on training
