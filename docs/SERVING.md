# Serving the model

Phase 10b: the container stack that runs the API in production.

```
client ──HTTPS──> nginx ──HTTP──> gateway ──HTTP──> vLLM ──> GPU
                   :80/:443        :8000            :8001
                   TLS, edge       auth, limits,    weights,
                   rate limit      SSE, schemas     batching
```

Three containers because they fail and restart on different timescales. The
gateway restarts in a second; vLLM holds 16 GB of weights and takes minutes. If
they were one image, every rate-limit tweak would cost a model reload.

## 1. Quick start

```bash
cd deployment
cp .env.example .env          # then put a real key in it
docker compose up -d          # gateway + nginx, no GPU needed
curl localhost/healthz
```

That runs the **echo backend** by default via `QM_SERVE_CONFIG` — canned replies,
no model, no GPU. It is the right thing to run first: it proves nginx, auth,
rate limiting and SSE all work before a GPU is involved.

With a merged checkpoint in `models/merged/qwen3-8b-kaa`:

```bash
docker compose --profile gpu up -d
```

`vllm` sits behind a profile so `docker compose up` still works on a machine
with no NVIDIA runtime. Without the profile, compose would fail on a service you
did not ask for.

## 2. Generating a key

```bash
python -c "import secrets; print('qm-' + secrets.token_urlsafe(32))"
```

Put it in `deployment/.env` as `QM_API_KEYS` (comma-separated for several).

Compose **refuses to start** if the variable is unset — the `:?` in the compose
file. That is deliberate: a gateway started with no keys and auth enabled
refuses every request, which is a confusing way to discover a missing `.env`.

Keys never appear in the image or in any committed file. They arrive from the
environment at run time, because anything baked into an image is readable by
everyone who can pull it.

## 3. What each piece is doing

### nginx

| Setting | Why |
|---|---|
| `proxy_buffering off` | Without it a token stream is delivered in one lump when generation finishes. The response is *correct*, so this looks like broken frontend code. |
| `proxy_http_version 1.1` | HTTP/1.0 has no chunked encoding, so a response of unknown length cannot stream at all. |
| `proxy_read_timeout 600s` | The default is 60s. Long generations get cut mid-answer and arrive as a truncated reply. |
| `resolver 127.0.0.11` + variable upstream | nginx resolves a hostname once at startup and caches it forever. A restarted `api` container can come back on a new IP, and nginx then serves 502 while the gateway sits there healthy. |
| `location = /metrics { deny all; }` | Prometheus scrapes over the compose network. Public metrics leak request volumes and model identity. |

The edge rate limit (30 r/s per IP) and the gateway's limit (60/min per key) are
not redundant. nginx sees IP addresses and stops a flood before it reaches
Python; the gateway sees API keys and enforces fair use between callers. Neither
substitutes for the other.

### The gateway image

65 MB, no torch, no CUDA, no weights. It runs as uid 10001, not root.

Two environment variables in it are load-bearing:

- `QM_PROJECT_ROOT=/app` — `paths.py` finds the project root by looking for a
  directory containing **both** `pyproject.toml` and `configs/`. An installed
  package has only the second, so without the override `--config
  serve/production.yaml` resolves under `site-packages` and fails.
- `QM_LOG_JSON=true` — one JSON object per line, for log shipping.

The healthcheck curls `$(hostname)`, not `localhost`. This matters more than it
looks: `configs/serve/dev.yaml` binds `127.0.0.1`, which is correct on a laptop
and unreachable from any other container. A healthcheck against `localhost`
passes in exactly that case — the container reports healthy, compose starts
nginx against it, and every request is a 502. Compose also passes
`--host 0.0.0.0` explicitly so any serve config works in a container.

### Liveness vs readiness

- `/healthz` — is the process alive. Does **not** check the backend.
- `/readyz` — can it actually serve. 503 while vLLM is still loading.

Only `/healthz` drives the container healthcheck. A container that restarts
because its *dependency* is down turns one outage into a crash loop, and vLLM
legitimately takes minutes to load. Verified: with vLLM stopped, `/readyz`
returns 503 and the container stays `healthy`.

## 4. Swapping in the fine-tuned model

`configs/serve/production.yaml` is the only file that changes:

```yaml
backend:
  kind: vllm
  model: qaraqalpaqmind      # must match --served-model-name in compose
```

The path on disk (`models/merged/qwen3-8b-kaa`) and the name clients send
(`qaraqalpaqmind`) are deliberately different, so replacing a checkpoint does
not change the API contract.

```bash
docker compose --profile gpu up -d --force-recreate vllm
```

The gateway keeps running throughout; it will report `not_ready` until vLLM
finishes loading.

## 5. TLS

`nginx/conf.d/qaraqalpaqmind.conf` ships with the HTTPS block commented out.
nginx refuses to start if `ssl_certificate` points at a file that does not
exist, so an uncommented block would mean the stack cannot run at all until
certificates exist.

```bash
# with certbot on the host
certbot certonly --standalone -d qaraqalpaqmind.example
cp /etc/letsencrypt/live/qaraqalpaqmind.example/{fullchain,privkey}.pem \
   deployment/nginx/certs/
```

Then uncomment the block and make port 80 redirect. `deployment/nginx/certs/`
is gitignored.

## 6. Verified

Run against the real stack, not asserted:

| Check | Result |
|---|---|
| `docker compose config` with no `QM_API_KEYS` | fails with the reason |
| `--profile gpu` off / on | `api, nginx` / `vllm, api, nginx` |
| `nginx -t` | passes, and no longer needs `api` to resolve at boot |
| `/healthz` through nginx | 200 |
| `/metrics` through nginx | 403 |
| No key / wrong key / right key | 401 / 401 / 200 |
| Streaming through nginx | chunks ~40 ms apart, not buffered |
| `/readyz` with vLLM down | 503, container stays healthy |
| Image size | 65 MB |
| Build context | 449 MB of `data/` excluded by `.dockerignore` |

## 7. Not done yet

- No TLS certificate is issued; the HTTPS block is commented out.
- The rate limiter holds state in process memory, so it is correct for one
  replica and wrong for several behind a load balancer — with N replicas the
  effective limit is N times the configured one. Move it to Redis before
  scaling out.
- vLLM has no authentication of its own. It is reachable by anything on the
  compose network, and is not published to the host. Do not put it on a shared
  network.
