"""Guards on the serving stack's configuration (Phase 10b).

These parse the deployment files rather than running Docker, so they are fast
and run everywhere. Every assertion corresponds to something that was actually
wrong while building the stack, or to a property whose loss is silent: a
published port, a buffered stream, a container that reports healthy while being
unreachable.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qaraqalpaqmind.common.paths import PROJECT_ROOT

DEPLOYMENT = PROJECT_ROOT / "deployment"
COMPOSE = DEPLOYMENT / "docker-compose.yml"
DOCKERFILE = DEPLOYMENT / "Dockerfile.api"
SITE_CONF = DEPLOYMENT / "nginx" / "conf.d" / "qaraqalpaqmind.conf"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _healthcheck_line() -> str:
    """The CMD line of the HEALTHCHECK, not the comment block explaining it.

    The comment above it discusses curl, healthchecks and localhost, so a
    naive substring search finds the prose first and the test passes or fails
    on documentation rather than on the directive.
    """
    return next(
        line
        for line in _dockerfile().splitlines()
        if not line.lstrip().startswith("#")
        and line.lstrip().startswith("CMD ")
        and "curl" in line
    )


# --- compose --------------------------------------------------------------


def test_the_gateway_is_not_published_to_the_host() -> None:
    # `expose` is internal to the network; `ports` opens a path around nginx,
    # and with it TLS and edge rate limiting.
    api = _compose()["services"]["api"]
    assert "ports" not in api
    assert api["expose"] == ["8000"]


def test_vllm_is_not_published_to_the_host() -> None:
    # vLLM has no authentication of its own. Anything that reaches it can use
    # the GPU.
    assert "ports" not in _compose()["services"]["vllm"]


def test_missing_api_keys_fails_the_stack_rather_than_opening_it() -> None:
    # The `:?` form makes compose refuse to start. Without it the gateway comes
    # up with auth on and no keys, refusing every request - which presents as a
    # broken deployment rather than a missing .env.
    env = _compose()["services"]["api"]["environment"]
    assert env["QM_API_KEYS"].startswith("${QM_API_KEYS:?")


def test_no_secret_is_written_into_the_compose_file() -> None:
    for line in COMPOSE.read_text(encoding="utf-8").splitlines():
        if "QM_API_KEYS" in line:
            # Every mention must be a variable reference or a comment.
            assert "${" in line or line.lstrip().startswith("#"), line


def test_the_gpu_service_is_behind_a_profile() -> None:
    # Otherwise `docker compose up` fails on a machine with no NVIDIA runtime,
    # for a service the user did not ask for.
    assert _compose()["services"]["vllm"]["profiles"] == ["gpu"]


def test_the_gateway_does_not_hard_depend_on_the_gpu_service() -> None:
    # `required: false` lets the profiled service be absent. Without it compose
    # errors with "depends on undefined service vllm" whenever gpu is off.
    assert _compose()["services"]["api"]["depends_on"]["vllm"]["required"] is False


def test_the_gateway_binds_all_interfaces_in_the_container() -> None:
    # configs/serve/dev.yaml binds 127.0.0.1 - right on a laptop, unreachable
    # from nginx. The override makes any serve config work in a container.
    command = _compose()["services"]["api"]["command"]
    assert command[command.index("--host") + 1] == "0.0.0.0"


def test_vllm_mounts_the_weights_read_only() -> None:
    volumes = _compose()["services"]["vllm"]["volumes"]
    weights = [v for v in volumes if v.startswith("../models")]
    assert weights and all(v.endswith(":ro") for v in weights)


# --- Dockerfile -----------------------------------------------------------


def test_the_image_runs_as_a_non_root_user() -> None:
    users = [line.split()[1] for line in _dockerfile().splitlines() if line.startswith("USER ")]
    assert users and users[-1] != "root"


def test_the_project_root_is_set_explicitly() -> None:
    # paths.py needs a directory holding BOTH pyproject.toml and configs/. The
    # image has only configs/, so without this every --config resolves under
    # site-packages and fails.
    assert "QM_PROJECT_ROOT=/app" in _dockerfile()


def test_the_healthcheck_does_not_use_loopback() -> None:
    # A config binding 127.0.0.1 is reachable on loopback and unreachable from
    # every other container. A localhost healthcheck passes in exactly that
    # case, so the container reports healthy and nginx 502s against it.
    line = _healthcheck_line()
    assert "$(hostname)" in line
    assert "localhost" not in line


def test_the_healthcheck_uses_liveness_not_readiness() -> None:
    # /readyz is 503 while vLLM loads. Restarting the gateway for that turns a
    # slow model load into a crash loop.
    line = _healthcheck_line()
    assert "/healthz" in line and "/readyz" not in line


def test_the_gateway_installs_only_the_serve_extra() -> None:
    # torch would add ~7 GB and minutes to every restart, for code that never
    # runs: the GPU work happens in the vLLM container.
    installs = [line for line in _dockerfile().splitlines() if "pip install" in line]
    assert any('".[serve]"' in line for line in installs)
    assert not any("[train]" in line or "[vllm]" in line for line in installs)


# --- the web UI -----------------------------------------------------------
#
# The one property worth guarding here is that the gateway key stays on the
# server. Next.js inlines any NEXT_PUBLIC_* variable into the browser bundle at
# build time, so the mistake is a one-word rename away and is invisible until
# someone opens devtools.

WEB = PROJECT_ROOT / "web"
DOCKERFILE_WEB = DEPLOYMENT / "Dockerfile.web"


def _components() -> list[Path]:
    """Every component, at any depth.

    Recursive on purpose: a hardcoded directory silently stops checking the
    moment a file moves, and these are the security guards.
    """
    found = [p for p in (WEB / "components").rglob("*.tsx")]
    assert found, "no components found - has the directory moved?"
    return found


def _one(name: str) -> Path:
    matches = [p for p in (WEB / "components").rglob(name)]
    assert len(matches) == 1, f"expected exactly one {name}, found {len(matches)}"
    return matches[0]


def _code_lines(path: Path) -> list[str]:
    """Source lines with comments dropped.

    The comments in these files name the exact anti-patterns being avoided, so
    a plain substring search finds the explanation and fails on documentation
    rather than on code.
    """
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith(("//", "*", "/*"))
    ]


def test_the_gateway_key_is_never_exposed_to_the_browser() -> None:
    sources = [*WEB.glob("**/*.ts"), *WEB.glob("**/*.tsx")]
    sources = [p for p in sources if "node_modules" not in p.parts and ".next" not in p.parts]
    assert sources, "no web sources found"

    for path in sources:
        for line in _code_lines(path):
            assert "NEXT_PUBLIC_" not in line, f"{path.name}: {line.strip()}"


def test_the_key_is_read_only_in_a_server_route() -> None:
    # A "use client" file reading process.env would be a build-time inline.
    route = (WEB / "app" / "api" / "chat" / "route.ts").read_text(encoding="utf-8")
    assert "QM_API_KEY" in route
    assert '"use client"' not in route
    assert 'runtime = "nodejs"' in route

    for path in _components():
        for line in _code_lines(path):
            assert "process.env" not in line, f"{path.name}: {line.strip()}"


def test_the_web_container_gets_a_single_key_not_the_list() -> None:
    # QM_API_KEYS is comma-separated. Passing it straight through would send
    # `Authorization: Bearer key-one,key-two` as soon as a second key existed,
    # and every request would 401 with nothing pointing at the comma.
    env = _compose()["services"]["web"]["environment"]
    assert env["QM_API_KEY"].startswith("${QM_WEB_API_KEY:?")


def test_the_web_container_is_not_published_to_the_host() -> None:
    web = _compose()["services"]["web"]
    assert "ports" not in web
    assert web["expose"] == ["3000"]


def test_the_web_image_copies_the_static_assets() -> None:
    # `output: "standalone"` does not include .next/static or public/. Missing
    # them serves HTML with no CSS or JS, which reads as a broken build.
    text = DOCKERFILE_WEB.read_text(encoding="utf-8")
    assert ".next/standalone" in text
    assert ".next/static" in text


def test_the_web_image_runs_as_a_non_root_user() -> None:
    users = [
        line.split()[1]
        for line in DOCKERFILE_WEB.read_text(encoding="utf-8").splitlines()
        if line.startswith("USER ")
    ]
    assert users and users[-1] != "root"


def test_the_web_healthcheck_does_not_use_loopback() -> None:
    line = next(
        line
        for line in DOCKERFILE_WEB.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("CMD ") and "fetch(" in line
    )
    assert "hostname()" in line
    assert "127.0.0.1" not in line and "localhost" not in line


def test_model_output_is_never_rendered_as_raw_html() -> None:
    # The markdown renderer builds React elements. dangerouslySetInnerHTML
    # anywhere near model output would turn a generated string into markup -
    # and that string is partly determined by whatever a user typed in.
    #
    # layout.tsx is the one allowed use: a constant theme bootstrap script that
    # must run before first paint, with no user input in it.
    for path in _components():
        for line in _code_lines(path):
            assert "dangerouslySetInnerHTML" not in line, f"{path.name}: {line.strip()}"


def test_links_in_model_output_are_restricted_to_http() -> None:
    # A model can emit `javascript:` or `data:` URLs, either because it learned
    # them or because someone asked it to. Rendering one as a clickable anchor
    # makes model output executable.
    markdown = _one("Markdown.tsx").read_text(encoding="utf-8")
    assert '["http:", "https:"]' in markdown
    assert "new URL(href)" in markdown
    # An anchor that opens a new tab needs both, or the opened page can reach
    # back through window.opener.
    assert 'rel="noopener noreferrer"' in markdown


def test_the_ui_declares_karakalpak() -> None:
    # lang="kaa" is what tells a screen reader which language to pronounce.
    layout = (WEB / "app" / "layout.tsx").read_text(encoding="utf-8")
    assert 'lang="kaa"' in layout


def test_ui_strings_use_the_2016_latin_orthography() -> None:
    # The 2016 standard uses acute letters and dotless i. If these have been
    # normalised away, the interface is written in a different orthography from
    # the one the model was trained on.
    strings = (WEB / "lib" / "strings.ts").read_text(encoding="utf-8")
    assert any(ch in strings for ch in "áóúǵń"), "no acute letters in the UI strings"
    assert "ı" in strings, "no dotless i in the UI strings"


def test_streaming_messages_are_memoised() -> None:
    # Every token appends to the last message and re-renders the list. Without
    # memo, a long conversation reparses every markdown tree per token and the
    # stream visibly stutters.
    for name in ("MessageItem.tsx", "Markdown.tsx", "CodeBlock.tsx"):
        text = _one(name).read_text(encoding="utf-8")
        assert "memo(" in text, name


def test_single_key_shortcuts_do_not_fire_while_typing() -> None:
    # "/" and "?" are ordinary characters. A global handler that ignores the
    # focused element makes them impossible to type into a message.
    shell = _one("ChatShell.tsx").read_text(encoding="utf-8")
    assert "isContentEditable" in shell
    assert "if (typing) return;" in shell


def test_dialogs_use_the_native_element() -> None:
    # <dialog> + showModal() gives focus trapping, Escape, page inertness and
    # correct semantics. A div modal reimplements those, usually incompletely.
    dialog = _one("Dialog.tsx").read_text(encoding="utf-8")
    assert "showModal()" in dialog
    assert "<dialog" in dialog


def test_icon_only_buttons_require_an_accessible_name() -> None:
    # An icon button with no name is unusable with a screen reader, and it is
    # easy to forget per call site - so the type makes it non-optional.
    button = _one("IconButton.tsx").read_text(encoding="utf-8")
    assert "label: string;" in button
    assert "aria-label={label}" in button


def test_reduced_motion_is_respected() -> None:
    css = (WEB / "app" / "globals.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in css
    # The streaming caret must stay visible when animation stops - it is the
    # only signal that generation is still running.
    block = css.split("prefers-reduced-motion: reduce")[1]
    assert "streaming-caret" in block and "opacity: 1" in block


# --- nginx ----------------------------------------------------------------


def _v1_block() -> str:
    return SITE_CONF.read_text(encoding="utf-8").split("location /v1/")[1].split("\n    }")[0]


def test_streaming_is_not_buffered() -> None:
    # With buffering on, a token stream arrives in one lump when generation
    # finishes. The response is correct, so this presents as broken frontend
    # code rather than as a proxy setting.
    block = _v1_block()
    assert "proxy_buffering off" in block
    assert "proxy_cache off" in block
    # HTTP/1.0 has no chunked encoding, so a response of unknown length cannot
    # stream at all.
    assert "proxy_http_version 1.1" in block


def test_streaming_timeouts_outlast_a_long_generation() -> None:
    # nginx's default proxy_read_timeout is 60s, which truncates a long answer.
    read_timeout = _v1_block().split("proxy_read_timeout")[1].split(";")[0].strip()
    assert int(read_timeout.rstrip("s")) >= 300


def test_metrics_are_not_public() -> None:
    text = SITE_CONF.read_text(encoding="utf-8")
    assert "deny all" in text.split("location = /metrics")[1].split("}")[0]


def test_upstream_is_re_resolved() -> None:
    # nginx caches a resolved upstream address for the life of the process. A
    # restarted api container on a new IP then 502s until nginx restarts, with
    # the gateway sitting there healthy.
    text = SITE_CONF.read_text(encoding="utf-8")
    assert "resolver 127.0.0.11" in text
    assert "proxy_pass http://$qm_api;" in text
    # An `upstream` block would resolve once and cache again, defeating the
    # above. Check for the directive, not the word - the comments discuss it.
    directives = [line.strip() for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any(line.startswith("upstream ") for line in directives)


def test_tls_block_is_commented_out_until_certificates_exist() -> None:
    # nginx refuses to start when ssl_certificate points at a missing file, so
    # an active block would mean the stack cannot run at all before certbot.
    for line in SITE_CONF.read_text(encoding="utf-8").splitlines():
        if "ssl_certificate" in line:
            assert line.lstrip().startswith("#"), line


# --- build context --------------------------------------------------------


def _dockerignore() -> list[str]:
    return (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").split()


def test_the_corpus_and_checkpoints_stay_out_of_the_build_context() -> None:
    # The daemon receives the whole context before the first instruction runs.
    # data/ alone is 449 MB and nothing COPYs it.
    for path in ("data/", "models/", ".git/", ".venv/"):
        assert path in _dockerignore()


def test_secrets_stay_out_of_the_build_context() -> None:
    # Anything in the context can be read by a later instruction and baked into
    # a layer, whether or not the Dockerfile currently copies it.
    for path in (".env", "*.pem", "*.key", "deployment/nginx/certs/"):
        assert path in _dockerignore()
