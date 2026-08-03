"""`qm serve` - run the API server.

    qm serve                                  echo backend, no model needed
    qm serve --config configs/serve/local_model.yaml
    qm serve --backend vllm --vllm-url http://localhost:8001/v1
"""

from __future__ import annotations

import typer
from rich.console import Console

from ..common.config import load_config
from ..common.logging import get_logger
from .config import BackendKind, ServeConfig

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Serve the model over an OpenAI-compatible API.", no_args_is_help=False)


@app.callback(invoke_without_command=True)
def serve(
    config_path: str = typer.Option("serve/dev.yaml", "--config", "-c"),
    backend: str | None = typer.Option(None, "--backend", help="echo | transformers | vllm"),
    model: str | None = typer.Option(None, "--model", "-m"),
    vllm_url: str | None = typer.Option(None, "--vllm-url"),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Start the API server."""
    settings = load_config(config_path, ServeConfig)

    # CLI flags override the file, so one config serves several situations
    # without being edited.
    overrides: dict[str, object] = {}
    backend_overrides: dict[str, object] = {}
    server_overrides: dict[str, object] = {}

    if backend is not None:
        backend_overrides["kind"] = BackendKind(backend)
    if model is not None:
        backend_overrides["model"] = model
    if vllm_url is not None:
        backend_overrides["vllm_url"] = vllm_url
    if host is not None:
        server_overrides["host"] = host
    if port is not None:
        server_overrides["port"] = port

    if backend_overrides:
        overrides["backend"] = settings.backend.model_copy(update=backend_overrides)
    if server_overrides:
        overrides["server"] = settings.server.model_copy(update=server_overrides)
    if overrides:
        settings = settings.model_copy(update=overrides)

    _announce(settings)

    import uvicorn

    from .app import create_app

    uvicorn.run(
        "qaraqalpaqmind.api.app:create_app" if reload else create_app(settings),
        factory=reload,
        host=settings.server.host,
        port=settings.server.port,
        reload=reload,
        log_config=None,  # the project's own logging is already configured
    )


def _announce(settings: ServeConfig) -> None:
    console.print(
        f"[bold cyan]QaraqalpaqMind API[/]  "
        f"backend=[green]{settings.backend.kind.value}[/] "
        f"model=[green]{settings.backend.model}[/]"
    )
    console.print(
        f"  http://{settings.server.host}:{settings.server.port}  "
        f"docs at /docs, health at /healthz"
    )

    if settings.auth.enabled:
        keys = settings.auth.load_keys()
        if keys:
            console.print(f"  auth: [green]on[/], {len(keys)} key(s) from ${settings.auth.keys_env_var}")
        else:
            console.print(
                f"  auth: [red]on but NO KEYS in ${settings.auth.keys_env_var}[/] - "
                "every request will be refused"
            )
    else:
        console.print("  auth: [yellow]off[/]")
        if settings.server.host in {"0.0.0.0", "::"}:
            console.print(
                "  [red]Warning:[/] auth is off and the server binds all interfaces. "
                "Anything that can reach this host can use the GPU."
            )

    if settings.backend.kind is BackendKind.ECHO:
        console.print(
            "  [bright_black]Echo backend: canned replies, no model. "
            "Use --backend transformers or vllm for real output.[/]"
        )
