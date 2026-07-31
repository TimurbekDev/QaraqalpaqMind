"""`qm` - the single command-line entrypoint for the whole pipeline.

Sub-command groups are added phase by phase:

    qm crawl   ...   Phase 2
    qm clean   ...   Phase 3
    qm dataset ...   Phase 4
    qm train   ...   Phases 5-7
    qm eval    ...   Phase 8
    qm rag     ...   Phase 9
    qm serve   ...   Phase 10

Groups are attached lazily so that `qm --help` works on a laptop with no torch
installed. Only the group you actually invoke imports its heavy dependencies.
"""

from __future__ import annotations

import typer

from . import __version__
from .common.logging import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer(
    name="qm",
    help="QaraqalpaqMind - build, train and serve a Karakalpak LLM.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="DEBUG|INFO|WARNING|ERROR"),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON lines to stderr."),
) -> None:
    """Global options applied before any sub-command runs."""
    setup_logging(level=log_level, use_json=json_logs)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"qaraqalpaqmind {__version__}")


@app.command()
def doctor() -> None:
    """Check that the environment is set up correctly (paths, optional deps, GPU)."""
    from .common import paths
    from .common.env import has_secret, project_env_file

    typer.secho(f"project root : {paths.PROJECT_ROOT}", fg=typer.colors.CYAN)
    typer.secho(f"data dir     : {paths.DATA_DIR}", fg=typer.colors.CYAN)
    typer.secho(f"models dir   : {paths.MODELS_DIR}", fg=typer.colors.CYAN)
    typer.secho(f"logs dir     : {paths.LOGS_DIR}", fg=typer.colors.CYAN)

    env_file = project_env_file()
    typer.secho(
        f".env         : {'loaded' if env_file.is_file() else 'not found'} ({env_file})",
        fg=typer.colors.CYAN if env_file.is_file() else typer.colors.YELLOW,
    )
    # Presence only. A token must never be printed, logged or echoed.
    for secret in ("HF_TOKEN", "WANDB_API_KEY"):
        configured = has_secret(secret)
        typer.secho(
            f"{secret:<13}: {'configured' if configured else 'not set'}",
            fg=typer.colors.GREEN if configured else typer.colors.YELLOW,
        )

    for label, module in (
        ("crawl", "httpx"),
        ("clean", "datasketch"),
        ("train", "torch"),
        ("rag", "qdrant_client"),
        ("serve", "fastapi"),
    ):
        try:
            __import__(module)
            typer.secho(f"[ok]      extra '{label}' available", fg=typer.colors.GREEN)
        except ImportError:
            typer.secho(f"[missing] extra '{label}'  ->  pip install -e '.[{label}]'", fg=typer.colors.YELLOW)


def _register_groups() -> None:
    """Attach phase sub-commands, skipping any whose extras are not installed.

    Import errors are swallowed on purpose: a machine with only `.[crawl]`
    should still get a working `qm crawl`, and a helpful message rather than a
    traceback for the groups it cannot load.
    """
    try:
        from .crawlers.cli import app as crawl_app

        app.add_typer(crawl_app, name="crawl")
    except ImportError as exc:
        logger.debug("crawl group unavailable", extra={"error": str(exc)})

    try:
        from .ingest.cli import app as ingest_app

        app.add_typer(ingest_app, name="ingest")
    except ImportError as exc:
        logger.debug("ingest group unavailable", extra={"error": str(exc)})

    try:
        from .cleaning.cli import app as clean_app

        app.add_typer(clean_app, name="clean")
    except ImportError as exc:
        logger.debug("clean group unavailable", extra={"error": str(exc)})

    try:
        from .dedup.cli import app as dedup_app

        app.add_typer(dedup_app, name="dedup")
    except ImportError as exc:
        logger.debug("dedup group unavailable", extra={"error": str(exc)})

    try:
        from .tokenizer.cli import app as tokenizer_app

        app.add_typer(tokenizer_app, name="tokenizer")
    except ImportError as exc:
        logger.debug("tokenizer group unavailable", extra={"error": str(exc)})

    try:
        from .schemas.cli import app as schema_app

        app.add_typer(schema_app, name="schema")
    except ImportError as exc:
        logger.debug("schema group unavailable", extra={"error": str(exc)})

    # Registered as each phase lands:
    # from .training.cli import app as train_app;  app.add_typer(train_app, name="train")


_register_groups()


if __name__ == "__main__":
    app()
