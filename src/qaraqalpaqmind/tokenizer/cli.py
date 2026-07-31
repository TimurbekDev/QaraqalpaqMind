"""`qm tokenizer` - measure how well the base tokenizer handles Karakalpak.

    qm tokenizer analyze     fertility, cross-language comparison, letter splits
    qm tokenizer count       exact token count of the training corpus
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from ..common.io import read_jsonl
from ..common.logging import get_logger
from ..common.paths import INTERIM_DIR
from ..dedup.pipeline import output_path
from ..ingest.manifest import read_manifest
from .fertility import (
    QWEN3_MODEL,
    FertilityReport,
    compare_languages,
    letter_split_rate,
    load_tokenizer,
    measure,
    relative_penalty,
)

logger = get_logger(__name__)
console = Console()

app = typer.Typer(help="Analyse tokenizer behaviour on Karakalpak.", no_args_is_help=True)

# dilmash is parallel: identical content in four languages, which controls for
# what is being said and isolates the tokenizer's behaviour.
_DILMASH = INTERIM_DIR / "hf_dilmash_parallel.jsonl.zst"
# Verified against the data, not guessed: dilmash labels Uzbek `uzn_Latn`
# (Northern Uzbek, NLLB convention), not `uzb_Latn`. Uzbek is the baseline that
# matters most here - it is Karakalpak's closest well-resourced relative, so the
# gap between them is the clearest measure of what the tokenizer is missing.
_PARALLEL_LANGS = {"eng_Latn": "eng", "rus_Cyrl": "rus", "uzn_Latn": "uzb"}


def _parallel_samples(limit: int) -> dict[str, list[str]]:
    """Karakalpak sentences with their translations, from dilmash.

    Quota per partner language, not "first N rows": dilmash stores its splits
    in order, so reading sequentially returns only kaa_eng pairs and the
    comparison silently loses Russian and Uzbek - the two baselines that matter
    most, Uzbek being the closest relative.
    """
    per_language = max(1, limit // len(_PARALLEL_LANGS))
    partners: dict[str, list[str]] = {label: [] for label in _PARALLEL_LANGS.values()}
    kaa: dict[str, list[str]] = {label: [] for label in _PARALLEL_LANGS.values()}

    for row in read_jsonl(_DILMASH):
        meta = row.get("meta", {})
        label = _PARALLEL_LANGS.get(str(meta.get("parallel_lang")))
        partner_text = meta.get("parallel_text")
        if not label or not isinstance(partner_text, str) or len(partner_text) < 20:
            continue
        if len(partners[label]) >= per_language:
            if all(len(v) >= per_language for v in partners.values()):
                break
            continue
        partners[label].append(partner_text)
        kaa[label].append(str(row["text"]))

    samples: dict[str, list[str]] = {
        "kaa": [text for texts in kaa.values() for text in texts]
    }
    samples.update({label: texts for label, texts in partners.items() if texts})
    return {label: texts for label, texts in samples.items() if texts}


@app.command()
def analyze(
    model: str = typer.Option(QWEN3_MODEL, "--model", "-m"),
    limit: int = typer.Option(3000, "--limit", "-n", min=100),
) -> None:
    """Measure fertility, compare languages, and report letter fragmentation."""
    console.print(f"Loading [cyan]{model}[/] tokenizer...")
    tokenizer = load_tokenizer(model)

    console.print(f"Sampling {limit:,} parallel sentences from dilmash...")
    samples = _parallel_samples(limit)
    reports = compare_languages(samples, tokenizer)
    penalties = relative_penalty(reports)

    table = Table(title=f"Fertility on parallel content ({model})")
    for column in ("language", "words", "tokens", "tok/word", "chars/tok", "1-tok words", "vs eng"):
        table.add_column(column, justify="right")

    for language in sorted(reports, key=lambda language: -reports[language].fertility):
        report = reports[language]
        penalty = penalties.get(language, 1.0)
        style = "red" if penalty > 1.5 else "yellow" if penalty > 1.2 else "green"
        table.add_row(
            language,
            f"{report.words:,}",
            f"{report.tokens:,}",
            f"[{style}]{report.fertility:.2f}[/]",
            f"{report.chars_per_token:.2f}",
            f"{report.single_token_rate:.1%}",
            f"[{style}]{penalty:.2f}x[/]",
        )
    console.print(table)

    # --- letter fragmentation ---
    rates = letter_split_rate(samples["kaa"][:1000], tokenizer)
    if rates:
        letters = Table(title="Karakalpak letters emitted as standalone tokens")
        letters.add_column("letter")
        letters.add_column("standalone rate", justify="right")
        for letter, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
            style = "red" if rate > 0.5 else "yellow" if rate > 0.2 else "green"
            letters.add_row(letter, f"[{style}]{rate:.1%}[/]")
        console.print(letters)
        console.print(
            "[bright_black]A high rate means the vocabulary contains no subword "
            "holding that letter, so every word using it pays an extra token.[/]"
        )

    kaa = reports.get("kaa")
    if kaa and kaa.worst_words:
        console.print("\n[bold]Worst-fragmented Karakalpak words[/]")
        for word, cost in kaa.worst_words[:10]:
            pieces = [
                tokenizer.decode([i]) for i in tokenizer.encode(" " + word, add_special_tokens=False)
            ]
            console.print(f"  {word:<28}{cost:>3} tokens  {'|'.join(p.strip() for p in pieces)}")

    _verdict(reports, penalties)


def _verdict(reports: dict[str, FertilityReport], penalties: dict[str, float]) -> None:
    kaa = reports.get("kaa")
    if kaa is None:
        return
    penalty = penalties.get("kaa", 1.0)

    console.print("\n[bold]What this means[/]")
    console.print(
        f"  Karakalpak costs [bold]{penalty:.2f}x[/] more tokens than English for the "
        "same content."
    )
    if penalty > 1.5:
        console.print(
            "  [red]Vocabulary extension is worth evaluating in Phase 5:[/] this "
            "inflates training cost, shrinks the effective context window, and "
            "denies the model morpheme-level units to generalise over."
        )
    elif penalty > 1.2:
        console.print(
            "  [yellow]Tolerable without vocabulary extension[/], but it raises the "
            "token budget and shortens effective context."
        )
    else:
        console.print("  [green]The base vocabulary handles Karakalpak acceptably.[/]")


@app.command()
def count(
    model: str = typer.Option(QWEN3_MODEL, "--model", "-m"),
    dataset: str = typer.Option("pretrain_v1", "--dataset", "-d"),
    limit: int | None = typer.Option(None, "--limit", "-n", min=100),
) -> None:
    """Count the corpus in real tokens, replacing the chars/3.1 estimate."""
    path = output_path(dataset)
    if not path.exists():
        raise typer.BadParameter(f"No dataset at {path}. Run `qm dedup run` first.")

    tokenizer = load_tokenizer(model)
    rows: Iterator[dict[str, Any]] = read_jsonl(path)
    if limit is not None:
        rows = itertools.islice(rows, limit)

    report = measure((str(row["text"]) for row in rows), tokenizer, dataset, worst_n=0)
    console.print(
        f"[bold]{report.documents:,}[/] documents, "
        f"[bold]{report.words:,}[/] words, "
        f"[bold green]{report.tokens:,}[/] real tokens "
        f"({report.fertility:.2f} tok/word, {report.chars_per_token:.2f} chars/tok)"
    )

    if limit is None:
        try:
            manifest = read_manifest(dataset)
        except (OSError, ValueError):
            return
        estimated = manifest.estimated_tokens
        drift = report.tokens / estimated if estimated else 0.0
        console.print(
            f"[bright_black]The chars/3.1 estimate said {estimated:,}; "
            f"the real count is {drift:.2f}x that.[/]"
        )
