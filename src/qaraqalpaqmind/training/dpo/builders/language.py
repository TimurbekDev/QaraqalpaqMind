"""Language-consistency preference pairs.

The failure this targets is specific and likely: a model whose Karakalpak is
thin will answer a Karakalpak question in Russian or English, because those are
the languages it knows best. Every base model with Karakalpak in the tail of its
distribution does this, and continued pretraining on 28.7M tokens does not
reliably stop it.

dilmash gives the pair for free. Ask a question in Karakalpak; the Karakalpak
answer is `chosen`, its own translation into Russian or English is `rejected`.
The two say exactly the same thing - they are translations of each other - so
the only dimension they differ in is the language they are in.

That is the tightest possible pair for this behaviour, and it is real human
translation rather than model output.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from ....common.io import read_jsonl
from ....common.logging import get_logger
from ....common.paths import PROCESSED_DIR
from ....schemas import PreferenceRecord, Provenance

logger = get_logger(__name__)

CRITERION = "language_consistency"

DILMASH = PROCESSED_DIR / "hf_dilmash_parallel.jsonl.zst"

_MIN_CHARS = 30
_MAX_CHARS = 800

# Karakalpak instructions whose answer is the text itself. Framing it as a task
# with a Karakalpak instruction is what makes answering in Russian a visible
# error rather than an alternative reading of the request.
_PROMPTS = (
    "Tómendegi mazmundı qaraqalpaq tilinde jazıp ber:",
    "Bul mazmundı qaraqalpaqsha bayan et:",
    "Usı maǵlıwmattı qaraqalpaq tilinde jetkerip ber:",
)

_LANGUAGE_NAMES = {"eng": "ingliz", "rus": "rus", "uzn": "ózbek", "uzb": "ózbek"}


@dataclass(slots=True)
class LanguageStats:
    read: int = 0
    emitted: int = 0
    skipped_length: int = 0
    by_rejected_language: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"language-consistency preferences: {self.emitted:,} pairs from "
            f"{self.read:,} translation pairs ({self.skipped_length:,} out of length range)"
        )


def build(
    *,
    limit: int | None = None,
    seed: int = 20260731,
    source_path: str | None = None,
) -> Iterator[PreferenceRecord]:
    """Yield pairs preferring a Karakalpak answer over its foreign translation."""
    rng = random.Random(seed)
    stats = LanguageStats()

    provenance = Provenance(
        source_id="dpo_language_consistency",
        source_url="https://huggingface.co/datasets/tahrirchi/dilmash",
        license="MIT",
        # Both sides are human-written translations; only the pairing is ours.
        synthetic=False,
        human_reviewed=False,
    )

    for row in read_jsonl(source_path or DILMASH):
        if limit is not None and stats.emitted >= limit:
            break

        meta = row.get("meta", {})
        foreign = meta.get("parallel_text")
        foreign_lang = meta.get("parallel_lang")
        kaa = str(row.get("text", ""))

        if not isinstance(foreign, str) or not isinstance(foreign_lang, str):
            continue
        stats.read += 1

        if not (_MIN_CHARS <= len(kaa) <= _MAX_CHARS):
            stats.skipped_length += 1
            continue
        if not (_MIN_CHARS <= len(foreign) <= _MAX_CHARS):
            stats.skipped_length += 1
            continue
        if kaa.strip() == foreign.strip():
            continue

        code = foreign_lang.split("_")[0].lower()
        instruction = _PROMPTS[rng.randrange(len(_PROMPTS))]

        yield PreferenceRecord(
            prompt=f"{instruction}\n\n{foreign}",
            chosen=kaa,
            rejected=foreign,
            criterion=CRITERION,
            provenance=provenance,
            meta={"rejected_language": code, "rejected_language_name": _LANGUAGE_NAMES.get(code)},
        )
        stats.emitted += 1
        stats.by_rejected_language[code] = stats.by_rejected_language.get(code, 0) + 1

    logger.info("Language preference builder finished", extra={"summary": stats.summary()})
    if stats.by_rejected_language:
        logger.info("Rejected languages", extra=dict(stats.by_rejected_language))
