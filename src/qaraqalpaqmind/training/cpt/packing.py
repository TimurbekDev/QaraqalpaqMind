"""Sequence packing for continued pretraining.

The median document in this corpus is 30 words. Padding each one to a 2048-token
sequence would spend well over 90% of the compute on padding tokens, so
documents are concatenated with an EOS separator and cut into fixed-length
chunks instead.

The separator matters. Without it the model learns that an article about Aral
Sea hydrology flows naturally into a court ruling, and never learns where a
document ends - which shows up later as generations that will not stop.

Packing is streamed: the corpus is 28.7M tokens and does not need to be
resident, and neither does its tokenised form.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from ...common.logging import get_logger

logger = get_logger(__name__)

_PROGRESS_EVERY = 50_000


@dataclass(slots=True)
class PackingStats:
    """What packing produced, for the run log."""

    documents: int = 0
    sequences: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    dropped_tail_tokens: int = 0

    @property
    def utilisation(self) -> float:
        """Share of tokenised input that reached a full sequence."""
        return self.tokens_out / self.tokens_in if self.tokens_in else 0.0

    def summary(self) -> str:
        return (
            f"{self.documents:,} documents -> {self.sequences:,} sequences "
            f"({self.tokens_out:,} of {self.tokens_in:,} tokens, "
            f"{self.utilisation:.1%} used, {self.dropped_tail_tokens:,} dropped)"
        )


def pack_documents(
    texts: Iterable[str],
    tokenizer: Any,
    sequence_length: int,
    *,
    add_eos: bool = True,
    stats: PackingStats | None = None,
) -> Iterator[dict[str, list[int]]]:
    """Stream fixed-length token sequences from a corpus.

    Yields dicts with `input_ids`, `attention_mask` and `labels`, which is what
    a causal language-modelling collator expects. Labels equal input_ids: for
    pretraining every token is a prediction target.

    The final partial buffer is discarded rather than padded. On a corpus of
    286k documents that is at most one sequence, and padding it would introduce
    the only padded example in the run.
    """
    tracker = stats if stats is not None else PackingStats()
    eos = tokenizer.eos_token_id
    if add_eos and eos is None:
        raise ValueError("tokenizer has no eos_token_id, so documents cannot be separated")

    buffer: list[int] = []

    for text in texts:
        if not text or not text.strip():
            continue
        tracker.documents += 1

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            token_ids.append(eos)
        tracker.tokens_in += len(token_ids)
        buffer.extend(token_ids)

        while len(buffer) >= sequence_length:
            chunk = buffer[:sequence_length]
            del buffer[:sequence_length]
            tracker.sequences += 1
            tracker.tokens_out += sequence_length
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * sequence_length,
                "labels": list(chunk),
            }

        if tracker.documents % _PROGRESS_EVERY == 0:
            logger.info("Packing progress", extra={"summary": tracker.summary()})

    tracker.dropped_tail_tokens = len(buffer)
    logger.info("Packing complete", extra={"summary": tracker.summary()})


def count_sequences(corpus_tokens: int, sequence_length: int) -> int:
    """How many training sequences a corpus of this size yields."""
    return corpus_tokens // sequence_length


def describe_schedule(
    corpus_tokens: int,
    sequence_length: int,
    effective_batch_size: int,
    num_epochs: float,
) -> dict[str, int | float]:
    """Work out the run's shape before committing a GPU to it.

    Being able to answer "how many steps, how long?" from a config alone is
    what stops a run being launched that would take a week nobody has.
    """
    sequences = count_sequences(corpus_tokens, sequence_length)
    steps_per_epoch = max(1, sequences // effective_batch_size)
    total_steps = max(1, int(steps_per_epoch * num_epochs))
    return {
        "sequences": sequences,
        "tokens_per_step": effective_batch_size * sequence_length,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "tokens_seen": int(corpus_tokens * num_epochs),
    }
