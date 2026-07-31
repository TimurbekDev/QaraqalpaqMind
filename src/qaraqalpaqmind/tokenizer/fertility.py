"""Tokenizer fertility analysis.

*Fertility* is tokens per word. It decides three things at once:

* **Training cost.** A corpus of N words costs N x fertility tokens to train on.
* **Context budget.** High fertility means fewer Karakalpak words fit in a
  context window, so the model sees less at once than it would in English.
* **Whether the base model can represent the language at all.** A tokenizer
  that shreds words into single characters gives the model no morpheme-level
  units to generalise over.

Qwen3's vocabulary was built without Karakalpak, so every number here is an
argument for or against extending it before continued pretraining.

The comparison uses `dilmash`, which is *parallel*: the same sentence in
Karakalpak, English, Russian and Uzbek. That controls for content, so a
fertility difference is a property of the tokenizer, not of what was said.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..common.logging import get_logger

logger = get_logger(__name__)

QWEN3_MODEL = "Qwen/Qwen3-8B"

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# The letters that distinguish Karakalpak Latin from the alphabets Qwen3 was
# trained on. If these consistently become standalone tokens, the tokenizer is
# treating Karakalpak as decorated ASCII rather than as a language.
KAA_LETTERS = "áǵńóúıÁǴŃÓÚ"


@dataclass(slots=True)
class FertilityReport:
    """Tokenizer behaviour over one body of text."""

    label: str
    documents: int = 0
    characters: int = 0
    words: int = 0
    tokens: int = 0
    single_token_words: int = 0
    words_over_four_tokens: int = 0
    worst_words: list[tuple[str, int]] = field(default_factory=list)

    @property
    def fertility(self) -> float:
        """Tokens per word. 1.0 would mean every word is one token."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def chars_per_token(self) -> float:
        """Characters per token. Higher is better - more text per unit of compute."""
        return self.characters / self.tokens if self.tokens else 0.0

    @property
    def single_token_rate(self) -> float:
        return self.single_token_words / self.words if self.words else 0.0

    @property
    def fragmented_rate(self) -> float:
        """Share of words costing more than four tokens."""
        return self.words_over_four_tokens / self.words if self.words else 0.0

    def summary(self) -> str:
        return (
            f"{self.label}: fertility={self.fertility:.2f} tok/word, "
            f"{self.chars_per_token:.2f} chars/tok, "
            f"1-token words {self.single_token_rate:.1%}, "
            f"badly split {self.fragmented_rate:.1%}"
        )


def load_tokenizer(model: str = QWEN3_MODEL) -> Any:
    """Load a Hugging Face tokenizer. No torch required."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def measure(
    texts: Iterable[str],
    tokenizer: Any,
    label: str,
    *,
    worst_n: int = 15,
) -> FertilityReport:
    """Measure fertility over a body of text."""
    report = FertilityReport(label=label)
    word_costs: dict[str, int] = {}
    # Per-word statistics need one encode call per word, which is far too slow
    # for a whole-corpus count. `worst_n=0` asks for document-level totals only.
    per_word = worst_n > 0
    cost_cache: dict[str, int] = {}

    for text in texts:
        if not text.strip():
            continue
        report.documents += 1
        report.characters += len(text)

        words = _WORD.findall(text)
        report.words += len(words)
        report.tokens += len(tokenizer.encode(text, add_special_tokens=False))

        if not per_word:
            continue

        for word in words:
            cost = cost_cache.get(word)
            if cost is None:
                # Encode with a leading space: that is how a word appears
                # mid-text to a byte-level BPE, and encoding it bare
                # understates the cost of the word-initial piece.
                cost = len(tokenizer.encode(" " + word, add_special_tokens=False))
                cost_cache[word] = cost
            if cost == 1:
                report.single_token_words += 1
            elif cost > 4:
                report.words_over_four_tokens += 1
            if len(word) > 3 and word_costs.get(word, 0) < cost:
                word_costs[word] = cost

    report.worst_words = sorted(word_costs.items(), key=lambda kv: -kv[1])[:worst_n]
    return report


def letter_split_rate(
    texts: Sequence[str], tokenizer: Any, letters: str = KAA_LETTERS
) -> dict[str, float]:
    """How often each distinctive letter is emitted as its own token.

    A rate near 1.0 means the tokenizer has no subword containing that letter,
    so every word using it pays an extra token and loses morpheme structure.
    """
    occurrences: collections.Counter[str] = collections.Counter()
    standalone: collections.Counter[str] = collections.Counter()

    for text in texts:
        pieces = [
            tokenizer.decode([token_id])
            for token_id in tokenizer.encode(text, add_special_tokens=False)
        ]
        for letter in letters:
            occurrences[letter] += text.count(letter)
        for piece in pieces:
            stripped = piece.strip()
            if len(stripped) == 1 and stripped in letters:
                standalone[stripped] += 1

    return {
        letter: standalone[letter] / occurrences[letter]
        for letter in letters
        if occurrences[letter] > 0
    }


def compare_languages(
    samples: dict[str, list[str]], tokenizer: Any
) -> dict[str, FertilityReport]:
    """Measure fertility for several languages over parallel content."""
    return {
        language: measure(texts, tokenizer, language)
        for language, texts in samples.items()
    }


def relative_penalty(reports: dict[str, FertilityReport], baseline: str = "eng") -> dict[str, float]:
    """How many times more tokens each language costs than the baseline.

    Computed on chars-per-token so that differences in word length between
    languages do not distort the comparison.
    """
    reference = reports.get(baseline)
    if reference is None or not reference.chars_per_token:
        return {}
    return {
        language: reference.chars_per_token / report.chars_per_token
        for language, report in reports.items()
        if report.chars_per_token
    }
