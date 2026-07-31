"""Quality filtering, with thresholds derived from this corpus.

**Do not import Gopher or C4 thresholds here.** They are tuned on English web
text and this corpus is neither English nor web-typical. Two examples of what
that would cost, both measured over 11,678 sampled documents:

* Gopher rejects documents whose mean word length exceeds 10 characters.
  Karakalpak is agglutinative: this corpus sits at a median of **6.98** and a
  99th percentile of **9.6**. Gopher's ceiling lands inside our legitimate
  distribution and would discard the most morphologically dense documents -
  exactly the material a Karakalpak model most needs.
* Common minimum-length rules demand 50+ words. The median document here is
  **30 words**, because two thirds of the corpus is sentence-level data from
  dilmash and karakalpak_corpus_v2. A 50-word floor would delete most of it.

So: hard rejection is reserved for text that is not usable prose at all, and
everything else becomes a *score* plus *flags* that later stages can weigh.
On a corpus of ~25M tokens, throwing away recoverable data is the expensive
mistake, not keeping some mediocre data.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import Field

from ..common.config import StrictModel
from ..common.logging import get_logger
from ..common.records import Quality
from ..preprocessing.script import analyse

logger = get_logger(__name__)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?…]\s*$")

# Characters that signal layout debris rather than language.
_JUNK_SYMBOLS = frozenset("#…•|→←※★☆■□▪▫©®™§¶‹›^~*_=+[]{}<>")

# Lines that are navigation, not prose. Karakalpak and Russian variants of the
# words that appear in site chrome across every source we crawled.
_BOILERPLATE_LINES = frozenset(
    {
        "bas bet", "jańalıqlar", "baylanıs", "hújjetler", "biz haqqımızda",
        "sayt xarita", "izlew", "kirisiw", "shıǵıw", "barlıq huqıqlar qorǵalǵan",
        "главная", "новости", "контакты", "поиск", "войти", "все права защищены",
        "home", "news", "contacts", "search", "login", "all rights reserved",
        "бас бет", "жаңалықлар", "байланыс", "ҳүжжетлер",
    }
)

# Function words that are unambiguous evidence of another language. None of
# these collide with Karakalpak vocabulary, which is what makes the test safe.
#
# This exists because the Karakalpak-likelihood heuristic cannot decide short
# text - it needs ~40 letters - and dilmash contains textbook rows whose
# "Karakalpak" column is actually English ("Group 1: A boy has a donkey.").
# Those are short, so they score 0.0 on the positive signal and would otherwise
# be kept on a soft penalty, teaching the model to produce English.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "of", "to", "in", "is", "was", "are", "were", "has",
        "have", "with", "that", "this", "from", "they", "their", "there",
        "which", "about", "would", "could", "should", "write", "read",
    }
)

_RUSSIAN_STOPWORDS: frozenset[str] = frozenset(
    {
        "и", "в", "не", "на", "что", "с", "по", "это", "как", "он", "она",
        "они", "был", "была", "было", "быть", "для", "или", "если", "также",
        "который", "которые", "его", "her", "их", "при", "уже",
    }
)

_SPAM_PATTERNS = (
    re.compile(r"\b(?:viagra|casino|porn|xxx|bitcoin\s+invest)\b", re.IGNORECASE),
    re.compile(r"(?:https?://\S+\s*){6,}"),  # link farm
    re.compile(r"(.)\1{20,}"),  # one character repeated 20+ times
)


class Flag(StrEnum):
    """Why a document lost points. Recorded, not silently applied."""

    TOO_SHORT = "too_short"
    TOO_FEW_WORDS = "too_few_words"
    WORD_LENGTH_OUTLIER = "word_length_outlier"
    LOW_ALPHA = "low_alpha"
    HIGH_SYMBOL = "high_symbol"
    HIGH_DIGIT = "high_digit"
    HIGH_UPPERCASE = "high_uppercase"
    REPETITIVE_LINES = "repetitive_lines"
    REPETITIVE_WORDS = "repetitive_words"
    BOILERPLATE = "boilerplate"
    SPAM = "spam"
    NOT_KARAKALPAK = "not_karakalpak"
    FOREIGN_LANGUAGE = "foreign_language"
    NO_SENTENCE_END = "no_sentence_end"
    FALLBACK_EXTRACTION = "fallback_extraction"


class FilterConfig(StrictModel):
    """Thresholds. Every default below was read off the corpus, not borrowed.

    Percentiles quoted are from 11,678 documents sampled across all sources.
    """

    # --- hard rejection: not usable prose in any downstream stage ---
    min_chars: int = Field(default=20, ge=1)
    min_words: int = Field(default=3, ge=1)  # corpus p1 = 3
    max_symbol_ratio: float = Field(default=0.10, ge=0.0, le=1.0)  # p99 = 0.033
    min_alpha_word_fraction: float = Field(default=0.50, ge=0.0, le=1.0)  # p1 = 0.55

    # Mean word length. Corpus p1 = 3.4, p50 = 6.98, p99 = 9.6.
    # The ceiling is 12.0, NOT Gopher's 10.0, which sits inside our p99.
    min_mean_word_length: float = Field(default=2.5, ge=0.0)
    max_mean_word_length: float = Field(default=12.0, ge=0.0)

    # --- soft signals: reduce the score, never reject outright ---
    max_digit_ratio: float = Field(default=0.30, ge=0.0, le=1.0)  # p99 = 0.161
    max_uppercase_ratio: float = Field(default=0.40, ge=0.0, le=1.0)  # p99 = 0.571
    max_duplicate_line_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    max_top_word_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    max_boilerplate_line_fraction: float = Field(default=0.30, ge=0.0, le=1.0)

    min_karakalpak_score: float = Field(default=0.30, ge=0.0, le=1.0)
    check_language: bool = True

    # Fraction of words that are unambiguous English or Russian function words.
    # 0.15 is deliberately high: Karakalpak text quotes foreign phrases, and a
    # rejection here is permanent.
    max_foreign_stopword_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    # Documents scoring below this are dropped by `should_keep`.
    min_quality_score: float = Field(default=0.40, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class DocumentStats:
    """Everything the filters need, computed in one pass over the text."""

    chars: int
    words: int
    lines: int
    mean_word_length: float
    alpha_word_fraction: float
    symbol_ratio: float
    digit_ratio: float
    uppercase_ratio: float
    duplicate_line_fraction: float
    top_word_fraction: float
    boilerplate_line_fraction: float
    ends_with_sentence: bool


def compute_stats(text: str) -> DocumentStats:
    """Measure a document once; every filter reads from the result."""
    words = _WORD.findall(text)
    tokens = text.split()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    letters = [c for c in text if c.isalpha()]

    if not text:
        return DocumentStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    word_counts = collections.Counter(w.lower() for w in words)
    line_counts = collections.Counter(lines)

    return DocumentStats(
        chars=len(text),
        words=len(words),
        lines=len(lines),
        mean_word_length=(sum(len(w) for w in words) / len(words)) if words else 0.0,
        alpha_word_fraction=(
            sum(1 for t in tokens if any(c.isalpha() for c in t)) / len(tokens) if tokens else 0.0
        ),
        symbol_ratio=sum(1 for c in text if c in _JUNK_SYMBOLS) / len(text),
        digit_ratio=sum(1 for c in text if c.isdigit()) / len(text),
        uppercase_ratio=(sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0,
        duplicate_line_fraction=(1 - len(line_counts) / len(lines)) if lines else 0.0,
        top_word_fraction=(word_counts.most_common(1)[0][1] / len(words)) if words else 0.0,
        boilerplate_line_fraction=(
            sum(1 for ln in lines if ln.lower().strip(":|-· ") in _BOILERPLATE_LINES) / len(lines)
            if lines
            else 0.0
        ),
        ends_with_sentence=bool(_SENTENCE_END.search(text.strip())),
    )


@dataclass(slots=True)
class Assessment:
    """Verdict for one document."""

    score: float
    flags: list[Flag] = field(default_factory=list)
    rejected: bool = False
    stats: DocumentStats | None = None

    def to_quality(self) -> Quality:
        return Quality(score=round(self.score, 3), flags=[f.value for f in self.flags])


def _is_spam(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SPAM_PATTERNS)


def foreign_stopword_ratio(text: str) -> float:
    """Fraction of words that are unambiguous English or Russian function words."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    foreign = sum(1 for w in words if w in _ENGLISH_STOPWORDS or w in _RUSSIAN_STOPWORDS)
    return foreign / len(words)


def assess(
    text: str,
    config: FilterConfig | None = None,
    *,
    extractor: str | None = None,
) -> Assessment:
    """Score a document and record why.

    `rejected` marks text that is not usable prose at all. Everything else gets
    a score in [0, 1]; `should_keep` applies the threshold, so a caller can
    keep more or less of the distribution without re-running the analysis.
    """
    cfg = config or FilterConfig()
    stats = compute_stats(text)
    flags: list[Flag] = []
    score = 1.0

    # --- hard rejections ---
    # Every applicable reason is collected, not just the first. A document of
    # pure symbols trips the word-count check too, and reporting only
    # "too_few_words" would misdescribe why the corpus lost it - which matters
    # when these counts are the evidence for tuning the thresholds.
    rejections: list[Flag] = []
    if stats.chars < cfg.min_chars:
        rejections.append(Flag.TOO_SHORT)
    if stats.words < cfg.min_words:
        rejections.append(Flag.TOO_FEW_WORDS)
    if stats.alpha_word_fraction < cfg.min_alpha_word_fraction:
        rejections.append(Flag.LOW_ALPHA)
    if stats.symbol_ratio > cfg.max_symbol_ratio:
        rejections.append(Flag.HIGH_SYMBOL)
    if stats.words and not (
        cfg.min_mean_word_length <= stats.mean_word_length <= cfg.max_mean_word_length
    ):
        rejections.append(Flag.WORD_LENGTH_OUTLIER)
    if _is_spam(text):
        rejections.append(Flag.SPAM)
    if cfg.check_language and foreign_stopword_ratio(text) > cfg.max_foreign_stopword_ratio:
        # A hard rejection, unlike the Karakalpak-likelihood signal, because
        # this evidence is positive and unambiguous rather than an absence.
        rejections.append(Flag.FOREIGN_LANGUAGE)

    if rejections:
        return Assessment(0.0, rejections, rejected=True, stats=stats)

    # --- soft signals ---
    if stats.digit_ratio > cfg.max_digit_ratio:
        flags.append(Flag.HIGH_DIGIT)
        score -= 0.20
    if stats.uppercase_ratio > cfg.max_uppercase_ratio:
        flags.append(Flag.HIGH_UPPERCASE)
        score -= 0.15
    if stats.duplicate_line_fraction > cfg.max_duplicate_line_fraction:
        flags.append(Flag.REPETITIVE_LINES)
        score -= 0.25
    if stats.top_word_fraction > cfg.max_top_word_fraction and stats.words > 20:
        flags.append(Flag.REPETITIVE_WORDS)
        score -= 0.20
    if stats.boilerplate_line_fraction > cfg.max_boilerplate_line_fraction:
        flags.append(Flag.BOILERPLATE)
        score -= 0.30
    if not stats.ends_with_sentence and stats.words > 30:
        # Long text with no terminal punctuation is usually a truncated
        # listing page rather than prose.
        flags.append(Flag.NO_SENTENCE_END)
        score -= 0.10
    if extractor == "fallback":
        # trafilatura declined this page; the fallback keeps more chrome.
        flags.append(Flag.FALLBACK_EXTRACTION)
        score -= 0.15

    if cfg.check_language:
        report = analyse(text[:3000])
        if report.karakalpak_score < cfg.min_karakalpak_score:
            flags.append(Flag.NOT_KARAKALPAK)
            score -= 0.40

    return Assessment(max(0.0, min(1.0, score)), flags, rejected=False, stats=stats)


def should_keep(assessment: Assessment, config: FilterConfig | None = None) -> bool:
    cfg = config or FilterConfig()
    return not assessment.rejected and assessment.score >= cfg.min_quality_score
