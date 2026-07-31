"""Karakalpak script and orthography detection.

Karakalpak is written in two scripts, and the Latin one has had three
incompatible orthographies in thirty years. Any corpus built from the web will
contain all of them mixed together, plus a large amount of Uzbek, Kazakh and
Russian that a naive "is it Cyrillic?" check would happily let through.

This module answers two questions cheaply, without a model:

    1. Which script is this text in?          -> `detect_script`
    2. Which Latin orthography variant?       -> `detect_orthography`
    3. Is it plausibly Karakalpak at all?     -> `karakalpak_score`

It is a *heuristic pre-filter*, not a language identifier. Phase 3 adds a
proper fastText classifier; this exists so that Phase 2 can decide whether a
domain is worth crawling before any model is trained.

References for the alphabets:
    * Cyrillic (1957-1994, still dominant in printed books)
    * Latin 1994  - used digraphs and letters like `ǵ`, `ń`, `ó`, `ú`, `á`
    * Latin 2009  - apostrophe forms `o'`, `g'` borrowed from Uzbek practice
    * Latin 2016  - current official standard, acute accents restored
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ..common.records import Script

__all__ = [
    "KAA_CYRILLIC_MARKERS",
    "KAA_LATIN_MARKERS",
    "Orthography",
    "Script",
    "ScriptReport",
    "analyse",
    "detect_orthography",
    "detect_script",
    "karakalpak_score",
]


class Orthography(StrEnum):
    """Which Latin Karakalpak spelling convention a sample follows."""

    LATIN_2016 = "latin_2016"  # á é í ó ú ǵ ń w  (current official)
    LATIN_2009 = "latin_2009"  # o' g' with apostrophes
    LATIN_1994 = "latin_1994"  # ä ö ü ñ ģ, Turkish-style diacritics
    CYRILLIC = "cyrillic"
    UNKNOWN = "unknown"


# --- Character inventories -------------------------------------------------
# Letters that are *distinctive*: they appear in Karakalpak and are rare or
# absent in the neighbouring languages we most need to reject.

# Present in Karakalpak Latin, absent from Uzbek Latin.
# NOTE: plain capital `I` must NOT be listed - it occurs in every Latin-script
# language and would inflate the score of any text at all. The discriminative
# capital is `İ` (the capital of Karakalpak dotted `i`; capital of dotless `ı`
# is the ordinary `I`).
KAA_LATIN_MARKERS: Final[frozenset[str]] = frozenset("áǵńóúıÁǴŃÓÚİ")

# Karakalpak Cyrillic letters that Uzbek Cyrillic does NOT have. These are the
# only Cyrillic letters that genuinely discriminate.
KAA_CYRILLIC_STRONG: Final[frozenset[str]] = frozenset("әңөүӘҢӨҮ")

# Karakalpak Cyrillic letters that Uzbek Cyrillic ALSO has. Weak evidence:
# counting these at full weight makes every Uzbek Cyrillic page look Karakalpak.
KAA_CYRILLIC_SHARED: Final[frozenset[str]] = frozenset("ғқўҳҒҚЎҲ")

KAA_CYRILLIC_MARKERS: Final[frozenset[str]] = KAA_CYRILLIC_STRONG | KAA_CYRILLIC_SHARED

# Kazakh Cyrillic has these; Karakalpak does not. Strong negative signal.
KAZ_ONLY_MARKERS: Final[frozenset[str]] = frozenset("ұіҰІ")

# Weight applied to letters shared with Uzbek Cyrillic.
_SHARED_MARKER_WEIGHT: Final[float] = 0.25

# Uzbek Latin's modifier-letter forms. Their presence suggests Uzbek text, or
# Karakalpak written in the 2009 apostrophe orthography - `detect_orthography`
# disambiguates using the Karakalpak markers.
UZB_LATIN_MARKERS: Final[frozenset[str]] = frozenset("ʻʼ`'")

# 1994-era Latin used Turkish/German style diacritics instead of acutes.
KAA_LATIN_1994_MARKERS: Final[frozenset[str]] = frozenset("äöüñģÄÖÜÑĢ")

# High-frequency Karakalpak function words. Cheap lexical confirmation that
# survives orthography differences reasonably well.
KAA_STOPWORDS_LATIN: Final[frozenset[str]] = frozenset(
    {
        "hám", "menen", "ushın", "bolıp", "boyınsha", "sonıń", "olardıń",
        "bul", "usı", "jáne", "yaki", "eger", "biraq", "sebebi", "arqalı",
        "haqqında", "keyin", "aldın", "jılı", "jılda", "respublikası",
        "qaraqalpaqstan", "mámleketlik", "xalıq", "jańalıqlar", "bolǵan",
    }
)

KAA_STOPWORDS_CYRILLIC: Final[frozenset[str]] = frozenset(
    {
        "ҳәм", "менен", "ушын", "болып", "бойынша", "соның", "олардың",
        "бул", "усы", "және", "яки", "егер", "бирақ", "себеби", "арқалы",
        "ҳаққында", "кейин", "алдын", "жылы", "жылда", "республикасы",
        "қарақалпақстан", "мәмлекетлик", "халық", "жаңалықлар", "болған",
    }
)

_MIN_LETTERS_FOR_CONFIDENCE: Final[int] = 40
_MIXED_THRESHOLD: Final[float] = 0.15


@dataclass(frozen=True, slots=True)
class ScriptReport:
    """Result of analysing one text sample."""

    script: Script
    orthography: Orthography
    karakalpak_score: float
    letters: int
    latin_ratio: float
    cyrillic_ratio: float
    marker_ratio: float
    stopword_hits: int

    @property
    def is_probably_karakalpak(self) -> bool:
        """Conservative gate for "worth crawling / worth keeping"."""
        return self.karakalpak_score >= 0.5 and self.letters >= _MIN_LETTERS_FOR_CONFIDENCE


def _classify_letters(text: str) -> tuple[int, int, int]:
    """Return (latin, cyrillic, other) letter counts."""
    latin = cyrillic = other = 0
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            other += 1
            continue
        if name.startswith("LATIN"):
            latin += 1
        elif name.startswith("CYRILLIC"):
            cyrillic += 1
        else:
            other += 1
    return latin, cyrillic, other


def detect_script(text: str) -> Script:
    """Classify the dominant writing system of `text`."""
    latin, cyrillic, other = _classify_letters(text)
    total = latin + cyrillic + other
    if total == 0:
        return Script.UNKNOWN
    if other / total > 0.5:
        return Script.OTHER

    minority = min(latin, cyrillic)
    if minority and minority / (latin + cyrillic) > _MIXED_THRESHOLD:
        return Script.MIXED
    return Script.LATIN if latin >= cyrillic else Script.CYRILLIC


def detect_orthography(text: str) -> Orthography:
    """Guess which Karakalpak spelling convention `text` follows."""
    script = detect_script(text)
    if script is Script.CYRILLIC:
        return Orthography.CYRILLIC
    if script not in {Script.LATIN, Script.MIXED}:
        return Orthography.UNKNOWN

    counts = Counter(text)
    acute = sum(counts[c] for c in KAA_LATIN_MARKERS)
    umlaut = sum(counts[c] for c in KAA_LATIN_1994_MARKERS)
    apostrophe = sum(counts[c] for c in UZB_LATIN_MARKERS)

    if acute and acute >= umlaut and acute >= apostrophe:
        return Orthography.LATIN_2016
    if umlaut and umlaut >= apostrophe:
        return Orthography.LATIN_1994
    if apostrophe:
        return Orthography.LATIN_2009
    return Orthography.UNKNOWN


def _tokenize(text: str) -> list[str]:
    return [
        token.strip(".,!?;:()[]«»\"'—–…").lower()
        for token in text.split()
        if any(ch.isalpha() for ch in token)
    ]


def analyse(text: str) -> ScriptReport:
    """Full script/orthography/Karakalpak-likelihood report for one sample.

    The score blends two independent signals so that neither alone can carry a
    false positive:

    * `marker_ratio`  - density of letters unique to Karakalpak
    * `stopword_hits` - count of high-frequency Karakalpak function words

    Kazakh-only letters subtract from the score, because Kazakh Cyrillic shares
    most of Karakalpak's distinctive letters and is the likeliest confusion.
    """
    latin, cyrillic, other = _classify_letters(text)
    letters = latin + cyrillic + other
    if letters == 0:
        return ScriptReport(
            script=Script.UNKNOWN,
            orthography=Orthography.UNKNOWN,
            karakalpak_score=0.0,
            letters=0,
            latin_ratio=0.0,
            cyrillic_ratio=0.0,
            marker_ratio=0.0,
            stopword_hits=0,
        )

    script = detect_script(text)
    counts = Counter(text)

    strong_count = sum(counts[c] for c in KAA_LATIN_MARKERS | KAA_CYRILLIC_STRONG)
    shared_count = sum(counts[c] for c in KAA_CYRILLIC_SHARED)
    kazakh_count = sum(counts[c] for c in KAZ_ONLY_MARKERS)
    marker_ratio = (strong_count + _SHARED_MARKER_WEIGHT * shared_count) / letters

    tokens = _tokenize(text)
    stopwords = KAA_STOPWORDS_LATIN | KAA_STOPWORDS_CYRILLIC
    stopword_hits = sum(1 for token in tokens if token in stopwords)

    # Karakalpak text runs roughly 4-9% distinctive letters; 4% saturates.
    marker_signal = min(marker_ratio / 0.04, 1.0)
    # Five function-word hits in a page is already decisive.
    stopword_signal = min(stopword_hits / 5.0, 1.0)
    kazakh_penalty = min(kazakh_count / max(letters * 0.005, 1.0), 1.0)

    # Lexical evidence outweighs orthographic evidence on purpose: Uzbek
    # Cyrillic shares four marker letters with Karakalpak, so a marker-only
    # match must stay below the 0.5 "probably Karakalpak" gate.
    score = max(0.0, 0.4 * marker_signal + 0.6 * stopword_signal - 0.6 * kazakh_penalty)

    return ScriptReport(
        script=script,
        orthography=detect_orthography(text),
        karakalpak_score=round(score, 3),
        letters=letters,
        latin_ratio=round(latin / letters, 3),
        cyrillic_ratio=round(cyrillic / letters, 3),
        marker_ratio=round(marker_ratio, 4),
        stopword_hits=stopword_hits,
    )


def karakalpak_score(text: str) -> float:
    """Shorthand for `analyse(text).karakalpak_score`."""
    return analyse(text).karakalpak_score
