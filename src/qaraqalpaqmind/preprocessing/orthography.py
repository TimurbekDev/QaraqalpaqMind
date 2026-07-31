"""Karakalpak orthography unification.

Karakalpak has been written four different ways in living memory, and the
corpus contains all of them:

    Cyrillic     Қарақалпақстан Республикасы     sud.uz /qqc/, kitapxana, books
    Latin 1994   Qaraqalpaqstan Respublikasï     umlaut/cedilla diacritics
    Latin 2009   Qaraqalpaqstan Respublikasi'    apostrophes: a' o' u' g' n'
    Latin 2016   Qaraqalpaqstan Respublikası     acutes: á ó ú ǵ ń  (current)

A model trained on the mixture learns four spellings for every word, on a
corpus that is already too small to spend on redundancy. So we normalise
everything to **Latin 2016**, the current official standard, while keeping the
original in the record so nothing is lost.

Cyrillic is not a legacy edge case here: the judiciary publishes *more*
articles in Cyrillic than Latin, and the entire literary e-library is Cyrillic.
Transliterating it is how that material becomes usable.

Transliteration is deterministic and reversible for the letters that matter.
The lossy cases are documented at `CYRILLIC_TO_LATIN`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from ..common.logging import get_logger
from .script import Orthography, detect_orthography

logger = get_logger(__name__)

# --- Latin 2009 (apostrophe) -> Latin 2016 (acute) ------------------------
# Confirmed against real corpus text from GlotCC:
#   "XII a'sirdegi tu'rk ... ko'rkem so'z sheberlerinin'"
#   -> "XII ásirdegi túrk ... kórkem sóz sheberleriniń"
_APOSTROPHE_PAIRS: Final[dict[str, str]] = {
    "a'": "á", "A'": "Á",
    "o'": "ó", "O'": "Ó",
    "u'": "ú", "U'": "Ú",
    "g'": "ǵ", "G'": "Ǵ",
    "n'": "ń", "N'": "Ń",
    "i'": "ı", "I'": "I",
}

# Every apostrophe-like character actually used in the wild. Uzbek keyboards
# and Word autocorrect produce all of these for the same intended letter.
_APOSTROPHES: Final[str] = "'‘’ʼʻ`´"

# --- Latin 1994 (umlaut) -> Latin 2016 (acute) ----------------------------
_UMLAUT_MAP: Final[dict[str, str]] = {
    "ä": "á", "Ä": "Á",
    "ö": "ó", "Ö": "Ó",
    "ü": "ú", "Ü": "Ú",
    "ñ": "ń", "Ñ": "Ń",
    "ģ": "ǵ", "Ģ": "Ǵ",
    "ğ": "ǵ", "Ğ": "Ǵ",
    "ı": "ı", "İ": "İ",
}

# --- Cyrillic -> Latin 2016 ----------------------------------------------
# Multi-character outputs are handled by ordinary replacement; the map is
# applied longest-key-first so digraph sources cannot be split.
#
# LOSSY CASES, deliberate:
#   ъ, ь  -> dropped. Karakalpak Latin has no hard/soft sign; they appear
#            almost exclusively in Russian loanwords.
#   щ     -> "shsh". Rare, Russian loanwords only.
#   е     -> "e" always. Russian-style word-initial "ye" is not Karakalpak.
#
# ц -> "c", NOT "ts". This was measured, not assumed: the natively-Latin
# sud.uz locale writes korrupciya (37), konstituciya (69), instanciya (35),
# investiciya (24), apellyaciya (17), kassaciya (3) and never the "ts" forms.
# Karakalpak Latin does use `c`, contrary to several published alphabet charts.
CYRILLIC_TO_LATIN: Final[dict[str, str]] = {
    "а": "a", "ә": "á", "б": "b", "в": "v", "г": "g", "ғ": "ǵ", "д": "d",
    "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ń", "о": "o", "ө": "ó",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ү": "ú", "ў": "w",
    "ф": "f", "х": "x", "ҳ": "h", "ц": "c", "ч": "ch", "ш": "sh",
    "щ": "shsh", "ъ": "", "ы": "ı", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "А": "A", "Ә": "Á", "Б": "B", "В": "V", "Г": "G", "Ғ": "Ǵ", "Д": "D",
    "Е": "E", "Ё": "Yo", "Ж": "J", "З": "Z", "И": "I", "Й": "Y", "К": "K",
    "Қ": "Q", "Л": "L", "М": "M", "Н": "N", "Ң": "Ń", "О": "O", "Ө": "Ó",
    "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ү": "Ú", "Ў": "W",
    "Ф": "F", "Х": "X", "Ҳ": "H", "Ц": "C", "Ч": "Ch", "Ш": "Sh",
    "Щ": "Shsh", "Ъ": "", "Ы": "I", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
}

# Cyrillic look-alikes that real Karakalpak typists produce instead of the
# standard letters. Found by counting characters in the sud.uz Cyrillic locale
# that the table above did not cover: 437 occurrences across 142 documents,
# every one of which would otherwise have survived transliteration untouched
# and corrupted the word it appeared in.
CYRILLIC_HOMOGLYPHS: Final[dict[str, str]] = {
    "ӯ": "ў", "Ӯ": "Ў",  # U+04EF u-with-macron, typed for ў  (158x)
    "ѳ": "ө", "Ѳ": "Ө",  # U+0473 fita, typed for ө           (161x)
    "ӊ": "ң", "Ӊ": "Ң",  # U+04CA en-with-tail, typed for ң    (118x)
    "һ": "ҳ", "Һ": "Ҳ",  # U+04BB shha, typed for ҳ
    "і": "и", "І": "И",  # U+0456 Ukrainian/Kazakh i
}

_HOMOGLYPH_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(k) for k in CYRILLIC_HOMOGLYPHS)
)


def fix_cyrillic_homoglyphs(text: str) -> str:
    """Map look-alike Cyrillic letters onto the standard Karakalpak ones."""
    return _HOMOGLYPH_PATTERN.sub(lambda m: CYRILLIC_HOMOGLYPHS[m.group(0)], text)

_CYRILLIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(sorted((re.escape(k) for k in CYRILLIC_TO_LATIN), key=len, reverse=True))
)

# English contraction suffixes. `n'` is both the Karakalpak `ń` and the middle
# of "don't", and the corpus contains real English - dilmash carries 100k
# English sentences and quoted English turns up throughout the web sources.
# Converting blindly turns "don't" into "dońt".
_CONTRACTION_SUFFIX: Final[str] = r"(?:t|s|d|m|ll|re|ve)\b"

_APOSTROPHE_PATTERN: Final[re.Pattern[str]] = re.compile(
    f"([aouginAOUGIN])[{re.escape(_APOSTROPHES)}](?!{_CONTRACTION_SUFFIX})"
)

_UMLAUT_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(k) for k in _UMLAUT_MAP)
)


def apostrophe_to_acute(text: str) -> str:
    """Latin 2009 -> Latin 2016: `a' o' u' g' n'` become `á ó ú ǵ ń`.

    Only the five letters that take an apostrophe in this orthography are
    converted, and English contractions are excluded explicitly: `n'` is both
    Karakalpak `ń` and the middle of "don't", so `'` followed by
    t/s/d/m/ll/re/ve at a word boundary is left alone.

    `O'Brien` still becomes `ÓBrien`, which is the residual cost of this
    heuristic. Karakalpak text contains far more `so'z`-shaped words than Irish
    surnames, so the trade favours conversion.
    """

    def replace(match: re.Match[str]) -> str:
        letter = match.group(1)
        return _APOSTROPHE_PAIRS.get(f"{letter}'", match.group(0))

    return _APOSTROPHE_PATTERN.sub(replace, text)


def umlaut_to_acute(text: str) -> str:
    """Latin 1994 -> Latin 2016: `ä ö ü ñ ģ` become `á ó ú ń ǵ`."""
    return _UMLAUT_PATTERN.sub(lambda m: _UMLAUT_MAP[m.group(0)], text)


def cyrillic_to_latin(text: str) -> str:
    """Karakalpak Cyrillic -> Latin 2016.

    Look-alike letters are folded onto the standard ones first; without that
    step they survive the mapping unchanged and corrupt their words.

    Non-Cyrillic characters pass through untouched, so mixed-script text
    (Latin headings over a Cyrillic body, or embedded URLs) survives.
    """
    return _CYRILLIC_PATTERN.sub(
        lambda m: CYRILLIC_TO_LATIN[m.group(0)], fix_cyrillic_homoglyphs(text)
    )


def to_latin2016(text: str, orthography: Orthography | None = None) -> str:
    """Normalise any Karakalpak spelling convention to Latin 2016.

    Args:
        text: Input in any of the four conventions.
        orthography: Force a conversion instead of detecting one. Useful when
            a whole source is known to use one convention and short samples
            would be misdetected.
    """
    if not text:
        return text

    resolved = orthography or detect_orthography(text)

    match resolved:
        case Orthography.CYRILLIC:
            converted = cyrillic_to_latin(text)
        case Orthography.LATIN_2009:
            converted = apostrophe_to_acute(text)
        case Orthography.LATIN_1994:
            converted = umlaut_to_acute(text)
        case _:
            converted = text

    # Mixed documents are common: a Cyrillic article can carry an apostrophe
    # spelling in a quoted title, and 1994 diacritics turn up inside otherwise
    # modern text. Running the cheap passes unconditionally costs little and
    # catches those.
    converted = umlaut_to_acute(apostrophe_to_acute(converted))

    # Compose so that `á` is one code point, not `a` + combining acute. Without
    # this, visually identical strings hash differently and dedup misses them.
    return unicodedata.normalize("NFC", converted)


def latin_to_cyrillic(text: str) -> str:
    """Latin 2016 -> Cyrillic, for round-trip testing and Cyrillic output.

    Not used in the training pipeline. It exists so the transliteration tables
    can be checked for self-consistency, which is how mapping typos surface.
    """
    reverse: dict[str, str] = {}
    for cyrillic, latin in CYRILLIC_TO_LATIN.items():
        if latin and latin not in reverse:
            reverse[latin] = cyrillic

    pattern = re.compile("|".join(sorted((re.escape(k) for k in reverse), key=len, reverse=True)))
    return pattern.sub(lambda m: reverse[m.group(0)], text)
