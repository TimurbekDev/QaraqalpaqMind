"""Text normalisation: make identical text identical.

Everything here exists to serve two goals:

1. **Repair** what transport and copy-paste broke - mojibake, control
   characters, decomposed accents, non-breaking spaces.
2. **Canonicalise** what is merely inconsistent - three kinds of quotation
   mark, five kinds of dash, URLs written six ways.

The second matters more than it looks. A deduplicator compares strings; two
paragraphs that differ only in curly versus straight quotes are two documents
to it and one document to a reader. Normalisation is what makes Phase 3's
dedup step able to see them as the same.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Literal

from pydantic import Field

from ..common.config import StrictModel
from ..common.logging import get_logger

logger = get_logger(__name__)


class Handling(StrEnum):
    """What to do with a class of token."""

    KEEP = "keep"
    MASK = "mask"
    REMOVE = "remove"


class NormalizeConfig(StrictModel):
    """Knobs for `normalize_text`. Defaults are what the corpus is built with."""

    fix_mojibake: bool = True
    # NFC by default, and NFKD would be a mistake here: it strips the acute
    # accents that carry meaning in Karakalpak, turning `á` into `a`.
    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    strip_control: bool = True
    fold_quotes: bool = True
    fold_dashes: bool = False
    urls: Handling = Handling.KEEP
    emails: Handling = Handling.MASK
    emoji: Handling = Handling.KEEP
    collapse_whitespace: bool = True
    max_consecutive_newlines: int = Field(default=2, ge=1)


URL_TOKEN = "<URL>"
EMAIL_TOKEN = "<EMAIL>"

_URL = re.compile(r"https?://\S+|www\.[^\s/$.?#][^\s]*", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Control characters except tab and newline, plus the invisible formatting
# characters. These are written as escapes on purpose: embedding them literally
# puts unprintable bytes in the source file, where no reviewer can see them.
#
# They matter because they survive every other cleaning step while being
# invisible, so two strings that read identically hash differently and dedup
# silently keeps both.
_CONTROL = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    "​-‏"  # zero-width space/non-joiner/joiner, LTR/RTL marks
    "﻿"  # byte-order mark appearing mid-document
    "‪-‮"  # bidirectional overrides
    "⁠-⁤"  # word joiner and invisible operators
    "]"
)

_QUOTE_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "«": '"', "»": '"', "″": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
})

_DASH_MAP = str.maketrans({"–": "-", "—": "-", "―": "-", "−": "-"})

# Unicode spaces that are not ASCII space. NBSP is the common one in pasted
# text and, like the control characters above, is invisible in a diff.
# Written as escapes so the source file stays readable and reviewable.
_SPACE_MAP = str.maketrans(
    dict.fromkeys(
        " "          # no-break space
        " "          # ogham space mark
        "           "
        " "          # narrow no-break space
        " "          # medium mathematical space
        "　",         # ideographic space
        " ",
    )
)

_HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)

_EMOJI = re.compile(
    "["
    "🀀-🫿"   # pictographs, emoticons, transport, symbols
    "☀-➿"           # misc symbols and dingbats
    "️‍"            # variation selector 16, zero-width joiner
    "]+"
)


def strip_control_chars(text: str) -> str:
    """Remove invisible characters that survive other cleaning and break hashing."""
    return _CONTROL.sub("", text)


def fold_quotes(text: str) -> str:
    """Collapse typographic quotation marks onto ASCII forms.

    Karakalpak sources use `«»` (Russian convention), `""` (Word autocorrect)
    and `""` interchangeably, often within one document.
    """
    return text.translate(_QUOTE_MAP)


def fold_dashes(text: str) -> str:
    """Collapse en/em dashes and the minus sign onto ASCII hyphen.

    Off by default: em dashes carry punctuation meaning that hyphens do not.
    """
    return text.translate(_DASH_MAP)


def normalize_spaces(text: str, *, max_newlines: int = 2) -> str:
    """Canonicalise whitespace without destroying paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_SPACE_MAP)
    text = _HORIZONTAL_RUN.sub(" ", text)
    text = _TRAILING_SPACE.sub("", text)
    text = re.sub(rf"\n{{{max_newlines + 1},}}", "\n" * max_newlines, text)
    return text.strip()


def handle_urls(text: str, mode: Handling) -> str:
    if mode is Handling.KEEP:
        return text
    return _URL.sub("" if mode is Handling.REMOVE else URL_TOKEN, text)


def handle_emails(text: str, mode: Handling) -> str:
    """Emails default to masking - they are personal data with no training value."""
    if mode is Handling.KEEP:
        return text
    return _EMAIL.sub("" if mode is Handling.REMOVE else EMAIL_TOKEN, text)


def handle_emoji(text: str, mode: Handling) -> str:
    if mode is Handling.KEEP:
        return text
    return _EMOJI.sub("" if mode is Handling.REMOVE else " ", text)


def fix_mojibake(text: str) -> str:
    """Repair text that was decoded with the wrong codec.

    Cyrillic is the usual victim: UTF-8 bytes read as cp1251 produce the
    familiar `ÐšÐ°Ñ€Ð°ÐºÐ°Ð»Ð¿Ð°Ðº` sequences. ftfy detects and reverses these.
    """
    try:
        import ftfy
    except ImportError:
        logger.debug("ftfy not installed; skipping mojibake repair")
        return text
    return str(ftfy.fix_text(text, normalization=None))


def normalize_text(text: str, config: NormalizeConfig | None = None) -> str:
    """Apply the full normalisation pipeline.

    Order matters: mojibake repair must run before unicode normalisation
    (it needs the broken byte sequences intact), and whitespace collapsing
    must run last (earlier steps create runs of spaces when they remove things).
    """
    if not text:
        return text

    cfg = config or NormalizeConfig()

    if cfg.fix_mojibake:
        text = fix_mojibake(text)
    if cfg.strip_control:
        text = strip_control_chars(text)

    text = unicodedata.normalize(cfg.unicode_form, text)

    if cfg.fold_quotes:
        text = fold_quotes(text)
    if cfg.fold_dashes:
        text = fold_dashes(text)

    text = handle_emails(text, cfg.emails)
    text = handle_urls(text, cfg.urls)
    text = handle_emoji(text, cfg.emoji)

    if cfg.collapse_whitespace:
        text = normalize_spaces(text, max_newlines=cfg.max_consecutive_newlines)

    return text
