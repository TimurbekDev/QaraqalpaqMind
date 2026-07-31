"""Tests for normalisation and quality filtering.

Invisible characters are constructed with `chr()` rather than written into the
source. Typing them literally puts unprintable bytes in the file, where no
reviewer can see them - an earlier draft of this file embedded a real null byte
and stopped parsing entirely.
"""

from __future__ import annotations

import pytest

from qaraqalpaqmind.cleaning.filters import (
    FilterConfig,
    Flag,
    assess,
    compute_stats,
    should_keep,
)
from qaraqalpaqmind.cleaning.normalize import (
    EMAIL_TOKEN,
    URL_TOKEN,
    Handling,
    NormalizeConfig,
    fold_quotes,
    normalize_spaces,
    normalize_text,
    strip_control_chars,
)

GOOD = (
    "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp ótti. "
    "Sessiyada mámleketlik hám jámiyetlik ómirdiń áhmiyetli máseleleri boyınsha "
    "qararlar qabıl etildi. Deputatlar tárepinen bir neshe nızam joybarları "
    "dodalanıp, olar boyınsha tiyisli sheshimler qabıllandı."
)

ZERO_WIDTH_SPACE = chr(0x200B)
ZERO_WIDTH_NON_JOINER = chr(0x200C)
LEFT_TO_RIGHT_MARK = chr(0x200E)
BYTE_ORDER_MARK = chr(0xFEFF)
WORD_JOINER = chr(0x2060)
NULL = chr(0x00)
VERTICAL_TAB = chr(0x0B)

NO_BREAK_SPACE = chr(0x00A0)
EM_SPACE = chr(0x2003)
NARROW_NO_BREAK_SPACE = chr(0x202F)
IDEOGRAPHIC_SPACE = chr(0x3000)


# --- normalisation --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "char"),
    [
        ("zero-width space", ZERO_WIDTH_SPACE),
        ("zero-width non-joiner", ZERO_WIDTH_NON_JOINER),
        ("left-to-right mark", LEFT_TO_RIGHT_MARK),
        ("byte-order mark", BYTE_ORDER_MARK),
        ("word joiner", WORD_JOINER),
        ("null", NULL),
        ("vertical tab", VERTICAL_TAB),
    ],
)
def test_invisible_characters_are_stripped(name: str, char: str) -> None:
    # These survive every other cleaning step while being invisible, so two
    # strings that read identically hash differently and dedup keeps both.
    assert strip_control_chars(f"bir{char}eki") == "bireki", name


def test_tabs_and_newlines_survive() -> None:
    assert strip_control_chars("a\nb") == "a\nb"
    assert strip_control_chars("a\tb") == "a\tb"


def test_quote_folding_covers_the_conventions_in_the_corpus() -> None:
    # Karakalpak sources mix Russian guillemets, Word's curly quotes and ASCII,
    # frequently within one document.
    assert fold_quotes(chr(0xAB) + "soz" + chr(0xBB)) == '"soz"'
    assert fold_quotes(chr(0x201C) + "soz" + chr(0x201D)) == '"soz"'
    assert fold_quotes(chr(0x201E) + "soz" + chr(0x201C)) == '"soz"'
    assert fold_quotes(chr(0x2018) + "soz" + chr(0x2019)) == "'soz'"


def test_whitespace_normalisation_preserves_paragraphs() -> None:
    assert normalize_spaces("bir     eki") == "bir eki"
    assert normalize_spaces("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize_spaces("a\n\nb", max_newlines=1) == "a\nb"
    assert normalize_spaces("  padded  ") == "padded"


@pytest.mark.parametrize(
    "space", [NO_BREAK_SPACE, EM_SPACE, NARROW_NO_BREAK_SPACE, IDEOGRAPHIC_SPACE]
)
def test_unicode_spaces_fold_to_ascii(space: str) -> None:
    # NBSP is the common one in pasted text and is invisible in a diff.
    assert normalize_spaces(f"bir{space}eki") == "bir eki"


def test_emails_are_masked_by_default() -> None:
    # Personal data with no training value.
    out = normalize_text("Baylanıs: info@ndpi.uz hám admin@karsu.uz")
    assert EMAIL_TOKEN in out
    assert "info@ndpi.uz" not in out


def test_urls_are_kept_by_default_and_maskable() -> None:
    text = "Derek https://kknews.uz/qq/1 boyınsha"
    assert "https://kknews.uz/qq/1" in normalize_text(text)
    masked = normalize_text(text, NormalizeConfig(urls=Handling.MASK))
    assert URL_TOKEN in masked
    assert "kknews" not in masked


def test_emoji_handling_is_configurable() -> None:
    wave = chr(0x1F44B)
    text = f"Salawmat {wave} dúnya"
    assert wave in normalize_text(text)
    assert wave not in normalize_text(text, NormalizeConfig(emoji=Handling.REMOVE))


def test_normalisation_is_idempotent() -> None:
    once = normalize_text(GOOD)
    assert normalize_text(once) == once


def test_normalisation_preserves_karakalpak_letters() -> None:
    # The acute-accented letters are exactly what a careless NFKD or an
    # over-eager accent stripper would destroy.
    out = normalize_text(GOOD + " úlken túrkiy")
    for letter in "áǵńóúı":
        assert letter in out, letter
    assert "Joqarǵı Keńesiniń" in out


def test_empty_input() -> None:
    assert normalize_text("") == ""


# --- statistics -----------------------------------------------------------


def test_stats_on_real_prose() -> None:
    stats = compute_stats(GOOD)
    assert stats.words > 30
    assert 5.0 < stats.mean_word_length < 9.0  # Karakalpak sits near 7
    assert stats.alpha_word_fraction > 0.9
    assert stats.ends_with_sentence


def test_stats_on_empty_text() -> None:
    stats = compute_stats("")
    assert stats.chars == 0
    assert stats.words == 0
    assert not stats.ends_with_sentence


# --- filtering ------------------------------------------------------------


def test_good_document_is_kept() -> None:
    result = assess(GOOD)
    assert not result.rejected
    assert result.score >= 0.9, result.flags
    assert should_keep(result)


def test_agglutinative_documents_are_not_rejected() -> None:
    # Gopher's 10-character ceiling sits inside this corpus's p99 of 9.6, so
    # importing it would discard the most morphologically dense Karakalpak -
    # exactly the material the model most needs. This sample has a mean word
    # length of ~9.5, which is realistic for legislative prose.
    dense = (
        "Qaraqalpaqstan Respublikasınıń mámleketlik basqarıw uyımlarınıń "
        "xızmetkerleriniń kásiplik tayarlıǵın jetilistiriw máselelerine "
        "baǵıshlanǵan mákemelerara jıynalıs bolıp ótkerildi. "
    ) * 3
    result = assess(dense, FilterConfig(check_language=False))
    assert not result.rejected, (result.flags, result.stats)


@pytest.mark.parametrize(
    ("text", "flag"),
    [
        ("qısqa gáp", Flag.TOO_FEW_WORDS),
        ("xxxxx", Flag.TOO_SHORT),
        ("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16", Flag.LOW_ALPHA),
        ("### | ### | ### | ### | ### | ### | ###", Flag.HIGH_SYMBOL),
        ("a a a a a a a a a a a a a a a a a a a a", Flag.WORD_LENGTH_OUTLIER),
    ],
)
def test_unusable_text_is_rejected(text: str, flag: Flag) -> None:
    result = assess(text)
    assert result.rejected
    assert flag in result.flags


def test_all_rejection_reasons_are_reported() -> None:
    # Symbol soup trips several checks at once. Reporting only the first would
    # misdescribe why the corpus lost the document, and these counts are the
    # evidence used to tune the thresholds.
    result = assess("### | ### | ### | ### | ### | ### | ###")
    assert {Flag.TOO_FEW_WORDS, Flag.LOW_ALPHA, Flag.HIGH_SYMBOL} <= set(result.flags)


def test_spam_is_rejected() -> None:
    assert assess("Buy viagra now! " + GOOD).rejected
    assert assess("a" * 30 + " " + GOOD).rejected


def test_boilerplate_is_flagged_not_rejected() -> None:
    # Recoverable with a different extractor, so it loses points rather than
    # disappearing. On a 25M-token corpus, discarding recoverable data is the
    # expensive mistake.
    nav = "Bas bet\nJańalıqlar\nBaylanıs\nHújjetler\nIzlew"
    result = assess(nav, FilterConfig(check_language=False))
    assert Flag.BOILERPLATE in result.flags
    assert not result.rejected


def test_repetition_is_penalised() -> None:
    repeated = ("Bul qatar tákirarlanadı hám tákirarlanadı.\n" * 12).strip()
    result = assess(repeated, FilterConfig(check_language=False))
    assert Flag.REPETITIVE_LINES in result.flags
    assert result.score < 0.9


def test_fallback_extraction_costs_points() -> None:
    plain = assess(GOOD)
    fallback = assess(GOOD, extractor="fallback")
    assert Flag.FALLBACK_EXTRACTION in fallback.flags
    assert fallback.score < plain.score


def test_non_karakalpak_is_flagged() -> None:
    russian = (
        "Состоялось очередное заседание Сената Олий Мажлиса Республики Узбекистан. "
        "На заседании обсуждались важные вопросы государственной жизни страны."
    )
    result = assess(russian)
    assert Flag.NOT_KARAKALPAK in result.flags
    assert result.score < 0.7


def test_language_check_can_be_disabled() -> None:
    english = "The regular session of the Supreme Council was held in Nukus this week again."
    assert assess(english).rejected
    assert not assess(english, FilterConfig(check_language=False)).rejected


@pytest.mark.parametrize(
    "foreign",
    [
        "Group 1: A boy has a donkey and the boy is happy about that.",
        "Love Lucy 6b Write about Lucy and read this to the class.",
        "Это заседание было проведено в городе, и на нем обсуждались вопросы.",
    ],
)
def test_foreign_language_is_rejected_outright(foreign: str) -> None:
    # Real rows from dilmash whose "Karakalpak" column is English. They are too
    # short for the Karakalpak-likelihood heuristic to judge, so without this
    # check they survived on a soft penalty and would have taught the model to
    # produce English.
    result = assess(foreign)
    assert result.rejected
    assert Flag.FOREIGN_LANGUAGE in result.flags


@pytest.mark.parametrize(
    "short_kaa",
    [
        "kirzadan islengen etik",
        "Tok deregi Elektrolit Anod Katod 51-súwret.",
        "7. Informaciya jetkerip beriw tezligi degende neni túsinesiz?",
    ],
)
def test_short_karakalpak_survives_the_foreign_check(short_kaa: str) -> None:
    # These score 0.0 on the positive Karakalpak signal purely because they are
    # too short to judge. Rejecting on an *absence* of evidence would delete
    # legitimate text; only positive foreign evidence rejects.
    result = assess(short_kaa)
    assert not result.rejected, result.flags


def test_karakalpak_quoting_english_is_kept() -> None:
    # A rejection is permanent, so the foreign threshold is deliberately high.
    mixed = (
        "Qaraqalpaqstan Respublikasınıń delegaciyası xalıqaralıq konferenciyada "
        'sóylegen sóziniń temasi "the future of small languages" bolıp tabıladı '
        "hám bul máselede tiyisli usınıslar berildi."
    )
    assert not assess(mixed).rejected


def test_assessment_converts_to_the_record_quality_field() -> None:
    quality = assess(GOOD).to_quality()
    assert quality.score is not None
    assert 0.0 <= quality.score <= 1.0
    assert isinstance(quality.flags, list)


def test_threshold_is_a_caller_decision() -> None:
    result = assess(GOOD, extractor="fallback")
    assert should_keep(result, FilterConfig(min_quality_score=0.5))
    assert not should_keep(result, FilterConfig(min_quality_score=0.99))
