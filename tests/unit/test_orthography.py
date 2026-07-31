"""Tests for Karakalpak orthography unification.

The spot-check cases below were verified against real corpus text from
sud.uz, shagalalab.com and GlotCC rather than from alphabet charts, because
the charts and the writing disagree (see the `ц` case).
"""

from __future__ import annotations

import unicodedata

import pytest

from qaraqalpaqmind.preprocessing.orthography import (
    CYRILLIC_HOMOGLYPHS,
    CYRILLIC_TO_LATIN,
    apostrophe_to_acute,
    cyrillic_to_latin,
    fix_cyrillic_homoglyphs,
    to_latin2016,
    umlaut_to_acute,
)
from qaraqalpaqmind.preprocessing.script import Orthography, detect_orthography


@pytest.mark.parametrize(
    ("cyrillic", "latin"),
    [
        ("Қарақалпақстан Республикасы", "Qaraqalpaqstan Respublikası"),
        ("Жоқарғы Кеңеси", "Joqarǵı Keńesi"),
        ("жазыў", "jazıw"),
        ("ҳәм", "hám"),
        ("мәмлекетлик", "mámleketlik"),
        ("бойынша", "boyınsha"),
        ("әлипбе", "álipbe"),
        ("өмир", "ómir"),
        ("үлкен", "úlken"),
        ("тәжирийбе", "tájiriybe"),
    ],
)
def test_cyrillic_transliteration(cyrillic: str, latin: str) -> None:
    assert cyrillic_to_latin(cyrillic) == latin


def test_cyrillic_c_not_ts() -> None:
    # Measured, not assumed. The natively-Latin sud.uz locale writes
    # korrupciya/konstituciya/instanciya and never the "ts" spellings, so
    # several published alphabet charts are wrong about this.
    assert cyrillic_to_latin("коррупция") == "korrupciya"
    assert cyrillic_to_latin("конституциялық") == "konstituciyalıq"
    assert cyrillic_to_latin("инстанция") == "instanciya"


@pytest.mark.parametrize(
    ("typed", "standard"),
    [("ӯ", "ў"), ("ѳ", "ө"), ("ӊ", "ң"), ("Ѳ", "Ө"), ("Ӊ", "Ң")],
)
def test_homoglyphs_are_folded(typed: str, standard: str) -> None:
    # 437 of these were found in 142 real documents; untouched they survive
    # transliteration and corrupt the surrounding word.
    assert fix_cyrillic_homoglyphs(typed) == standard


def test_homoglyphs_transliterate_correctly() -> None:
    assert cyrillic_to_latin("Ассалаӯма") == "Assalawma"
    assert cyrillic_to_latin("ѳмир") == "ómir"
    assert cyrillic_to_latin("тѳлеӊ") == "tóleń"


def test_apostrophe_orthography() -> None:
    # Real GlotCC text: "XII a'sirdegi tu'rk ... ko'rkem so'z sheberlerinin'"
    assert apostrophe_to_acute("a'sirdegi") == "ásirdegi"
    assert apostrophe_to_acute("tu'rk") == "túrk"
    assert apostrophe_to_acute("ko'rkem so'z") == "kórkem sóz"
    assert apostrophe_to_acute("sheberlerinin'") == "sheberleriniń"
    assert apostrophe_to_acute("g'arezsiz") == "ǵarezsiz"


@pytest.mark.parametrize("mark", ["'", "‘", "’", "ʼ", "ʻ", "`"])
def test_all_apostrophe_variants_are_handled(mark: str) -> None:
    # Uzbek keyboards and word processors emit all of these for one letter.
    assert apostrophe_to_acute(f"so{mark}z") == "sóz"


@pytest.mark.parametrize(
    "english",
    ["don't", "isn't", "can't", "won't", "it's", "we're", "I've", "I'm", "he'd", "they'll"],
)
def test_english_contractions_are_not_mangled(english: str) -> None:
    # The corpus contains real English: dilmash carries 100k English sentences
    # and quoted English appears throughout the web sources. Without the
    # contraction guard, "don't" became "dońt".
    assert apostrophe_to_acute(english) == english


def test_apostrophe_after_unaffected_letters_is_left_alone() -> None:
    assert apostrophe_to_acute("s'more") == "s'more"
    assert apostrophe_to_acute("rock 'n roll") == "rock 'n roll"


def test_known_residual_cost_of_the_heuristic() -> None:
    # Documented, accepted: Irish surnames lose their apostrophe. Karakalpak
    # text has far more `so'z`-shaped words than O'-names.
    assert apostrophe_to_acute("O'Brien") == "ÓBrien"


def test_umlaut_orthography() -> None:
    assert umlaut_to_acute("äyel ökimet üshin") == "áyel ókimet úshin"
    assert umlaut_to_acute("ñ ģ") == "ń ǵ"


def test_to_latin2016_dispatches_on_detected_orthography() -> None:
    assert to_latin2016("Қарақалпақстан").startswith("Qaraqalpaqstan")
    assert to_latin2016("a'sirdegi tu'rk tilles xalıqlar") == "ásirdegi túrk tilles xalıqlar"
    assert to_latin2016("Qaraqalpaqstan Respublikası") == "Qaraqalpaqstan Respublikası"


def test_to_latin2016_can_be_forced() -> None:
    # Short samples get misdetected; a source known to use one convention
    # should be able to say so.
    assert to_latin2016("so'z", orthography=Orthography.LATIN_2009) == "sóz"


def test_output_is_nfc_composed() -> None:
    # Without NFC, `á` may be a + combining acute, which hashes differently
    # from the composed form and silently defeats deduplication.
    result = to_latin2016("ә")
    assert result == unicodedata.normalize("NFC", result)
    assert len(result) == 1


def test_mixed_script_survives() -> None:
    converted = cyrillic_to_latin("Қарақалпақ https://kaa.wikipedia.org tili")
    assert "https://kaa.wikipedia.org" in converted
    assert converted.startswith("Qaraqalpaq")


def test_empty_and_non_karakalpak_input() -> None:
    assert to_latin2016("") == ""
    assert to_latin2016("plain english text") == "plain english text"
    assert cyrillic_to_latin("12345 !?") == "12345 !?"


def test_mapping_tables_are_internally_consistent() -> None:
    # Every homoglyph must resolve to a letter the main table can handle.
    for standard in CYRILLIC_HOMOGLYPHS.values():
        assert standard in CYRILLIC_TO_LATIN, f"{standard!r} is unmapped"


def test_detect_then_convert_round_trip_for_the_current_standard() -> None:
    text = "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń sessiyası"
    assert detect_orthography(text) is Orthography.LATIN_2016
    assert to_latin2016(text) == text
