"""Tests for Karakalpak script / orthography detection.

The samples are the same sentence rendered in each language that the detector
must tell apart. Karakalpak's neighbours share most of its alphabet, so these
negatives are the whole point of the module.
"""

from __future__ import annotations

import pytest

from qaraqalpaqmind.preprocessing.script import (
    Orthography,
    Script,
    analyse,
    detect_orthography,
    detect_script,
    karakalpak_score,
)

KAA_LATIN = (
    "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp ótti. "
    "Sessiyada mámleketlik hám jámiyetlik ómirdiń áhmiyetli máseleleri boyınsha "
    "qararlar qabıl etildi."
)

KAA_CYRILLIC = (
    "Қарақалпақстан Республикасы Жоқарғы Кеңесиниң гезекли сессиясы болып өтти. "
    "Сессияда мәмлекетлик ҳәм жәмийетлик өмирдиң әҳмийетли мәселелери бойынша "
    "қарарлар қабыл етилди."
)

UZB_LATIN = (
    "Oʻzbekiston Respublikasi Oliy Majlisi Senati yigʻilishi boʻlib oʻtdi. "
    "Yigʻilishda davlat va jamiyat hayotining muhim masalalari boʻyicha "
    "qarorlar qabul qilindi."
)

UZB_CYRILLIC = (
    "Ўзбекистон Республикаси Олий Мажлиси Сенати йиғилиши бўлиб ўтди. "
    "Йиғилишда давлат ва жамият ҳаётининг муҳим масалалари бўйича "
    "қарорлар қабул қилинди."
)

KAZ_CYRILLIC = (
    "Қазақстан Республикасының Парламенті Сенатының кезекті отырысы өтті. "
    "Отырыста мемлекеттік және қоғамдық өмірдің маңызды мәселелері бойынша "
    "шешімдер қабылданды."
)

RUS = "Состоялось очередное заседание Сената Олий Мажлиса Республики Узбекистан."

ENG = "The regular session of the Supreme Council was held in Nukus this week."


def test_script_detection() -> None:
    assert detect_script(KAA_LATIN) is Script.LATIN
    assert detect_script(KAA_CYRILLIC) is Script.CYRILLIC
    assert detect_script(RUS) is Script.CYRILLIC
    assert detect_script("") is Script.UNKNOWN
    assert detect_script("　　　") is Script.UNKNOWN


def test_orthography_detection() -> None:
    assert detect_orthography(KAA_LATIN) is Orthography.LATIN_2016
    assert detect_orthography(KAA_CYRILLIC) is Orthography.CYRILLIC
    assert detect_orthography(UZB_LATIN) is Orthography.LATIN_2009
    assert detect_orthography("Ózbekistan äyel ökimet üshin") is Orthography.LATIN_1994


def test_karakalpak_is_accepted() -> None:
    for sample in (KAA_LATIN, KAA_CYRILLIC):
        report = analyse(sample)
        assert report.is_probably_karakalpak, report
        assert report.karakalpak_score >= 0.8


@pytest.mark.parametrize(
    ("name", "sample"),
    [("uzb_latin", UZB_LATIN), ("uzb_cyrillic", UZB_CYRILLIC), ("kaz", KAZ_CYRILLIC), ("rus", RUS)],
)
def test_neighbours_are_rejected(name: str, sample: str) -> None:
    report = analyse(sample)
    assert not report.is_probably_karakalpak, f"{name} falsely accepted: {report}"


def test_plain_latin_scores_zero() -> None:
    # Regression: capital `I` was once in the marker set, which gave every
    # English page a non-zero Karakalpak score.
    assert karakalpak_score(ENG) == 0.0
    assert karakalpak_score("I INSIST THIS IS ENGLISH TEXT WITH MANY CAPITAL I LETTERS") == 0.0


def test_marker_only_text_stays_below_the_gate() -> None:
    # Distinctive letters with no Karakalpak function words must not pass.
    markers_only = "ǵǵǵ áá ńń óó úú ıı " * 10
    assert analyse(markers_only).karakalpak_score <= 0.45


def test_empty_input_is_safe() -> None:
    report = analyse("")
    assert report.letters == 0
    assert report.karakalpak_score == 0.0
    assert not report.is_probably_karakalpak
