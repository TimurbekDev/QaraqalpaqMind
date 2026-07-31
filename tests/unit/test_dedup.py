"""Tests for exact and near-duplicate detection."""

from __future__ import annotations

import pytest

from qaraqalpaqmind.dedup.blocklist import build_blocklist, is_contaminated
from qaraqalpaqmind.dedup.exact import canonical_form, content_hash, find_exact_duplicates
from qaraqalpaqmind.dedup.minhash import (
    MinHashConfig,
    duplicate_ids,
    find_near_duplicates,
    shingles,
)

ARTICLE = (
    "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp ótti. "
    "Sessiyada mámleketlik hám jámiyetlik ómirdiń áhmiyetli máseleleri boyınsha "
    "qararlar qabıl etildi. Deputatlar tárepinen bir neshe nızam joybarları "
    "dodalanıp, olar boyınsha tiyisli sheshimler qabıllandı. Jıynaq juwmaǵında "
    "tiyisli tapsırmalar berildi hám olardıń orınlanıwı baqlawǵa alındı."
)


# --- canonical form and exact hashing -------------------------------------


def test_formatting_differences_collapse() -> None:
    assert content_hash("Bul  gáp.") == content_hash("bul gáp")
    assert content_hash("Bul\ngáp!") == content_hash("BUL GÁP")


def test_accented_letters_stay_distinct() -> None:
    # `sóz` and `soz` are different words; folding them would merge documents
    # that are genuinely different.
    assert content_hash("sóz") != content_hash("soz")
    assert canonical_form("Joqarǵı") == "joqarǵı"


def test_sentence_case_matches_lower_case() -> None:
    # The common real-world case: the same text as a heading and as body copy.
    assert content_hash("Islam tariyxı boyınsha") == content_hash("islam tariyxı boyınsha")


def test_dotted_and_dotless_i_stay_distinct() -> None:
    # They are different letters in Karakalpak, as in Turkish.
    assert canonical_form("ı") != canonical_form("i")


def test_all_caps_is_a_documented_limit_not_a_silent_bug() -> None:
    # `.upper()` maps BOTH `i` and `ı` to `I`, so uppercase Karakalpak is
    # genuinely ambiguous and no folding recovers it. Mapping `I` back to `ı`
    # was tried: it fixes rare all-caps documents while breaking the common
    # case above. All-caps duplicates are left to the near-duplicate pass.
    with_dotless = "respublikası"
    assert canonical_form(with_dotless.upper()) != canonical_form(with_dotless)


def test_hash_is_stable_and_short() -> None:
    assert content_hash(ARTICLE) == content_hash(ARTICLE)
    assert len(content_hash(ARTICLE)) == 32


def test_exact_duplicates_keep_the_first_seen() -> None:
    # The caller sorts by source quality first, so "first" means "best source":
    # the dilmash copy of a sentence survives and the GlotCC copy goes.
    reformatted = "  " + ARTICLE.replace(". ", ".\n\n") + "  "
    winners, duplicates = find_exact_duplicates(
        [("best", ARTICLE), ("worse", reformatted), ("other", "Basqa tekst pútkilley bar.")]
    )
    assert winners["best"] == "best"
    assert winners["worse"] == "best"
    assert winners["other"] == "other"
    assert duplicates["best"] == ["worse"]


def test_empty_input() -> None:
    winners, duplicates = find_exact_duplicates([])
    assert winners == {}
    assert duplicates == {}


# --- shingling ------------------------------------------------------------


def test_long_documents_use_word_shingles() -> None:
    result = shingles(ARTICLE)
    assert len(result) > 20
    assert any(" " in s for s in result)


def test_short_documents_use_character_shingles() -> None:
    # Two thirds of this corpus is sentence-level. A 5-word shingle over an
    # 8-word sentence yields four shingles, which is far too few to estimate
    # similarity from; character shingles give ~40.
    short = "Bul qısqa gáp bolıp tabıladı."
    result = shingles(short)
    assert len(result) > 15, result
    assert all(len(s) <= MinHashConfig().char_shingle for s in result)


def test_shingles_on_tiny_and_empty_input() -> None:
    assert shingles("") == set()
    assert shingles("bir") == {"bir"}


# --- near-duplicate detection ---------------------------------------------


def test_identical_documents_cluster() -> None:
    clusters = find_near_duplicates([("a", ARTICLE), ("b", ARTICLE)])
    assert duplicate_ids(clusters) == {"b"}


def test_documents_differing_by_a_sentence_cluster() -> None:
    # The realistic case: a wiki page re-scraped a year later, or a ruling
    # republished with one paragraph added.
    extended = ARTICLE + " Sonıń menen birge, qosımsha máseleler dodalandı."
    clusters = find_near_duplicates([("original", ARTICLE), ("extended", extended)])
    assert duplicate_ids(clusters) == {"extended"}


def test_unrelated_documents_do_not_cluster() -> None:
    other = (
        "Ájiniyaz atındaǵı Nókis mámleketlik pedagogikalıq institutında jańa oqıw "
        "jılına tayarlıq jumısları juwmaqlanbaqta. Studentler ushın jatasxana "
        "jaǵdayları jaqsılandı hám kitapxana qorı toltırıldı."
    )
    clusters = find_near_duplicates([("a", ARTICLE), ("b", other)])
    assert duplicate_ids(clusters) == set()


def test_first_document_wins_the_cluster() -> None:
    clusters = find_near_duplicates([("keep_me", ARTICLE), ("drop_me", ARTICLE)])
    assert len(clusters) == 1
    assert clusters[0].winner == "keep_me"
    assert clusters[0].size == 2


def test_chains_collapse_into_one_cluster() -> None:
    a = ARTICLE
    b = ARTICLE + " Qosımsha bir gáp bar."
    c = ARTICLE + " Qosımsha bir gáp bar hám taǵı bir gáp."
    clusters = find_near_duplicates([("a", a), ("b", b), ("c", c)])
    assert len(duplicate_ids(clusters)) == 2


def test_threshold_is_configurable() -> None:
    partly_similar = ARTICLE[: len(ARTICLE) // 2] + " Pútkilley basqa mazmun jazıldı bul jerde."
    strict = find_near_duplicates(
        [("a", ARTICLE), ("b", partly_similar)], MinHashConfig(threshold=0.95)
    )
    loose = find_near_duplicates(
        [("a", ARTICLE), ("b", partly_similar)], MinHashConfig(threshold=0.30)
    )
    assert len(duplicate_ids(strict)) <= len(duplicate_ids(loose))


def test_results_are_deterministic() -> None:
    # A fixed seed matters: a rerun must reproduce the same corpus.
    pairs = [("a", ARTICLE), ("b", ARTICLE + " Qosımsha."), ("c", "Basqa tekst pútkilley bar.")]
    assert duplicate_ids(find_near_duplicates(pairs)) == duplicate_ids(find_near_duplicates(pairs))


# --- benchmark contamination ---------------------------------------------


def test_blocklist_catches_benchmark_sentences(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from qaraqalpaqmind.common.io import write_jsonl

    held_out = tmp_path / "flores.jsonl"
    write_jsonl(
        held_out,
        [{"text": "Bul benchmark gápi bolıp tabıladı hám qorǵalıwı kerek."}],
    )

    blocked = build_blocklist([held_out])
    assert blocked

    training = "Kirisiw gápi. Bul benchmark gápi bolıp tabıladı hám qorǵalıwı kerek. Aqırı."
    assert is_contaminated(training, blocked)
    assert not is_contaminated("Pútkilley baylanıssız mazmun jazılǵan bul jerde.", blocked)


def test_short_sentences_are_not_blocked(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Short sentences recur naturally; blocking them would delete legitimate
    # text without protecting the benchmark.
    from qaraqalpaqmind.common.io import write_jsonl

    held_out = tmp_path / "flores.jsonl"
    write_jsonl(held_out, [{"text": "Awa. Yaq."}])
    assert build_blocklist([held_out]) == set()


def test_missing_blocklist_blocks_nothing() -> None:
    assert not is_contaminated(ARTICLE, set())


def test_missing_held_out_file_is_survivable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert build_blocklist([tmp_path / "absent.jsonl"]) == set()


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_blank_text_is_never_contaminated(text: str) -> None:
    assert not is_contaminated(text, {"deadbeef"})
