"""Tests for the ingest layer: wikitext cleaning, HF mapping, the run loop.

No network. Wikipedia parsing is exercised against a synthetic MediaWiki dump
written to a temporary bz2 file, which is exactly the shape of the real thing.
"""

from __future__ import annotations

import bz2
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qaraqalpaqmind.common.io import read_jsonl
from qaraqalpaqmind.common.records import Document, Script
from qaraqalpaqmind.crawlers.core.registry import SourceSpec, load_registry
from qaraqalpaqmind.ingest.base import Ingester
from qaraqalpaqmind.ingest.huggingface import MAPPINGS, HFMapping, HuggingFaceIngester
from qaraqalpaqmind.ingest.wikipedia import WikipediaIngester, clean_wikitext

KAA = "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp ótti. " * 4


def _spec(**overrides: object) -> SourceSpec:
    base: dict[str, object] = {
        "id": "wiki_kaa",
        "name": "Test",
        "kind": "wiki",
        "access": "dump",
        "url": "https://example.org/",
        "legal": "open_license",
        "license": "CC-BY-SA-4.0",
        "update_frequency": "weekly",
        "est_size_mb": 1.0,
        "quality": 4,
    }
    return SourceSpec.model_validate(base | overrides)


# --- wikitext cleaning ----------------------------------------------------


def test_templates_and_links_are_stripped() -> None:
    cleaned = clean_wikitext("{{Infobox|a=1}} Ájiniyaz [[qaraqalpaq]] [[shayır|aqın]] boldı.")
    assert "Infobox" not in cleaned
    assert "[[" not in cleaned
    assert "qaraqalpaq" in cleaned
    assert "aqın" in cleaned


def test_refs_tables_files_and_categories_are_dropped() -> None:
    cleaned = clean_wikitext(
        "Tekst<ref>Derek 1</ref> dawamı.\n"
        "{| class=wikitable\n|- \n| kesteler | joq\n|}\n"
        "[[File:Sample.jpg|thumb|caption]]\n"
        "[[Category:Qaraqalpaqstan]]\n"
    )
    assert "Derek 1" not in cleaned
    assert "wikitable" not in cleaned
    assert "Sample.jpg" not in cleaned
    assert "Category" not in cleaned
    assert "Tekst" in cleaned and "dawamı" in cleaned


def test_headings_become_plain_lines() -> None:
    cleaned = clean_wikitext("== Tariyxı ==\nMazmun bar.")
    assert "==" not in cleaned
    assert "Tariyxı" in cleaned


def test_reference_sections_are_cut() -> None:
    # These sections are link lists, not language, and on a small wiki they
    # would otherwise dominate the short-article end of the corpus.
    cleaned = clean_wikitext("Mazmun bar.\n\nSiltemeler\n\nhttp://a.uz\nhttp://b.uz")
    assert "Mazmun bar." in cleaned
    assert "a.uz" not in cleaned


def test_whitespace_is_collapsed() -> None:
    assert "\n\n\n" not in clean_wikitext("bir\n\n\n\n\neki")
    assert "  " not in clean_wikitext("bir     eki")


# --- Wikipedia dump parsing ----------------------------------------------


def _dump(tmp_path: Path, pages: str) -> Path:
    xml = f"""<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
      <siteinfo><sitename>Wikipedia</sitename></siteinfo>
      {pages}
    </mediawiki>"""
    target = tmp_path / "test-pages-articles.xml.bz2"
    target.write_bytes(bz2.compress(xml.encode()))
    return target


def _page(title: str, text: str, ns: str = "0", redirect: bool = False) -> str:
    redirect_tag = '<redirect title="Elsewhere" />' if redirect else ""
    return f"""<page>
      <title>{title}</title><ns>{ns}</ns><id>1</id>{redirect_tag}
      <revision><id>2</id><text xml:space="preserve">{text}</text></revision>
    </page>"""


def test_dump_parsing_emits_articles(tmp_path: Path) -> None:
    dump = _dump(tmp_path, _page("Ájiniyaz", KAA) + _page("Ádebiyat", KAA))
    documents = list(WikipediaIngester(_spec(), dump_path=dump).documents())

    assert len(documents) == 2
    assert documents[0].meta["title"] == "Ájiniyaz"
    assert documents[0].text.startswith("Ájiniyaz")
    assert documents[0].source_url == "https://kaa.wikipedia.org/wiki/Ájiniyaz"
    assert documents[0].script is Script.LATIN
    assert documents[0].license == "CC-BY-SA-4.0"


def test_dump_parsing_skips_redirects_namespaces_and_stubs(tmp_path: Path) -> None:
    dump = _dump(
        tmp_path,
        _page("Keep", KAA)
        + _page("Redirect", KAA, redirect=True)
        + _page("Talk:Something", KAA, ns="1")
        + _page("Stub", "qısqa"),
    )
    titles = [d.meta["title"] for d in WikipediaIngester(_spec(), dump_path=dump).documents()]
    assert titles == ["Keep"]


def test_dump_parsing_respects_limit(tmp_path: Path) -> None:
    dump = _dump(tmp_path, "".join(_page(f"A{i}", KAA) for i in range(10)))
    assert len(list(WikipediaIngester(_spec(), dump_path=dump).documents(limit=3))) == 3


# --- the ingest run loop --------------------------------------------------


class _FakeIngester(Ingester):
    def __init__(self, spec: SourceSpec, target: Path, count: int = 5) -> None:
        super().__init__(spec)
        self._target = target
        self._count = count

    def output_path(self) -> Path:
        return self._target

    def documents(self, limit: int | None = None) -> Iterator[Document]:
        for index in range(self._count if limit is None else min(limit, self._count)):
            yield Document.create(
                text=f"{KAA} nomer {index}",
                source_id=self.spec.id,
                license=self.spec.license,
                script=Script.LATIN,
            )


def test_run_writes_corpus_and_counts(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl.zst"
    manifest = _FakeIngester(_spec(), target, count=5).run()

    assert target.exists()
    assert manifest.documents == 5
    assert manifest.characters > 0
    assert manifest.estimated_tokens > 0
    assert manifest.scripts == {"latin": 5}
    assert manifest.sha256 and manifest.size_bytes
    assert len(list(read_jsonl(target))) == 5


def test_run_dry_run_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl.zst"
    manifest = _FakeIngester(_spec(), target, count=5).run(dry_run=True)

    assert not target.exists()
    assert manifest.documents == 5
    assert manifest.path is None
    assert manifest.sha256 is None


def test_written_records_validate_as_documents(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl.zst"
    _FakeIngester(_spec(), target, count=3).run()
    for row in read_jsonl(target):
        assert Document.model_validate(row).source_id == "wiki_kaa"


# --- Hugging Face mapping -------------------------------------------------


def test_every_enabled_hf_source_has_a_mapping() -> None:
    # Only enabled sources need one. `nllb_kaa` is registered but disabled for
    # its CC-BY-NC licence, so we have deliberately never mapped its schema.
    enabled = {s.id for s in load_registry().enabled_sources() if s.access.value == "hf"}
    assert enabled <= set(MAPPINGS), f"missing mappings for {enabled - set(MAPPINGS)}"


def test_unmapped_hf_sources_are_disabled() -> None:
    for spec in load_registry().sources:
        if spec.access.value == "hf" and spec.id not in MAPPINGS:
            assert not spec.enabled, f"{spec.id} is enabled but has no ingest mapping"


def test_flores_mapping_is_marked_held_out_and_gated() -> None:
    mapping = MAPPINGS["flores_plus_kaa"]
    assert mapping.held_out
    assert mapping.gated


def test_pair_extraction_takes_the_karakalpak_side() -> None:
    spec = _spec(id="hf_dilmash_parallel", access="hf", kind="parallel", license="MIT")
    ingester = HuggingFaceIngester(spec)
    mapping = MAPPINGS["hf_dilmash_parallel"]
    now = datetime.now(tz=UTC)

    forward = ingester._from_pair(
        {"src_lang": "kaa_Latn", "src_sent": KAA, "tgt_lang": "eng_Latn", "tgt_sent": "English side"},
        mapping, "kaa_eng", now,
    )
    assert forward is not None
    assert forward.text.startswith("Qaraqalpaqstan")
    assert forward.meta["parallel_lang"] == "eng_Latn"

    # Karakalpak on the target side must work identically.
    reverse = ingester._from_pair(
        {"src_lang": "eng_Latn", "src_sent": "English side", "tgt_lang": "kaa_Latn", "tgt_sent": KAA},
        mapping, "kaa_eng", now,
    )
    assert reverse is not None
    assert reverse.text.startswith("Qaraqalpaqstan")


def test_pair_without_karakalpak_is_dropped() -> None:
    spec = _spec(id="hf_dilmash_parallel", access="hf", kind="parallel", license="MIT")
    result = HuggingFaceIngester(spec)._from_pair(
        {"src_lang": "eng_Latn", "src_sent": "a", "tgt_lang": "rus_Cyrl", "tgt_sent": "b"},
        MAPPINGS["hf_dilmash_parallel"], "x", datetime.now(tz=UTC),
    )
    assert result is None


def test_missing_text_column_names_the_actual_columns() -> None:
    # A silent empty corpus is the failure mode this guards against.
    mapping = HFMapping(repo="x/y", text_fields=("text",))
    with pytest.raises(KeyError, match="content"):
        HuggingFaceIngester._pick_text({"content": "hello"}, mapping)
