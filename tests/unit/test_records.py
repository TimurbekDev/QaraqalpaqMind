"""Tests for the universal record envelope and dataset manifests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from qaraqalpaqmind.common.records import Document, Quality, Script
from qaraqalpaqmind.ingest.manifest import Manifest


def _doc(**overrides: object) -> Document:
    kwargs: dict[str, object] = {
        "text": "Qaraqalpaqstan Respublikası Joqarǵı Keńesi",
        "source_id": "wiki_kaa",
        "license": "CC-BY-SA-4.0",
        "source_url": "https://kaa.wikipedia.org/wiki/Test",
    }
    return Document.create(**(kwargs | overrides))  # type: ignore[arg-type]


def test_id_is_stable_for_identical_content() -> None:
    assert _doc().id == _doc().id


def test_id_changes_with_text_or_origin() -> None:
    base = _doc()
    assert _doc(text="basqa tekst").id != base.id
    assert _doc(source_url="https://kaa.wikipedia.org/wiki/Other").id != base.id
    assert _doc(source_id="news_kknews").id != base.id


def test_blank_text_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        _doc(text="   \n\t ")


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Document(
            id="x",
            text="t",
            source_id="s",
            fetched_at=datetime.now(tz=UTC),
            license="MIT",
            unexpected="boom",  # type: ignore[call-arg]
        )


def test_provenance_defaults_are_filled() -> None:
    document = _doc()
    assert document.lang == "kaa"
    assert document.fetched_at.tzinfo is not None
    assert document.quality == Quality()


def test_counts() -> None:
    document = _doc(text="bir eki úsh")
    assert document.char_count() == 11
    assert document.word_count() == 3
    assert document.estimated_tokens() == round(11 / 3.1)


def test_script_enum_is_shared_with_the_detector() -> None:
    # There must be exactly one Script definition; a second one drifted once and
    # broke ingestion with "'other' is not a valid Script".
    from qaraqalpaqmind.preprocessing.script import Script as DetectorScript

    assert DetectorScript is Script
    assert {s.value for s in Script} >= {"latin", "cyrillic", "mixed", "other", "unknown"}


def test_round_trips_through_json() -> None:
    document = _doc(script=Script.CYRILLIC, meta={"title": "Sınaq"})
    restored = Document.model_validate(document.model_dump(mode="json"))
    assert restored == document


# --- manifests ------------------------------------------------------------


def test_manifest_hashes_the_file(tmp_path: Path) -> None:
    target = tmp_path / "corpus.jsonl"
    target.write_text('{"a":1}\n', encoding="utf-8")

    manifest = Manifest.build(
        source_id="wiki_kaa",
        path=target,
        documents=1,
        characters=10,
        words=2,
        estimated_tokens=3,
        license="CC-BY-SA-4.0",
        source_url="https://example.org",
        scripts={"latin": 1},
    )
    assert manifest.sha256 and len(manifest.sha256) == 64
    assert manifest.size_bytes == target.stat().st_size
    assert "wiki_kaa" in manifest.summary()


def test_dry_run_manifest_has_no_file_fields() -> None:
    manifest = Manifest.build(
        source_id="wiki_kaa",
        path=None,
        documents=5,
        characters=50,
        words=10,
        estimated_tokens=16,
        license="CC-BY-SA-4.0",
        source_url="https://example.org",
    )
    assert manifest.path is None
    assert manifest.sha256 is None
    assert manifest.documents == 5


def test_manifest_is_frozen() -> None:
    manifest = Manifest.build(
        source_id="s",
        path=None,
        documents=1,
        characters=1,
        words=1,
        estimated_tokens=1,
        license="MIT",
        source_url="https://example.org",
    )
    with pytest.raises(ValidationError):
        manifest.documents = 2  # type: ignore[misc]
