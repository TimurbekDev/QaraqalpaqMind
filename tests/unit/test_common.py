"""Smoke tests for the shared foundation. No network, no GPU, runs in <1s."""

from __future__ import annotations

from pathlib import Path

import pytest

from qaraqalpaqmind.common import (
    PROJECT_ROOT,
    count_lines,
    read_jsonl,
    text_sha1,
    write_jsonl,
)
from qaraqalpaqmind.common.config import StrictModel, load_config, load_raw


def test_project_root_points_at_repo() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


_RECORDS = [
    {"id": "1", "text": "Qaraqalpaqstan Respublikası", "lang": "kaa"},
    {"id": "2", "text": "Қарақалпақстан Республикасы", "script": "cyrillic"},
]


@pytest.mark.parametrize("name", ["sample.jsonl", "sample.jsonl.gz", "sample.jsonl.zst"])
def test_jsonl_roundtrip_preserves_unicode(tmp_path: Path, name: str) -> None:
    # Every compression path must be exercised. Writing zstd worked long before
    # reading it did, because its stream_reader has no readline().
    target = tmp_path / name
    assert write_jsonl(target, _RECORDS) == 2
    assert list(read_jsonl(target)) == _RECORDS
    assert count_lines(target) == 2


@pytest.mark.parametrize("name", ["big.jsonl.zst", "big.jsonl.gz"])
def test_compressed_roundtrip_spans_buffer_boundaries(tmp_path: Path, name: str) -> None:
    # A handful of records fits in one buffer and can hide framing bugs.
    records = [{"i": i, "text": "Qaraqalpaqstan Respublikası " * 20} for i in range(5_000)]
    target = tmp_path / name
    assert write_jsonl(target, records) == 5_000
    assert list(read_jsonl(target)) == records


def test_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    target = tmp_path / "broken.jsonl"
    target.write_text('{"a": 1}\nnot json\n\n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in read_jsonl(target)] == [1, 2]


def test_text_hash_is_stable() -> None:
    assert text_sha1("salawmat") == text_sha1("salawmat")
    assert text_sha1("salawmat") != text_sha1("salawmat ")


class _Demo(StrictModel):
    name: str
    retries: int = 3


def test_config_inheritance_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "base.yaml").write_text("name: base\nretries: 1\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "_base_: base.yaml\nname: ${QM_TEST_NAME:fallback}\n", encoding="utf-8"
    )

    cfg = load_config(tmp_path / "child.yaml", _Demo)
    assert (cfg.name, cfg.retries) == ("fallback", 1)

    monkeypatch.setenv("QM_TEST_NAME", "overridden")
    assert load_config(tmp_path / "child.yaml", _Demo).name == "overridden"


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("name: x\nretrys: 9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retrys"):
        load_config(tmp_path / "bad.yaml", _Demo)


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.yaml")
