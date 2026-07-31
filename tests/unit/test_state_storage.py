"""Tests for the resumable crawl state and the content-addressed blob store."""

from __future__ import annotations

from pathlib import Path

from qaraqalpaqmind.crawlers.core.state import CrawlState, UrlStatus, url_hash
from qaraqalpaqmind.crawlers.core.storage import RawStore

SOURCE = "news_example"


def _state(tmp_path: Path) -> CrawlState:
    return CrawlState(tmp_path / "crawl.db")


def test_add_urls_is_idempotent(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        first = state.add_urls(SOURCE, ["https://a.uz/1", "https://a.uz/2"])
        second = state.add_urls(SOURCE, ["https://a.uz/2", "https://a.uz/3"])
        assert (first, second) == (2, 1)
        assert state.stats(SOURCE) == {UrlStatus.PENDING.value: 3}


def test_frontier_hands_out_shallowest_first(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        state.add_urls(SOURCE, ["https://a.uz/deep"], depth=2)
        state.add_urls(SOURCE, ["https://a.uz/shallow"], depth=0)
        assert [r.url for r in state.next_pending(SOURCE)] == [
            "https://a.uz/shallow",
            "https://a.uz/deep",
        ]


def test_marking_moves_urls_out_of_the_frontier(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        state.add_urls(SOURCE, ["https://a.uz/1", "https://a.uz/2", "https://a.uz/3"])
        state.mark_fetched(
            url_hash("https://a.uz/1"),
            http_status=200,
            content_hash="abc",
            content_path="news_example/ab/abc.html",
            content_type="text/html",
        )
        state.mark_failed(url_hash("https://a.uz/2"), error="HTTP 500", http_status=500)
        state.mark_skipped(url_hash("https://a.uz/3"), reason="robots.txt denied")

        assert state.stats(SOURCE) == {
            UrlStatus.FETCHED.value: 1,
            UrlStatus.FAILED.value: 1,
            UrlStatus.SKIPPED.value: 1,
        }
        assert state.next_pending(SOURCE) == []


def test_state_survives_reopening(tmp_path: Path) -> None:
    # This is the whole point of the module: Ctrl-C then re-run must resume.
    db = tmp_path / "crawl.db"
    with CrawlState(db) as state:
        state.add_urls(SOURCE, ["https://a.uz/1", "https://a.uz/2"])
        state.mark_fetched(
            url_hash("https://a.uz/1"),
            http_status=200,
            content_hash="abc",
            content_path="p",
            content_type="text/html",
        )

    with CrawlState(db) as reopened:
        assert [r.url for r in reopened.next_pending(SOURCE)] == ["https://a.uz/2"]
        assert reopened.stats(SOURCE)[UrlStatus.FETCHED.value] == 1


def test_retry_failed_respects_the_attempt_ceiling(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        state.add_urls(SOURCE, ["https://a.uz/1"])
        for _ in range(3):
            state.mark_failed(url_hash("https://a.uz/1"), error="boom")

        assert state.retry_failed(SOURCE, max_attempts=3) == 0
        assert state.retry_failed(SOURCE, max_attempts=4) == 1
        assert [r.url for r in state.next_pending(SOURCE)] == ["https://a.uz/1"]


def test_known_reports_existing_urls(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        state.add_urls(SOURCE, ["https://a.uz/1"])
        assert state.known(["https://a.uz/1", "https://a.uz/2"]) == {"https://a.uz/1"}


def test_iter_fetched_yields_provenance(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        state.add_urls(SOURCE, ["https://a.uz/1"])
        state.mark_fetched(
            url_hash("https://a.uz/1"),
            http_status=200,
            content_hash="abc",
            content_path="news_example/ab/abc.html",
            content_type="text/html",
        )
        rows = list(state.iter_fetched(SOURCE))
        assert len(rows) == 1
        assert rows[0]["url"] == "https://a.uz/1"
        assert rows[0]["content_path"] == "news_example/ab/abc.html"
        assert rows[0]["fetched_at"]


def test_sources_are_isolated(tmp_path: Path) -> None:
    with _state(tmp_path) as state:
        state.add_urls("source_a", ["https://a.uz/1"])
        state.add_urls("source_b", ["https://b.uz/1"])
        assert len(state.next_pending("source_a")) == 1
        assert state.stats("source_b") == {UrlStatus.PENDING.value: 1}


# --- storage --------------------------------------------------------------


def test_store_is_content_addressed_and_sharded(tmp_path: Path) -> None:
    store = RawStore(SOURCE, root=tmp_path)
    blob = store.store(b"<html>salawmat</html>", "text/html; charset=utf-8")

    assert blob.was_new
    assert blob.absolute_path.exists()
    assert blob.absolute_path.suffix == ".html"
    # Sharded by the first two hex chars, so no directory holds 30k files.
    assert blob.absolute_path.parent.name == blob.content_hash[:2]
    assert blob.relative_path.startswith(f"{SOURCE}/")


def test_identical_bytes_are_stored_once(tmp_path: Path) -> None:
    store = RawStore(SOURCE, root=tmp_path)
    first = store.store(b"same", "text/html")
    second = store.store(b"same", "text/html")

    assert first.content_hash == second.content_hash
    assert first.was_new and not second.was_new
    assert len(list(store.root.rglob("*.html"))) == 1


def test_stored_bytes_round_trip(tmp_path: Path) -> None:
    store = RawStore(SOURCE, root=tmp_path)
    payload = "Qaraqalpaqstan Respublikası".encode()
    blob = store.store(payload, "text/html")
    assert store.read(blob.relative_path) == payload


def test_extension_follows_content_type(tmp_path: Path) -> None:
    store = RawStore(SOURCE, root=tmp_path)
    assert store.store(b"%PDF-1.4", "application/pdf").absolute_path.suffix == ".pdf"
    assert store.store(b"<x/>", "application/xml").absolute_path.suffix == ".xml"
    assert store.store(b"\x00\x01", "image/webp").absolute_path.suffix == ".bin"


def test_no_partial_files_are_left_behind(tmp_path: Path) -> None:
    store = RawStore(SOURCE, root=tmp_path)
    store.store(b"payload", "text/html")
    assert list(store.root.rglob("*.part")) == []
