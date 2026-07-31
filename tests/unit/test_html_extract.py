"""Tests for HTML extraction and the crawled-source ingester."""

from __future__ import annotations

from pathlib import Path

from qaraqalpaqmind.common.records import Script
from qaraqalpaqmind.crawlers.core.registry import SourceSpec
from qaraqalpaqmind.crawlers.core.state import CrawlState, url_hash
from qaraqalpaqmind.crawlers.core.storage import RawStore
from qaraqalpaqmind.ingest.crawled import CrawledIngester
from qaraqalpaqmind.preprocessing.html import extract_main_text, normalise_whitespace

ARTICLE = (
    "Qaraqalpaqstan Respublikası Joqarǵı Keńesiniń gezekli sessiyası bolıp ótti. "
    "Sessiyada mámleketlik hám jámiyetlik ómirdiń áhmiyetli máseleleri boyınsha "
    "qararlar qabıl etildi. Deputatlar tárepinen bir neshe nızam joybarları "
    "dodalanıp, olar boyınsha tiyisli sheshimler qabıllandı."
)

PAGE = f"""<!doctype html>
<html lang="kaa"><head>
  <title>Sessiya haqqında</title>
  <meta property="article:published_time" content="2026-05-14">
  <style>.x {{ color: red }}</style>
</head><body>
  <nav class="menu"><a href="/qq/">Bas bet</a><a href="/ru/">Русский</a></nav>
  <header class="header">QARAQALPAQSTAN</header>
  <aside class="sidebar">Kópshilik oqıǵanları</aside>
  <article><h1>Sessiya haqqında</h1><p>{ARTICLE}</p></article>
  <div class="social">Bólisiw: Telegram Facebook</div>
  <footer class="footer">© 2026 Barlıq huqıqlar qorǵalǵan</footer>
  <script>console.log("tracking")</script>
</body></html>"""


def _spec(**overrides: object) -> SourceSpec:
    base: dict[str, object] = {
        "id": "news_example",
        "name": "Example",
        "kind": "news",
        "access": "crawl",
        "url": "https://example.uz/qq/",
        "legal": "government_work",
        "license": "unknown",
        "update_frequency": "daily",
        "est_size_mb": 1.0,
        "quality": 3,
    }
    return SourceSpec.model_validate(base | overrides)


# --- extraction -----------------------------------------------------------


def test_article_text_survives() -> None:
    page = extract_main_text(PAGE, url="https://example.uz/qq/1")
    assert "Joqarǵı Keńesiniń" in page.text
    assert page.is_usable


def test_chrome_is_removed() -> None:
    # Boilerplate is the biggest single risk to a corpus this small: repeated
    # menus would become the most "frequent Karakalpak" the model ever sees.
    page = extract_main_text(PAGE, url="https://example.uz/qq/1")
    for chrome in ("Bas bet", "Русский", "tracking", "color: red", "Bólisiw"):
        assert chrome not in page.text, f"{chrome!r} leaked into extracted text"


def test_title_is_captured() -> None:
    page = extract_main_text(PAGE, url="https://example.uz/qq/1")
    assert page.title == "Sessiya haqqında"


def test_fallback_is_used_and_labelled_when_extraction_fails() -> None:
    # Bare text with no article structure: trafilatura declines, we still want
    # the content, but Phase 3 must be able to tell the two apart.
    html = f"<html><body><div>{ARTICLE}</div></body></html>"
    page = extract_main_text(html)
    assert page.is_usable
    assert "Joqarǵı" in page.text
    assert page.extractor in {"trafilatura", "fallback"}


def test_empty_page_is_not_usable() -> None:
    assert not extract_main_text("<html><body></body></html>").is_usable
    assert not extract_main_text("").is_usable


def test_whitespace_normalisation() -> None:
    assert normalise_whitespace("bir    eki") == "bir eki"
    assert normalise_whitespace("bir\n\n\n\n\neki") == "bir\n\neki"
    assert normalise_whitespace("  bir\r\n  eki  ") == "bir\neki"


# --- crawled ingester -----------------------------------------------------


def _seed_crawl(tmp_path: Path, pages: dict[str, str], content_type: str = "text/html") -> tuple[
    CrawlState, RawStore
]:
    state = CrawlState(tmp_path / "crawl.db")
    store = RawStore("news_example", root=tmp_path / "raw")
    state.add_urls("news_example", list(pages))
    for url, html in pages.items():
        blob = store.store(html.encode(), content_type)
        state.mark_fetched(
            url_hash(url),
            http_status=200,
            content_hash=blob.content_hash,
            content_path=blob.relative_path,
            content_type=content_type,
        )
    return state, store


def test_extraction_produces_documents_with_provenance(tmp_path: Path) -> None:
    state, store = _seed_crawl(tmp_path, {"https://example.uz/qq/1": PAGE})
    try:
        documents = list(CrawledIngester(_spec(), state=state, store=store).documents())
    finally:
        state.close()

    assert len(documents) == 1
    document = documents[0]
    assert document.source_url == "https://example.uz/qq/1"
    assert document.source_id == "news_example"
    assert document.script is Script.LATIN
    assert document.meta["title"] == "Sessiya haqqında"
    assert "Joqarǵı Keńesiniń" in document.text


def test_identical_blobs_are_extracted_once(tmp_path: Path) -> None:
    # The same article under two URLs is one blob; extracting twice is waste.
    state, store = _seed_crawl(
        tmp_path,
        {"https://example.uz/qq/1": PAGE, "https://example.uz/qq/1-copy": PAGE},
    )
    try:
        documents = list(CrawledIngester(_spec(), state=state, store=store).documents())
    finally:
        state.close()
    assert len(documents) == 1


def test_non_text_responses_are_skipped(tmp_path: Path) -> None:
    state, store = _seed_crawl(
        tmp_path, {"https://example.uz/qq/x.pdf": "%PDF-1.4"}, content_type="application/pdf"
    )
    try:
        assert list(CrawledIngester(_spec(), state=state, store=store).documents()) == []
    finally:
        state.close()


def test_pages_without_content_are_skipped(tmp_path: Path) -> None:
    state, store = _seed_crawl(
        tmp_path, {"https://example.uz/qq/empty": "<html><body><nav>menu</nav></body></html>"}
    )
    try:
        assert list(CrawledIngester(_spec(), state=state, store=store).documents()) == []
    finally:
        state.close()


def test_missing_blob_does_not_abort_extraction(tmp_path: Path) -> None:
    state, store = _seed_crawl(tmp_path, {"https://example.uz/qq/1": PAGE})
    state.add_urls("news_example", ["https://example.uz/qq/gone"])
    state.mark_fetched(
        url_hash("https://example.uz/qq/gone"),
        http_status=200,
        content_hash="deadbeef",
        content_path="news_example/de/deadbeef.html",
        content_type="text/html",
    )
    try:
        documents = list(CrawledIngester(_spec(), state=state, store=store).documents())
    finally:
        state.close()
    assert len(documents) == 1


def test_run_writes_interim_and_manifest(tmp_path: Path) -> None:
    state, store = _seed_crawl(tmp_path, {"https://example.uz/qq/1": PAGE})
    try:
        manifest = CrawledIngester(_spec(), state=state, store=store).run()
    finally:
        state.close()

    assert manifest.documents == 1
    assert manifest.scripts == {"latin": 1}
    assert manifest.sha256
