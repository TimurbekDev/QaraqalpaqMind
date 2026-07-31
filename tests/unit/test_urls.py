"""Tests for URL normalisation, scoping and discovery."""

from __future__ import annotations

import pytest

from qaraqalpaqmind.crawlers.core.registry import SourceSpec
from qaraqalpaqmind.crawlers.core.urls import (
    extract_links,
    in_scope,
    is_fetchable_document,
    normalise_url,
    parse_sitemap,
    url_suffix,
)


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://Example.UZ/qq/news", "https://example.uz/qq/news"),
        ("https://example.uz/qq/news#comments", "https://example.uz/qq/news"),
        ("https://example.uz/qq//news///a", "https://example.uz/qq/news/a"),
        ("https://example.uz/qq/news?utm_source=tg&id=7", "https://example.uz/qq/news?id=7"),
        ("https://example.uz/qq/news?fbclid=x", "https://example.uz/qq/news"),
        ("https://example.uz:443/qq/", "https://example.uz/qq/"),
        ("http://example.uz:80/qq/", "http://example.uz/qq/"),
        ("https://example.uz", "https://example.uz/"),
    ],
)
def test_normalise_url(raw: str, expected: str) -> None:
    assert normalise_url(raw) == expected


def test_normalise_url_sorts_remaining_query_params() -> None:
    # Same page, two param orders, must collapse to one frontier entry.
    a = normalise_url("https://example.uz/qq/?b=2&a=1")
    b = normalise_url("https://example.uz/qq/?a=1&b=2")
    assert a == b


@pytest.mark.parametrize(
    "raw",
    ["", "#top", "mailto:a@b.c", "tel:+998", "javascript:void(0)", "data:text/plain,x", "ftp://x.uz/f"],
)
def test_normalise_url_rejects_non_http(raw: str) -> None:
    assert normalise_url(raw) is None


def test_normalise_url_resolves_relative_against_base() -> None:
    assert (
        normalise_url("../maqala/1", base="https://example.uz/qq/news/index.html")
        == "https://example.uz/qq/maqala/1"
    )


def test_url_suffix_and_asset_detection() -> None:
    assert url_suffix("https://example.uz/a/photo.JPG") == ".jpg"
    assert url_suffix("https://example.uz/qq/news") == ""
    assert not is_fetchable_document("https://example.uz/style.css")
    assert is_fetchable_document("https://example.uz/qq/news")
    assert is_fetchable_document("https://example.uz/qq/doc.pdf")


def test_in_scope_requires_the_declared_host() -> None:
    spec = _spec()
    assert in_scope("https://example.uz/qq/news", spec)
    assert in_scope("https://cdn.example.uz/qq/news", spec)  # subdomain allowed
    assert not in_scope("https://evil.uz/qq/news", spec)


def test_in_scope_enforces_locale_paths() -> None:
    # This is what keeps a Karakalpak crawl out of the Russian edition.
    spec = _spec(allowed_paths=["/qq/"], denied_paths=["/ru/", "/uz/"])
    assert in_scope("https://example.uz/qq/news/1", spec)
    assert not in_scope("https://example.uz/ru/news/1", spec)
    assert not in_scope("https://example.uz/uz/news/1", spec)
    assert not in_scope("https://example.uz/en/news/1", spec)


def test_in_scope_denied_beats_allowed() -> None:
    spec = _spec(allowed_paths=["/qq/"], denied_paths=["/qq/tag/"])
    assert in_scope("https://example.uz/qq/news", spec)
    assert not in_scope("https://example.uz/qq/tag/aral", spec)


def test_in_scope_rejects_assets() -> None:
    assert not in_scope("https://example.uz/qq/logo.png", _spec())


def test_extract_links_normalises_and_dedups() -> None:
    html = """
    <html><body>
      <a href="/qq/news/1">bir</a>
      <a href="/qq/news/1?utm_source=tg">bir again</a>
      <a href="https://example.uz/qq/news/2#top">eki</a>
      <a href="mailto:a@b.c">mail</a>
      <a>no href</a>
    </body></html>
    """
    links = extract_links(html, "https://example.uz/qq/")
    assert links == ["https://example.uz/qq/news/1", "https://example.uz/qq/news/2"]


def test_parse_sitemap_urlset() -> None:
    xml = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.uz/qq/1</loc></url>
      <url><loc>https://example.uz/qq/2</loc></url>
    </urlset>"""
    locations, is_index = parse_sitemap(xml)
    assert locations == ["https://example.uz/qq/1", "https://example.uz/qq/2"]
    assert not is_index


def test_parse_sitemap_index() -> None:
    xml = """<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.uz/sm-1.xml</loc></sitemap>
    </sitemapindex>"""
    locations, is_index = parse_sitemap(xml)
    assert locations == ["https://example.uz/sm-1.xml"]
    assert is_index


def test_parse_sitemap_survives_malformed_xml() -> None:
    # Real sitemaps in our registry are frequently truncated or mis-encoded.
    broken = "<urlset><url><loc>https://example.uz/qq/1</loc></url><url><loc>htt"
    locations, is_index = parse_sitemap(broken)
    assert locations == ["https://example.uz/qq/1"]
    assert not is_index
