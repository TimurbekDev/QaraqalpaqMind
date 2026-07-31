"""URL normalisation, scope rules, and link/sitemap discovery.

Normalisation exists to stop the frontier exploding. The same article reachable
as `?utm_source=telegram`, `#comments` and `//qq//news/` must collapse to one
URL, or a 1,800-article site becomes a 40,000-request crawl.

Scope rules exist to stop us leaving the language. Every Tier 2 source in the
registry is a multilingual site where only one locale prefix is Karakalpak;
without `allowed_paths` we would silently mirror the Russian edition.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from ...common.logging import get_logger
from .registry import SourceSpec

logger = get_logger(__name__)

# Query parameters that never change the article a URL serves.
_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        # analytics and referral tagging
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "fbclid", "gclid", "yclid", "msclkid", "igshid",
        "_ga", "_gl", "mc_cid", "mc_eid", "ref", "referrer", "source",
        "share", "from", "spm",
        # Comment permalinks and alternate renderings of the SAME article.
        # Measured on shagalalab.com: `?showComment=<timestamp>` produced 12
        # frontier entries for one post, so we fetched it twelve times and got
        # twelve near-identical documents. Blogger and WordPress both do this.
        "showComment", "showcomment", "replytocom", "commentpage",
        "amp", "output", "print", "print_preview", "format",
    }
)

# Extensions we never want as documents.
_BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff",
        ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".wav",
        ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz",
        ".css", ".js", ".json", ".woff", ".woff2", ".ttf", ".eot", ".map",
        ".exe", ".dmg", ".apk", ".iso",
    }
)

# Documents we DO want, handled by a different extractor than HTML.
DOCUMENT_SUFFIXES: frozenset[str] = frozenset({".pdf", ".doc", ".docx", ".rtf", ".odt", ".txt"})

_MULTI_SLASH = re.compile(r"/{2,}")
_LOC_PATTERN = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)
_INDEX_PATTERN = re.compile(r"<sitemapindex|<sitemap>", re.IGNORECASE)
_MAX_SITEMAP_CHARS = 8_000_000


def normalise_url(url: str, base: str | None = None) -> str | None:
    """Canonicalise `url`, resolving it against `base` if it is relative.

    Returns None for anything that is not a fetchable http(s) URL.
    """
    raw = url.strip()
    if not raw or raw.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
        return None

    if base:
        raw = urljoin(base, raw)

    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None

    host = parts.netloc.lower()
    # Drop redundant default ports so :443 and bare host are one URL.
    if host.endswith(":80") and parts.scheme == "http":
        host = host[:-3]
    elif host.endswith(":443") and parts.scheme == "https":
        host = host[:-4]

    path = _MULTI_SLASH.sub("/", parts.path) or "/"

    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS]
    query = urlencode(sorted(kept))

    # Fragments are client-side only; they never select different content.
    return urlunsplit((parts.scheme.lower(), host, path, query, ""))


def url_suffix(url: str) -> str:
    """Lowercased file extension of a URL's path, or '' if it has none."""
    path = urlsplit(url).path
    _, dot, suffix = path.rpartition(".")
    if not dot or "/" in suffix:
        return ""
    return f".{suffix.lower()}"


def is_fetchable_document(url: str) -> bool:
    """True if the URL looks like HTML or a text document, not an asset."""
    suffix = url_suffix(url)
    return suffix not in _BINARY_SUFFIXES


def in_scope(url: str, spec: SourceSpec) -> bool:
    """Does `url` belong to the corpus this source is declared to cover?

    A URL is in scope when it is on the source's host, is not an asset, is not
    under a denied path, and - when `allowed_paths` is set - sits under one of
    them. `allowed_paths` is how we stay inside the Karakalpak locale.
    """
    parts = urlsplit(url)
    spec_host = urlsplit(str(spec.url)).netloc.lower()

    host = parts.netloc.lower()
    if host != spec_host and not host.endswith(f".{spec_host}"):
        return False

    if not is_fetchable_document(url):
        return False

    path = parts.path or "/"
    if any(path.startswith(denied) for denied in spec.denied_paths):
        return False

    return not (
        spec.allowed_paths
        and not any(path.startswith(allowed) for allowed in spec.allowed_paths)
    )


def extract_links(html: str, base_url: str) -> list[str]:
    """Normalised absolute links from an HTML document, de-duplicated in order."""
    from selectolax.parser import HTMLParser

    seen: set[str] = set()
    links: list[str] = []
    for node in HTMLParser(html).css("a[href]"):
        href = node.attributes.get("href")
        if not href:
            continue
        normalised = normalise_url(href, base=base_url)
        if normalised and normalised not in seen:
            seen.add(normalised)
            links.append(normalised)
    return links


def parse_sitemap(xml_text: str) -> tuple[list[str], bool]:
    """Extract locations from a sitemap or sitemap index.

    Returns `(locations, is_index)`.

    Scanning for `<loc>` rather than parsing the XML is deliberate, for two
    reasons. First, security: sitemaps are attacker-controllable remote input,
    and `xml.etree` is documented as vulnerable to entity-expansion attacks
    ("billion laughs"), which a text scan cannot be. Second, robustness: the
    sitemaps on our registered sources are frequently truncated, mis-encoded or
    served with the wrong content type, and a strict parser would reject
    documents we can read perfectly well. We only ever need the `<loc>` values.
    """
    if len(xml_text) > _MAX_SITEMAP_CHARS:
        logger.warning("Sitemap truncated", extra={"chars": len(xml_text)})
        xml_text = xml_text[:_MAX_SITEMAP_CHARS]

    locations = [loc.strip() for loc in _LOC_PATTERN.findall(xml_text)]
    return locations, bool(_INDEX_PATTERN.search(xml_text))
