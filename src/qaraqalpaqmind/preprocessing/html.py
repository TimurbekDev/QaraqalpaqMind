"""HTML to readable text.

A news page is perhaps 5% article and 95% navigation, sidebars, cookie banners
and footer links. On a corpus this small that ratio is not a nuisance, it is an
existential problem: unextracted boilerplate would be the single most repeated
"Karakalpak text" in the dataset, and the model would learn menus.

`trafilatura` does the real work. This module wraps it with a fallback that
still produces something usable when extraction fails, and with the metadata we
need for provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..common.logging import get_logger

logger = get_logger(__name__)

_MIN_EXTRACTED_CHARS = 120
_WHITESPACE_RUN = re.compile(r"[ \t ]{2,}")
_BLANK_RUN = re.compile(r"\n{3,}")

# Elements that never contain article text.
_STRIP_SELECTORS = (
    "script, style, noscript, template, svg, iframe, form",
    "nav, header, footer, aside",
    "[role=navigation], [role=banner], [role=contentinfo], [role=search]",
    ".menu, .nav, .navbar, .sidebar, .footer, .header, .breadcrumb",
    ".cookie, .banner, .advert, .ads, .social, .share, .comments",
)


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Main content of one HTML page, plus what we could learn about it."""

    text: str
    title: str | None = None
    published_at: str | None = None
    author: str | None = None
    extractor: str = "trafilatura"

    @property
    def is_usable(self) -> bool:
        return len(self.text) >= _MIN_EXTRACTED_CHARS


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines without touching content."""
    text = _WHITESPACE_RUN.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def extract_main_text(html: str, url: str | None = None) -> ExtractedPage:
    """Pull the main article out of an HTML document.

    Falls back to stripped-body text when trafilatura declines, which it does on
    listing pages and on markup it considers too thin. The fallback is marked in
    `extractor` so Phase 3 can weight it differently - fallback output carries
    much more boilerplate.
    """
    page = _try_trafilatura(html, url)
    if page is not None and page.is_usable:
        return page

    text = _fallback_text(html)
    return ExtractedPage(text=normalise_whitespace(text), extractor="fallback")


def _try_trafilatura(html: str, url: str | None) -> ExtractedPage | None:
    try:
        import trafilatura
        from trafilatura.settings import use_config
    except ImportError:
        logger.warning("trafilatura missing; using fallback extraction only")
        return None

    config = use_config()
    # Karakalpak pages are short by the standards trafilatura was tuned on;
    # its default minimum would discard legitimate three-paragraph articles.
    config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "120")
    config.set("DEFAULT", "MIN_OUTPUT_SIZE", "120")

    try:
        text = trafilatura.extract(
            html,
            url=url,
            config=config,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            # deduplicate=False deliberately. Trafilatura's deduplicator keeps a
            # PROCESS-GLOBAL LRU of seen paragraphs, so whether a page survives
            # depends on what was extracted before it - two identical articles
            # yield text for the first and nothing for the second. That is
            # hidden, order-dependent data loss. Deduplication is Phase 3's job,
            # where it is explicit, corpus-wide and tunable.
            deduplicate=False,
        )
    except Exception as exc:
        logger.debug("trafilatura failed", extra={"url": url, "error": str(exc)})
        return None

    if not text:
        return None

    title = author = published = None
    try:
        metadata = trafilatura.extract_metadata(html, default_url=url)
        if metadata is not None:
            title = metadata.title
            author = metadata.author
            published = metadata.date
    except Exception as exc:
        logger.debug("metadata extraction failed", extra={"url": url, "error": str(exc)})

    return ExtractedPage(
        text=normalise_whitespace(text),
        title=title,
        published_at=published,
        author=author,
    )


def _fallback_text(html: str) -> str:
    """Strip chrome and return whatever body text remains."""
    from selectolax.parser import HTMLParser

    tree = HTMLParser(html)
    for selector in _STRIP_SELECTORS:
        for node in tree.css(selector):
            node.decompose()

    body = tree.body
    return body.text(separator="\n", strip=True) if body else ""
