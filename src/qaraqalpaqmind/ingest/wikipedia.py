"""Karakalpak Wikipedia ingester, from the official Wikimedia dump.

We use the live dump rather than the Hugging Face `wikimedia/wikipedia`
snapshot on purpose. That snapshot is pinned to 2023-11-01, when kaa.wikipedia
had ~4,070 articles; the wiki has since grown to over 15,000. Taking the stale
copy would throw away roughly three quarters of the corpus.

The trade is that we must parse MediaWiki XML and strip wikitext ourselves.
"""

from __future__ import annotations

import bz2
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..common.logging import get_logger
from ..common.paths import raw_source_dir
from ..common.records import Document, Script
from ..crawlers.core.registry import SourceSpec
from ..preprocessing.script import detect_script
from .base import Ingester

logger = get_logger(__name__)

DUMP_BASE = "https://dumps.wikimedia.org/kaawiki/latest"
DUMP_FILE = "kaawiki-latest-pages-articles.xml.bz2"
CHECKSUM_FILE = "kaawiki-latest-sha1sums.txt"

# The `latest/` directory serves symlinks named `-latest-`, but the checksum
# file lists the real, DATED targets (e.g. kaawiki-20260701-pages-articles...).
# Matching on the literal "latest" name therefore never hits.
_DATED_DUMP = re.compile(r"kaawiki-\d{8}-pages-articles\.xml\.bz2$")

_MIN_CHARS = 200
_DOWNLOAD_CHUNK = 1 << 20

# Wikitext leftovers that `strip_code` does not remove.
_FILE_LINK = re.compile(r"\[\[(?:File|Image|Fayl|Súwret):[^\]]*\]\]", re.IGNORECASE)
_CATEGORY = re.compile(r"\[\[(?:Category|Kategoriya|Kategoriya):[^\]]*\]\]", re.IGNORECASE)
_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACE_RUN = re.compile(r"[ \t]{2,}")

# Section headings whose content is link-farm, not prose.
_DROP_SECTIONS = {
    "siltemeler", "derekler", "ádebiyatlar", "sıltemeler", "paydalanılǵan ádebiyatlar",
    "sonı da qarań", "usını da qarań", "references", "external links", "see also",
    "сілтемелер", "дереклер", "әдебиятлар", "пайдаланылған әдебиятлар",
}


class WikipediaIngester(Ingester):
    """Streams `data/raw/wiki_kaa/*.xml.bz2` into `Document`s."""

    def __init__(self, spec: SourceSpec, *, dump_path: Path | None = None) -> None:
        super().__init__(spec)
        self._dump_path = dump_path or (raw_source_dir(spec.id) / DUMP_FILE)

    # --- acquisition ------------------------------------------------------

    def download(self, *, force: bool = False) -> Path:
        """Fetch the dump if it is not already on disk, and verify its checksum."""
        if self._dump_path.exists() and not force:
            logger.info(
                "Dump already present",
                extra={"path": str(self._dump_path), "bytes": self._dump_path.stat().st_size},
            )
            return self._dump_path

        url = f"{DUMP_BASE}/{DUMP_FILE}"
        temp = self._dump_path.with_suffix(self._dump_path.suffix + ".part")
        logger.info("Downloading dump", extra={"url": url})

        with httpx.stream(
            "GET", url, timeout=120.0, follow_redirects=True,
            headers={"User-Agent": "QaraqalpaqMindBot/0.1 (+dump ingest)"},
        ) as response:
            response.raise_for_status()
            with temp.open("wb") as handle:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK):
                    handle.write(chunk)

        temp.replace(self._dump_path)
        self._verify(self._dump_path)
        return self._dump_path

    def _verify(self, path: Path) -> None:
        """Check the dump against Wikimedia's published SHA-1.

        We parse this XML with the standard library, which is only defensible
        because the bytes are an official Wikimedia artefact fetched over TLS
        and confirmed against their checksum. A checksum mismatch is fatal.
        """
        try:
            response = httpx.get(
                f"{DUMP_BASE}/{CHECKSUM_FILE}",
                timeout=60.0,
                follow_redirects=True,
                headers={"User-Agent": "QaraqalpaqMindBot/0.1 (+dump ingest)"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "UNVERIFIED DUMP: could not fetch checksums", extra={"error": str(exc)}
            )
            return

        expected = snapshot = None
        for line in response.text.splitlines():
            parts = line.split()
            if len(parts) == 2 and _DATED_DUMP.search(parts[1]):
                expected, snapshot = parts[0], parts[1]
                break

        if expected is None:
            logger.warning("UNVERIFIED DUMP: no checksum entry", extra={"file": DUMP_FILE})
            return

        import hashlib

        digest = hashlib.sha1()
        with path.open("rb") as handle:
            while chunk := handle.read(_DOWNLOAD_CHUNK):
                digest.update(chunk)

        actual = digest.hexdigest()
        if actual != expected:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"Dump checksum mismatch: expected {expected}, got {actual}. "
                "The file has been deleted; re-run the download."
            )
        logger.info("Dump checksum verified", extra={"sha1": actual, "snapshot": snapshot})

    # --- parsing ----------------------------------------------------------

    def documents(self, limit: int | None = None) -> Iterator[Document]:
        if not self._dump_path.exists():
            self.download()

        fetched_at = datetime.fromtimestamp(self._dump_path.stat().st_mtime, tz=UTC)
        emitted = skipped_redirect = skipped_short = 0

        with bz2.open(self._dump_path, "rb") as handle:
            for _, element in ET.iterparse(handle, events=("end",)):
                if _localname(element.tag) != "page":
                    continue

                try:
                    if _child_text(element, "ns") != "0":
                        continue
                    if element.find("{*}redirect") is not None:
                        skipped_redirect += 1
                        continue

                    title = _child_text(element, "title") or ""
                    wikitext = _revision_text(element)
                    if not wikitext:
                        continue

                    text = clean_wikitext(wikitext)
                    if len(text) < _MIN_CHARS:
                        skipped_short += 1
                        continue

                    yield Document.create(
                        text=f"{title}\n\n{text}" if title else text,
                        source_id=self.spec.id,
                        license=self.spec.license,
                        source_url=_article_url(title),
                        fetched_at=fetched_at,
                        script=_script_of(text),
                        meta={"title": title, "dump": DUMP_FILE},
                    )
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return
                finally:
                    # iterparse keeps every element alive otherwise, and the
                    # dump does not fit in memory uncompressed.
                    element.clear()

        logger.info(
            "Wikipedia parse finished",
            extra={"emitted": emitted, "redirects": skipped_redirect, "too_short": skipped_short},
        )


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    child = element.find(f"{{*}}{name}")
    return child.text if child is not None else None


def _revision_text(page: ET.Element) -> str | None:
    revision = page.find("{*}revision")
    if revision is None:
        return None
    body = revision.find("{*}text")
    return body.text if body is not None else None


def _article_url(title: str) -> str:
    return f"https://kaa.wikipedia.org/wiki/{title.replace(' ', '_')}"


def _script_of(text: str) -> Script:
    return detect_script(text[:4000])


def clean_wikitext(wikitext: str) -> str:
    """Reduce MediaWiki markup to readable prose.

    `mwparserfromhell.strip_code()` handles templates and links, but leaves
    tables, refs, file captions and category lines behind - all of which are
    metadata rather than language, and all of which would otherwise dominate
    the short-article end of a small wiki.
    """
    text = _REF.sub(" ", wikitext)
    text = _TABLE.sub(" ", text)
    text = _FILE_LINK.sub(" ", text)
    text = _CATEGORY.sub(" ", text)

    try:
        import mwparserfromhell

        text = str(mwparserfromhell.parse(text).strip_code(normalize=True, collapse=True))
    except ImportError:  # pragma: no cover - the extra is declared in pyproject
        logger.warning("mwparserfromhell missing; falling back to regex stripping")
        text = re.sub(r"\{\{[^}]*\}\}", " ", text)
        text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)

    text = _HTML_TAG.sub(" ", text)
    text = _HEADING.sub(r"\1", text)
    text = _drop_trailing_sections(text)
    text = _SPACE_RUN.sub(" ", text)
    return _BLANK_RUN.sub("\n\n", text).strip()


def _drop_trailing_sections(text: str) -> str:
    """Cut everything from the first references/see-also heading onward."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip().lower().rstrip(":") in _DROP_SECTIONS:
            return "\n".join(lines[:index]).strip()
    return text
