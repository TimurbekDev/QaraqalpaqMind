"""Shared crawling machinery.

Source modules in `crawlers/sources/` compose these; none of them opens a
socket, writes a file or invents a politeness rule of its own.
"""

from __future__ import annotations

from .crawler import Crawler, CrawlStats
from .fetcher import Fetcher, FetchOutcome, FetchResult
from .rate_limit import HostRateLimiter
from .registry import (
    AccessMethod,
    LegalStatus,
    SourceKind,
    SourceRegistry,
    SourceSpec,
    UpdateFrequency,
    load_registry,
)
from .robots import RobotsCache, RobotsRules
from .state import CrawlState, UrlRecord, UrlStatus, url_hash
from .storage import RawStore, StoredBlob
from .urls import extract_links, in_scope, normalise_url, parse_sitemap

__all__ = [
    "AccessMethod",
    "CrawlState",
    "CrawlStats",
    "Crawler",
    "FetchOutcome",
    "FetchResult",
    "Fetcher",
    "HostRateLimiter",
    "LegalStatus",
    "RawStore",
    "RobotsCache",
    "RobotsRules",
    "SourceKind",
    "SourceRegistry",
    "SourceSpec",
    "StoredBlob",
    "UpdateFrequency",
    "UrlRecord",
    "UrlStatus",
    "extract_links",
    "in_scope",
    "load_registry",
    "normalise_url",
    "parse_sitemap",
    "url_hash",
]
