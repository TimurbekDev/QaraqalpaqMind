"""Bulk source ingestion: Wikipedia dumps and Hugging Face datasets.

Ingesters and crawlers are two ways to reach the same destination - a stream of
`Document`s in `data/interim/` with a manifest beside it. Phase 3 consumes that
destination and never needs to know which produced a given file.
"""

from __future__ import annotations

from .base import Ingester
from .manifest import Manifest, load_all_manifests, read_manifest, write_manifest

__all__ = [
    "Ingester",
    "Manifest",
    "load_all_manifests",
    "read_manifest",
    "write_manifest",
]
