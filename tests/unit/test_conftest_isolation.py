"""Guard the guard: prove the test suite cannot write into the real data tree.

This exists because a fixture once overwrote a real ingest manifest. If the
isolation fixture is ever removed or renamed, these fail loudly.
"""

from __future__ import annotations

from pathlib import Path

from qaraqalpaqmind.common.paths import PROJECT_ROOT
from qaraqalpaqmind.crawlers.core.registry import SourceSpec
from qaraqalpaqmind.ingest.manifest import manifest_path


def _real_data_dir() -> Path:
    return PROJECT_ROOT / "data"


def test_manifest_writes_land_outside_the_repo() -> None:
    target = manifest_path("wiki_kaa")
    assert _real_data_dir() not in target.parents, (
        f"tests would write a manifest into the real data tree at {target}"
    )


def test_interim_writes_land_outside_the_repo() -> None:
    from qaraqalpaqmind.ingest.base import Ingester

    class _Probe(Ingester):
        def documents(self, limit: int | None = None):  # type: ignore[no-untyped-def]
            yield from ()

    spec = SourceSpec.model_validate(
        {
            "id": "wiki_kaa",
            "name": "T",
            "kind": "wiki",
            "access": "dump",
            "url": "https://example.org/",
            "legal": "open_license",
            "license": "CC0-1.0",
            "update_frequency": "rarely",
            "est_size_mb": 0.0,
            "quality": 3,
        }
    )
    assert _real_data_dir() not in _Probe(spec).output_path().parents
