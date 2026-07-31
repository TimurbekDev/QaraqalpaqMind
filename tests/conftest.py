"""Shared pytest fixtures.

The important one is `_isolate_writable_dirs`: it is autouse, and it stops the
test suite from writing into the real `data/` tree. Without it, any test that
exercises `Ingester.run()` writes a manifest to `data/manifests/<source_id>.json`
and silently overwrites the record of a genuine ingest - which is exactly what
happened once, replacing a 10,199-document Wikipedia manifest with a 3-document
one from a fixture.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

_REDIRECTED = (
    ("qaraqalpaqmind.common.paths", "MANIFESTS_DIR"),
    ("qaraqalpaqmind.ingest.manifest", "MANIFESTS_DIR"),
    ("qaraqalpaqmind.common.paths", "INTERIM_DIR"),
    ("qaraqalpaqmind.ingest.base", "INTERIM_DIR"),
    ("qaraqalpaqmind.common.paths", "RAW_DIR"),
)


@pytest.fixture(autouse=True)
def _isolate_writable_dirs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point every project-writable directory at a per-test temporary tree."""
    sandbox = tmp_path_factory.mktemp("qm_sandbox")

    for module_name, attribute in _REDIRECTED:
        target = sandbox / attribute.removesuffix("_DIR").lower()
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(f"{module_name}.{attribute}", target, raising=False)

    yield sandbox
