"""Ingester contract and shared plumbing.

An *ingester* turns one registered source into a stream of `Document`s and
writes them to `data/interim/<source_id>.jsonl.zst`, alongside a manifest.

Ingesters are for sources we obtain in bulk - Wikipedia dumps, Hugging Face
datasets - as distinct from crawlers, which fetch page by page. Both end up in
the same place in the same shape, so Phase 3 sees one uniform input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from ..common.io import write_jsonl
from ..common.logging import get_logger
from ..common.paths import INTERIM_DIR, ensure_dir
from ..common.records import Document
from ..crawlers.core.registry import SourceSpec
from .manifest import Manifest, write_manifest

logger = get_logger(__name__)

_PROGRESS_EVERY = 2_000


class Ingester(ABC):
    """Base class for bulk source loaders."""

    def __init__(self, spec: SourceSpec) -> None:
        self.spec = spec

    @abstractmethod
    def documents(self, limit: int | None = None) -> Iterator[Document]:
        """Yield documents for this source. Must be lazy - corpora exceed RAM."""

    def output_path(self) -> Path:
        ensure_dir(INTERIM_DIR)
        return INTERIM_DIR / f"{self.spec.id}.jsonl.zst"

    def run(self, limit: int | None = None, *, dry_run: bool = False) -> Manifest:
        """Ingest the source, write interim JSONL and a manifest.

        `dry_run` counts and measures without writing the corpus, which is how
        you check a source is what you think it is before spending disk on it.
        """
        target = self.output_path()
        counters = _Counters()

        stream = self._counted(self.documents(limit), counters)
        if dry_run:
            for _ in stream:
                pass
        else:
            write_jsonl(target, (doc.model_dump(mode="json") for doc in stream))

        manifest = Manifest.build(
            source_id=self.spec.id,
            path=None if dry_run else target,
            documents=counters.documents,
            characters=counters.characters,
            words=counters.words,
            estimated_tokens=counters.tokens,
            license=self.spec.license,
            source_url=str(self.spec.url),
            scripts=counters.scripts,
        )

        if not dry_run:
            write_manifest(manifest)
        logger.info("Ingest complete", extra={"source": self.spec.id, "docs": counters.documents})
        return manifest

    def _counted(self, stream: Iterator[Document], counters: _Counters) -> Iterator[Document]:
        for document in stream:
            counters.add(document)
            if counters.documents % _PROGRESS_EVERY == 0:
                logger.info(
                    "Ingest progress",
                    extra={"source": self.spec.id, "documents": counters.documents},
                )
            yield document


class _Counters:
    """Running totals gathered in one pass, so nothing is read twice."""

    __slots__ = ("characters", "documents", "scripts", "tokens", "words")

    def __init__(self) -> None:
        self.documents = 0
        self.characters = 0
        self.words = 0
        self.tokens = 0
        self.scripts: dict[str, int] = {}

    def add(self, document: Document) -> None:
        self.documents += 1
        self.characters += document.char_count()
        self.words += document.word_count()
        self.tokens += document.estimated_tokens()
        key = document.script.value
        self.scripts[key] = self.scripts.get(key, 0) + 1
