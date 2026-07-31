"""Dataset manifests: the reason a checkpoint is reproducible.

A manifest records exactly what a dataset file contains and where it came
from - row count, character and token totals, script mix, licence, and the
sha256 of the bytes. Manifests are the only thing under `data/` that is
committed to git, because they are what lets someone six months from now
answer "which data produced this model, and may we publish it?".
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel, ConfigDict, Field

from ..common.io import file_sha256
from ..common.logging import get_logger
from ..common.paths import MANIFESTS_DIR, ensure_dir

logger = get_logger(__name__)


class Manifest(BaseModel):
    """Immutable description of one ingested dataset file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    created_at: datetime
    license: str
    source_url: str

    path: str | None = Field(default=None, description="Repo-relative path; None for a dry run.")
    sha256: str | None = None
    size_bytes: int | None = None

    documents: int
    characters: int
    words: int
    estimated_tokens: int = Field(
        description="Cheap chars/3.1 approximation, available without a tokenizer."
    )
    measured_tokens: int | None = Field(
        default=None,
        description=(
            "Exact count from the real tokenizer, written by `qm tokenizer count`. "
            "The estimate understated this corpus by 35%, so anything sizing a "
            "training run must prefer this when it is present."
        ),
    )
    tokenizer: str | None = Field(default=None, description="Model whose tokenizer was used.")
    scripts: dict[str, int] = Field(default_factory=dict)

    @property
    def tokens(self) -> int:
        """Best available token count: measured if known, estimated otherwise."""
        return self.measured_tokens if self.measured_tokens is not None else self.estimated_tokens

    @classmethod
    def build(
        cls,
        *,
        source_id: str,
        path: Path | None,
        documents: int,
        characters: int,
        words: int,
        estimated_tokens: int,
        license: str,
        source_url: str,
        scripts: dict[str, int] | None = None,
    ) -> Manifest:
        return cls(
            source_id=source_id,
            created_at=datetime.now(tz=UTC),
            license=license,
            source_url=source_url,
            path=path.as_posix() if path else None,
            sha256=file_sha256(path) if path and path.exists() else None,
            size_bytes=path.stat().st_size if path and path.exists() else None,
            documents=documents,
            characters=characters,
            words=words,
            estimated_tokens=estimated_tokens,
            scripts=scripts or {},
        )

    def summary(self) -> str:
        mb = (self.size_bytes or 0) / 1_048_576
        return (
            f"{self.source_id}: {self.documents:,} docs, {self.characters:,} chars, "
            f"~{self.estimated_tokens:,} tokens, {mb:.1f} MB compressed, "
            f"scripts={self.scripts or 'n/a'}"
        )


def manifest_path(source_id: str) -> Path:
    return ensure_dir(MANIFESTS_DIR) / f"{source_id}.json"


def write_manifest(manifest: Manifest) -> Path:
    target = manifest_path(manifest.source_id)
    target.write_bytes(
        orjson.dumps(manifest.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )
    logger.info("Manifest written", extra={"path": str(target)})
    return target


def read_manifest(source_id: str) -> Manifest:
    payload: dict[str, Any] = orjson.loads(manifest_path(source_id).read_bytes())
    return Manifest.model_validate(payload)


def load_all_manifests() -> list[Manifest]:
    """Every ingest manifest on disk, newest first."""
    manifests: list[Manifest] = []
    for file in sorted(MANIFESTS_DIR.glob("*.json")):
        if file.name.startswith("source_audit_"):
            continue
        try:
            manifests.append(Manifest.model_validate(orjson.loads(file.read_bytes())))
        except ValueError:
            logger.debug("Not an ingest manifest", extra={"path": str(file)})
    return sorted(manifests, key=lambda m: m.created_at, reverse=True)
