"""Hugging Face dataset ingesters.

One generic loader driven by a declarative mapping per source, rather than a
module per dataset. The mappings state what we currently believe a dataset's
schema to be; `inspect_schema()` exists because that belief should be checked
against the real thing rather than assumed, and the loader fails with the
actual field names when a mapping is wrong.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..common.logging import get_logger
from ..common.records import Document
from ..preprocessing.script import detect_script
from .base import Ingester

logger = get_logger(__name__)

_MIN_CHARS = 20

# Language codes that mean Karakalpak across the datasets we use.
KAA_CODES: frozenset[str] = frozenset({"kaa", "kaa_Latn", "kaa_Cyrl", "karakalpak"})


@dataclass(frozen=True, slots=True)
class HFMapping:
    """How to read one Hugging Face dataset into our record schema."""

    repo: str
    splits: tuple[str, ...] = ("train",)
    config: str | None = None
    # Explicit file globs, for repos that expose no per-language builder config.
    # Takes precedence over `config` when set.
    data_files: tuple[str, ...] | None = None
    # Candidate text columns, tried in order. The first present one is used.
    text_fields: tuple[str, ...] = ("text", "content", "sentence", "raw_content")
    streaming: bool = False
    held_out: bool = False
    gated: bool = False
    # (src_lang_col, src_text_col, tgt_lang_col, tgt_text_col) for parallel corpora.
    pair_fields: tuple[str, str, str, str] | None = None
    meta_fields: tuple[str, ...] = field(default_factory=tuple)
    # Columns carrying real provenance, preferred over the dataset's Hub URL.
    url_field: str | None = None
    lang_conf_field: str | None = None
    notes: str = ""


MAPPINGS: dict[str, HFMapping] = {
    "hf_karakalpak_corpus_v2": HFMapping(
        repo="bekan/karakalpak_corpus_v2_m",
        text_fields=("text",),
        notes="135,667 sentences. Sentence-level, so documents are single sentences.",
    ),
    "hf_dilmash_parallel": HFMapping(
        repo="tahrirchi/dilmash",
        # Verified 2026-07-31: this dataset has no "train" split. Its splits are
        # the language pairs themselves.
        splits=("kaa_eng", "kaa_rus", "kaa_uzb"),
        pair_fields=("src_lang", "src_sent", "tgt_lang", "tgt_sent"),
        notes="300k pairs. We emit the Karakalpak side and keep the translation in meta.",
    ),
    "flores_plus_kaa": HFMapping(
        repo="openlanguagedata/flores_plus",
        config="kaa_Latn",
        # Verified against the Hub, not assumed: kaa_Latn ships `devtest` only.
        # Most FLORES+ languages have both dev and devtest; this one does not.
        splits=("devtest",),
        text_fields=("text",),
        held_out=True,
        gated=True,
        notes=(
            "BENCHMARK, never merged into training data. GATED on the Hub: accept the "
            "terms at huggingface.co/datasets/openlanguagedata/flores_plus, then set "
            "HF_TOKEN in .env."
        ),
    ),
    "glotcc_kaa": HFMapping(
        repo="cis-lmu/GlotCC-V1",
        # Verified 2026-07-31 from the Hub file listing. There is no per-language
        # builder config; the data are plain parquet files per language-script.
        # Both scripts are taken - GlotCC is one of the few sources of Karakalpak
        # Cyrillic web text.
        data_files=("v1.0/kaa-Latn/*.parquet", "v1.0/kaa-Cyrl/*.parquet"),
        streaming=True,
        text_fields=("content", "text"),
        url_field="warc-target-uri",
        lang_conf_field="identification-prob",
        meta_fields=(
            "identification-language",
            "quality-warnings",
            "categories",
            "warc-date",
            "num-sents",
        ),
        notes=(
            "CommonCrawl derived, both scripts. Carries per-document WARC provenance. "
            "NOTE: much of it is scraped kaa.wikipedia, so it overlaps wiki_kaa heavily "
            "- Phase 3 dedup must run across sources, not only within them."
        ),
    ),
}


def inspect_schema(source_id: str, rows: int = 3) -> list[dict[str, Any]]:
    """Return the first few raw rows of a dataset, to check a mapping is right.

    Guessing schemas is how ingest pipelines silently produce empty corpora.
    """
    mapping = require_mapping(source_id)
    dataset = _load(mapping, mapping.splits[0], streaming=True)
    return [row for _, row in zip(range(rows), dataset, strict=False)]


def _load(mapping: HFMapping, split: str, *, streaming: bool) -> Any:
    """Open one split, using explicit data files when the repo has no configs."""
    from datasets import load_dataset

    if mapping.data_files is not None:
        return load_dataset(
            mapping.repo, data_files=list(mapping.data_files), split=split, streaming=streaming
        )
    return load_dataset(mapping.repo, mapping.config, split=split, streaming=streaming)


def require_mapping(source_id: str) -> HFMapping:
    try:
        return MAPPINGS[source_id]
    except KeyError as exc:
        known = ", ".join(sorted(MAPPINGS))
        raise KeyError(f"No Hugging Face mapping for '{source_id}'. Known: {known}") from exc


class HuggingFaceIngester(Ingester):
    """Loads a registered Hugging Face dataset into `Document`s."""

    def documents(self, limit: int | None = None) -> Iterator[Document]:
        mapping = require_mapping(self.spec.id)
        if mapping.held_out:
            logger.warning(
                "Ingesting a HELD-OUT benchmark. This must never enter a training split.",
                extra={"source": self.spec.id},
            )

        import os

        if mapping.gated and not (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")):
            raise RuntimeError(
                f"'{self.spec.id}' is a gated dataset on the Hugging Face Hub.\n"
                f"  1. Accept the terms at https://huggingface.co/datasets/{mapping.repo}\n"
                "  2. Create a token at https://huggingface.co/settings/tokens\n"
                "  3. Put HF_TOKEN=<token> in .env, or export it in your shell"
            )

        fetched_at = datetime.now(tz=UTC)
        emitted = 0

        for split in mapping.splits:
            dataset = _load(mapping, split, streaming=mapping.streaming)
            for row in dataset:
                document = (
                    self._from_pair(row, mapping, split, fetched_at)
                    if mapping.pair_fields
                    else self._from_text(row, mapping, split, fetched_at)
                )
                if document is None:
                    continue
                yield document
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _from_text(
        self, row: dict[str, Any], mapping: HFMapping, split: str, fetched_at: datetime
    ) -> Document | None:
        text = self._pick_text(row, mapping)
        if text is None or len(text.strip()) < _MIN_CHARS:
            return None

        meta: dict[str, Any] = {"split": split, "repo": mapping.repo}
        meta.update({key: row[key] for key in mapping.meta_fields if key in row})
        if mapping.held_out:
            meta["held_out"] = True

        # A per-document origin URL is worth far more than the dataset's Hub
        # page: it is what makes withdrawal-on-request and cross-source dedup
        # possible at all.
        source_url = None
        if mapping.url_field:
            candidate = row.get(mapping.url_field)
            source_url = candidate if isinstance(candidate, str) else None
        source_url = source_url or f"https://huggingface.co/datasets/{mapping.repo}"

        document = Document.create(
            text=text.strip(),
            source_id=self.spec.id,
            license=self.spec.license,
            source_url=source_url,
            fetched_at=fetched_at,
            script=detect_script(text),
            meta=meta,
        )

        if mapping.lang_conf_field:
            confidence = row.get(mapping.lang_conf_field)
            if isinstance(confidence, int | float) and 0.0 <= confidence <= 1.0:
                document = document.model_copy(update={"lang_conf": float(confidence)})
        return document

    def _from_pair(
        self, row: dict[str, Any], mapping: HFMapping, split: str, fetched_at: datetime
    ) -> Document | None:
        """Emit the Karakalpak side of a translation pair, keeping its partner."""
        assert mapping.pair_fields is not None
        src_lang_col, src_text_col, tgt_lang_col, tgt_text_col = mapping.pair_fields

        src_lang, tgt_lang = row.get(src_lang_col), row.get(tgt_lang_col)
        src_text, tgt_text = row.get(src_text_col), row.get(tgt_text_col)

        if src_lang in KAA_CODES:
            text, other_text, other_lang = src_text, tgt_text, tgt_lang
        elif tgt_lang in KAA_CODES:
            text, other_text, other_lang = tgt_text, src_text, src_lang
        else:
            return None

        if not isinstance(text, str) or len(text.strip()) < _MIN_CHARS:
            return None

        return Document.create(
            text=text.strip(),
            source_id=self.spec.id,
            license=self.spec.license,
            source_url=f"https://huggingface.co/datasets/{mapping.repo}",
            fetched_at=fetched_at,
            script=detect_script(text),
            meta={
                "split": split,
                "repo": mapping.repo,
                "parallel_lang": other_lang,
                "parallel_text": other_text,
            },
        )

    @staticmethod
    def _pick_text(row: dict[str, Any], mapping: HFMapping) -> str | None:
        for name in mapping.text_fields:
            value = row.get(name)
            if isinstance(value, str):
                return value
        raise KeyError(
            f"None of {mapping.text_fields} found in {mapping.repo}. "
            f"Actual columns: {sorted(row)}. Fix the mapping in ingest/huggingface.py."
        )
