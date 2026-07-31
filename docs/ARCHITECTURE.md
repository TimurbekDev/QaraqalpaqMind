# Architecture

## 1. The data pipeline is one directed flow

```
                 ┌─────────────┐
   internet ───► │  crawlers/  │──► data/raw/<source_id>/         immutable bytes
                 └─────────────┘        (html, pdf, json dumps)
                                             │
                 ┌────────────────┐          ▼
                 │ preprocessing/ │──► data/interim/<source_id>.jsonl.zst
                 └────────────────┘        (text extracted, still dirty)
                                             ▲
                 ┌─────────────┐             │
   dumps, HF ──► │   ingest/   │─────────────┘  + data/manifests/<source_id>.json
                 └─────────────┘        (bulk sources skip raw/: the upstream
                                         artefact is already the archive, and
                                         is checksum-verified instead)
                                             │
                 ┌─────────────┐             ▼
                 │  cleaning/  │──► data/processed/<source_id>/part-*.jsonl.zst
                 └─────────────┘        (quality-filtered, language-verified)
                                             │
                 ┌─────────────┐             ▼
                 │   dedup/    │──► data/datasets/pretrain/*.jsonl.zst
                 └─────────────┘        + data/manifests/<name>.json   ◄── committed
                                             │
                 ┌──────────────────┐        ▼
                 │ training/cpt ────┼──► models/cpt/
                 │ training/sft ────┼──► models/sft/
                 │ training/dpo ────┼──► models/dpo/ ──► models/merged/
                 └──────────────────┘        │
                                             ▼
                 ┌─────────────┐      ┌───────────┐      ┌──────────┐
                 │ evaluation/ │◄─────│   vLLM    │◄─────│  api/    │◄── web/
                 └─────────────┘      └───────────┘      └──────────┘
                                             ▲
                                        ┌────┴────┐
                                        │  rag/   │◄── Qdrant
                                        └─────────┘
```

**Invariants**

1. `data/raw/` is written once and never modified. If extraction logic changes, re-run preprocessing — never re-crawl.
2. Each arrow is a separate CLI command with its own config file. Any stage can be re-run in isolation.
3. Every artefact that feeds a training run is recorded in a manifest (sha256, row count, token count, source licence). A checkpoint without a manifest is not reproducible and is treated as scrap.

## 2. Naming conventions

| Thing | Convention | Example |
|---|---|---|
| Source id | `<kind>_<domain-slug>` | `wiki_kaa`, `news_qmuz`, `gov_karakalpakstan` |
| Raw file | `<sha1-of-url>.<ext>` | `data/raw/news_qmuz/a3f9….html` |
| Interim / processed | `<source_id>.jsonl.zst`, sharded as `part-00003.jsonl.zst` | |
| Dataset release | `<task>_<version>` | `pretrain_v1`, `sft_v2`, `dpo_v1` |
| Manifest | `data/manifests/<dataset>.json` | `pretrain_v1.json` |
| Checkpoint | `models/<stage>/<base>-<dataset>-<date>` | `models/cpt/qwen3-8b-pretrain_v1-20260801` |
| Config | `configs/<stage>/<descriptor>.yaml` | `configs/cpt/qwen3_8b_lora.yaml` |

## 3. Universal record schema

Every record in `interim/`, `processed/` and `datasets/pretrain/` carries the same envelope. Phase 4 formalises the task-specific schemas; this is the substrate they all share.

```json
{
  "id": "sha1 of source_url + chunk index",
  "text": "…the actual content…",
  "source_id": "news_qmuz",
  "source_url": "https://…",
  "fetched_at": "2026-08-01T09:12:44Z",
  "lang": "kaa",
  "lang_conf": 0.97,
  "script": "latin",
  "license": "CC-BY-SA-4.0",
  "meta": { "title": "…", "author": null, "published_at": "2024-03-11" },
  "quality": { "score": 0.81, "flags": [] }
}
```

`lang`/`script` are separate fields on purpose: Karakalpak exists in **Latin** (official since 1994, several orthography revisions) and **Cyrillic** (most pre-1994 books, many older sites). Both are valuable training data, but they must be identifiable so that transliteration and script-balance decisions happen deliberately in Phase 3, not by accident.

## 4. Dependency direction

```
common/  ←── everything          (common imports nothing from the project)
crawlers/ ─► common
preprocessing/ ─► common
cleaning/ ─► common
dedup/ ─► common
training/ ─► common
evaluation/ ─► common, rag (optional)
rag/ ─► common
api/ ─► common, rag
```

No cycles. `common/` never imports torch, httpx or qdrant — that is what keeps `qm --help` fast on a machine with no GPU stack.

## 5. Configuration model

* One YAML per run under `configs/`.
* `_base_:` gives inheritance, so variants only state their deltas.
* `${ENV_VAR}` / `${ENV_VAR:default}` pulls secrets from the environment.
* Every config is validated into a frozen pydantic model with `extra="forbid"` — a typo like `learing_rate` fails in the first second, not after a 40-hour run.

See [`src/qaraqalpaqmind/common/config.py`](../src/qaraqalpaqmind/common/config.py).

## 6. Why Python 3.12 and not 3.13/3.14

`torch`, `vllm`, `flash-attn`, `deepspeed` and `bitsandbytes` publish wheels for 3.12 first; newer interpreters lag by 6–12 months and force source builds against CUDA. The project pins `>=3.12,<3.13` in `pyproject.toml`.
