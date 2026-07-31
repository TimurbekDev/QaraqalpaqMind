# Roadmap

One phase at a time. A phase is done when its deliverables exist, run, and are covered by at least a smoke test.

| # | Phase | Deliverables | Gate to pass before moving on |
|---|---|---|---|
| 1 | **Architecture** | Folder tree, `common/` (paths, logging, config, io), `pyproject.toml`, `qm` CLI, docs | `qm doctor` runs; unit tests green |
| 2 | **Data collection** | Source audit (legality, licence, size, update cadence), `crawlers/core` + ≥6 source modules, resumable state | ≥50 MB raw Karakalpak text landed in `data/raw/` |
| 3 | **Cleaning** | HTML/PDF extraction, unicode/ftfy normalisation, kaa language-ID model, quality scorer, exact + MinHash dedup | ✅ `pretrain_v1`: 286,434 docs, ~21.3M unique tokens |
| 4 | **Dataset formats** | JSONL schemas for all 11 task types + validators; tokenizer fertility study on Qwen3 | `pretrain_v1` manifest with token count |
| 5 | **Continued pretraining** | Qwen3-8B CPT configs (LoRA + full), DeepSpeed/FSDP, launch scripts | Loss curve down; kaa perplexity beats base model |
| 6 | **SFT** | Instruction datasets (8 categories), TRL `SFTTrainer` scripts, chat template | Model answers in Karakalpak, follows format |
| 7 | **DPO** | Preference pair construction, `DPOTrainer`, win-rate eval | Win rate > SFT baseline on held-out prompts |
| 8 | **Evaluation** | KaaBench: grammar, translation, history, geography, math, code, reasoning, dialogue, factual QA | Reproducible score table for every checkpoint |
| 9 | **RAG** | Qdrant ingest, chunking, embeddings, reranker, citation assembly | Grounded answers with working citations |
| 10 | **Deployment** | vLLM server, FastAPI gateway (SSE streaming, auth, rate limit, metrics), Nginx, Compose | `docker compose up` serves a streaming chat endpoint |
| 11 | **Web UI** | Next.js chat: streaming, markdown, code highlight, upload, history, dark mode, mobile | Usable end-to-end from a browser |
| 12 | **Automation** | Scheduled crawl → clean → train → eval → deploy, single-command scripts | Nightly refresh runs unattended |
| 13 | **Documentation** | Install, train, infer, deploy, troubleshoot, scale | A stranger can reproduce the model from the docs |

## Corpus: final numbers

Phase 3 is complete, so the corpus size is now measured end to end rather than
estimated at any stage.

| Stage | Documents | Characters | ~Tokens |
|---|---:|---:|---:|
| Ingested (`data/interim/`) | 358,077 | 77.6 M | 25.2 M |
| Cleaned (`data/processed/`) | 345,872 | 77.1 M | 24.9 M |
| **Deduplicated (`pretrain_v1`)** | **286,434** | **66.0 M** | **21.3 M** |

Deduplication removed 56,331 exact and 3,107 near duplicates — 17.2% of the
cleaned corpus. Where it came from is informative:

| Source | Removed | Why |
|---|---:|---|
| `hf_dilmash_parallel` | 20.7% | the same Karakalpak sentence appears in the kaa_eng, kaa_rus **and** kaa_uzb splits |
| `gov_sud_latin` | 48.9% | archive and pagination pages re-rendering the same rulings |
| `blog_shagalalab` | 45.7% | Blogger date archives re-rendering posts |
| `gov_qrdsm` | 37.3% | multilingual site serving overlapping content |
| `glotcc_kaa` | 27.3% | scraped `kaa.wikipedia`, overlapping `wiki_kaa` |

**~21.3 M unique tokens is the number Phase 5 must be planned against.**

## Current position

**Phase 1 complete. Phase 2 in progress.**

| Step | State |
|---|---|
| 2.1 Source audit — legality, quality, size, cadence, script | ✅ done, [docs/SOURCES.md](SOURCES.md) |
| 2.2 `crawlers/core` — fetcher, robots, rate limiter, resume state, `qm crawl` | ✅ done |
| 2.3 Tier 1 ingesters — Wikipedia dump + HF datasets | ✅ done |
| 2.4 Tier 2 crawls + HTML extraction | ✅ done (sud.uz still filling) |
| 2.5 kknews permission, then crawl at scale | ⏳ blocked on a decision |
| 2.6 PDF/OCR path — National Library archive | stretch |

**Corpus in hand: 357,347 documents, 75.8 M characters, ~24.4 M estimated tokens**
— measured by running the pipeline, not projected. See
[SOURCES.md](SOURCES.md#1-the-headline-number).

The ~250 MB opening estimate did not survive contact, in two stages:

* **Tier 1** — MADLAD-400 ships no Karakalpak at all (−60 MB) and GlotCC-V1 has
  172 documents rather than 30 MB (−28 MB). 174 MB estimated → 71 MB measured.
* **Tier 2** — 78 MB estimated → ~11 MB projected. Karakalpak institutional
  articles run 1,500–3,200 characters, not the 5–10 K assumed.

Planning figure after dedup: **20–25 M unique tokens**. Phase 5 is therefore an
adaptation task, and LoRA over full fine-tuning is a requirement rather than a
preference.
