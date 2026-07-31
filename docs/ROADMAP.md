# Roadmap

One phase at a time. A phase is done when its deliverables exist, run, and are covered by at least a smoke test.

| # | Phase | Deliverables | Gate to pass before moving on |
|---|---|---|---|
| 1 | **Architecture** | Folder tree, `common/` (paths, logging, config, io), `pyproject.toml`, `qm` CLI, docs | `qm doctor` runs; unit tests green |
| 2 | **Data collection** | Source audit (legality, licence, size, update cadence), `crawlers/core` + ≥6 source modules, resumable state | ≥50 MB raw Karakalpak text landed in `data/raw/` |
| 3 | **Cleaning** | HTML/PDF extraction, unicode/ftfy normalisation, kaa language-ID model, quality scorer, exact + MinHash dedup | `data/processed/` with a measured keep-rate report |
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

## Current position

**Phase 1 complete.** Next: Phase 2 — source audit before a single line of scraper code.
