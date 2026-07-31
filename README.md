# QaraqalpaqMind

An open-source Large Language Model stack for the **Karakalpak** language (*qaraqalpaq tili*, ISO 639-3 `kaa`, ~2 million speakers, Turkic / Kipchak-Nogai branch, Latin script since 1994 with a large Cyrillic legacy corpus).

Base model: **Qwen3-8B** → Continued Pretraining → SFT → DPO → RAG → vLLM serving → Next.js chat UI.

> **Status:** Phase 1 complete, Phase 2 in progress. See [docs/ROADMAP.md](docs/ROADMAP.md).
> The Karakalpak data landscape is audited in [docs/SOURCES.md](docs/SOURCES.md) — ~250 MB
> of text is realistically obtainable, which is an *adaptation* budget, not a from-scratch one.

---

## Repository layout

```
QaraqalpaqMind/
├── configs/                 # every run is driven by a YAML file here
│   ├── crawl/               #   per-source crawl rules (domains, rate limits, selectors)
│   ├── clean/               #   filter thresholds, dedup params, language-ID cutoffs
│   ├── tokenizer/           #   vocabulary-extension + tokenizer-fertility experiments
│   ├── cpt/                 #   continued-pretraining hyperparameters
│   ├── sft/                 #   supervised fine-tuning hyperparameters
│   ├── dpo/                 #   preference-optimisation hyperparameters
│   ├── eval/                #   benchmark suites and decoding settings
│   ├── rag/                 #   chunking, embedding model, Qdrant collection, reranker
│   └── serve/               #   vLLM engine args, API limits, auth
│
├── data/                    # NEVER committed to git (see .gitignore)
│   ├── raw/<source_id>/     # byte-identical copies of what the crawler fetched
│   ├── interim/             # text extracted from raw, still dirty
│   ├── processed/           # cleaned + deduplicated shards
│   ├── datasets/            # final, train-ready JSONL
│   │   ├── pretrain/        #   CPT corpus
│   │   ├── sft/             #   instruction / chat data
│   │   ├── dpo/             #   (prompt, chosen, rejected) triples
│   │   └── eval/            #   held-out benchmark splits
│   └── manifests/           # COMMITTED: sha256 + row counts + licence per dataset
│
├── src/qaraqalpaqmind/      # all importable Python lives here (src-layout)
│   ├── common/              # paths, logging, config loading, JSONL I/O
│   ├── crawlers/            # Phase 2: polite, resumable source-specific fetchers
│   │   ├── core/            #   shared HTTP client, robots.txt, rate limiter, state store
│   │   └── sources/         #   one module per source (wikipedia, news, gov, pdf, ...)
│   ├── preprocessing/       # Phase 3a: HTML→text, PDF→text, OCR fixes, segmentation
│   ├── cleaning/            # Phase 3b: quality heuristics, spam/ads/boilerplate filters
│   ├── dedup/               # Phase 3c: exact hashing + MinHash-LSH near-duplicate removal
│   ├── tokenizer/           # Phase 4b: fertility analysis, vocabulary extension
│   ├── training/
│   │   ├── cpt/             # Phase 5: continued pretraining on raw Karakalpak text
│   │   ├── sft/             # Phase 6: TRL SFTTrainer, chat templates, packing
│   │   └── dpo/             # Phase 7: TRL DPOTrainer, reference model handling
│   ├── evaluation/          # Phase 8: KaaBench harness, metrics, LLM-as-judge
│   ├── rag/
│   │   ├── ingest/          # Phase 9a: chunking, embedding, Qdrant upsert
│   │   └── retrieve/        # Phase 9b: hybrid search, reranking, citation assembly
│   └── api/                 # Phase 10: FastAPI gateway
│       ├── routes/          #   /v1/chat/completions, /v1/rag, /healthz
│       ├── middleware/      #   auth, rate limiting, request logging, metrics
│       └── schemas/         #   pydantic request/response contracts
│
├── web/                     # Phase 11: Next.js + React + Tailwind chat UI
│
├── deployment/              # Phase 10/12: how it runs in production
│   ├── docker/              #   Dockerfiles (api, vllm, worker, web)
│   ├── compose/             #   docker-compose stacks (dev, gpu, full)
│   ├── nginx/               #   TLS termination, SSE-safe reverse proxy config
│   └── systemd/             #   unit files for bare-metal GPU hosts
│
├── scripts/                 # Phase 12: one-command shell/python entrypoints
├── benchmarks/              # Phase 8: KaaBench task files (COMMITTED, small, human-checked)
├── tests/                   # pytest: unit (fast, no network) + integration (marked slow)
├── notebooks/               # exploratory analysis only; nothing here is a dependency
├── docs/                    # Phase 13: installation, training, deployment, troubleshooting
├── logs/                    # runtime logs, rotated; crawl/ train/ serve/ subdirs
└── models/                  # checkpoints: base/ cpt/ sft/ dpo/ merged/ gguf/
```

### Why the folders are split this way

| Concern | Rule |
|---|---|
| **Code vs. artefacts** | `src/` is versioned and reviewed; `data/`, `models/`, `logs/` are reproducible outputs and stay out of git. Only `data/manifests/` is committed — it records *which* dataset a checkpoint was trained on, via sha256, without storing the bytes. |
| **src-layout** | Package sits in `src/` so tests import the *installed* package, not stray files in the CWD. Catches missing-`__init__` and packaging bugs before release. |
| **Stage-per-directory data flow** | `raw → interim → processed → datasets` is a one-way pipeline. Each stage is re-runnable from the previous one, so a bug in the cleaner never forces a re-crawl. `raw/` is immutable. |
| **Config out of code** | No hyperparameter is ever hardcoded. A training run is fully described by one YAML in `configs/` + one dataset manifest → reproducible six months later. |
| **Optional dependency groups** | A crawl machine installs `.[crawl]`; a GPU node installs `.[train]`. Nobody drags 3 GB of CUDA wheels onto a scraper box. |
| **preprocessing ≠ cleaning ≠ dedup** | Different failure modes, different tests. *Preprocessing* converts formats (HTML/PDF → text). *Cleaning* judges quality of text that is already text. *Dedup* is a corpus-level operation that needs the whole set in scope. Keeping them apart lets you re-run only the layer you changed. |
| **crawlers/core vs. crawlers/sources** | Politeness, retries, resume state and robots.txt handling are written once in `core/`. Adding a new site is then a ~40-line module in `sources/`. |
| **benchmarks/ committed, data/ not** | Evaluation sets are small, hand-verified, and must be diffable in review. Training corpora are huge and machine-generated. |

---

## Quick start

```bash
# 1. Python 3.12 (NOT 3.13/3.14 - torch, vLLM and flash-attn have no wheels there yet)
py -3.12 -m venv .venv          # Windows
# python3.12 -m venv .venv      # Linux

.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux

# 2. Install the core package in editable mode
pip install -e ".[dev]"

# 3. Configure
copy .env.example .env          # Windows
# cp .env.example .env          # Linux

# 4. Verify
qm version
qm doctor
```

`qm doctor` prints which optional extras are installed. Install them as you reach each phase:

```bash
pip install -e ".[crawl]"   # Phase 2
pip install -e ".[clean]"   # Phase 3
pip install -e ".[train]"   # Phase 5  (Linux + CUDA)
pip install -e ".[rag]"     # Phase 9
pip install -e ".[serve]"   # Phase 10
```

---

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Project architecture & shared foundation | ✅ done |
| 2 | Data collection (crawlers, legality, source audit) | 🔨 audit done, crawlers next |
| 3 | Cleaning & deduplication pipeline | |
| 4 | Dataset schemas (JSONL) + tokenizer analysis | |
| 5 | Continued pretraining (Qwen3-8B) | |
| 6 | Supervised fine-tuning (TRL) | |
| 7 | Preference optimisation (DPO) | |
| 8 | KaaBench evaluation suite | |
| 9 | RAG (Qdrant + reranking + citations) | |
| 10 | Deployment (vLLM, FastAPI, Docker, Nginx) | |
| 11 | Web UI (Next.js) | |
| 12 | Automation (one-command pipelines) | |
| 13 | Documentation | |

## Licence

Apache-2.0 for the code. Dataset licences are tracked per source in `data/manifests/`; model weights inherit the Qwen3 licence.
