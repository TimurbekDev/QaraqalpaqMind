# QaraqalpaqMind

An open-source Large Language Model stack for the **Karakalpak** language (*qaraqalpaq tili*, ISO 639-3 `kaa`, ~2 million speakers, Turkic / Kipchak-Nogai branch, Latin script since 1994 with a large Cyrillic legacy corpus).

Base model: **Qwen3-8B** → Continued Pretraining → SFT → DPO → RAG → vLLM serving → Next.js chat UI.

> **Status:** Phases 1–7 complete (data → CPT → SFT → DPO), deployment-ready for a
> single RTX 4090. See [docs/ROADMAP.md](docs/ROADMAP.md) and
> [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). No training run has been executed yet.
>
> **Corpus: 286,434 unique documents / 66.0 M characters / 28.7 M Qwen3 tokens**
> (`pretrain_v1`), measured end to end by running the pipeline — crawl, clean, dedup —
> and counted with the real tokenizer, not estimated.
> See [docs/SOURCES.md](docs/SOURCES.md) and [docs/TOKENIZER.md](docs/TOKENIZER.md).
>
> An opening ~250 MB projection did not survive contact: MADLAD-400 ships no Karakalpak
> at all, GlotCC-V1 has 172 documents rather than 30 MB, and the live Karakalpak web is
> ~11 MB rather than 78 MB. Deduplication then removed a further 17.2%.
>
> That makes this an **adaptation** project, not a from-scratch one. Qwen3-8B saw ~36 T
> tokens; 28.7 M Karakalpak tokens is enough to teach a model that already knows
> Kazakh, Uzbek and Turkish about this Kipchak relative — and not enough for
> full-parameter fine-tuning, which is why Phase 5 uses LoRA.
>
> Karakalpak costs **1.88× more tokens than English** for identical content. Uzbek, its
> closest well-resourced relative, costs 1.75× — so most of that is agglutinative
> morphology rather than a Karakalpak-shaped hole in Qwen3's vocabulary.

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
│   │   └── sources/         #   one module per crawled site
│   ├── ingest/              # Phase 2: bulk loaders — Wikipedia dumps, HF datasets
│   │                        #   (crawlers fetch page by page; ingesters pull in bulk;
│   │                        #    both emit the same Document records + a manifest)
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

**Python 3.12 or 3.13.** 3.12 is a hard floor: `common/config.py` uses PEP 695
generics, which 3.11 cannot parse. 3.13 works — torch has supported it since 2.5.

```bash
# 1. Toolchain. uv installs its own private 3.12 and never touches your system Python.
winget install --id astral-sh.uv -e     # Windows;  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv --python 3.12

# 2. Install
uv pip install -e ".[dev,crawl,ingest]"

# 3. Configure
cp .env.example .env                    # Windows: copy .env.example .env

# 4. Verify
uv run qm version
uv run qm doctor
```

<details>
<summary>Without uv (stdlib venv + pip)</summary>

```bash
py -3.12 -m venv .venv          # Windows;  Linux: python3.12 -m venv .venv
.venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install -e ".[dev,crawl,ingest]"
```
</details>

`qm doctor` prints which optional extras are installed. Add them as you reach each phase:

| Extra | Phase | For |
|---|---|---|
| `.[crawl]` | 2 | httpx, trafilatura, selectolax, pypdf |
| `.[ingest]` | 2 | datasets, huggingface-hub, mwparserfromhell |
| `.[clean]` | 3 | datasketch, ftfy, fasttext |
| `.[train]` | 5 | torch, transformers, trl, peft, accelerate, bitsandbytes (Linux + CUDA) |
| `.[flash]` | 5 | flash-attn — optional, compiles from source, see [DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| `.[distributed]` | 5 | deepspeed — multi-GPU only |
| `.[rag]` | 9 | qdrant-client, sentence-tdoransformers |
| `.[serve]` | 10 | vllm, fastapi, uvicorn |

---

## Building the corpus

```bash
# What is registered, and why each source is on or off
qm crawl list --all
qm ingest list

# Bulk sources: Wikipedia dumps and HF datasets. No crawling, open licences.
qm ingest run wiki_kaa
qm ingest all --bulk-only

# Crawl the live Karakalpak web, politely and resumably
qm crawl all --max-pages 600            # concurrent across hosts
qm crawl status                         # resume-safe; Ctrl-C and re-run
qm crawl run gov_sud_latin -n 100       # one source

# Turn crawled HTML into interim documents (re-runnable, never re-fetches)
qm ingest run gov_sud_latin
```

Every source must be declared in [`configs/crawl/sources.yaml`](configs/crawl/sources.yaml)
with its licence, legal basis and rate limit before anything fetches it — so those decisions
are reviewed in a diff rather than improvised inside a scraper. `respect_robots` is on for
every source and a unit test enforces it.

---

## Training on a GPU

`data/` is not in git, so a fresh clone has no corpus. See
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** for the full RunPod guide; the short
version:

```bash
bash scripts/runpod_setup.sh            # install, caches, preflight
qm ingest pull timurbek-saburov/qaraqalpaqmind-data   # or rebuild: qm ingest all --bulk-only
qm train preflight --config configs/cpt/qwen3_8b_qlora_24gb.yaml
qm train cpt       --config configs/cpt/qwen3_8b_qlora_24gb.yaml
```

Verified for a single **RTX 4090 (24 GB)**: peak ~13 GB, ~10.8 GB headroom,
~1,750 steps, ~6–10 h. Runs resume automatically after a pod reclaim.

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Project architecture & shared foundation | ✅ done |
| 2 | Data collection (source audit, crawlers, ingesters) | 🔨 Tier 1 ingested, Tier 2 crawling |
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
