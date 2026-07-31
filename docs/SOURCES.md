# Karakalpak data source audit

**Audit date: 2026-07-31.** Every claim below was produced by a live probe, not by assumption.
Reproduce with:

```bash
python scripts/audit_sources.py --include-disabled
```

The machine-readable version is [`configs/crawl/sources.yaml`](../configs/crawl/sources.yaml); the raw probe output is written to `data/manifests/source_audit_<date>.json`.

---

## 1. The headline number

| | MB of Karakalpak text |
|---|---|
| Open-licence bulk datasets (no crawling) | ~174 |
| Verified live web, crawlable today | ~78 |
| Blocked / needs permission / needs OCR | ~60 |
| **Realistically obtainable, tier 1 + 2** | **~250 MB** |

250 MB of text is roughly **60–75 million Qwen3 tokens** after cleaning and deduplication — and expect to lose 30–50% of the web portion in Phase 3.

**Calibrate expectations now:** Qwen3-8B saw on the order of 36 trillion tokens. Karakalpak is not going to be *learned* from scratch here. The realistic goal for continued pretraining is **adaptation** — teaching a model that already knows Kazakh, Uzbek, Turkish and Russian that this closely-related Kipchak language exists, and grounding it in Karakalpak orthography, morphology and Karakalpakstan-specific facts. That is achievable with 50–100M tokens. Building a Karakalpak model from zero is not.

This is also why Phase 5 will default to LoRA over full fine-tuning: with this little data, full-parameter training on 8B weights overfits and induces catastrophic forgetting of the multilingual ability we are relying on.

---

## 2. Tier 1 — open licence, bulk download, zero crawl risk

Start here. No robots.txt to honour, no rate limits, no legal ambiguity.

| Source | Licence | Size | Quality | Updates | Script | Notes |
|---|---|---|---|---|---|---|
| [Karakalpak Wikipedia](https://kaa.wikipedia.org) dumps | CC-BY-SA-4.0 | ~22 MB | 4/5 | weekly | Latin | ~11.5k articles (Mar 2026), up from <2k in 2023 |
| [bekan/karakalpak_corpus_v2_m](https://huggingface.co/datasets/bekan/karakalpak_corpus_v2_m) | MIT | 21.4 MB | 4/5 | rarely | Latin | 135,667 sentences, ~2.2M words |
| [tahrirchi/dilmash](https://huggingface.co/datasets/tahrirchi/dilmash) | MIT | ~40 MB | 5/5 | rarely | Latin | 300k parallel pairs, kaa↔en/ru/uz |
| [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) kaa | ODC-BY-1.0 | ~60 MB | 2/5 | static | both | Common Crawl; noisy, needs our own language ID |
| [GlotCC-V1](https://huggingface.co/datasets/cis-lmu/GlotCC-V1) kaa | CC0-1.0 | ~30 MB | 2/5 | rarely | both | Document-level CC; overlaps MADLAD heavily |
| [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) kaa_Latn | CC-BY-SA-4.0 | 0.5 MB | 5/5 | rarely | Latin | **HELD OUT — never train on this** |

### Legality

All six carry explicit open licences. Two caveats that matter:

- **CC-BY-SA (Wikipedia, FLORES+)** is share-alike. It applies to *redistribution of the text*. Whether trained model weights are a derivative work is legally unsettled; the practical convention — followed by essentially every open LLM — is that weights are not, but the *dataset* we publish must carry the CC-BY-SA notice and attribution. We will do that in `data/manifests/`.
- **`allenai/nllb`** (mined bitext, ~25 MB) is **CC-BY-NC**. It is registered but **disabled**. Non-commercial licensing propagates to anything you claim is derived from it, so including it would force a non-commercial release of the whole model. Not worth 25 MB of machine-mined pairs. If the project later accepts an NC release, flip `enabled: true`.

### The FLORES+ contamination trap

FLORES+ is the standard translation benchmark for low-resource languages, so it is also *in* several of the web corpora above — MADLAD-400 and GlotCC both scraped pages that quote it. If it leaks into pretraining, Phase 8 translation scores become fiction.

Phase 3 will build a blocklist of normalised FLORES+ sentence hashes and strip exact and near-duplicate matches from every training split. This is not optional.

---

## 3. Tier 2 — verified live web

Every row was fetched on 2026-07-31, text-extracted, and scored by [`preprocessing/script.py`](../src/qaraqalpaqmind/preprocessing/script.py). `kaa` is that detector's confidence (0–1).

| Source | Type | kaa | Script | Size est. | Quality | Updates | robots |
|---|---|---|---|---|---|---|---|
| [kknews.uz/qq](https://kknews.uz/qq/) — Qaraqalpaqstan Xabar Agentligi | news | **1.00** | Latin | ~30 MB | 4/5 | daily | allow |
| [joqargikenes.uz/qr](https://joqargikenes.uz/qr/) — parliament | gov | **0.94** | Latin | ~10 MB | 5/5 | weekly | none |
| [qaraqalpaqstan.sud.uz/qqc](https://qaraqalpaqstan.sud.uz/qqc/) — judiciary | gov | **1.00** | Cyrillic | ~5 MB | 5/5 | weekly | allow |
| [qaraqalpaqstan.sud.uz/qql](https://qaraqalpaqstan.sud.uz/qql/) — judiciary | gov | **1.00** | Latin | ~4 MB | 5/5 | weekly | allow |
| [ndpi.uz/qq](https://ndpi.uz/qq/) — Nukus Pedagogical Institute | edu | **1.00** | Latin | ~5 MB | 4/5 | weekly | allow |
| [qrdsm.uz](https://qrdsm.uz/) — Ministry of Health | gov | **0.88** | Latin | ~3 MB | 4/5 | weekly | allow |
| [shagalalab.com](https://www.shagalalab.com/) — software blog | blog | **1.00** | Cyrillic | ~1 MB | 4/5 | rarely | allow |

### Measured volume, from sitemaps

Not guesses — these come from parsing each site's `wp-sitemap.xml`:

| Site | URLs found |
|---|---|
| kknews.uz | 1,785 posts in one sitemap child alone (children 1–2 return HTTP 500) |
| qaraqalpaqstan.sud.uz | 496 Cyrillic + 348 Latin posts |
| qrdsm.uz | 352 posts |
| ndpi.uz | 300 posts + 169 pages |
| kkmi.uz | 87 posts + 103 pages under `/qq/` |

### Three things this audit found that guesswork would have missed

**1. Locale segments are inconsistent and cannot be assumed.**
`kknews.uz` uses `/qq/`. `joqargikenes.uz` uses **`/qr/`**. `qaraqalpaqstan.sud.uz` uses **`/qql/`** for Latin and **`/qqc/`** for Cyrillic. Guessing `/kaa/` would have found nothing anywhere.

**2. A locale segment does not prove the content is Karakalpak.**
`kkmi.uz/qq/` renders *Russian* despite the `/qq/` path. The crawler must score every page, never trust the URL. This is exactly why the audit runs a detector instead of checking paths.

**3. The judiciary publishes the same register in both scripts.**
`/qql/` (348 posts, Latin) and `/qqc/` (496 posts, Cyrillic) are the same institution, same content type. That is near-parallel Latin↔Cyrillic data from a single source — the ideal training and validation set for the Phase 3 transliterator. Note that the *Cyrillic* locale has more articles, which contradicts the assumption that Cyrillic Karakalpak is only a legacy concern.

### Legality of Tier 2

**Government and state-media texts (kknews, joqargikenes, sud.uz, qrdsm)** — official works of a state body. Uzbekistan's copyright law, like most, excludes official documents, laws and court decisions from copyright protection. State news agency output is a weaker case than legislation, but is published for public dissemination. We treat all four as `government_work`.

**Educational institutions (ndpi.uz) and the blog (shagalalab.com)** — ordinary copyright applies. We classify these `fair_use_train`: crawlable, robots-clean, and used for model training rather than republication. Note that this is a *training* justification, not a redistribution one — we will not publish verbatim corpora from these sources, only the trained weights and a manifest of URLs and hashes.

**Crawler etiquette we commit to, regardless of legal necessity:**
- honour `robots.txt` for every source (`respect_robots: true` is the default and no source overrides it)
- 2–4 second delay per domain, single connection, no parallel hammering of one host
- an honest User-Agent with a project URL and a contact address
- only `/qq/`-style Karakalpak locale paths; we do not mirror the Russian or Uzbek editions we have no use for
- store provenance (source URL, fetch timestamp) with every document, so anything can be removed on request

A takedown request means we remove the source and re-run. Cheap, because the pipeline is re-runnable by design.

---

## 4. Tier 3 — blocked, dead, or wrong language

Recorded so nobody investigates them twice.

| Source | Verdict | Detail |
|---|---|---|
| `sovminrk.gov.uz` (Council of Ministers) | **BLOCKED** | HTTP 503 on all paths **and** expired TLS certificate. Host exists — re-probe monthly, high value if it returns |
| `karsu.uz` (Karakalpak State University) | **BLOCKED** | JS SPA: HTTP 200, zero extractable text, empty sitemap urlset. Needs headless browser; deferred |
| `kkmi.uz/qq/` (Medical Institute) | **SUSPECT** | Locale says `/qq/`, page renders Russian (kaa=0.00). 190 URLs exist — needs per-article probing |
| `lex.uz` (national legislation) | **NO KAA** | `/kaa` and `/qq` both 404. Uzbek + Russian only |
| `uza.uz` (national news agency) | **NO KAA** | `/kaa`, `/qq`, `/kk` all 404. No Karakalpak edition exists |
| `erkinqq.uz`, `erkinqaraqalpaqstan.uz` | **DNS FAIL** | *Erkin Qaraqalpaqstan*, the republic's flagship paper since 1924, appears to have **no website**. Its archive exists as scanned PDFs at the National Library (`rlps.natlib.uz`) → OCR project, Phase 2 stretch goal |
| `karakalpakstan.uz`, `nukusnews.uz`, `aralinfo.uz`, `qrtrk.uz`, `kk.uz` | **DNS FAIL** | Do not resolve |

### Requires permission before use

| Source | Size | Why held back |
|---|---|---|
| [kitapxana.com](http://kitapxana.com/) — literature e-library | ~25 MB | **Highest-quality Karakalpak prose on the open web** (kaa=1.00). Literary works are in copyright unless the author died >70 years ago. Ask the operators first |
| [sozlik.com](https://sozlik.com/) — dictionaries | ~3 MB | Client-rendered; API not mapped. Same operators as kitapxana — request both together |
| [jw.org/kaa](https://www.jw.org/kaa/) | ~15 MB | Terms forbid redistribution, and the register is narrow enough to visibly skew the model's voice. Excluded by policy |

**⚠️ kitapxana.com transport warning:** the library is served over **plain HTTP only**. The HTTPS vhost on the same hostname resolves to a *different site* (sozlik.com). Fetching `https://kitapxana.com/` silently returns the wrong corpus. The registry pins `http://` deliberately — and because plain HTTP offers no integrity guarantee, anything collected there must be hash-manifested and spot-checked.

### On Telegram channels

You asked about public Telegram channels. The honest position:

- Reading a **public** channel's history through the Telegram API is technically straightforward and the content is publicly visible.
- Telegram's Terms of Service restrict bulk scraping, and the API's terms are stricter than "the content is public".
- Channel posts are user-generated content with no licence grant, and often contain personal data.
- Quality is poor for pretraining: short, code-switched Karakalpak/Russian/Uzbek, heavy emoji, and forwarded duplicates.

**Recommendation: skip it.** The legal exposure and cleaning cost are high, and the marginal value over Tier 1+2 is low. If you want it later, the defensible path is channel-owner permission, not scraping. No Telegram source is registered.

---

## 5. Ecosystem finding: existing Karakalpak NLP work

The audit surfaced a small but real Karakalpak civic-tech cluster, all apparently by the same people (Shagala Lab):

- **[qaraqalpaq.uz](https://qaraqalpaq.uz/)** — an orthography checker and a **Cyrillic↔Latin transliterator**, including an **old→new alphabet converter**
- **[kitapxana.com](http://kitapxana.com/)** — the literature e-library
- **[sozlik.com](https://sozlik.com/)** — dictionaries
- **[shagalalab.com](https://www.shagalalab.com/)** — Karakalpak Android keyboard (`QqKeyboard`), weather app, blog
- **[from-to.uz](https://from-to.uz/)** — Karakalpak↔Uzbek translation and Cyrillic↔Latin conversion

Two consequences:

1. **Phase 3 gets a reference implementation.** Karakalpak has three incompatible Latin orthographies (1994 umlauts, 2009 apostrophes, 2016 acutes) plus Cyrillic. `qaraqalpaq.uz`'s converter is a public reference for the normalisation rules we have to implement.
2. **These are the people to contact** about kitapxana and sozlik licensing. One conversation potentially unlocks ~28 MB of the highest-quality text available.

Prior art worth reading before Phase 5: **[Open Language Data Initiative: Advancing Low-Resource Machine Translation for Karakalpak](https://arxiv.org/abs/2409.04269)** (arXiv:2409.04269) — the paper behind `dilmash`. It documents what already worked for Karakalpak MT, and its FLORES+ split is our Phase 8 benchmark.

---

## 6. Script and orthography: the decision that shapes everything downstream

Karakalpak text on the web is split across **four writing conventions**:

| Convention | Where it appears | Example |
|---|---|---|
| Latin 2016 (official) | all verified Tier 2 Latin sites | `Qaraqalpaqstan Respublikası` |
| Latin 2009 | older pages, Uzbek-influenced | `Qaraqalpaqstan Respublikasi` |
| Latin 1994 | rare, legacy | umlauts: `ä ö ü ñ` |
| Cyrillic | sud.uz `/qqc/`, kitapxana, shagalalab | `Қарақалпақстан Республикасы` |

**Cyrillic is not a legacy edge case.** The judiciary publishes *more* articles in Cyrillic than Latin, and the entire literary e-library is Cyrillic. Any plan that throws Cyrillic away discards the best prose.

The Phase 3 decision, stated now so Phase 5 is not surprised:

1. Keep both scripts, tagged (`script` is a first-class field in the record schema — see [ARCHITECTURE.md](ARCHITECTURE.md#3-universal-record-schema)).
2. Normalise all Latin variants to **Latin 2016**, the current official standard.
3. Transliterate Cyrillic → Latin 2016 to produce a second copy, keeping the original.
4. Train on a mix. The model should *understand* Cyrillic and *generate* Latin 2016, which matches how Karakalpak is actually used today.

The `/qql/` + `/qqc/` pair from sud.uz is how we will validate step 3 — near-parallel same-institution text in both scripts.

---

## 7. What the detector does, and what it does not

[`preprocessing/script.py`](../src/qaraqalpaqmind/preprocessing/script.py) is a **heuristic pre-filter**, not a language identifier. It blends two independent signals:

- **marker density** — letters unique to Karakalpak
- **function-word hits** — high-frequency Karakalpak grammatical words

Lexical evidence is weighted higher (0.6 vs 0.4) on purpose. Two calibration failures found and fixed while building this audit, both instructive:

1. **Capital `I` was in the marker set.** It occurs in every Latin-script language, so every English page scored non-zero. The genuinely distinctive capital is `İ`.
2. **Uzbek Cyrillic shares four of eight Cyrillic markers** (`ғ қ ў ҳ`). Before the fix, `kitobxon.com` — an Uzbek book site — scored **0.50** and would have been accepted. Splitting markers into Karakalpak-only (`ә ң ө ү`) and shared-with-Uzbek (weighted 0.25), then requiring lexical confirmation, dropped it to **0.10**. Verified Karakalpak sites stayed at 0.88–1.00.

Both failures are locked in as regression tests in [`tests/unit/test_script.py`](../tests/unit/test_script.py), alongside negative samples in Uzbek Latin, Uzbek Cyrillic, Kazakh, Russian and English.

**Limits:** it is character- and word-frequency based. It will not reliably catch machine-translated Karakalpak, will misjudge very short texts (<40 letters), and cannot detect code-switching within a document. Phase 3 replaces it with a trained fastText classifier for per-document filtering; this stays as the cheap first pass.

---

## 8. Crawl order

1. **Tier 1 bulk datasets** — no crawling, ~174 MB, immediate
2. **kknews.uz** — largest live source, ~1,800+ articles
3. **joqargikenes.uz** — formal register, no sitemap so link-following
4. **sud.uz `/qqc/` + `/qql/`** — both scripts, enables the transliterator
5. **ndpi.uz, qrdsm.uz, shagalalab.com** — smaller, straightforward
6. **Ask permission** for kitapxana.com + sozlik.com
7. **Re-probe** sovminrk.gov.uz monthly; per-article probe kkmi.uz
8. **Stretch:** National Library PDF archive → OCR pipeline
