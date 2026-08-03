# Preference optimisation

SFT teaches the model what a good answer looks like. DPO teaches it which of two
answers is better — and in particular, corrects failures that supervised
examples cannot express.

```bash
qm dpo build      # assemble preference pairs
qm dpo inspect    # check what the pairs actually teach
qm train dpo --config configs/dpo/qwen3_8b_qlora_24gb.yaml
```

**Built and verified: 18,336 pairs.** No shortfall on the two main criteria —
unlike SFT, where six of nine tasks are seed-starved.

---

## 1. Why pairs must differ in exactly one dimension

DPO raises the likelihood of `chosen` relative to `rejected`. **Every dimension
the two differ in is taught**, including the ones nobody intended. If `chosen`
is also longer, DPO learns "longer is better". If it happens to be about a
different topic, DPO learns a topic preference.

This is the design constraint behind every builder here, and it is why two of
the three are strong and one is weak.

## 2. The three real-signal builders

| Criterion | Share | Pairs | How the sides differ |
|---|---:|---:|---|
| `language_consistency` | 43.6% | 8,000 | **only** the language |
| `orthography` | 43.6% | 8,000 | **only** the spelling convention |
| `response_quality` | 12.7% | 2,336 | quality — but also content |

### Language consistency — the tightest pair available

The failure it targets is specific and likely: **a model whose Karakalpak is
thin answers a Karakalpak question in Russian or English**, because those are
the languages it knows best. 28.7 M tokens of continued pretraining does not
reliably stop that.

dilmash gives the pair for free. The Karakalpak sentence is `chosen`; its own
human translation is `rejected`. The two say *exactly the same thing* — they are
translations of each other — so language is the only difference.

```
prompt    Tómendegi mazmundı qaraqalpaq tilinde jazıp ber: …
chosen    Biraq sońǵı waqıtları biziń jámiyetimizde Rudin hám onıń barlıq …
rejected  However, recently, our society has witnessed the emergence of …
```

### Orthography — generated, so nothing else can vary

Reuses the Phase 6 grammar corruption. Both sides come from the *same sentence*,
so they cannot differ in content, length, register or topic — only in which
spelling convention was used.

```
chosen    … rayonındaǵı "Tebinbulaq" káni negizinde kán-metallurgiya …
rejected  … rayonindagi "Tebinbulaq" käni negizinde ka'n-metallurgiya …
```

Prompts vary across "write this correctly" phrasings rather than only
"correct this", so the preference generalises to free generation rather than
only to correction requests.

### Response quality — the weak one, and why it is still here

Pairs a kept document against one the cleaner rejected. **The two sides are
different documents, so they differ in content as well as quality.** That is
stated rather than glossed over; it is the weakest signal of the three.

It is included because the failure it targets — a model that emits navigation
menus and raw wikitext — is real and visible, and the alternative is no quality
signal at all until a trained model exists to sample from.

**A bug caught by inspecting the output.** The first build produced a pair whose
`chosen` side was an arithmetic table:

```
10 + 6 = 16
10 + 7 = 17
10 + 8 = 18 …
```

It had passed the Phase 3 cleaner, and the builder's only requirement was
"was kept". But *surviving cleaning* and *being worth imitating* are different
bars, and conflating them would have taught the model to emit multiplication
tables. A prose gate now requires the chosen side to have real sentence
structure, low repetition, few digits and terminal punctuation. That cut the
criterion from 4,000 pairs to 2,336 — the shortfall is reported rather than
padded with junk.

## 3. On-policy pairs — written, not run

`builders/on_policy.py` samples several completions per prompt from the SFT
model, scores them, and pairs best against worst. This is what DPO was designed
for: correcting *this* model's *actual* failures rather than teaching a general
preference.

The scorer is rule-based — language consistency, orthography, repetition,
length — because a Karakalpak reward model does not exist and an LLM judge would
be scoring a language it barely knows, measuring its own weakness rather than
the candidate's.

**It requires a trained SFT checkpoint and a GPU, and has not been executed.**
The other three builders were run against real data; this one is written from
the same spec and is unverified. Its scoring logic is unit-tested with an
injected sampler, but the generation path is not. Treat its first run as a
debugging session.

## 4. Training choices

| Setting | DPO | SFT | Why |
|---|---|---|---|
| learning rate | **5e-6** | 2e-5 | Refines a working model; a large rate erases the instruction-following SFT installed. Validator rejects >1e-5 |
| LoRA `r` | 16 | 32 | Refinement pass — a large adapter mostly adds room to over-optimise |
| epochs | 1 | 3 | Preference training degrades quickly. Validator rejects >3 |
| batch | 1×16 | 2×8 | DPO holds four sequences per example: chosen and rejected, policy and reference |

**`beta = 0.1`** controls how far the policy may drift from the reference. Low
values allow large drift and, on a small preference set, **reward hacking** —
the model finds a degenerate output that scores well and collapses onto it.
Raise it if outputs degrade.

**No second copy of the weights.** With LoRA the reference model is this same
model with adapters disabled (`ref_model=None`). That is what makes DPO fit on
one 24 GB card at all.

A default-construction bug was caught by the tests here: `DPOConfig` inherited
the shared optimiser defaults, whose 1e-4 learning rate its own validator
rejects — so `DPOConfig()` could not be constructed at all. It now carries
DPO-appropriate defaults.

## 5. What has not been done

- **No run executed.** No GPU at 24 GB+.
- **On-policy builder unverified**, as above.
- **No helpfulness or factuality preferences.** Both need either human ranking
  or a trained model to sample from. Fabricating them from rules would teach the
  rules, not the preference.
