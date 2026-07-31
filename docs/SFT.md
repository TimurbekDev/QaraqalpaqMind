# Supervised fine-tuning

CPT teaches the language. SFT teaches the model to be *useful* in it.

```bash
qm sft seeds      # what hand-authored data exists
qm sft plan       # how large a balanced mixture the data supports
qm sft build      # assemble it
qm sft inspect    # check how records render as chat
qm train sft --config configs/sft/qwen3_8b_qlora_24gb.yaml
```

---

## 1. The honest state of the data

**Three of nine tasks have real data. Six have seed sets of 5–12 examples.**

| Task | Source | Available | Kind |
|---|---|---:|---|
| translation | dilmash, both directions | ~430,000 | human-translated |
| grammar | reversed orthography normalisation | unlimited | generated, labelled synthetic |
| summarization | Wikipedia lead sections | ~2,500 | human-written, heuristically paired |
| instruction | `seeds/instruction.jsonl` | 12 | hand-authored |
| reasoning | `seeds/reasoning.jsonl` | 8 | hand-authored |
| math | `seeds/math.jsonl` | 8 | hand-authored |
| coding | `seeds/coding.jsonl` | 6 | hand-authored |
| qa | `seeds/qa.jsonl` | 6 | hand-authored |
| conversation | `seeds/conversation.jsonl` | 5 | hand-authored |

`qm sft plan` reports what that means:

> Target size in config: 50,000. **Proportion-respecting maximum: 60**, limited by `qa`.

Sixty records. The pipeline is complete and tested; **the data for six tasks is
not, and no amount of engineering fixes that.** It needs native speakers writing
examples. `seeds/README.md` explains the format and the two honest paths to
scaling it.

Building at the configured 50,000 target produces a mixture that is 90%
translation and grammar — a translator with a spell-checker attached, not an
assistant. That is reported as a shortfall rather than hidden.

## 2. What the builders do

### Translation — real data, capped

215,417 dilmash pairs, emitted in **both directions**. A model trained only
kaa→X translates *out of* Karakalpak and not into it, and into is the direction
a Karakalpak assistant needs.

Reads from `data/processed/`, not `data/interim/`. That distinction was a bug
found by reading built output: interim is pre-normalisation, so its Karakalpak
still carries `senin biraq ózin` where the standard is `seniń biraq óziń`.
Those would have become training *targets*, teaching inconsistent spelling.
Processed is also already quality-filtered, so the rows whose "Karakalpak"
column is actually English are gone.

Capped at 30% of the mixture despite being able to supply 90%. Letting
abundance decide the mixture is how a translation engine gets trained by
accident.

### Grammar — generated from real error distribution

Karakalpak has no error-annotated corpus. But Phase 3 normalised everything to
Latin 2016, and **the errors people actually make are the older conventions
that normalisation removed.** Running it backwards produces pairs whose error
distribution is real rather than invented:

```
correct     Qaraqalpaqstan Respublikasınıń paytaxtı — Nókis
apostrophe  Qaraqalpaqstan Respublikasi'ni'n' paytaxti' — No'kis   (2009)
umlaut      Qaraqalpaqstan Respublikasınıń paytaxtı — Nökis        (1994)
stripped    Qaraqalpaqstan Respublikasiniń paytaxti — Nokis        (no diacritics)
mixed       conventions varying word to word, as transitional text does
```

The *correct* side is genuine corpus text; the *incorrect* side is generated,
so every record carries `synthetic: true`. A corrector trained on invented
errors and presented as trained on observed ones is one nobody can audit.

Corruption is seeded, so a rebuild reproduces the same dataset.

### Summarization — Wikipedia lead sections

An article's opening paragraph is a summary written by someone who read the
article. It is the only genuine Karakalpak summarisation supervision available
without paying annotators.

Its weakness is stated rather than hidden: a lead is an *introduction*, not an
abstract. It over-represents definitions and dates. Records are marked
`human_reviewed: false`, and pairs are dropped when compression falls outside
2–50%.

## 3. Mixture assembly

Three things the trainer must not be left to do:

1. **Cap per task.** Proportions are chosen, not inherited from whatever was
   cheap to obtain.
2. **Deduplicate across tasks.** Grammar and summarisation both draw on
   `pretrain_v1`, so cross-task duplication is expected.
3. **Check contamination.** FLORES+ sentences reaching the SFT set would
   invalidate the Phase 8 benchmark exactly as they would in pretraining.

Train/validation splitting happens **per task**, before shuffling. Splitting the
concatenated stream would produce a validation set made entirely of whichever
task landed last.

Builders are **interleaved**, not concatenated — otherwise translation exhausts
the caps before the seed sets are read at all.

## 4. Training choices

| Setting | SFT | CPT | Why the difference |
|---|---|---|---|
| LoRA `r` | 32 | 64 | SFT teaches a response style on far less data; a large adapter mostly adds overfitting risk |
| learning rate | 2e-5 | 1e-4 | Every SFT example is an exact target, so it overfits faster |
| epochs | 3 | 2 | Validator rejects >5 |
| sequence length | 1024 | 2048 | Conversations, not documents; packing is off |
| packing | off | on | Packing SFT would blend one example's answer into the next |

**Completion-only loss.** Loss is computed on assistant turns only. Without it
the model learns to generate user questions as readily as answers, which at
inference looks like the model continuing your turn instead of replying to it.

**The chat template comes from the tokenizer.** Hand-assembling a prompt string
is how a model ends up unable to follow the format it is later served with.

**The CPT adapter is merged, not stacked.** Two live adapters would have to be
carried through serving, DPO and evaluation. Merging makes the language
knowledge part of the weights the SFT adapter trains against.

## 5. What has not been done

- **No run has been executed.** No GPU available at 24 GB+. Every number here
  is from measurement or standard practice, none from a loss curve.
- **Six tasks need real data.** This is the blocking item for a usable
  assistant, and it is a human-authoring problem, not an engineering one.
- **No preference data yet.** That is Phase 7.
