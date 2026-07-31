# Qwen3 tokenizer on Karakalpak

Measured 2026-07-31 with `qm tokenizer analyze` and `qm tokenizer count`
against `Qwen/Qwen3-8B` (vocabulary 151,643).

Reproduce with:

```bash
qm tokenizer analyze -n 6000     # cross-language fertility
qm tokenizer count               # exact corpus token count
```

---

## 1. Fertility, on parallel content

The comparison uses `dilmash`, which is **parallel** — the same sentence in
Karakalpak, Uzbek, Russian and English. Content is therefore held constant, so
every difference below is a property of the tokenizer.

| Language | Words | Tokens | Tokens/word | Chars/token | 1-token words | vs. English |
|---|---:|---:|---:|---:|---:|---:|
| **Karakalpak** | 97,969 | 359,037 | **3.66** | **2.27** | **12.1%** | **1.88×** |
| Uzbek | 36,602 | 113,782 | 3.11 | 2.44 | 23.3% | 1.75× |
| Russian | 33,079 | 95,236 | 2.88 | 2.81 | 38.7% | 1.52× |
| English | 40,363 | 61,109 | 1.51 | 4.27 | 92.6% | 1.00× |

**Karakalpak costs 1.88× more tokens than English for identical content.**

The number that reframes this, though, is Uzbek at **1.75×**. Uzbek is
Karakalpak's closest well-resourced relative, is written in the same Latin
script, and is far better represented in any web-scale training mix — yet it
pays almost the same penalty. So most of the cost is **agglutinative
morphology**, not a Karakalpak-shaped hole in the vocabulary. Turkic languages
build long words from stacked suffixes, and a BPE vocabulary trained mostly on
analytic languages has no units for them.

That bounds what vocabulary extension can buy: the gap between Karakalpak and
Uzbek is 7%, not 88%.

## 2. Where the Karakalpak-specific cost is

| Letter | Emitted as its own token |
|---|---:|
| `ǵ` | **100.0%** |
| `Á` | **100.0%** |
| `Ń` | **100.0%** |
| `ń` | 99.8% |
| `ó` | 74.7% |
| `ú` | 68.1% |
| `Ú` | 60.0% |
| `ı` | 42.3% |
| `Ó` | 31.3% |
| `á` | 22.5% |

A rate near 100% means the vocabulary contains **no subword at all** holding
that letter. Every word using `ǵ` or `ń` — and they are among the most common
letters in Karakalpak — is split at that point, losing the morpheme boundary
the model would otherwise generalise over.

`Ǵ` (U+01F4) is worse still. It does not survive as a single token; it falls
back to **raw UTF-8 bytes**:

```
Ǵárezsizliktiń   11 tokens   |��|á|rez|s|iz|li|kt|i|ń
QOZǴALÍSTAǴÍ     11 tokens   Q|O|Z|��|AL|Í|STA|��|Í
```

Uppercase Karakalpak is therefore disproportionately expensive — headings,
institution names and legal citations all pay it.

## 3. Worst-fragmented words

```
suyetuǵınsúymeytuǵınımızdı   14 tokens   su|yet|u|ǵ|ı|ns|ú|ym|ey|tu|ǵ|ın|ımız|dı
shólkemlestiriwshileriniń    11 tokens   sh|ó|lk|em|lest|iri|w|sh|iler|ini|ń
huqıqbuzarlıqlardıń          11 tokens   hu|q|ı|qb|uz|ar|lı|ql|ard|ı|ń
```

`shólkemlestiriwshileriniń` ("of the organisers") is one word carrying five
morphemes. Qwen3 spends 11 tokens on it and splits it in places that do not
correspond to any of them.

## 4. The corpus, in real tokens

```
286,434 documents · 7,967,803 words · 28,683,631 tokens
3.60 tokens/word · 2.30 chars/token
```

**The `chars / 3.1` estimate used throughout Phases 2 and 3 said 21,297,069.
The real count is 1.35× that.**

This is not good news. The same text costs 35% more compute than planned, and
carries no more information for it. Every token-budget figure in earlier
documents should be read as understating the training cost by about a third.

## 5. What this means for Phase 5

**The corpus is ~28.7 M Qwen3 tokens.** For reference, Qwen3-8B saw on the
order of 36 trillion. This is an adaptation budget — roughly 0.00008% of
pretraining — which confirms the plan but tightens it:

* **LoRA, not full fine-tuning.** 28.7 M tokens cannot support updating 8 B
  parameters without destroying the multilingual transfer the whole approach
  depends on.
* **Vocabulary extension is worth evaluating, but is not the main lever.** The
  Uzbek comparison caps the realistic gain at roughly 7% on fertility. The
  targeted win is narrower and cheaper: adding subwords containing `ǵ`, `ń`,
  `Ǵ` and the uppercase accented forms would remove the byte-fallback and the
  100%-isolation cases specifically. Any extension requires resizing the
  embedding matrix and training the new rows, which on this data budget is
  itself a risk.
* **Effective context is ~1.9× smaller than the nominal window.** A 4,096-token
  context holds roughly as much Karakalpak as a 2,180-token context holds
  English. Sequence-length choices in Phase 5 must be made in Karakalpak
  tokens, not in English intuition.
* **Measure fertility again after any orthography change.** Phase 3 normalises
  everything to Latin 2016; if a future decision keeps Cyrillic instead, these
  numbers do not transfer.
