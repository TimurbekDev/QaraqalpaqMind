# Seed datasets

Hand-authored Karakalpak examples for the tasks that have **no existing
corpus**: instruction following, dialogue, question answering, reasoning,
mathematics and coding.

Translation, grammar and summarisation are not here — they are derived from
real data by the builders in
[`training/sft/builders/`](../src/qaraqalpaqmind/training/sft/builders/).

These files are committed, for the same reason `benchmarks/` is: they are
small, hand-checked, and must be diffable in review.

## Format

One JSON object per line. Give the task and its fields; provenance is filled in
by the loader, so contributing does not require knowing the record schema.

```json
{"task": "instruction", "instruction": "...", "output": "..."}
{"task": "qa", "context": "...", "question": "...", "answer": "..."}
{"task": "math", "problem": "...", "solution": "...", "answer": "..."}
```

Fields per task are listed by:

```bash
qm schema list
qm schema example instruction     # a full worked record
```

Validate before committing — an invalid row is skipped with a warning at load
time, which is easy to miss:

```bash
qm schema validate seeds/instruction.jsonl
qm sft seeds                       # counts per task
```

## Current contents

| File | Task | Records |
|---|---|---:|
| `instruction.jsonl` | instruction | 12 |
| `reasoning.jsonl` | reasoning | 8 |
| `math.jsonl` | math | 8 |
| `coding.jsonl` | coding | 6 |
| `conversation.jsonl` | conversation | 5 |
| `qa.jsonl` | qa | 6 |

**This is a seed set, not a dataset.** Forty-five examples will not teach
instruction following on their own. They exist to establish the format, to give
the mixture builder something real to sample, and to serve as the reference
style for whatever scales them up.

## Scaling up

Two honest paths, and the difference matters:

1. **Human authoring.** Native speakers write more examples in this format.
   Highest quality, and the only route that adds genuinely new knowledge.
   Records stay `synthetic: false`, `human_reviewed: true`.

2. **Model-assisted generation.** Use a strong model to expand these seeds,
   then have a native speaker review the output. Records **must** be written
   with `synthetic: true` and a `generator` name — the schema enforces that a
   generator cannot be named without the synthetic flag.

Never mix the two silently. A corpus that cannot distinguish written examples
from generated ones cannot be audited later, and quality problems become
untraceable.

## Writing guidance

- **Answer in Karakalpak.** Prompts and answers both. A model trained on English
  answers to Karakalpak questions learns to switch languages.
- **Use the current orthography** (Latin 2016: `á ǵ ń ó ú ı`). The grammar
  builder generates the older conventions deliberately; seeds should not.
- **Prefer local knowledge.** Karakalpakstan geography, history, literature and
  institutions are what no other model will know. Generic world facts are
  already covered by the base model.
- **Show the working** in reasoning and maths. The `reasoning` field exists so
  the model learns to derive rather than to recall.
- **Keep answers honest.** If an example asserts a fact, it should be one. These
  become training targets.
