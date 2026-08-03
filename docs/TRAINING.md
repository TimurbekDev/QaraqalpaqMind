# Continued pretraining

Every number in `configs/cpt/` is justified here. They are tuned for **this**
corpus — 28.7 M Karakalpak tokens against an 8 B model — and copying them to a
larger corpus without rereading this page is a mistake.

```bash
qm train configs                                   # what is available
qm train plan --config configs/cpt/qwen3_8b_qlora_24gb.yaml   # shape of the run
qm train cpt  --config configs/cpt/qwen3_8b_qlora_24gb.yaml   # run it
```

`qm train plan` needs no GPU. Run it first, always — launching a job that would
take a week nobody has is a mistake worth catching in one second.

---

## 1. Why Qwen3-8B

| Requirement | Why Qwen3-8B |
|---|---|
| Already knows related languages | Kazakh, Uzbek, Turkish and Russian are all in its training mix. Karakalpak is Kipchak-Nogai — the same branch as Kazakh — so cross-lingual transfer does most of the work that 28.7 M tokens cannot |
| Permissive licence | Apache-2.0. No restriction on releasing derived weights |
| Fits one consumer GPU | 8 B in 4-bit is ~5 GB, leaving room for adapters and activations on a 24 GB card |
| Long context | 32 K native, which matters more than usual given Karakalpak's 1.88× token penalty |

The alternative worth naming is a Turkic-specialised base. None at this size has
a comparable licence and general-capability floor, and starting from a weaker
model would mean teaching reasoning as well as language on a 28.7 M-token budget.

## 2. Why LoRA, not full fine-tuning

**28.7 M tokens against 8 B parameters is ~280 parameters per training token.**
That is the textbook setup for catastrophic forgetting: the model fits
Karakalpak and loses the Kazakh, Uzbek and Russian knowledge that makes the
transfer work — destroying the reason a multilingual base was chosen.

LoRA constrains the update to a low-rank subspace. The base weights are frozen,
so what the model already knows cannot be overwritten, only added to.

`configs/cpt/qwen3_8b_full_multigpu.yaml` exists for completeness and says so at
the top. The config validator **rejects** full fine-tuning above 5e-5, because
the LoRA default of 1e-4 applied to every parameter would wreck the base model.

### LoRA settings

| Setting | Value | Reasoning |
|---|---|---|
| `r` | 64 | Generous by instruction-tuning standards, deliberately. Adapting a whole language is a larger change than teaching a response style; too low a rank cannot express it |
| `alpha` | 128 | 2× rank, the usual ratio — scaling of 2.0 |
| `dropout` | 0.05 | Light. The corpus is small enough that heavy dropout wastes signal |
| `target_modules` | all 7 linear layers | Restricting to `q_proj`/`v_proj` is an instruction-tuning convention. For **language** adaptation the MLP matrices (`gate`, `up`, `down`) hold most of the lexical knowledge that has to change |

That gives ~189 M trainable parameters, **2.36% of the model**.

## 3. Sequence length, in Karakalpak tokens

`sequence_length: 2048`.

Measured: Karakalpak runs **2.30 chars/token** against English's 4.27
([TOKENIZER.md](TOKENIZER.md)). So 2048 Karakalpak tokens ≈ 4,700 characters ≈
what 1,100 English tokens would hold.

**Sequence lengths must be chosen in Karakalpak tokens, not English intuition.**
A "short" 2048-token window here is not short.

Longer would mostly concatenate unrelated documents — the median document in
this corpus is 30 words — and cost quadratic attention for no gain.

## 4. Packing

The median document is 30 words. Padding each to 2048 tokens would spend **over
90% of the compute on padding**. So documents are concatenated with an EOS
separator and cut into fixed-length chunks.

The separator is not optional. Without it the model learns that an article on
Aral Sea hydrology flows naturally into a court ruling, and never learns where a
document ends — which surfaces later as generations that will not stop.

The final partial buffer is **dropped, not padded**: at most one sequence out of
14,005, and padding it would introduce the only padded example in the run.

## 5. Optimiser and schedule

| Setting | Value | Reasoning |
|---|---|---|
| `learning_rate` | 1e-4 | LoRA updates only a low-rank projection and tolerates — needs — a rate 1–2 orders above full fine-tuning's 1e-5 |
| `adam_beta2` | 0.95 | LLM pretraining convention, not the 0.999 default. Faster adaptation of the second moment on noisy gradients |
| `lr_scheduler` | cosine | Standard; smooth decay suits a fixed step budget |
| `warmup_ratio` | 0.03 | ~52 steps of 1,750. Enough to avoid an early destabilising step |
| `min_lr_ratio` | 0.1 | Never decay to exactly zero — the last steps would contribute nothing |
| `max_grad_norm` | 1.0 | Standard clipping |
| `weight_decay` | 0.01 | Light; adapters are small |

**Epochs: 2.** More invites memorisation on a corpus this small; fewer leaves
the adapter undertrained. Watch validation loss and stop when it turns — the
config saves the best checkpoint by `eval_loss`, not the last.

## 6. Memory and hardware

### `qwen3_8b_qlora_24gb.yaml` — the default

| Component | VRAM |
|---|---|
| 4-bit NF4 base weights | ~5.0 GB |
| LoRA adapters (r=64) + gradients | ~0.7 GB |
| AdamW states (adapters only) | ~0.6 GB |
| Activations, checkpointed, seq 2048 batch 1 | ~4–6 GB |
| Logits, batch 1 × seq 2048 × 151,936 vocab | ~1.7 GB |
| **Total** | **~13–15 GB** |

Run shape: 32,768 tokens/step → 875 steps/epoch → **1,750 steps** for 2 epochs.

### The evaluation batch size is a separate setting, and its default is 8

Everything above scales with `per_device_batch_size` — except that it doesn't
govern evaluation. `TrainingArguments.per_device_eval_batch_size` defaults to
**8** independently, so a config that sets only the train batch trains fine at
batch 1 and then OOMs at the first `eval_steps` boundary, hours in.

The logits tensor is why. It is `batch × sequence × vocabulary` and the loss
upcasts it to fp32:

```
1 × 2048 × 151,936 × 4 bytes = 1.16 GiB
8 × 2048 × 151,936 × 4 bytes = 9.27 GiB   ← one allocation, on a 24 GB card
```

Two settings prevent it, and every config under `configs/` sets both:

```yaml
runtime:
  per_device_eval_batch_size: 1
  prediction_loss_only: true   # discard eval logits; keep only eval_loss
```

`prediction_loss_only` matters on its own. Without it the `Trainer` gathers
logits from every eval batch to hand to a `compute_metrics` function — 1.16 GiB
per element, accumulated across the whole eval set — when the only metric here
is `eval_loss`, which the model already returns.

`qm train plan` prints train and eval logits size and refuses configs above
6 GB. `tests/unit/test_oom_guards.py` fails if a shipped config regresses.

### `qwen3_8b_lora_a100.yaml`

bf16 base instead of 4-bit: removes quantisation error from the forward pass and
trains faster, at ~16 GB for weights. Batch 4 × accum 8 = 65,536 tokens/step.
Uses `flash_attention_2`.

### DeepSpeed

- **ZeRO-2** (`deepspeed_zero2.json`) — shards gradients and optimizer states.
  Right for multi-GPU **LoRA**, where only adapters carry optimizer state.
- **ZeRO-3** (`deepspeed_zero3.json`) — also shards parameters. Needed only for
  full-parameter training.

Batch size and precision are `"auto"` in the JSON and set in the YAML. Two
places to configure one number is how they end up disagreeing.

## 7. Catastrophic forgetting

The risk this whole design is built against. Mitigations, in order of strength:

1. **LoRA** — base weights frozen. The primary defence.
2. **Two epochs, not ten.**
3. **Multilingual replay** — `data.replay_dataset` + `replay_ratio`. Off by
   default. Turn it on if Phase 8 shows English or Russian ability dropping:
   mixing 5–10% of the model's original distribution back in is the standard
   fix.

Phase 8 must evaluate **non-Karakalpak** ability too. A model that gained
Karakalpak and lost everything else is not a success, and measuring only the
target language would hide it.

## 8. What is not settled

- **Vocabulary extension.** `ǵ`, `Á`, `Ń` are always standalone tokens and `Ǵ`
  falls back to raw bytes. Adding subwords would help — but Uzbek pays 1.75×
  against Karakalpak's 1.88×, so the ceiling on the gain is ~7%, and any
  extension means resizing embeddings and training the new rows on a data
  budget that is already thin. Deferred, deliberately.
- **The actual run has not been executed.** Every number above is derived from
  measurement or standard practice; none is yet confirmed by a loss curve on
  this corpus.
