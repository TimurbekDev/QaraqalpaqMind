# RunPod deployment

Target: **1× RTX 4090 (24 GB) · 31 GB RAM · 8 vCPU · CUDA 12.8 · Ubuntu 24.04 ·
RunPod PyTorch 2.8.0 template.**

---

## 1. Commands to run after connecting

```bash
cd /workspace
git clone https://github.com/TimurbekDev/QaraqalpaqMind QaraqalpaqMind
cd QaraqalpaqMind

bash scripts/runpod_setup.sh
```

That script is idempotent and does everything below. If you prefer to run the
steps yourself:

```bash
# 1. Caches on the persistent volume, NOT the ephemeral container filesystem
export HF_HOME=/workspace/.cache/huggingface
export TRITON_CACHE_DIR=/workspace/.cache/triton
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR"
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

# 2. Install. torch is NOT reinstalled: the template's CUDA build already
#    satisfies >=2.5, and replacing it risks a mismatched wheel.
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[train,ingest,clean]"

# 3. Environment
cp .env.example .env
#    Edit .env: set HF_TOKEN, and HF_HOME=/workspace/.cache/huggingface

# 4. Data - see section 3
qm ingest pull Timurbek/qaraqalpaqmind-data

# 5. Verify before spending money on a download
qm doctor
qm train preflight --config configs/cpt/qwen3_8b_qlora_24gb.yaml
qm train plan      --config configs/cpt/qwen3_8b_qlora_24gb.yaml

# 6. Train
qm train cpt --config configs/cpt/qwen3_8b_qlora_24gb.yaml
```

## 2. Why `HF_HOME` matters more than it looks

The default cache is `~/.cache/huggingface`. On a container that is the
**ephemeral overlay filesystem**. Qwen3-8B is a 16.4 GB download; leave the
default and you re-download it on every pod restart, paying for the time each
time.

`qm train preflight` flags this explicitly rather than letting you discover it.

## 3. Getting the corpus onto the pod

`data/` is not in git — corpora are reproducible outputs, and a repo is not a
data store. So a fresh clone has **no training data** and `qm train cpt` will
stop with a message saying so.

Three options, in order of preference:

**a) Pull the exact corpus you built.** Push once from the machine that built
it, pull on every GPU host. Manifests travel with the data, so the sha256 and
token count describing a run are never separated from it.

```bash
# On the machine that has the data:
qm ingest push Timurbek/qaraqalpaqmind-data     # private by default

# On the pod:
qm ingest pull Timurbek/qaraqalpaqmind-data
```

The repo defaults to **private**: several crawled sources have an `unknown`
redistribution licence, so publishing is a deliberate decision.

**b) Rebuild from public sources only.** No crawling, ~93% of the corpus
(everything except the 4.8 M characters of crawled institutional web):

```bash
qm ingest all --bulk-only && qm clean all && qm dedup run
```

**c) Copy the file directly** — it is 35 MB:

```bash
runpodctl send data/datasets/pretrain/pretrain_v1.jsonl.zst
```

## 4. Resource estimates

### VRAM — measured from the actual 189 M LoRA parameter count

| Component | GB |
|---|---:|
| 4-bit base weights (nf4 + double quant) | 5.4 |
| LoRA adapters, 189 M × 2 bytes | 0.38 |
| Gradients, 189 M × 2 bytes | 0.38 |
| AdamW states, 189 M × 8 bytes | 1.51 |
| Activations (checkpointed, seq 2048, bs 1) | 2.5–4.0 |
| CUDA context + fragmentation | ~1.5 |
| **Peak** | **11.7–13.2** |
| **Headroom on 24 GB** | **~10.8** |

### Everything else

| Resource | Estimate |
|---|---|
| System RAM | ~8–12 GB peak. Packing holds tokenised sequences in memory; 31 GB is ample |
| Disk — base model | 16.4 GB (one-time, cached on the volume) |
| Disk — corpus | 35 MB compressed |
| Disk — per checkpoint | **1.89 GB** (0.38 adapter + 1.51 optimizer state) |
| Disk — checkpoints kept | 5.7 GB (`save_total_limit: 3`) |
| **Disk total** | **~25 GB**; provision 50 GB+ |
| Steps | 1,750 (28.7 M tokens, 32,768 tokens/step, 2 epochs) |
| Runtime | **~6–10 h** at an estimated 12–20 s/step |
| Cost | **~$4–8** at RTX 4090 rates around $0.70/h |

The runtime figure is an estimate from step count and typical QLoRA throughput
for an 8 B model at sequence length 2048. **It has not been measured on this
hardware.** Check the rate after 20 steps and extrapolate.

## 5. Interruption and resume

Every training config sets `resume_from_checkpoint: auto`. After a pod reclaim,
re-running the same command continues from the newest checkpoint in
`output_dir` — and says so loudly in the log:

```
WARNING  RESUMING an interrupted run - not starting from scratch  step=800
```

Checkpoints land every 200 steps (~35–65 min), so at most that much work is
lost. Inspect what exists at any time:

```bash
qm train preflight --config configs/cpt/qwen3_8b_qlora_24gb.yaml
```

`models/` must be on the persistent volume. If you cloned into `/workspace`, it
already is.

## 6. Configuration verification

Every setting below was checked against this hardware.

| Setting | Value | Verdict |
|---|---|---|
| `per_device_batch_size` | 1 | Safe. 2 would likely fit — unmeasured, so not the default |
| `gradient_accumulation_steps` | 16 | 32,768 tokens/step |
| `sequence_length` | 2048 | Karakalpak tokens; ≈1,100 English-token equivalent |
| `bf16` / `fp16` | true / false | Correct: 4090 is sm_89, bf16 native, no loss scaling needed |
| `attn_implementation` | `sdpa` | PyTorch-native, no build step. flash-attn optional — see below |
| `gradient_checkpointing` | true | Required; `use_cache` is false to match |
| `optim` | `adamw_torch_fused` | Fused kernel, CUDA-only, correct here |
| `lr_scheduler` | cosine, 3% warmup | ~52 warmup steps of 1,750 |
| `save_steps` | 200 | 1.89 GB each, 3 kept |
| `eval_steps` | 200 | 1% validation split |
| `dataloader_num_workers` | 2 | Right for 8 vCPU; packing is done before the loader |
| `deepspeed` | null | Correct — single GPU needs no sharding |

**flash-attention is deliberately optional.** It has no prebuilt wheel for most
torch/CUDA combinations, so pip compiles it from source: 10–30 minutes, high
RAM, and it fails outright if torch is not importable at build time. Putting it
in the `train` extra would make a routine install fail. The 4090 supports it, so
if you want it:

```bash
pip install -e ".[flash]" --no-build-isolation
# then set model.attn_implementation: flash_attention_2
```

Expect ~20–30% faster steps and lower activation memory. `preflight` errors if a
config asks for it and it is not installed.

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **No corpus after clone** | **High** | `preflight` and the setup script both check; three transfer options in §3 |
| HF cache on ephemeral disk | High | `HF_HOME` set by the setup script; `preflight` warns |
| Pod reclaimed mid-run | High | `resume_from_checkpoint: auto`; checkpoints every 200 steps |
| torch replaced by a mismatched wheel | Medium | Template's 2.8.0 satisfies `>=2.5` so pip skips it; the setup script prints before/after and warns on change |
| flash-attn build failure | Medium | Moved out of `train` into its own extra |
| `models/` on ephemeral disk | Medium | Clone into `/workspace`; §5 |
| Disk exhaustion | Medium | ~25 GB needed; `preflight` checks free space |
| OOM | **Low** | ~10.8 GB headroom measured. Raising batch size is the only likely cause |
| Gated FLORES+ without `HF_TOKEN` | Low | Only affects evaluation; fails with instructions |
| Non-editable install breaking paths | Low | `PROJECT_ROOT` now falls back to cwd search; `QM_PROJECT_ROOT` overrides |

## 8. Deployment checklist

```
[ ] Pod created: RTX 4090, PyTorch 2.8.0 template, 50 GB+ volume at /workspace
[ ] Repo cloned into /workspace (not into ~, which is ephemeral)
[ ] bash scripts/runpod_setup.sh completed without errors
[ ] HF_HOME points at /workspace/.cache/huggingface   (qm doctor)
[ ] HF_TOKEN set in .env if pulling data or gated datasets
[ ] Corpus present: data/datasets/pretrain/pretrain_v1.jsonl.zst
[ ] qm train preflight exits 0
[ ] qm train plan shows ~1,750 steps and a measured token count
[ ] nvidia-smi shows the 4090 idle before starting
[ ] Started inside tmux/screen, or nohup, so an SSH drop does not kill it
[ ] After 20 steps: check s/step, VRAM in nvidia-smi, and that loss is falling
```

Start under `tmux` so a dropped SSH session does not kill the run:

```bash
tmux new -s cpt
qm train cpt --config configs/cpt/qwen3_8b_qlora_24gb.yaml
# detach: Ctrl-B then D    reattach: tmux attach -t cpt
```

## 9. Requirements verification

| Package | Constraint | Template ships | Action |
|---|---|---|---|
| Python | `>=3.12,<3.14` | 3.12 (Ubuntu 24.04) | OK — was `<3.13`, widened |
| torch | `>=2.5` | 2.8.0 + CUDA 12.8 | **Not reinstalled** — satisfies the bound |
| transformers | `>=4.48` | — | installed |
| trl | `>=0.13` | — | installed |
| peft | `>=0.14` | — | installed |
| accelerate | `>=1.2` | — | installed |
| bitsandbytes | `>=0.45` | — | installed; needs CUDA, present |
| deepspeed | — | — | **moved to `[distributed]`** — not needed on one GPU |
| flash-attn | — | — | **moved to `[flash]`** — optional, source build |
