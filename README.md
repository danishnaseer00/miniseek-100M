<p align="center">

# Miniseek

**A from-scratch decoder-only transformer measurement lab**

Scale-up study: fix the token budget, grow the model, and watch validation loss fall.

`v0.1` · `dense baseline` · `101.4M control` · `WikiText-103` · `PyTorch` · `Modal A10G`

</p>

---

## tl;dr

- **Phase 1** — dense baseline, 17.7M params, 5 epochs (~590M tokens).
- **Phase 2** — dense control, 101.4M params, same ~590M tokens.
- Scaling the model **5.7×** at a **fixed token budget** lowers validation loss
  **3.658 → 3.169** and perplexity **38.8 → 23.8** (see figure below).

| Metric | Phase 1 (17.7M) | Phase 2 (101.4M) |
|--------|----------------:|-----------------:|
| Parameters | 17,686,016 | 101,360,512 |
| Tokens per epoch | 117,919,088 | 117,919,088 |
| Epochs | 5 | 5 |
| Effective batch | 16 × 1024 tok | 8 × grad-accum 4 → 32 |
| Final train loss | 3.6944 | 2.9586 |
| Final val loss | 3.6582 | **3.1694** |
| Final val perplexity | 38.79 | **23.79** |
| GPU cost | ~$6.00 | **$17.02** (≤ $24 cap) |

![Training curves](figures/loss.png)

Regenerate with `python scripts/loss.py` (writes `figures/loss.png`).
*Heads-up: Phase-1 train-loss for epochs 1–4 in the figure is an estimate —
only the epoch-5 value (3.6944) was logged. All val-loss curves are exact.*

---

## What the model learned — and what it didn't (and why)

Full samples live in [`response_lm.txt`](response_lm.txt). Key observations from
both checkpoints:

### ✅ What it learned

- **English syntax.** Sentences are grammatical, punctuated, and spelled
  correctly — it clearly captured word order, morphology, and agreement rules.
- **WikiText's genre.** The model reproduces encyclopedic prose structure,
  including `==== Heading ====` and `= = Reception = =` section-markup patterns.
- **Not memorization.** A training-data overlap probe (anchor-token matching
  against the 118M-token train split) finds **0–1 token** exact matches — the
  output is *generated*, never regurgitated.
- **Healthy train/val gap at 101.4M.** Train 2.96 vs val 3.17 means it is
  learning generalizable structure (Perplexity-Intuition: gap << 1).

### ❌ What it did not learn

- **No factual retrieval** — knowledge probing (temperature 0.1, top-k 5):
  - *"The capital of France is"* → "the capital of the Kingdom of France"
  - *"The Earth revolves around"* → "the city of the city"
  - *"World War II began in"* → "the late 1950s"
  - *"Water … boils at"* → "0 degrees altitude"
  - Score: **0/10**.
- **No arithmetic.** *"2 + 2 is equal to"* → "the 0.5c4 + 2 −".
- **No long-range coherence.** Topics drift without warning (Roman Empire →
  a 1980s pop song). This is a language model, not a reasoner.

### 🔍 Why

1. **Data distribution.** WikiText-103 is encyclopedic *prose*, never written as
   Q&A. The phrase *"The capital of France is"* effectively never appears in
   training, so retrieval can't be learned. A GPT-2-sized model only recalls
   facts when the fact **shape** (question → answer) is in the data.
2. **Not enough tokens.** GPT-2 (117M, ≈ our size) needed **~10B tokens** of
   diverse web text before factual recall appeared. 590M tokens of *one* corpus
   is roughly 1/17 of that — *capacity-limited and data-limited at once.*
3. **Wrong objective.** Next-token prediction optimizes *plausible text*, not
   *correct answers*. Decoding settings (temperature/top-k) tune style, not
   knowledge — they never add facts.
4. **Baseline literacy gap.** 23.8 validation perplexity is still high; a fluent
   LM needs ppl ≲ 15. Facts only emerge as perplexity falls toward that range.

### 🚀 What would fix it (next steps)

- **More data, diverse** — multi-domain web + books (a 1B-token pipeline),
  not repeated WikiText epochs (repeats stop buying generalization).
- **Higher scale** — the phase-2 curve was still descending at epoch 5; a
  101M model on 2B+ tokens is the direct follow-up.
- **Eval you actually want** — QA/summarization evals, not story prompts, when
  the training distribution is encyclopedic.

---

## Model architecture

Decoder-only stack: pre-norm attention + SwiGLU feed-forward, RoPE positional
encoding, RMSNorm, tied input/output embeddings, GPT-2 BPE vocabulary (50,257).

| Component | Phase 1 (17.7M) | Phase 2 (101.4M) |
|-----------|-----------------|------------------|
| Embedding dim | 256 | 704 |
| Layers | 6 | 12 |
| Attention heads | 8 × head 32 | 8 × head 88 |
| FFN hidden (SwiGLU) | 704 | 1664 |
| Max sequence len | 1024 | 1024 |
| Norm | RMSNorm (eps 1e-6) | RMSNorm (eps 1e-6) |
| Positional | RoPE (base 10 000) | RoPE (base 10 000) |
| Dropout | 0.1 | 0.1 |
| Tied embeddings | yes | yes |

The vocabulary dominates parameter count: in Phase 1, the shared
embedding/LM-head table (50 257 × 256) is 72.7% of all parameters. Configs
declare `expected_params_million` and the loader raises if the built model
disagrees (17.69 / 101.36).

## Dataset & token accounting

- **Dataset:** WikiText-103 (raw split), GPT-2 BPE, no filtering/deduplication.
- **Train split:** 1,165,029 documents → 117,919,088 tokens → 115,155 windows.
- **Phase 1:** batch 16 → 7,198 gradient steps/epoch → 590M tokens total.
- **Phase 2:** batch 8 × `grad_accum_steps 4` = **effective batch 32** (verified
  byte-for-byte equivalent to a real batch-32 step) → 3,599 optimizer steps/epoch.

## Training details

- **Hardware:** single NVIDIA A10G via Modal (~36 min/epoch @ 17.7M,
  ~1.6 h/epoch @ 101.4M).
- **Optimizer:** AdamW (β 0.9/0.95, wd 0.01), LR 6e-4 → 6e-5 cosine, warmup
  100 steps (Phase 1) / 200 steps (Phase 2), grad clip 1.0, AMP autocast.
- **Logging:** per-epoch JSONL to `logs/<run>/training_log.jsonl`; per-epoch +
  `best_model.pt` + `latest.pt` checkpoints on the `miniseek-data` volume.

### Budget-aware training (Phase 2, cap = $24)

- Cost cap tracked in a persistent ledger (`<log_dir>/cost_ledger.json`),
  cumulative across every run and resume. Phase 2 spent **$17.02**.
- Stops cleanly before/within an epoch that would exceed the cap after
  projecting GPU time × `usd_per_hour: 2.0` (measured $6/3h on A10G in Phase 1).
- Saves full state (model + optimizer + scheduler + scaler + RNG + step) to
  `latest.pt` — resumable across Modal accounts:
  `modal run scripts/modal_train.py --config-name phase2_100m.yaml --resume latest.pt`.

## Project structure

```
miniseek/
├─ configs/                  # phase1_debug, phase2_100m, smoke_realdata
├─ figures/                  # loss.png (from scripts/loss.py)
├─ scripts/
│  ├─ train.py               # Local CPU training entrypoint (smoke tests)
│  ├─ modal_train.py         # Modal app (download, train, generate, ckpt_history)
│  └─ loss.py                # Phase 1 vs Phase 2 comparison figure
├─ src/
│  ├─ model.py               # Decoder transformer (config-driven)
│  ├─ attention.py           # Causal multi-head attention
│  ├─ rope.py                # Rotary position embeddings
│  ├─ norm_activation.py     # RMSNorm, SwiGLU
│  ├─ tokenizer.py           # GPT-2 BPE wrapper
│  ├─ data.py                # WikiText-103 dataloader / .npz cache
│  ├─ train.py               # Training loop + JSONL logging + budget ledger
│  └─ generate.py            # Sampling
├─ tests/                    # Unit tests (params, forward, shapes)
├─ probes.txt                # Knowledge-probe prompts (used by generate)
├─ response_lm.txt           # Human-readable model samples (Phase 1 + 2)
└─ data/ logs/ checkpoints/  # Local caches (git-ignored)
```

## Quick start

```bash
pip install -r requirements.txt

# Unit tests
python tests/test_model.py

# Local CPU smoke run
python scripts/train.py --config configs/smoke_realdata.yaml

# Regenerate the comparison figure
python scripts/loss.py

# Sample from the Phase-2 checkpoint
modal run scripts/modal_train.py::generate \
  --prompts-file probes.txt --temperature 0.1 --top-k 5 --no-memorization-check
```

The `.modalignore` keeps `data/`, `checkpoints/`, `logs/`, `scratch/` and
`*.npz` out of the source image; datasets and checkpoints live on the persistent
`miniseek-data` volume at `/data`.

## Roadmap

```
Phase 1: Dense baseline (17.7M)             ✓ complete (val 3.658)
Phase 2: Dense control (101.4M)             ✓ complete (val 3.169, $17.02)
Phase 3: Experiments (MLA, MoE, MTP)
Phase 4: Final Miniseek (100–150M, 1B+ tokens)
```

## References

- GPT-2 / nanoGPT (Karpathy)
- LLaMA architecture (RoPE, RMSNorm, SwiGLU)
- WikiText-103 benchmark
- Chinchilla scaling laws (data × model-size trade-offs)