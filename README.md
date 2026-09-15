<div align="center">

# Miniseek-100M

**A from-scratch decoder-only language model, trained and measured.**

From a 17.7M-parameter baseline to a 101.4M-parameter model — first on
WikiText-103 at a fixed token budget, then continued on a self-built ~1B-token
science corpus — with full cost accounting throughout.

`v0.1` · 17.7M → 101.4M params · WikiText-103 + 1B science corpus · Modal A10G

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![101.4M params](https://img.shields.io/badge/params-101.4M-F4A261?style=flat-square)
![1B science corpus](https://img.shields.io/badge/data-1B+science--corpus-7B8794?style=flat-square)

</div>

---

## Overview

This repository builds a small language model entirely in PyTorch and measures
its behavior rigorously at every stage:

- **Phase 1** — a 17.7M-parameter dense baseline (5 epochs ≈ 590M tokens of
  WikiText-103).
- **Phase 2** — a 101.4M-parameter dense control trained on the same WikiText
  token budget, isolating the effect of scale.
- **Phase 3** — continued pretraining of the Phase-2 model on a ~1B-token
  science corpus, curated from FineWeb-Edu (life sciences, physical sciences,
  mathematics, environmental science).

The emphasis throughout is on reproducible, honestly-reported results:
validation losses, perplexities, GPU costs, and qualitative samples are all
recorded. Full model samples live in [`response_lm.txt`](response_lm.txt).

## Results

### WikiText pretraining (Phases 1–2)

Scaling the model **5.7×** at a fixed token budget lowered validation loss from
**3.658 → 3.169** and perplexity from **38.8 → 23.8**.

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

Left panel: Phases 1–2 on WikiText-103. Right panel: Phase-3 continuation on
the 1B-token science corpus (val measured on the science holdout, so the panels
are not directly comparable). Regenerate with `python scripts/loss.py` (writes
`figures/loss.png`).
*Heads-up: Phase-1 train-loss for epochs 1–4 in the figure is an estimate —
only the epoch-5 value (3.6944) was logged. All val-loss curves are exact.*

### Science-corpus continued pretraining (Phase 3, ~1B tokens)

The Phase-2 model is resumed on a curated ~1B-token science corpus built from
[`mdonigian/fineweb-edu-curated`](https://huggingface.co/datasets/mdonigian/fineweb-edu-curated),
filtered to four science groups — life sciences, physical sciences,
mathematics, environmental. Chemistry is excluded; documents outside 32–16,000
tokens are dropped; 2% of tokens are held out for validation
(`data-layer/corpus.py`).

Training ran in two segments (configs `1b_corpus.yaml`, `1b_corpus_seg2.yaml`),
the second with a higher LR restart, under the same $30 budget guard:

| Run | Segment | Epoch | Train loss | Val loss | Val ppl | Cumulative GPU cost |
|-----|---------|------:|-----------:|---------:|--------:|--------------------:|
| `logs/1b_seg1` | seg1 (LR 6e-6) | 1 | 3.8116 | 3.7849 | 44.03 | $27.63 |
| `logs/1b_seg2` | seg2-lr (LR 3e-5) | 2 | 3.5968 | **3.5589** | **35.13** | $27.98 |

Final checkpoint: `/data/checkpoints/phase3_seg2_lr`. Note that Phase-3
validation is measured against the science corpus holdout, *not* WikiText-103,
so it is not directly comparable to the Phase-1/2 numbers above.

## Sample responses

Unedited output from the 101.4M model after the 1B science-corpus continuation
pass (checkpoint `phase3_seg2_lr`, temperature 0.8 / top-k 100 / top-p 0.95).
Full samples for every checkpoint are in [`response_lm.txt`](response_lm.txt).

```
Q: DNA is the molecule that carries
A: DNA is the molecule that carries genes.
   A protein called RNA can be used to express proteins from one generation of
   molecules and has been shown to possess a significant role in natural
   selection, as well as defense against disease.

Q: Hemoglobin is the protein that transports
A: Hemoglobin is the protein that transports oxygen from your body to the
   muscles of your feet. It is a valuable drug for maintaining muscle control
   and reducing inflammation, as well as for improving overall health.

Q: Antibodies are produced by the immune system to
A: Antibodies are produced by the immune system to fight against toxins and
   diseases. While there is a strong connection between these two
   neurotransmitters in the brain, they have been linked primarily through our
   nervous system.

Q: Kinetic energy is the energy an object possesses due to its
A: Kinetic energy is the energy an object possesses due to its motion. In the
   above example, when the Sun and Moon are in orbit around Earth it takes a
   few minutes for them all together to form one new planet that's within 1
   billion light-years from us.

Q: Gravity is an attractive force between two objects that depends on their
A: Gravity is an attractive force between two objects that depends on their
   mass and its position. For example, if you are in a situation where the
   force of gravity is greater than 2-D it will be stronger with respect to
   the rest sides of your body.

Q: Photosynthesis is the process by which plants convert
A: Photosynthesis is the process by which plants convert light into water and
   provide oxygen to the soil. The process involves a careful application of
   chemical processes, such as adding salts or additives like fluoride-based
   herbicides or pesticides.
```

The model produces fluent, topic-aware first sentences on subjects it saw in
the science corpus (DNA, immune system, energy, gravity, photosynthesis),
continuing with its own extrapolations. On subjects outside the corpus it
diverts freely — here are unedited examples showing this clearly:

```
Q: The value of pi to four decimal places is
A: The value of pi to four decimal places is 6. A square in the bottom right
   corner, in which one square represents a percentage value and another, with
   two units representing two percent values.

Q: The Pythagorean theorem states that in a right triangle
A: The Pythagorean theorem states that in a right triangle the base of the
   square root of the xi is multiplied by 1. This gives us "a perfect and
   correct way to find, for example, the circle around our right corner:"

Q: The derivative of the function f(x) = x squared with respect to x is
A: The derivative of the function f(x) = x squared with respect to x is equal
   to f2.5x + 3p, where hz is defined by a constant between 0 and 1.

Q: Euler's number e is approximately equal to
A: Euler's number e is approximately equal to the sum of its square roots.
   This formula expresses the total amount of sum by itself, in this case an
   integer x^2 + ax+1.

Q: Plate tectonics explains the movement of the Earth's
A: Plate tectonics explains the movement of the Earth's crust through a single
   cell. The soil can be exposed to water and other contaminants as well as
   sediment and nutrients in soil to form the formation or deposition of
   organic matter.
```

The first two outputs about pi and the Pythagorean theorem are clearly
fabricated, yet the prose reads as fluently as the correct biology answers
above — a hallmark of a 101M-parameter model trained on 1.6B tokens of
scientific text that has learned genre well without learning facts.


## Findings

The model consistently produced fabricated facts while maintaining fluent,
grammatically correct English. Two primary factors explain this:

- **Compute bound.** At 101.4M parameters with roughly 1.6B training tokens,
  the model sits far below the data-to-parameter ratio at which factual recall
  begins to emerge in decoder-only LMs (GPT-2 needed ~10B tokens at a similar
  scale).
- **Data bound.** The training corpus — WikiText-103 plus a curated 1B-token
  science subset — is narrow and contains no question-answer pairs, giving the
  model little exposure to the retrieval patterns that yield factual answers.

Despite these constraints, the model learned consistent English grammar, proper
punctuation, and the rhetorical conventions of scientific prose, producing
readable, genre-appropriate text throughout.

## Model architecture

Decoder-only stack: pre-norm attention + SwiGLU feed-forward, RoPE positional
encoding, RMSNorm, tied input/output embeddings, GPT-2 BPE vocabulary (50,257).

| Component | Phase 1 (17.7M) | Phase 2/3 (101.4M) |
|-----------|-----------------|--------------------|
| Embedding dim | 256 | 704 |
| Layers | 6 | 12 |
| Attention heads | 8 × head 32 | 8 × head 88 |
| FFN hidden (SwiGLU) | 704 | 1664 |
| Max sequence len | 1024 | 1024 |
| Norm | RMSNorm (eps 1e-6) | RMSNorm (eps 1e-6) |
| Positional | RoPE (base 10 000) | RoPE (base 10 000) |
| Dropout | 0.1 | 0.1 |
| Tied embeddings | yes | yes |

The vocabulary dominates the parameter count: in Phase 1, the shared
embedding/LM-head table (50,257 × 256) is 72.7% of all parameters. Configs
declare `expected_params_million` and the loader raises if the built model
disagrees (17.69 / 101.36).

## Data & token accounting

- **WikiText-103** (`wikitext-103-raw-v1`), GPT-2 BPE, no filtering or
  deduplication: 1,165,029 train docs → 117,919,088 tokens → 115,155 windows.
  Phase 1: batch 16 → 590M tokens total. Phase 2: effective batch 32
  (verified byte-for-byte equivalent to a real batch-32 step) → 590M tokens.
- **Science corpus** (`data-layer/`, target `data/science1b`): streams
  `mdonigian/fineweb-edu-curated`, keeps life/physical sciences, mathematics
  and environmental documents, excludes chemistry, targets **1B train tokens**
  with a 2% validation holdout.

## Training details

- **Hardware:** single NVIDIA A10G via Modal (~36 min/epoch at 17.7M params,
  ~1.6 h/epoch at 101.4M on WikiText).
- **Optimizer:** AdamW (β 0.9/0.95, wd 0.01), cosine LR (Phase 1/2: 6e-4 → 6e-5,
  warmup 100/200 steps; Phase 3 seg1: 6e-6 → 6e-7; seg2: 3e-5 → 3e-6),
  grad clip 1.0, AMP autocast.
- **Logging:** per-epoch JSONL to `logs/<run>/training_log.jsonl`; per-epoch +
  `best_model.pt` + `latest.pt` checkpoints on the `miniseek-data` volume.

### Budget-aware training (cap = $24 / $30)

- Cost cap tracked in a persistent ledger (`<log_dir>/cost_ledger.json`),
  accumulated across every run and resume. Phase 2 spent **$17.02**; Phase 3
  (science corpus) brought the cumulative total to **$27.98**.
- Training stops cleanly before/within an epoch that would exceed the cap, after
  projecting GPU time × `usd_per_hour: 2.0` (measured at $6/3h on A10G in
  Phase 1).
- Full state (model + optimizer + scheduler + scaler + RNG + step) is saved to
  `latest.pt` and is resumable across Modal accounts:
  `modal run scripts/modal_train.py --config-name phase2_100m.yaml --resume latest.pt`.

## Project structure

```
miniseek/
├─ configs/                  # phase1/2, 1b_corpus (+seg2), smoke, debug
├─ data-layer/               # Science-corpus builder (corpus, shards, filters, tokenization)
├─ figures/                  # loss.png (from scripts/loss.py)
├─ scripts/
│  ├─ train.py               # Local CPU training entrypoint (smoke tests)
│  ├─ modal_train.py         # Modal app (corpus build, train, generate, ckpt_history)
│  └─ loss.py                # Phase 1 vs Phase 2 comparison figure
├─ src/
│  ├─ model.py               # Decoder transformer (config-driven)
│  ├─ attention.py           # Causal multi-head attention
│  ├─ rope.py                # Rotary position embeddings
│  ├─ norm_activation.py     # RMSNorm, SwiGLU
│  ├─ tokenizer.py           # GPT-2 BPE wrapper
│  ├─ data.py                # Dataloaders (WikiText + local corpus) / .npz cache
│  ├─ train.py               # Training loop + JSONL logging + budget ledger
│  └─ generate.py            # Sampling
├─ tests/                    # Unit tests (params, forward, shapes)
├─ response_lm.txt           # Human-readable model samples (Phases 1–3)
└─ data/ logs/ checkpoints/  # Local caches (git-ignored)
```

## Quick start

```bash
pip install -r requirements.txt

# Unit tests
python tests/test_model.py

# Local CPU smoke run
python scripts/train.py --config configs/smoke.yaml

# Regenerate the comparison figure
python scripts/loss.py

# Build the ~1B-token science corpus (local machine)
python data-layer/corpus.py --out ./data/science1b

# Sample from the Phase-3 checkpoint
modal run scripts/modal_train.py::generate \
  --prompts-file probes.txt --temperature 0.1 --top-k 5 --no-memorization-check
```

The `.modalignore` keeps `data/`, `checkpoints/`, `logs/`, `scratch/` and
`*.npz` out of the source image; datasets and checkpoints live on the persistent
`miniseek-data` volume at `/data`.


## References

- GPT-2 / nanoGPT (Karpathy)
- LLaMA architecture (RoPE, RMSNorm, SwiGLU)
- WikiText-103 benchmark
- FineWeb-Edu (`mdonigian/fineweb-edu-curated`)
- Chinchilla scaling laws (data × model-size trade-offs)