# Miniseek

Incremental LLM research project - building a decoder transformer from scratch with experimental architectures.

## Project Structure

```
miniseek/
├── configs/          # Training configurations
├── data/             # Dataset cache
├── src/              # Source code
│   ├── model.py      # Decoder transformer
│   ├── attention.py  # Causal attention
│   ├── layers.py     # RMSNorm, SwiGLU, RoPE
│   ├── tokenizer.py  # BPE tokenizer wrapper
│   ├── data.py       # WikiText-103 dataloader
│   ├── train.py      # Training loop
│   └── generate.py   # Text generation
├── scripts/          # Training scripts
├── tests/            # Unit tests
└── checkpoints/      # Saved models
```

## Phase 1: Baseline (20M Params)

**Model:**
- 6 layers, 256 dim, 8 heads
- RoPE positional encoding
- RMSNorm + SwiGLU
- ~21M parameters

**Dataset:** WikiText-103 (~100M tokens)

**Budget:** Modal Account #1 (~$30)

### Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download WikiText-103:
```bash
python scripts/download_data.py
```

3. Run model test:
```bash
python tests/test_model.py
```

4. Train baseline:
```bash
python scripts/train.py --config configs/phase1_debug.yaml
```

## Architecture Evolution

```
Phase 1: Vanilla Decoder (baseline)
    ↓
Phase 2: 100M Baseline (control)
    ↓
Phase 3: Experiments (MLA, MoE, MTP)
    ↓
Phase 4: Final Miniseek (100-150M)
```

## Compute Budget

- **Modal #1:** 20M debugging
- **Modal #2:** 100M baseline  
- **AWS:** Architecture experiments + final training

## References

- GPT-2 / nanoGPT (Karpathy)
- LLaMA architecture (RoPE, RMSNorm, SwiGLU)
- FineWeb-Edu dataset
- WikiText-103 benchmark
