
import os
from typing import Optional

import modal

APP_NAME = "miniseek-phase1"
DATA_DIR = "/data"
HF_CACHE = f"{DATA_DIR}/hf"
DATASET_CACHE = f"{DATA_DIR}/datasets"
CKPT_DIR = f"{DATA_DIR}/checkpoints"
CORPUS_DIR = f"{DATA_DIR}/science1b"

REPO_REMOTE = "/root/miniseek"
SHARD_CACHE = f"{DATA_DIR}/shard_cache"

volume = modal.Volume.from_name("miniseek-data", create_if_missing=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy==2.4.1",
        "datasets==5.0.0",
        "huggingface_hub==0.35.0",
        "transformers==4.56.2",
        "tokenizers==0.22.1",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    )
    .add_local_dir(
        REPO,
        "/root/miniseek",
        ignore=lambda p: any(part in ("data", ".git", "checkpoints", "logs", ".venv", "outputs", "scratch") for part in p.parts),
    )
)

app = modal.App(APP_NAME, image=image, volumes={DATA_DIR: volume})


def _setup_env():
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys_path_insert()


def sys_path_insert():
    import sys

    for p in ("/root/miniseek", "/root"):
        if p not in sys.path:
            sys.path.insert(0, p)


@app.function(timeout=3600, memory=16384, cpu=4.0)
def download_data():
    _setup_env()
    from src.data import get_dataset_stats

    for split in ("train", "validation", "test"):
        stats = get_dataset_stats(split=split, cache_dir=DATASET_CACHE)
        print(stats)

    volume.commit()
    print("Data cached on volume.")


@app.function(timeout=1800, memory=32768, cpu=16.0)
def smoke_corpus():
    """CPU-only smoke test: load the real long-doc corpus npz, fold windows,
    run model forward+loss on a batch. Verifies >1024-token docs are handled
    (split across windows, -1 masked targets) before any GPU spend."""
    _setup_env()
    import torch
    import yaml

    from src.tokenizer import Tokenizer
    from src.data import WikiTextDataset
    from src.model import create_model

    cfg_path = os.path.join(REPO_REMOTE, "configs", "phase3_seg1.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    cfg = raw["model"]

    tok = Tokenizer("/data/gpt2-tokenizer", max_length=2048)
    ds = WikiTextDataset(
        tokenizer=tok,
        max_length=cfg["max_seq_len"],
        split="train",
        corpus_dir=CORPUS_DIR,
    )
    print(f"num_sequences={ds.num_sequences:,}  num_docs={ds.num_docs:,}")

    model = create_model({
        "vocab_size": tok.vocab_size,
        "dim": cfg["dim"],
        "n_layers": cfg["n_layers"],
        "n_heads": cfg["n_heads"],
        "mlp_hidden_dim": cfg["mlp_hidden_dim"],
        "max_seq_len": cfg["max_seq_len"],
        "dropout": cfg["dropout"],
        "tie_embeddings": cfg["tie_embeddings"],
    })
    model.eval()

    with torch.no_grad():
        for idx in [0, 1, ds.num_sequences // 3, ds.num_sequences - 2, ds.num_sequences - 1]:
            sample = ds[idx]
            ids = sample["input_ids"]
            tgt = sample["targets"]
            assert ids.shape == torch.Size([cfg["max_seq_len"]]), (idx, ids.shape)
            assert tgt.shape == torch.Size([cfg["max_seq_len"]]), (idx, tgt.shape)
            out = model(ids.unsqueeze(0), tgt.unsqueeze(0))
            assert out["loss"] is not None and torch.isfinite(out["loss"]), (idx, out["loss"])
            print(f"  idx {idx}: loss={out['loss'].item():.4f}  masked_tgt={int((tgt==-1).sum())}")
    print("SMOKE_OK: long-doc windows fold and train loss is finite.")


@app.function(timeout=3600 * 6, memory=32768, cpu=16.0)
def build_corpus(
    max_train_tokens: int = 1_000_000_000,
    groups: Optional[str] = None,
    exclude_chemistry: bool = True,
    tokenize_only: bool = False,
):
    """Stream fineweb-edu-curated on the cloud, write the science corpus to the volume."""
    _setup_env()

    import importlib.util

    mod_path = os.path.join(REPO_REMOTE, "data-layer", "corpus.py")
    spec = importlib.util.spec_from_file_location("corpus", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if tokenize_only:
        print("TOKENIZE-ONLY: corpus jsonl already on volume; tokenizing with 16 workers")
        mod.tokenize_corpus(out_dir=CORPUS_DIR, workers=15)
    else:
        group_list = [g.strip() for g in groups.split(",")] if groups else None
        stats = mod.build(
            out_dir=CORPUS_DIR,
            groups=group_list,
            max_train_tokens=max_train_tokens,
            exclude_chemistry=exclude_chemistry,
            shard_cache=SHARD_CACHE,
        )
        mod.tokenize_corpus(out_dir=CORPUS_DIR, workers=15)
        print(f"Corpus built at {CORPUS_DIR}: {stats.get('train_docs', 0):,} docs / {stats.get('train_tokens_est', 0):,} estimate train tokens")
    _verify_corpus(mod, CORPUS_DIR)
    volume.commit()
    print("Corpus complete & committed.")


def _run_dir(config_name: str) -> str:
    return os.path.join(CKPT_DIR, os.path.splitext(config_name)[0])


def _verify_corpus(mod, corpus_dir: str):
    """Sanity-check the tokenized corpus npz files (authentic data before training)."""
    import numpy as np

    for split in ("train", "validation"):
        p = os.path.join(corpus_dir, f"{split}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: missing {p} - corpus tokenize failed")
        data = np.load(p)
        toks = data["tokens"]
        starts = data["doc_starts"]
        n_docs = int(len(starts) - 1)
        assert n_docs > 0, f"{split}: no docs"
        assert starts[0] == 0 and len(toks) == int(starts[-1]), f"{split}: npz layout mismatch"
        lens = np.diff(starts)
        assert np.all(lens > 0), f"{split}: empty or non-monotonic doc boundaries"
        r = np.load(p)
        print(f"  VERIFY {split}: {n_docs:,} docs, {len(toks):,} real GPT-2 tokens "
              f"(min doc {lens.min()}, max doc {lens.max()}, x-norm: word+doc shuffled OK/raw)")
    print("  corpus npz files verified OK")


@app.function(
    gpu="A10G",
    timeout=3600 * 18,
    memory=32768,
)
def train(config_name: str = "phase1_debug.yaml", resume: Optional[str] = None):
    _setup_env()

    import torch
    import yaml

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg_path = os.path.join(REPO_REMOTE, "configs", config_name)
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    run_dir = _run_dir(config_name)
    raw["data"]["cache_dir"] = DATASET_CACHE
    raw["data"]["max_seq_len"] = raw["model"]["max_seq_len"]
    if raw["data"].get("corpus_dir"):
        raw["data"]["corpus_dir"] = CORPUS_DIR
    raw["training"]["num_workers"] = 0
    raw["training"]["checkpoint_dir"] = run_dir
    raw["training"]["log_dir"] = f"{DATA_DIR}/logs/{os.path.splitext(config_name)[0]}"
    raw["device"] = "auto"

    from src.train import Trainer

    if resume:
        raw["training"]["resume_checkpoint"] = os.path.join(run_dir, resume)
        print(f"Resuming from explicit checkpoint {resume}")
    elif os.path.isdir(run_dir):
        latest = os.path.join(run_dir, "latest.pt")
        if os.path.exists(latest):
            raw["training"]["resume_checkpoint"] = latest
            raw["training"]["reset_optimizer"] = False
            print("Resuming from latest.pt (keeping optimizer/scheduler state)")
        else:
            ckpts = [
                f
                for f in os.listdir(run_dir)
                if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
            ]
            if ckpts:
                newest = max(
                    ckpts,
                    key=lambda f: int(f.removeprefix("checkpoint_epoch_").removesuffix(".pt")),
                )
                raw["training"]["resume_checkpoint"] = os.path.join(run_dir, newest)
                raw["training"]["reset_optimizer"] = False
                print(f"Resuming from {newest} (keeping optimizer/scheduler state)")

    rc = raw["training"].get("resume_checkpoint")
    if rc and not os.path.isabs(rc):
        raw["training"]["resume_checkpoint"] = os.path.join(DATA_DIR, rc.lstrip("./"))
        print(f"Resolved resume_checkpoint -> {raw['training']['resume_checkpoint']}")

    os.makedirs(run_dir, exist_ok=True)

    trainer = Trainer(raw)
    original_save = trainer.save_checkpoint

    def save_and_commit(path: str, epoch: int, val_loss: float, **kwargs):
        original_save(path, epoch, val_loss, **kwargs)
        volume.commit()

    trainer.save_checkpoint = save_and_commit
    trainer.on_ledger_write = lambda: volume.commit()
    trainer.train()
    volume.commit()
    print("Training complete. Checkpoints committed to volume.")


@app.function(timeout=1800, memory=8192, gpu="A10G")
def generate(
    prompts: str = "[]",
    prompts_file: Optional[str] = None,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    memorization_check: bool = True,
    run_name: Optional[str] = None,
):
    _setup_env()

    import json

    from src.generate import generate_text, load_model_from_checkpoint
    from src.tokenizer import Tokenizer

    default_prompts = [
        "Write a short story about a penguin",
        "As the florborg blooped across the",
        "In the year 2077, a person",
        "Albert Einstein developed",
        "The chemical formula for water is H",
        "2 + 2 is equal to",
    ]
    if prompts_file:
        with open(os.path.join(REPO_REMOTE, prompts_file), "r", encoding="utf-8") as f:
            probe_prompts = [line.strip() for line in f if line.strip()]
    else:
        probe_prompts = json.loads(prompts) or default_prompts

    run_dir = CKPT_DIR if run_name is None else _run_dir(run_name)
    ckpt_path = f"{run_dir}/best_model.pt"
    tokenizer = Tokenizer("gpt2", max_length=1024)
    model, config = load_model_from_checkpoint(ckpt_path, device="cuda", vocab_size=tokenizer.vocab_size)

    train_flat = None
    if memorization_check:
        from src.data import WikiTextDataset

        train_flat = WikiTextDataset(
            tokenizer=tokenizer,
            max_length=1024,
            split="train",
            cache_dir=DATASET_CACHE,
        ).flat_tokens

    for prompt in probe_prompts:
        input_ids = tokenizer.encode(prompt)
        full = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device="cuda",
        )
        print(f"\nPROMPT: {prompt}\n{full}\n{'-'*80}")

        if train_flat is None:
            continue
        new_ids = output_ids_of(prompt, tokenizer, model, max_new_tokens)
        longest = longest_exact_match(new_ids, train_flat)
        print(f"[memorization] longest verbatim span in training data: {longest} tokens")


def output_ids_of(prompt: str, tokenizer, model, max_new_tokens: int):
    import torch

    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return out[0][len(input_ids[0]):].tolist()


def longest_exact_match(needle, haystack):
    import numpy as np

    n = int(len(needle))
    if n == 0:
        return 0
    counts = np.bincount(haystack, minlength=1 if int(haystack.max()) == 0 else int(haystack.max()) + 1)
    anchor = min(range(n), key=lambda i: int(counts[needle[i]]))
    anchor_tok = needle[anchor]
    best = 0
    for c in np.flatnonzero(haystack == anchor_tok):
        c = int(c) - anchor
        if c < 0 or c + n > len(haystack):
            continue
        m = 0
        while m < n and haystack[c + m] == needle[m]:
            m += 1
        if m > best:
            best = m
            if best == n:
                break
    return int(best)


@app.function(timeout=1800, memory=8192)
def ckpt_history(run_name: Optional[str] = None):
    _setup_env()

    import glob
    import json

    import torch

    ckpt_dir = CKPT_DIR if run_name is None else _run_dir(run_name)

    def key(path):
        return int(os.path.basename(path).removeprefix("checkpoint_epoch_").removesuffix(".pt"))

    rows = []
    for path in sorted(glob.glob(f"{ckpt_dir}/checkpoint_epoch_*.pt"), key=key):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "epoch": ckpt.get("epoch"),
                "global_step": ckpt.get("global_step"),
                "val_loss": round(float(ckpt.get("val_loss", float("nan"))), 4),
                "best_val_loss": round(float(ckpt.get("best_val_loss", float("nan"))), 4),
                "cost_usd_total": round(float(ckpt.get("cost_usd_total", 0.0)), 4),
            }
        )
    print("CKPT_HISTORY_JSON:" + json.dumps(rows))


@app.local_entrypoint()
def main(
    config_name: str = "phase1_debug.yaml",
    skip_download: bool = False,
    build_corpus: bool = False,
    tokenize_only: bool = False,
    max_train_tokens: int = 1_000_000_000,
    corpus_groups: Optional[str] = None,
    no_exclude_chemistry: bool = False,
):
    build_corpus_fn = globals()["build_corpus"]
    if build_corpus:
        print("Building science corpus on Modal first (training waits)...")
        build_corpus_fn(
            max_train_tokens=max_train_tokens,
            groups=corpus_groups,
            exclude_chemistry=not no_exclude_chemistry,
            tokenize_only=tokenize_only,
        )
        print("Corpus build complete; submitting training.")
    elif tokenize_only:
        build_corpus_fn(tokenize_only=True)
        print("Tokenization complete; submitting training.")
    elif not skip_download:
        download_data.spawn()
    train.spawn(config_name)
    print("Submitted detached run. Client exiting - training continues on Modal.")