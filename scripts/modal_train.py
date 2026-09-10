"""Modal app: persist WikiText-103 + checkpoints on a volume, then train Miniseek on an A10G.

Modal SDK 1.5.2 API (Image.add_local_dir, gpu="<string>", App-level default image/volumes).

Usage:
    modal run scripts/modal_train.py                           # download data, then train (phase1_debug.yaml)
    modal run scripts/modal_train.py --skip-download            # train only, reuse volume
    modal run scripts/modal_train.py --config-name smoke_realdata.yaml
"""

import os
from typing import Optional

import modal

APP_NAME = "miniseek-phase1"
DATA_DIR = "/data"
HF_CACHE = f"{DATA_DIR}/hf"
DATASET_CACHE = f"{DATA_DIR}/datasets"
CKPT_DIR = f"{DATA_DIR}/checkpoints"

REPO_REMOTE = "/root/miniseek"

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
        ignore=lambda p: any(part in ("data", ".git", "checkpoints", "logs", ".venv", "outputs") for part in p.parts),
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


def _run_dir(config_name: str) -> str:
    return os.path.join(CKPT_DIR, os.path.splitext(config_name)[0])


@app.function(
    gpu="A10G",
    timeout=3600 * 12,
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
            print("Resuming from latest.pt")
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
                print(f"Resuming from {newest}")

    trainer = Trainer(raw)

    os.makedirs(run_dir, exist_ok=True)

    original_save = trainer.save_checkpoint

    def save_and_commit(path: str, epoch: int, val_loss: float, **kwargs):
        original_save(path, epoch, val_loss, **kwargs)
        volume.commit()

    trainer.save_checkpoint = save_and_commit
    trainer.on_ledger_write = lambda: volume.commit()
    trainer.train()
    volume.commit()
    print("Training complete. Checkpoints committed to volume.")


@app.function(timeout=1800, memory=8192, cpu=2.0)
def generate(
    prompts: str = "[]",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
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
    probe_prompts = json.loads(prompts) or default_prompts

    run_dir = CKPT_DIR if run_name is None else _run_dir(run_name)
    ckpt_path = f"{run_dir}/best_model.pt"
    tokenizer = Tokenizer("gpt2", max_length=1024)
    model, config = load_model_from_checkpoint(ckpt_path, device="cpu", vocab_size=tokenizer.vocab_size)

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
            device="cpu",
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
def main(config_name: str = "phase1_debug.yaml", skip_download: bool = False):
    if not skip_download:
        download_data.spawn()
    train.spawn(config_name)
    print("Submitted detached run. Client exiting - training continues on Modal.")