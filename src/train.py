import json
import math
import os
import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from .model import create_model
from .tokenizer import Tokenizer
from .data import create_dataloader

_INT_RE = re.compile(r"[-+]?\d+$")
_FLOAT_RE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip().replace("_", "")
        if _INT_RE.match(s):
            return int(s)
        if _FLOAT_RE.match(s):
            return float(s)
    return value


def flatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    model_section = config.get("model") or {}
    data_section = config.get("data") or {}

    model_seq = model_section.get("max_seq_len")
    data_seq = data_section.get("max_seq_len")
    if model_seq is not None and data_seq is not None and int(model_seq) != int(data_seq):
        raise ValueError(
            f"model.max_seq_len ({model_seq}) != data.max_seq_len ({data_seq}); "
            "keep a single max_seq_len in the model section"
        )

    merged: Dict[str, Any] = {}
    for section in ("model", "training", "data", "wandb"):
        merged.update(config.get(section, {}) or {})
    merged.update({k: v for k, v in config.items() if k not in ("model", "training", "data", "wandb")})
    return {k: _coerce(v) for k, v in merged.items()}


class Trainer:
    def __init__(self, raw_config: Dict[str, Any]):
        self.config = flatten_config(raw_config)
        cfg = self.config

        requested = cfg.get("device", "auto")
        if torch.cuda.is_available():
            self.device = "cuda" if requested in ("auto", "cuda") else requested
        else:
            self.device = "cpu" if requested in ("auto", "cuda") else requested
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported device: {self.device}")

        self.use_amp = cfg.get("mixed_precision", True) and self.device == "cuda"
        print(f"Device: {self.device} | Mixed precision: {self.use_amp}")

        self.tokenizer = Tokenizer(
            tokenizer_name=cfg.get("tokenizer", "gpt2"),
            max_length=cfg.get("max_seq_len", 2048),
        )

        model_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "dim": cfg.get("dim", 512),
            "n_layers": cfg.get("n_layers", 8),
            "n_heads": cfg.get("n_heads", 8),
            "mlp_hidden_dim": cfg.get("mlp_hidden_dim", 1376),
            "max_seq_len": cfg.get("max_seq_len", 2048),
            "dropout": cfg.get("dropout", 0.0),
            "tie_embeddings": cfg.get("tie_embeddings", True),
        }
        self.model = create_model(model_config).to(self.device)

        total_params = self.model.count_parameters()
        print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.get("learning_rate", 3e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )

        num_workers = cfg.get("num_workers", 0 if self.device == "cpu" else 4)
        mk_loader = lambda split, shuffle: create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=cfg.get("batch_size", 8),
            max_length=cfg.get("max_seq_len", 2048),
            split=split,
            shuffle=shuffle,
            num_workers=num_workers,
            cache_dir=cfg.get("cache_dir", "./data"),
            max_docs=cfg.get("max_docs"),
            dataset=cfg.get("dataset", "wikitext-103-raw-v1"),
        )

        self.train_loader = mk_loader("train", shuffle=True)
        self.val_loader = mk_loader("validation", shuffle=False)

        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = steps_per_epoch * cfg.get("num_epochs", 10)
        lr = cfg.get("learning_rate", 3e-4)
        eta_min = cfg.get("min_lr", lr * 0.1)

        warmup_steps = int(cfg.get("warmup_steps", 0))
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        if warmup_steps > 0:
            self.warmup_steps = warmup_steps
            warmup = LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=eta_min,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            self.warmup_steps = 0
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=eta_min,
            )

        self.scaler = GradScaler(enabled=self.use_amp)

        self.wandb = None
        if cfg.get("use_wandb", False):
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=cfg.get("wandb_project", "miniseek"),
                    entity=cfg.get("wandb_entity"),
                    config=cfg,
                )
            except ImportError:
                print("wandb not installed - skipping logging")

        self.log_dir = cfg.get("log_dir", "./logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training_log.jsonl")
        print(f"Training log: {self.log_file}")

        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float("inf")

        resume = cfg.get("resume_checkpoint")
        if resume and os.path.exists(resume):
            self.load_checkpoint(resume)

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.start_epoch = checkpoint.get("epoch", 1) + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Checkpoint loaded from {path}, step {self.global_step}")

    def _write_log(self, record: Dict[str, Any]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0

        self.train_loader.dataset.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, targets)
                loss = outputs["loss"]

            self.scaler.scale(loss).backward()

            if self.config.get("grad_clip", 0.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})

            if self.wandb is not None:
                self.wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step,
                })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            output = self.model(input_ids, targets)
            total_loss += output["loss"].item()

        avg_loss = total_loss / max(1, len(self.val_loader))
        perplexity = math.exp(min(avg_loss, 100))
        if self.wandb is not None:
            self.wandb.log({
                "val/loss": avg_loss,
                "val/perplexity": perplexity,
            })

        return avg_loss, perplexity

    def train(self):
        num_epochs = self.config.get("num_epochs", 10)
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        save_every = self.config.get("save_every", 1)

        for epoch in range(self.start_epoch, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            train_loss = self.train_epoch(epoch)
            val_loss, perplexity = self.validate()

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")

            if self.wandb is not None:
                self.wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss,
                    "val/perplexity": perplexity,
                })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, "best_model.pt"),
                    epoch,
                    val_loss,
                )

            if epoch % save_every == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                    epoch,
                    val_loss,
                )

            self._write_log({
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "perplexity": round(float(perplexity), 4),
                "best_val_loss": round(float(self.best_val_loss), 6),
                "checkpoint_dir": checkpoint_dir,
            })

        if self.wandb is not None:
            self.wandb.finish()

        print("\nTraining complete!")


def train_from_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()