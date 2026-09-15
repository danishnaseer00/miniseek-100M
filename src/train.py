import json
import math
import os
import random
import re
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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


class CostLedger:

    def __init__(self, path: str):
        self.path = path
        self._data = {"total_gpu_seconds": 0.0, "total_cost_usd": 0.0}
        self.total_gpu_seconds = 0.0
        self.total_cost_usd = 0.0
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                self.total_gpu_seconds = float(self._data.get("total_gpu_seconds", 0.0))
                self.total_cost_usd = float(self._data.get("total_cost_usd", 0.0))
            except Exception as e:
                print(f"WARNING: could not read cost ledger {path}: {e}")

    def add_seconds(self, seconds: float, usd_per_sec: float):
        self.total_gpu_seconds += seconds
        self.total_cost_usd += seconds * usd_per_sec

    def save(self):
        self._data["total_gpu_seconds"] = self.total_gpu_seconds
        self._data["total_cost_usd"] = self.total_cost_usd
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        os.replace(tmp, self.path)


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
        self.reset_optimizer = bool(cfg.get("reset_optimizer", False))
        print(f"Device: {self.device} | Mixed precision: {self.use_amp} | Reset optimizer: {self.reset_optimizer}")

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

        expected_m = cfg.get("expected_params_million")
        if expected_m and abs(total_params / 1e6 - expected_m) > max(0.5, expected_m * 0.02):
            raise ValueError(
                f"Expected {expected_m}M params but model has {total_params/1e6:.2f}M. "
                "Fix the config before training."
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
            corpus_dir=cfg.get("corpus_dir"),
        )

        self.train_loader = mk_loader("train", shuffle=True)
        self.val_loader = mk_loader("validation", shuffle=False)

        self.steps_per_epoch = max(1, len(self.train_loader))
        self.grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1) or 1))
        self.steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.grad_accum_steps)
        )
        self._init_optimizer()
        self._init_scheduler()

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

        # --- budget / cost accounting -----------------------------------
        self.ledger = CostLedger(os.path.join(self.log_dir, "cost_ledger.json"))
        self.budget_usd = float(cfg.get("budget_usd", 0.0) or 0.0)
        self.usd_per_sec = float(cfg.get("usd_per_hour", 2.0)) / 3600.0
        self.allow_over_budget = bool(cfg.get("allow_over_budget", False))
        self.on_ledger_write = None
        self._run_gpu_seconds = 0.0
        self._run_started = time.monotonic()
        self.epoch_secs: list = []
        self._last_periodic_sync_step = 0
        print(
            f"Budget: cap=${self.budget_usd:.2f} | spent=${self.ledger.total_cost_usd:.4f} "
            f"({self.ledger.total_gpu_seconds:.0f}s) | rate=${self.usd_per_sec*3600:.2f}/h | "
            f"allow_over_budget={self.allow_over_budget}"
        )

        self.global_step = 0
        self.start_epoch = 1
        self.resume_step = 0
        self.best_val_loss = float("inf")

        resume = cfg.get("resume_checkpoint")
        if resume and os.path.exists(resume):
            self.load_checkpoint(resume)

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        epoch_in_progress: Optional[int] = None,
        step_in_epoch: Optional[int] = None,
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = {
            "torch_rng_state": torch.get_rng_state(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()

        torch.save({
            "epoch": epoch,
            "epoch_in_progress": epoch_in_progress,
            "step_in_epoch": step_in_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "rng": rng,
        }, path)
        print(f"Checkpoint saved to {path}")

    def _init_optimizer(self):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )

    def _init_scheduler(self):
        start_epoch = int(getattr(self, "start_epoch", 1))
        num_epochs = int(self.config.get("num_epochs", 10))
        remaining = max(1, num_epochs - start_epoch + 1)
        total_steps = self.steps_per_epoch * remaining
        lr = self.config.get("learning_rate", 3e-4)
        eta_min = self.config.get("min_lr", lr * 0.1)
        warmup_steps = int(self.config.get("warmup_steps", 0))
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))

        if warmup_steps > 0:
            self.warmup_steps = warmup_steps
            initial_lr = self.config.get("initial_lr")
            start_factor = max(1e-6, min(1.0, initial_lr / lr)) if initial_lr else 1e-3
            warmup = LinearLR(
                self.optimizer,
                start_factor=start_factor,
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

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if not self.reset_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        rng = checkpoint.get("rng", {})
        try:
            if "torch_rng_state" in rng:
                torch.set_rng_state(rng["torch_rng_state"])
            if "random_state" in rng:
                random.setstate(rng["random_state"])
            if "numpy_random_state" in rng:
                np.random.set_state(rng["numpy_random_state"])
            if "torch_cuda_rng_state" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda_rng_state"])
        except (TypeError, ValueError, RuntimeError) as e:
            print(f"WARNING: could not restore RNG state from checkpoint ({e}); proceeding with fresh RNG.")

        if self.reset_optimizer:
            self.global_step = 0
            override = self.config.get("start_epoch")
            if override is not None:
                self.start_epoch = int(override)
                self.resume_step = 0
                print(
                    f"reset_optimizer: fresh optimizer+scheduler; forced start_epoch={self.start_epoch}, step 0"
                )
            else:
                self.start_epoch = 1
                self.resume_step = 0
                print("reset_optimizer: fresh optimizer+scheduler at low LR; counters reset to epoch 1, step 0")
            self.best_val_loss = float("inf")
            self._init_optimizer()
            self._init_scheduler()
        else:
            self.global_step = checkpoint.get("global_step", 0)
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

            epoch = checkpoint.get("epoch", 1)
            in_progress = checkpoint.get("epoch_in_progress")
            step_in_epoch = checkpoint.get("step_in_epoch") or 0
            if in_progress is not None:
                self.start_epoch = int(in_progress)
                self.resume_step = int(step_in_epoch)
            else:
                self.start_epoch = int(epoch) + 1
                self.resume_step = 0

            ckpt_steps = checkpoint.get("steps_per_epoch")
            if ckpt_steps and ckpt_steps != self.steps_per_epoch:
                print(
                    f"WARNING: checkpoint steps/epoch={ckpt_steps} != current {self.steps_per_epoch}; "
                    "cosine schedule may misalign."
                )
            print(
                f"Checkpoint loaded from {path}: resume epoch {self.start_epoch} at step "
                f"{self.resume_step}/{self.steps_per_epoch}, global step {self.global_step}"
            )

    # ------------------------------------------------------------------ logging / ledger
    def _write_log(self, record: Dict[str, Any]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _sync_ledger(self):
        self.ledger.add_seconds(self._run_gpu_seconds, self.usd_per_sec)
        self._run_gpu_seconds = 0.0
        self.ledger.save()
        if self.on_ledger_write is not None:
            try:
                self.on_ledger_write()
            except Exception as e:
                print(f"WARNING: ledger sync callback failed: {e}")

    def _budget_exhausted(self) -> bool:
        if not (self.budget_usd > 0) or self.allow_over_budget:
            return False
        spent = self.ledger.total_cost_usd + self._run_gpu_seconds * self.usd_per_sec
        return spent >= self.budget_usd

    def _projected_next_epoch_cost(self) -> float:
        if not self.epoch_secs:
            return 0.0
        avg = sum(self.epoch_secs[-3:]) / len(self.epoch_secs[-3:])
        return avg * 1.05 * self.usd_per_sec

    def _budget_projected_over(self) -> bool:
        if not (self.budget_usd > 0) or self.allow_over_budget:
            return False
        return (self.ledger.total_cost_usd + self._projected_next_epoch_cost()) > self.budget_usd

    # ------------------------------------------------------------------ training
    def train_epoch(self, epoch: int, resume_step: int) -> Tuple[float, bool]:
        """Train one epoch. Returns (avg_loss_so_far, budget_stopped)."""
        self.model.train()
        total_loss = 0.0
        n = 0
        epoch_t0 = time.monotonic()

        self.train_loader.dataset.set_epoch(epoch)

        save_every_steps = int(self.config.get("save_every_steps", 500) or 500)
        budget_check_every = int(self.config.get("budget_check_every", 25) or 25)
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        latest_path = os.path.join(checkpoint_dir, "latest.pt")

        pbar = tqdm(
            desc=f"Epoch {epoch}",
            total=self.steps_per_epoch,
            initial=resume_step // self.grad_accum_steps,
        )
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx < resume_step:
                continue

            is_boundary = (batch_idx + 1) % self.grad_accum_steps == 0 or batch_idx == len(
                self.train_loader
            ) - 1

            if batch_idx % self.grad_accum_steps == 0:
                self.optimizer.zero_grad()

            step_t0 = time.monotonic()
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(input_ids, targets)
                loss = outputs["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            total_loss += outputs["loss"].item()
            n += 1
            self._run_gpu_seconds += time.monotonic() - step_t0

            if not is_boundary:
                continue

            if self.config.get("grad_clip", 0.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"],
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": total_loss / max(1, n), "lr": self.scheduler.get_last_lr()[0]})

            if self.wandb is not None:
                self.wandb.log({
                    "train/loss": total_loss / max(1, n),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": self.global_step,
                })

            if self.global_step % save_every_steps == 0:
                self.save_checkpoint(
                    latest_path,
                    epoch,
                    float("nan"),
                    epoch_in_progress=epoch,
                    step_in_epoch=batch_idx + 1,
                )
                self._sync_ledger()

            if self.global_step % budget_check_every == 0 and self._budget_exhausted():
                print(
                    f"\nBUDGET STOP at step {self.global_step}: spent "
                    f"${self.ledger.total_cost_usd + self._run_gpu_seconds*self.usd_per_sec:.3f} "
                    f"vs cap ${self.budget_usd:.2f}. Saving state."
                )
                self.save_checkpoint(
                    latest_path,
                    epoch,
                    float("nan"),
                    epoch_in_progress=epoch,
                    step_in_epoch=batch_idx + 1,
                )
                self._sync_ledger()
                return total_loss / max(1, n), True

        self.epoch_secs.append(time.monotonic() - epoch_t0)
        return total_loss / max(1, n), False

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

        budget_stopped = False
        epoch = int(self.start_epoch)

        while epoch <= num_epochs:
            if self._budget_projected_over():
                budget_stopped = True
                self._sync_ledger()
                break

            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            resume_step = self.resume_step if epoch == self.start_epoch else 0
            self.resume_step = 0

            train_loss, budget_stopped = self.train_epoch(epoch, resume_step)
            if budget_stopped:
                break

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

            self._sync_ledger()
            self._write_log({
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "perplexity": round(float(perplexity), 4),
                "best_val_loss": round(float(self.best_val_loss), 6),
                "cost_usd_total": round(float(self.ledger.total_cost_usd), 5),
                "gpu_seconds_total": round(float(self.ledger.total_gpu_seconds), 1),
                "checkpoint_dir": checkpoint_dir,
            })
            epoch += 1

        self._sync_ledger()
        if self.wandb is not None:
            self.wandb.finish()

        print(f"\nGPU time this session: {time.monotonic() - self._run_started:.0f}s")
        print(
            f"Total cost (ledger): ${self.ledger.total_cost_usd:.4f} "
            f"over {self.ledger.total_gpu_seconds:.0f}s GPU"
        )
        if budget_stopped:
            print(
                "Stopped early due to budget.\n"
                "To resume: rerun with --checkpoint <checkpoint_dir>/latest.pt. "
                "If the cap is already spent, raise budget_usd or set allow_over_budget: true."
            )
        else:
            print("Training complete!")