import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any
import yaml
import os
from tqdm import tqdm
import wandb
from datetime import datetime

from .model import create_model
from .tokenizer import Tokenizer
from .data import create_dataloader


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = Tokenizer(
            tokenizer_name=config.get("tokenizer", "gpt2"),
            max_length=config.get("max_seq_len", 2048),
        )
        
        model_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "dim": config.get("dim", 512),
            "n_layers": config.get("n_layers", 8),
            "n_heads": config.get("n_heads", 8),
            "mlp_hidden_dim": config.get("mlp_hidden_dim", 1376),
            "max_seq_len": config.get("max_seq_len", 2048),
            "dropout": config.get("dropout", 0.0),
            "tie_embeddings": config.get("tie_embeddings", True),
        }
        self.model = create_model(model_config).to(self.device)
        
        total_params = self.model.count_parameters()
        print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )
        
        self.train_loader = create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=config.get("batch_size", 8),
            max_length=config.get("max_seq_len", 2048),
            split="train",
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            cache_dir=config.get("cache_dir", "./data"),
        )
        
        self.val_loader = create_dataloader(
            tokenizer=self.tokenizer,
            batch_size=config.get("batch_size", 8),
            max_length=config.get("max_seq_len", 2048),
            split="validation",
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            cache_dir=config.get("cache_dir", "./data"),
        )
        
        total_steps = len(self.train_loader) * config.get("num_epochs", 10)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.get("min_lr", 3e-5),
        )
        
        self.scaler = GradScaler(enabled=config.get("mixed_precision", True))
        
        if config.get("use_wandb", False):
            wandb.init(
                project=config.get("wandb_project", "miniseek"),
                config=config,
            )
        
        self.global_step = 0
        self.best_val_loss = float("inf")
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        print(f"Checkpoint loaded from {path}, step {self.global_step}")
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.get("mixed_precision", True)):
                outputs = self.model(input_ids, targets)
                loss = outputs["loss"]
            
            self.scaler.scale(loss).backward()
            
            if self.config.get("grad_clip", 0.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})
            
            if self.config.get("use_wandb", False):
                wandb.log({
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
            
            outputs = self.model(input_ids, targets)
            total_loss += outputs["loss"].item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity.item()
    
    def train(self):
        num_epochs = self.config.get("num_epochs", 10)
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        save_every = self.config.get("save_every", 1)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch(epoch)
            val_loss, perplexity = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            
            if self.config.get("use_wandb", False):
                wandb.log({
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
                    val_loss
                )
            
            if epoch % save_every == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                    epoch,
                    val_loss
                )
        
        if self.config.get("use_wandb", False):
            wandb.finish()
        
        print("\nTraining complete!")


def train_from_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    trainer.train()
