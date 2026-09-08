import yaml
import argparse
import torch
from src.train import train_from_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Miniseek model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (YAML)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    print(f"Training with config: {args.config}")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print("\nConfig:")
    print(yaml.dump(config, default_flow_style=False))
    
    if args.checkpoint:
        config["resume_checkpoint"] = args.checkpoint
        print(f"Resuming from checkpoint: {args.checkpoint}")
    
    from src.train import Trainer
    trainer = Trainer(config)
    trainer.train()
