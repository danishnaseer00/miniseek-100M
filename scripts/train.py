import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import Trainer, flatten_config


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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto/cpu/cuda)",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw_config = yaml.safe_load(f)

    if args.checkpoint:
        raw_config["resume_checkpoint"] = args.checkpoint
        print(f"Resuming from checkpoint: {args.checkpoint}")

    if args.device:
        raw_config["device"] = args.device

    print("Effective config:")
    print(yaml.dump(flatten_config(raw_config), default_flow_style=False))

    trainer = Trainer(raw_config)
    trainer.train()