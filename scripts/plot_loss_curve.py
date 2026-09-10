"""Plot paper-style training curves for Miniseek.

Usage:
    python scripts/plot_loss_curve.py [path/to/training_log.jsonl]

With a JSONL log (epoch, train_loss, val_loss, perplexity) it reads from the
log. Without arguments it plots the phase-1 real-data baseline, whose
per-epoch val losses were read from the volume checkpoints (ckpt_history())
and whose only recorded train-loss point is the epoch-5 run-end value.
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE = {
    "epochs": [1, 2, 3, 4, 5],
    "global_steps": [7198, 14396, 21594, 28792, 35990],
    "val_loss": [3.9735, 3.826, 3.7438, 3.6793, 3.6582],
    "final_train_loss": 3.6944,
}


def load_from_log(path: str):
    epochs, train, val = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            epochs.append(rec["epoch"])
            train.append(rec["train_loss"])
            val.append(rec["val_loss"])
    return epochs, train, val


def main():
    fname = os.path.join(OUT_DIR, "loss_curve.png")
    if len(sys.argv) > 1:
        epochs, train, val = load_from_log(sys.argv[1])
        source = os.path.basename(sys.argv[1])
    else:
        epochs = BASELINE["epochs"]
        val = BASELINE["val_loss"]
        train = [None] * 4 + [BASELINE["final_train_loss"]]
        source = "checkpoint-per-epoch val loss + epoch-5 train loss"

    ppl = [math.exp(l) for l in val]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": True,
    })

    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax1.twinx()

    colors = {"train": "#1f77b4", "val": "#d62728", "ppl": "#7f7f7f"}

    ax1.plot(epochs, val, color=colors["val"], marker="o", ms=5, lw=1.6, label="Val loss")
    ax1.scatter(epochs[-1], train[-1], color=colors["train"], marker="s", s=36, zorder=5)
    ax1.plot([], [], color=colors["train"], marker="s", lw=0, label="Train loss (final)")

    ax2.plot(epochs, ppl, color=colors["ppl"], ls="--", lw=1.2, marker="^", ms=4.5, label="Val perpl.")

    alpha = 0.06
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        ax1.axvspan(e0, e1, color="#888888", alpha=alpha)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax2.set_ylabel("Perplexity")
    ax1.set_ylim(3.55, 4.05)
    ax1.set_xticks(epochs)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9)

    ax1.annotate(
        f"val 38.79",
        xy=(5, 3.6582), xytext=(4.45, 3.615),
        color=colors["val"], fontsize=8.5,
    )
    ax1.annotate(
        f"train 3.69 (run end)",
        xy=(5, 3.6944), xytext=(4.15, 3.935),
        color=colors["train"], fontsize=8.5,
    )

    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    fig.suptitle("Miniseek - dense baseline (17.7M params, WikiText-103)", fontsize=12)
    fig.text(0.015, 0.015, f"source: {source}", fontsize=7, color="#666666")

    fig.savefig(fname, dpi=200)
    fig.savefig(os.path.join(OUT_DIR, "loss_curve.pdf"))
    print("Wrote", fname)
    print("steps:", BASELINE["global_steps"])
    print("val loss:", val)
    print("val perpl.:", [round(v, 2) for v in ppl])
    print("final train loss:", train[-1])


if __name__ == "__main__":
    main()