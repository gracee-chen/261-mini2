#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-model comparison: performance, training time, and generalisation ability.

Reads two sources of data that are produced automatically during training and
evaluation:
  checkpoints/<model>/training_log.json   -- written by train/trainer.py
  results/metrics/<model>_metrics.json    -- written by evaluation/metrics/compute_metrics.py

Outputs (saved to --output-dir)
--------
  summary_table.txt       plain-text comparison table
  loss_curves.png         per-epoch train / val loss for every model
  metrics_bar.png         bar chart: mIoU, mDice, Pixel Accuracy
  training_time_bar.png   total training time per model
  generalization_bar.png  val_loss - train_loss gap (lower = better generalised)

Usage
-----
    cd /path/to/261-mini2

    # Step 1 – train all models (trainer saves training_log.json automatically)
    python train/train_unet.py          --voc-root ./VOCtrainval_06-Nov-2007
    python train/train_deeplabv3plus.py --voc-root ./VOCtrainval_06-Nov-2007
    python train/train_sam2.py          --voc-root ./VOCtrainval_06-Nov-2007 --sam2-ckpt ...
    python train/train_dinov2.py        --voc-root ./VOCtrainval_06-Nov-2007

    # Step 2 – evaluate all models (compute_metrics saves *_metrics.json)
    python evaluation/metrics/compute_metrics.py \\
        --model-type unet deeplabv3plus sam2 dinov2 \\
        --checkpoint checkpoints/unet/best.pth \\
                     checkpoints/deeplabv3plus/best.pth \\
                     checkpoints/sam2/best.pth \\
                     checkpoints/dinov2/best.pth \\
        --voc-root ./VOCtrainval_06-Nov-2007

    # Step 3 – compare
    python evaluation/compare.py \\
        --models unet deeplabv3plus sam2 dinov2 \\
        --checkpoint-dirs checkpoints/unet checkpoints/deeplabv3plus \\
                          checkpoints/sam2  checkpoints/dinov2 \\
        --metrics-dir results/metrics \\
        --output-dir  results/compare
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_training_log(checkpoint_dir: str) -> dict | None:
    path = os.path.join(checkpoint_dir, "training_log.json")
    if not os.path.exists(path):
        print(f"  [warn] training_log.json not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_metrics(metrics_dir: str, model_name: str) -> dict | None:
    path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    if not os.path.exists(path):
        print(f"  [warn] metrics file not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def build_summary(models, training_logs, metrics_dicts) -> list[dict]:
    """Merge training log + eval metrics into one row per model."""
    rows = []
    for name in models:
        log = training_logs.get(name)
        met = metrics_dicts.get(name)
        row = {"model": name}

        if log:
            row["total_time_min"]      = round(log["total_time_sec"] / 60, 1)
            row["avg_epoch_time_sec"]  = log["avg_epoch_time_sec"]
            row["best_val_loss"]       = log["best_val_loss"]
            row["final_train_loss"]    = log["final_train_loss"]
            row["final_val_loss"]      = log["final_val_loss"]
            row["generalization_gap"]  = log["generalization_gap"]
        else:
            row.update({k: float("nan") for k in
                        ["total_time_min", "avg_epoch_time_sec",
                         "best_val_loss", "final_train_loss",
                         "final_val_loss", "generalization_gap"]})

        if met:
            row["mIoU"]           = met["mIoU"]
            row["mDice"]          = met["mDice"]
            row["pixel_accuracy"] = met["pixel_accuracy"]
            row["HD95"]           = met["HD95"]
        else:
            row.update({k: float("nan")
                        for k in ["mIoU", "mDice", "pixel_accuracy", "HD95"]})

        rows.append(row)
    return rows


def print_summary_table(rows: list[dict]) -> str:
    """Return and print a formatted comparison table."""
    header = (
        f"{'Model':<18} {'mIoU':>7} {'mDice':>7} {'PixAcc':>7} "
        f"{'HD95':>7} {'Time(m)':>8} {'Gen.Gap':>9}"
    )
    sep = "-" * len(header)
    lines = ["", "=" * len(header),
             "  Cross-Model Comparison",
             "=" * len(header),
             header, sep]

    for r in rows:
        def f(v): return f"{v:.4f}" if not np.isnan(v) else "  N/A "
        def g(v): return f"{v:.1f}"  if not np.isnan(v) else "  N/A "
        lines.append(
            f"{r['model']:<18} {f(r['mIoU']):>7} {f(r['mDice']):>7} "
            f"{f(r['pixel_accuracy']):>7} {f(r['HD95']):>7} "
            f"{g(r['total_time_min']):>8} {f(r['generalization_gap']):>9}"
        )

    lines += [sep,
              "Gen.Gap = final_val_loss - final_train_loss  "
              "(lower = better generalised)",
              "=" * len(header), ""]
    table = "\n".join(lines)
    print(table)
    return table


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_loss_curves(models, training_logs, output_path: str):
    """Per-epoch train and val loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    for i, name in enumerate(models):
        log = training_logs.get(name)
        if not log or not log.get("epochs"):
            continue
        epochs     = [e["epoch"]      for e in log["epochs"]]
        train_loss = [e["train_loss"] for e in log["epochs"]]
        val_loss   = [e["val_loss"]   for e in log["epochs"]]
        c = COLORS[i % len(COLORS)]
        axes[0].plot(epochs, train_loss, color=c, label=name)
        axes[1].plot(epochs, val_loss,   color=c, label=name, linestyle="--")

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Loss Curves", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.close()


def _bar_chart(models, values, ylabel, title, output_path,
               higher_is_better=True):
    """Generic grouped bar chart for scalar metrics."""
    colors = [COLORS[i % len(COLORS)] for i in range(len(models))]
    x      = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.8), 4))

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    arrow = "higher = better" if higher_is_better else "lower = better"
    ax.set_xlabel(f"({arrow})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.close()


def plot_metrics_bar(models, rows, output_path: str):
    """Grouped bar chart: mIoU, mDice, Pixel Accuracy."""
    metrics  = ["mIoU", "mDice", "pixel_accuracy"]
    labels   = ["mIoU", "mDice", "Pixel Accuracy"]
    x        = np.arange(len(models))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 4))

    for mi, (key, label) in enumerate(zip(metrics, labels)):
        vals = [r[key] if not np.isnan(r[key]) else 0 for r in rows]
        bars = ax.bar(x + mi * width, vals, width,
                      label=label, color=COLORS[mi], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Performance Metrics Comparison (higher = better)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.close()


def plot_training_time(models, rows, output_path: str):
    vals = [r["total_time_min"] for r in rows]
    _bar_chart(models, vals, "Total Training Time (minutes)",
               "Training Time Comparison (lower = faster)",
               output_path, higher_is_better=False)


def plot_generalization(models, rows, output_path: str):
    vals = [r["generalization_gap"] for r in rows]
    _bar_chart(models, vals, "val_loss - train_loss",
               "Generalisation Gap (lower = better generalised)",
               output_path, higher_is_better=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare performance, training time, and generalisation "
                    "across all trained segmentation models."
    )
    p.add_argument("--models",          nargs="+", required=True,
                   help="Model names in display order, e.g. unet deeplabv3plus sam2 dinov2")
    p.add_argument("--checkpoint-dirs", nargs="+", required=True,
                   help="Checkpoint directories in the same order as --models")
    p.add_argument("--metrics-dir",     default="results/metrics",
                   help="Directory containing {model}_metrics.json files")
    p.add_argument("--output-dir",      default="results/compare")
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.models) != len(args.checkpoint_dirs):
        raise ValueError("--models and --checkpoint-dirs must have the same length")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    training_logs  = {}
    metrics_dicts  = {}
    for name, ckpt_dir in zip(args.models, args.checkpoint_dirs):
        training_logs[name] = load_training_log(ckpt_dir)
        metrics_dicts[name] = load_metrics(args.metrics_dir, name)

    rows = build_summary(args.models, training_logs, metrics_dicts)

    # Summary table
    table = print_summary_table(rows)
    with open(os.path.join(args.output_dir, "summary_table.txt"), "w") as f:
        f.write(table)
    print(f"Saved -> {os.path.join(args.output_dir, 'summary_table.txt')}")

    # Plots
    plot_loss_curves(
        args.models, training_logs,
        os.path.join(args.output_dir, "loss_curves.png"),
    )
    plot_metrics_bar(
        args.models, rows,
        os.path.join(args.output_dir, "metrics_bar.png"),
    )
    plot_training_time(
        args.models, rows,
        os.path.join(args.output_dir, "training_time_bar.png"),
    )
    plot_generalization(
        args.models, rows,
        os.path.join(args.output_dir, "generalization_bar.png"),
    )

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
