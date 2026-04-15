#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute and visualise the pixel-level confusion matrix for a trained model.

Rows = ground-truth class, Columns = predicted class.
Diagonal = correctly classified pixels.
Label 255 (ignore/boundary) is excluded.

Usage
-----
    cd /path/to/261-mini2

    python evaluation/metrics/confusion_matrix.py \\
        --model-type  unet \\
        --checkpoint  checkpoints/unet/best.pth \\
        --voc-root    ./VOCtrainval_06-Nov-2007 \\
        --output-dir  results/metrics
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset.voc_dataset import VOC_CLASSES, NUM_CLASSES, get_datasets
from evaluation.metrics.compute_metrics import load_model, run_inference

IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# Confusion matrix computation
# --------------------------------------------------------------------------- #

def build_confusion_matrix(preds, targets, num_classes=NUM_CLASSES,
                            ignore=IGNORE_INDEX) -> np.ndarray:
    """
    Build a (num_classes, num_classes) confusion matrix.
    cm[i, j] = number of pixels with true label i predicted as j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        valid = (t != ignore)
        p_v   = p[valid].ravel()
        t_v   = t[valid].ravel()
        mask  = (p_v < num_classes) & (t_v < num_classes)
        np.add.at(cm, (t_v[mask], p_v[mask]), 1)
    return cm


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(cm: np.ndarray, class_names, normalize=True,
                          save_path=None, title="Confusion Matrix"):
    """
    Plot a confusion-matrix heatmap.

    Parameters
    ----------
    normalize  : row-normalise (show fraction instead of counts)
    save_path  : file path to save; None -> plt.show()
    """
    if normalize:
        row_sums   = cm.sum(axis=1, keepdims=True)
        cm_plot    = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt        = ".2f"
        cbar_label = "Fraction (row-normalised)"
    else:
        cm_plot    = cm
        fmt        = "d"
        cbar_label = "Pixel count"

    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.03, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    # Annotate diagonal cells
    for i in range(n):
        val   = cm_plot[i, i]
        color = "white" if val > 0.5 else "black"
        text  = f"{val:.2f}" if normalize else str(int(val))
        ax.text(i, i, text, ha="center", va="center",
                fontsize=6, color=color, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved -> {save_path}")
        plt.close()
    else:
        plt.show()


def plot_all_confusion_matrices(cm_list, class_names, normalize=True,
                                save_path=None):
    """
    Plot multiple confusion matrices stacked vertically (one subplot per model).

    Parameters
    ----------
    cm_list    : list of (model_name, cm_array) tuples
    class_names: list of class label strings
    normalize  : row-normalise (show fraction instead of counts)
    save_path  : file path to save; None -> plt.show()
    """
    n = len(class_names)
    n_models = len(cm_list)

    fig, axes = plt.subplots(n_models, 1,
                              figsize=(14, 12 * n_models))
    if n_models == 1:
        axes = [axes]

    for ax_i, (model_name, cm) in enumerate(cm_list):
        ax = axes[ax_i]
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_plot  = np.where(row_sums > 0, cm / row_sums, 0.0)
            fmt      = ".2f"
        else:
            cm_plot = cm
            fmt     = "d"

        im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else None)
        fig.colorbar(im, ax=ax,
                     label="Fraction (row-normalised)" if normalize else "Pixel count",
                     fraction=0.03, pad=0.04)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"Confusion Matrix — {model_name}")

        # Annotate diagonal cells
        for i in range(n):
            val   = cm_plot[i, i]
            color = "white" if val > 0.5 else "black"
            text  = f"{val:.2f}" if normalize else str(int(val))
            ax.text(i, i, text, ha="center", va="center",
                    fontsize=6, color=color, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Combined confusion matrices saved -> {save_path}")
        plt.close()
    else:
        plt.show()


def print_confusion_summary(cm: np.ndarray, class_names):
    """For each class, print the most common misclassification."""
    print(f"\n{'='*58}")
    print("  Top Confusion per Class")
    print(f"{'='*58}")
    print(f"  {'True class':<14}  {'Confused with':<14}  {'Error rate':>10}")
    print(f"  {'-'*50}")
    for i, name in enumerate(class_names):
        row     = cm[i].copy()
        row[i]  = 0
        if row.sum() == 0 or cm[i].sum() == 0:
            continue
        j        = row.argmax()
        err_rate = row[j] / cm[i].sum()
        print(f"  {name:<14}  {class_names[j]:<14}  {err_rate:>10.2%}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Compute and plot confusion matrix")
    p.add_argument("--model-type",  nargs="+", required=True,
                   choices=["unet", "deeplabv3plus", "sam2", "dinov2"])
    p.add_argument("--checkpoint",  nargs="+", required=True)
    p.add_argument("--voc-root",    required=True)
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-dir",  default="results/metrics")
    p.add_argument("--no-normalize", action="store_true",
                   help="Show raw counts instead of row-normalised fractions")
    p.add_argument("--sam2-ckpt",   default=None)
    p.add_argument("--sam2-cfg",    default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    if len(args.model_type) != len(args.checkpoint):
        raise ValueError("--model-type and --checkpoint must have the same length")

    os.makedirs(args.output_dir, exist_ok=True)
    normalize = not args.no_normalize

    cm_list = []  # (model_name, cm) for combined plot

    for model_type, ckpt_path in zip(args.model_type, args.checkpoint):
        image_size = 224 if model_type == "dinov2" else args.image_size
        _, val_ds  = get_datasets(args.voc_root, image_size=image_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

        model  = load_model(model_type, ckpt_path, device,
                            sam2_ckpt=args.sam2_ckpt, sam2_cfg=args.sam2_cfg)
        preds, targets = run_inference(model, val_loader, device)

        cm = build_confusion_matrix(preds, targets)
        cm_list.append((model_type, cm))
        print_confusion_summary(cm, VOC_CLASSES)

        # Individual PNG
        save_path = os.path.join(args.output_dir,
                                 f"{model_type}_confusion_matrix.png")
        plot_confusion_matrix(
            cm, VOC_CLASSES,
            normalize=normalize,
            save_path=save_path,
            title=f"Confusion Matrix — {model_type}",
        )

    # Combined figure when multiple models are given
    if len(cm_list) > 1:
        combined_path = os.path.join(args.output_dir,
                                     "all_confusion_matrices.png")
        plot_all_confusion_matrices(
            cm_list, VOC_CLASSES,
            normalize=normalize,
            save_path=combined_path,
        )


if __name__ == "__main__":
    main()
