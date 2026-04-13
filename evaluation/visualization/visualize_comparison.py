#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best / Worst comparison visualisation.

For a given model (or pair of models):
  1. Compute per-image mIoU (or person-class IoU) on the entire val set.
  2. Rank images by that score.
  3. Show Top-3 best and Top-3 worst as side-by-side plots:
       Input | Ground Truth | Model Prediction(s)

The "person" class (index 15) focus is motivated by the assignment requirement
to consider human class performance.

Usage
-----
    cd /path/to/261-mini2

    # Single model — ranked by overall mIoU
    python evaluation/visualization/visualize_comparison.py \\
        --voc-root   ./VOCtrainval_06-Nov-2007 \\
        --model-type unet \\
        --checkpoint checkpoints/unet/best.pth \\
        --output-dir results/visualization

    # Two models compared, ranked by person IoU
    python evaluation/visualization/visualize_comparison.py \\
        --voc-root    ./VOCtrainval_06-Nov-2007 \\
        --model-type  unet deeplabv3plus \\
        --checkpoint  checkpoints/unet/best.pth \\
                      checkpoints/deeplabv3plus/best.pth \\
        --metric      person \\
        --output-dir  results/visualization
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

from dataset.voc_dataset import VOC_CLASSES, NUM_CLASSES, get_datasets
from evaluation.metrics.compute_metrics import load_model, run_inference
from evaluation.visualization.visualize_mosaic import denorm, mask_to_rgb, clean_mask

IGNORE_INDEX = 255
PERSON_CLASS = 15     # VOC_CLASSES[15] == "person"
CMAP         = plt.cm.get_cmap("tab20", 21)


# --------------------------------------------------------------------------- #
# Per-image metrics
# --------------------------------------------------------------------------- #

def per_image_miou(pred: np.ndarray, target: np.ndarray,
                   num_classes=NUM_CLASSES) -> float:
    """Mean IoU for a single image (ignores label 255)."""
    ious = []
    for c in range(num_classes):
        valid = (target != IGNORE_INDEX)
        pc = (pred == c) & valid
        tc = (target == c) & valid
        u  = (pc | tc).sum()
        if u > 0:
            ious.append((pc & tc).sum() / u)
    return float(np.mean(ious)) if ious else 0.0


def per_image_class_iou(pred: np.ndarray, target: np.ndarray, cls: int) -> float:
    """IoU for a specific class on a single image."""
    valid = (target != IGNORE_INDEX)
    pc = (pred == cls) & valid
    tc = (target == cls) & valid
    u  = (pc | tc).sum()
    return float((pc & tc).sum() / u) if u > 0 else np.nan


# --------------------------------------------------------------------------- #
# Plot one set of results (top-3 or worst-3)
# --------------------------------------------------------------------------- #

def plot_set(val_ds, indices_scores, model_preds_dict, model_names,
             title: str, save_path: str, score_label: str):
    """
    Draw a grid: rows = images, columns = [Input | GT | model1 | model2 | ...]

    Parameters
    ----------
    indices_scores  : list of (dataset_index, score) tuples, length 3
    model_preds_dict: {model_name: list_of_pred_arrays_for_all_val_images}
    """
    col_titles = ["Input", "Ground Truth"] + model_names
    n_cols = len(col_titles)
    n_rows = len(indices_scores)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=9, fontweight="bold")

    for row_i, (ds_idx, score) in enumerate(indices_scores):
        img_t, mask_t = val_ds[ds_idx]
        img_np  = denorm(img_t)
        gt_mask = clean_mask(mask_t)

        score_str = f"{score:.3f}" if not np.isnan(score) else "N/A"
        axes[row_i, 0].set_ylabel(f"idx={ds_idx}\n{score_label}={score_str}",
                                   fontsize=7, rotation=0, labelpad=60)

        axes[row_i, 0].imshow(img_np);               axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(mask_to_rgb(gt_mask)); axes[row_i, 1].axis("off")

        for col_j, mname in enumerate(model_names):
            pred = model_preds_dict[mname][ds_idx]
            axes[row_i, col_j + 2].imshow(mask_to_rgb(pred))
            axes[row_i, col_j + 2].axis("off")

    patches = [
        mpatches.Patch(color=CMAP(i / 20.0), label=f"{i}: {VOC_CLASSES[i]}")
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=6,
               bbox_to_anchor=(0.5, -0.03), frameon=True)

    plt.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {save_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Top-3 / Worst-3 segmentation comparison")
    p.add_argument("--voc-root",    required=True)
    p.add_argument("--model-type",  nargs="+", required=True,
                   choices=["unet", "deeplabv3plus", "sam2", "dinov2"])
    p.add_argument("--checkpoint",  nargs="+", required=True)
    p.add_argument("--image-size",  type=int, default=256)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--metric",      default="miou",
                   choices=["miou", "person"],
                   help="Ranking metric: overall mIoU or person-class IoU")
    p.add_argument("--output-dir",  default="results/visualization")
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

    image_size = 224 if all(m == "dinov2" for m in args.model_type) else args.image_size
    _, val_ds  = get_datasets(args.voc_root, image_size=image_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Collect predictions for every model
    all_preds   = {}
    all_targets = None

    for model_type, ckpt_path in zip(args.model_type, args.checkpoint):
        model  = load_model(model_type, ckpt_path, device,
                            sam2_ckpt=args.sam2_ckpt, sam2_cfg=args.sam2_cfg)
        preds, targets = run_inference(model, val_loader, device)
        all_preds[model_type] = preds
        if all_targets is None:
            all_targets = targets

    # Score each image using the primary model
    primary_model = args.model_type[0]
    score_label   = "mIoU" if args.metric == "miou" else "Person IoU"

    scores = []
    for i in range(len(val_ds)):
        p = all_preds[primary_model][i]
        t = all_targets[i]
        s = per_image_miou(p, t) if args.metric == "miou" \
            else per_image_class_iou(p, t, PERSON_CLASS)
        scores.append((i, s))

    valid_scores  = [(i, s) for i, s in scores if not np.isnan(s)]
    sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)

    top3   = sorted_scores[:3]
    worst3 = sorted_scores[-3:][::-1]

    print(f"\nTop-3 ({score_label}, model={primary_model}):")
    for i, s in top3:
        print(f"  idx={i:4d}  {score_label}={s:.4f}")
    print(f"\nWorst-3 ({score_label}, model={primary_model}):")
    for i, s in worst3:
        print(f"  idx={i:4d}  {score_label}={s:.4f}")

    model_names = args.model_type

    plot_set(
        val_ds, top3, all_preds, model_names,
        title=f"Top-3 Best Results  [{primary_model}  {score_label}]",
        save_path=os.path.join(args.output_dir, f"top3_{args.metric}.png"),
        score_label=score_label,
    )
    plot_set(
        val_ds, worst3, all_preds, model_names,
        title=f"Top-3 Worst Results  [{primary_model}  {score_label}]",
        save_path=os.path.join(args.output_dir, f"worst3_{args.metric}.png"),
        score_label=score_label,
    )


if __name__ == "__main__":
    main()
