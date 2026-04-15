#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Best / Worst comparison visualisation — per model.

For each model, generates ONE image with:
  - Left side (3 rows): Top-3 best images (Input | GT | Prediction)
  - Right side (3 rows): Top-3 worst images (Input | GT | Prediction)

Output filename: {model_type}_best_worst_{metric}.png

Usage
-----
    cd /path/to/261-mini2

    python evaluation/visualization/visualize_comparison.py \
        --voc-root   ./VOCtrainval_06-Nov-2007 \
        --model-type unet deeplabv3plus \
        --checkpoint checkpoints/unet/best.pth \
                     checkpoints/deeplabv3plus/best.pth \
        --metric     miou \
        --output-dir results/visualization
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
# Plot one model: best (left) + worst (right) in a single figure
# --------------------------------------------------------------------------- #

def plot_best_worst(val_ds, top3, worst3, preds, model_type,
                    metric_name, score_label, save_path):
    """
    Draw a 3-row x 6-column grid for a single model.

    Left 3 columns  = best  images: Input | GT | Prediction
    Right 3 columns = worst images: Input | GT | Prediction
    """
    n_rows = 3
    n_cols = 6   # 3 sub-columns x 2 sides

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 3.0 * n_rows))

    # Column titles
    left_titles  = ["Input", "Ground Truth", "Prediction"]
    right_titles = ["Input", "Ground Truth", "Prediction"]
    for j, t in enumerate(left_titles):
        axes[0, j].set_title(f"Best — {t}", fontsize=9, fontweight="bold")
    for j, t in enumerate(right_titles):
        axes[0, j + 3].set_title(f"Worst — {t}", fontsize=9, fontweight="bold")

    def _fill_side(col_offset, indices_scores):
        for row_i, (ds_idx, score) in enumerate(indices_scores):
            img_t, mask_t = val_ds[ds_idx]
            img_np  = denorm(img_t)
            gt_mask = clean_mask(mask_t)
            pred    = preds[ds_idx]

            score_str = f"{score:.3f}" if not np.isnan(score) else "N/A"

            ax_inp = axes[row_i, col_offset]
            ax_gt  = axes[row_i, col_offset + 1]
            ax_pr  = axes[row_i, col_offset + 2]

            ax_inp.imshow(img_np);              ax_inp.axis("off")
            ax_gt.imshow(mask_to_rgb(gt_mask)); ax_gt.axis("off")
            ax_pr.imshow(mask_to_rgb(pred));    ax_pr.axis("off")

            ax_inp.set_ylabel(f"idx={ds_idx}\n{score_label}={score_str}",
                              fontsize=7, rotation=0, labelpad=60)

    _fill_side(0, top3)
    _fill_side(3, worst3)

    # Legend
    patches = [
        mpatches.Patch(color=CMAP(i / 20.0), label=f"{i}: {VOC_CLASSES[i]}")
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=6,
               bbox_to_anchor=(0.5, -0.03), frameon=True)

    plt.suptitle(f"{model_type} — Top-3 Best vs Worst  [{score_label}]",
                 fontsize=11, y=1.01)
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

    # val_ds for display uses the default image_size
    _, val_ds = get_datasets(args.voc_root, image_size=args.image_size)

    score_label = "mIoU" if args.metric == "miou" else "Person IoU"

    for model_type, ckpt_path in zip(args.model_type, args.checkpoint):
        img_size = 224 if model_type == "dinov2" else args.image_size
        _, ds_m  = get_datasets(args.voc_root, image_size=img_size)
        loader   = DataLoader(ds_m, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
        model    = load_model(model_type, ckpt_path, device,
                              sam2_ckpt=args.sam2_ckpt, sam2_cfg=args.sam2_cfg)
        preds, targets = run_inference(model, loader, device)

        # Resize predictions/targets to common display size if needed
        if img_size != args.image_size:
            def _resize_mask(arr, size):
                t = torch.from_numpy(arr.astype(np.int64)).unsqueeze(0).unsqueeze(0).float()
                r = F.interpolate(t, size=(size, size), mode="nearest")
                return r.squeeze().long().numpy()
            preds   = [_resize_mask(p, args.image_size) for p in preds]
            targets = [_resize_mask(t, args.image_size) for t in targets]

        # Score each image
        scores = []
        for i in range(len(val_ds)):
            p = preds[i]
            t = targets[i]
            s = per_image_miou(p, t) if args.metric == "miou" \
                else per_image_class_iou(p, t, PERSON_CLASS)
            scores.append((i, s))

        valid_scores  = [(i, s) for i, s in scores if not np.isnan(s)]
        sorted_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)

        top3   = sorted_scores[:3]
        worst3 = sorted_scores[-3:][::-1]

        print(f"\n[{model_type}] Top-3 ({score_label}):")
        for i, s in top3:
            print(f"  idx={i:4d}  {score_label}={s:.4f}")
        print(f"[{model_type}] Worst-3 ({score_label}):")
        for i, s in worst3:
            print(f"  idx={i:4d}  {score_label}={s:.4f}")

        save_path = os.path.join(args.output_dir,
                                 f"{model_type}_best_worst_{args.metric}.png")
        plot_best_worst(val_ds, top3, worst3, preds, model_type,
                        args.metric, score_label, save_path)


if __name__ == "__main__":
    main()
