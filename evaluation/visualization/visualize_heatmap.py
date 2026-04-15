#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-class IoU heatmap across models.

Rows    = 21 VOC classes
Columns = models

Reads *_metrics.json files produced by compute_metrics.py.

Usage
-----
    cd /path/to/261-mini2

    python evaluation/visualization/visualize_heatmap.py \
        --metrics-dir results/metrics \
        --models unet deeplabv3plus dinov2 sam2 \
        --output results/visualization/perclass_iou_heatmap.png
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dataset.voc_dataset import VOC_CLASSES, NUM_CLASSES


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_iou_per_class(metrics_path: str) -> np.ndarray:
    """Load iou_per_class from a metrics JSON file. Returns array of length NUM_CLASSES."""
    with open(metrics_path, "r") as f:
        data = json.load(f)
    values = data["iou_per_class"]
    arr = np.array(values, dtype=np.float64)
    # JSON encodes NaN as null; json.load turns those into None
    return arr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Per-class IoU heatmap across models")
    p.add_argument("--metrics-dir", required=True,
                   help="Directory containing <model>_metrics.json files")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names (matching <model>_metrics.json filenames)")
    p.add_argument("--output", default="results/visualization/perclass_iou_heatmap.png")
    return p.parse_args()


def main():
    args = parse_args()

    # Build matrix: (num_classes, num_models)
    n_classes = NUM_CLASSES
    n_models  = len(args.models)
    matrix    = np.full((n_classes, n_models), np.nan, dtype=np.float64)

    for col, model_name in enumerate(args.models):
        path = os.path.join(args.metrics_dir, f"{model_name}_metrics.json")
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, column will be NaN")
            continue
        iou = load_iou_per_class(path)
        matrix[:len(iou), col] = iou

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, 2.0 * n_models), 10))

    # Mask NaN for proper colormap handling
    masked = np.ma.masked_invalid(matrix)

    cmap = plt.cm.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="0.85")  # gray for NaN

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="IoU", fraction=0.03, pad=0.04)

    # Axes
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(args.models, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f"{i}: {VOC_CLASSES[i]}" for i in range(n_classes)], fontsize=8)
    ax.set_xlabel("Model")
    ax.set_ylabel("VOC Class")
    ax.set_title("Per-class IoU Heatmap", fontsize=13)

    # Annotate each cell with the IoU value
    for i in range(n_classes):
        for j in range(n_models):
            val = matrix[i, j]
            if np.isnan(val):
                text = "NaN"
                color = "gray"
            else:
                text = f"{val:.2f}"
                color = "white" if val > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved -> {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
