#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mosaic-style qualitative visualisation: show multiple models side-by-side
on the same set of test images.

Layout
------
  Rows    = test images (randomly sampled from val set)
  Columns = [Input | Ground Truth | Model 1 | Model 2 | ...]

Usage
-----
    cd /path/to/261-mini2

    python evaluation/visualization/visualize_mosaic.py \\
        --voc-root   ./VOCtrainval_06-Nov-2007 \\
        --models     unet:checkpoints/unet/best.pth \\
                     deeplabv3plus:checkpoints/deeplabv3plus/best.pth \\
                     dinov2:checkpoints/dinov2/best.pth \\
        --num-images 6 \\
        --output     results/visualization/mosaic.png
"""

import argparse
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset.voc_dataset import VOC_CLASSES, NUM_CLASSES, get_datasets
from evaluation.metrics.compute_metrics import load_model

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
IGNORE_INDEX = 255
CMAP = plt.cm.get_cmap("tab20", 21)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def denorm(tensor):
    """(3,H,W) normalised tensor -> (H,W,3) numpy RGB in [0,1]."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """(H,W) class index -> (H,W,3) coloured RGB in [0,1].  255 -> black."""
    rgb = CMAP(mask / 20.0)[:, :, :3].astype(np.float32)
    rgb[mask == IGNORE_INDEX] = 0.0
    return rgb


def clean_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    """(1,H,W) uint8 tensor -> (H,W) long, values >20 mapped to 255."""
    m = mask_tensor.squeeze(0).long().numpy()
    m[m > 20] = IGNORE_INDEX
    return m


@torch.no_grad()
def predict_single(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """(1,3,H,W) -> (H,W) class-index numpy array."""
    x      = image_tensor.to(device)
    logits = model(x)
    if hasattr(logits, "logits"):
        logits = logits.logits
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(logits, size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Mosaic visualisation of segmentation results")
    p.add_argument("--voc-root",   required=True)
    p.add_argument("--models",     nargs="+", required=True,
                   help="Format: model_type:checkpoint_path  "
                        "e.g.  unet:checkpoints/unet/best.pth")
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     default="results/visualization/mosaic.png")
    p.add_argument("--sam2-ckpt",  default=None)
    p.add_argument("--sam2-cfg",   default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    return p.parse_args()


def main():
    args   = parse_args()
    random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    # Parse model specs
    model_specs = []
    for spec in args.models:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Bad format: '{spec}'.  Expected model_type:checkpoint_path")
        model_specs.append((parts[0], parts[1]))

    # Load all models
    models = {}
    for model_type, ckpt_path in model_specs:
        img_size = 224 if model_type == "dinov2" else args.image_size
        models[model_type] = {
            "model":      load_model(model_type, ckpt_path, device,
                                     sam2_ckpt=args.sam2_ckpt,
                                     sam2_cfg=args.sam2_cfg),
            "image_size": img_size,
        }

    # Load val dataset
    _, val_ds = get_datasets(args.voc_root, image_size=args.image_size)
    indices   = random.sample(range(len(val_ds)), min(args.num_images, len(val_ds)))

    col_names = ["Input", "Ground Truth"] + [m for m, _ in model_specs]
    n_cols    = len(col_names)
    n_rows    = len(indices)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for j, name in enumerate(col_names):
        axes[0, j].set_title(name, fontsize=10, fontweight="bold")

    for row_i, ds_idx in enumerate(indices):
        img_t, mask_t = val_ds[ds_idx]
        img_np  = denorm(img_t)
        gt_mask = clean_mask(mask_t)

        axes[row_i, 0].imshow(img_np);          axes[row_i, 0].axis("off")
        axes[row_i, 1].imshow(mask_to_rgb(gt_mask)); axes[row_i, 1].axis("off")

        # Green border around Ground Truth cell
        gt_rect = matplotlib.patches.Rectangle(
            (0, 0), gt_mask.shape[1] - 1, gt_mask.shape[0] - 1,
            linewidth=3, edgecolor="lime", facecolor="none",
        )
        axes[row_i, 1].add_patch(gt_rect)

        for col_j, (model_type, _) in enumerate(model_specs):
            info = models[model_type]

            if info["image_size"] != args.image_size:
                img_input = F.interpolate(
                    img_t.unsqueeze(0),
                    size=(info["image_size"], info["image_size"]),
                    mode="bilinear", align_corners=False,
                )
            else:
                img_input = img_t.unsqueeze(0)

            pred = predict_single(info["model"], img_input, device)

            if pred.shape[0] != args.image_size:
                pred = F.interpolate(
                    torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                    size=(args.image_size, args.image_size),
                    mode="nearest",
                ).squeeze().long().numpy()

            axes[row_i, col_j + 2].imshow(mask_to_rgb(pred))
            axes[row_i, col_j + 2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=CMAP(i / 20.0), label=f"{i}: {VOC_CLASSES[i]}")
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=6,
               bbox_to_anchor=(0.5, -0.02), frameon=True, title="VOC Classes")

    plt.suptitle("Mosaic — Qualitative Segmentation Results", fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Mosaic saved -> {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
