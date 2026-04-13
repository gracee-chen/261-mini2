# -*- coding: utf-8 -*-
"""
Dataset exploration script for Pascal VOC 2007 Segmentation.

Usage:
    python dataset/explore.py --root /path/to/VOCtrainval_06-Nov-2007

The root directory should contain the VOCtrainval_06-Nov-2007/ folder
(i.e. the folder downloaded from Kaggle and unzipped).
"""

import argparse
import os
import sys

# Allow running from either the project root or the dataset/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.voc_dataset import (
    VOC_CLASSES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    get_datasets,
    mask_to_class_index,
)


# --------------------------------------------------------------------------- #
# Visualisation helpers
# --------------------------------------------------------------------------- #
def show_sample(image: torch.Tensor, mask: torch.Tensor, title: str = "",
                save_path: str = None):
    """
    Display one (image, mask) pair side by side.

    Parameters
    ----------
    image : (3, H, W) float tensor  – normalised ImageNet image
    mask  : (1, H, W) or (H, W) uint8/long tensor
    save_path : if provided, save the figure to this path instead of showing
    """
    # Denormalise image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    # Prepare mask
    mask_np = mask_to_class_index(mask).numpy()
    mask_display = mask_np.copy()
    mask_display[mask_display == 255] = 0   # treat ignore as background

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if title:
        fig.suptitle(title, fontsize=12)

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    seg_map = axes[1].imshow(mask_display, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("Segmentation Mask (cleaned)")
    axes[1].axis("off")

    cbar = plt.colorbar(seg_map, ax=axes[1], ticks=range(21))
    cbar.ax.set_yticklabels([f"{i}: {VOC_CLASSES[i]}" for i in range(21)])
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
        plt.close()
    else:
        plt.show()


def class_distribution(loader: DataLoader, split: str = "train"):
    """Count pixel-level class frequency across the entire split."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, masks in loader:
        for mask in masks:
            cls_mask = mask_to_class_index(mask).numpy()
            valid = cls_mask[cls_mask != 255]
            counts += np.bincount(valid, minlength=NUM_CLASSES)

    print(f"\nPixel-level class distribution ({split}):")
    total = counts.sum()
    for i, (name, cnt) in enumerate(zip(VOC_CLASSES, counts)):
        pct = 100 * cnt / total if total > 0 else 0
        print(f"  {i:2d}  {name:<15s}  {cnt:>12,}  ({pct:5.2f}%)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Explore Pascal VOC 2007 dataset")
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get("VOC_ROOT", "./VOCtrainval_06-Nov-2007"),
        help="Path containing the VOCtrainval_06-Nov-2007/ folder",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=2,
                        help="Number of random samples to visualise")
    parser.add_argument("--dist", action="store_true",
                        help="Compute pixel-level class distribution (slow)")
    parser.add_argument("--save-dir", default=None,
                        help="Save sample figures to this directory (for Colab)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Class info
    # ------------------------------------------------------------------ #
    print(f"Number of classes: {NUM_CLASSES}")
    print("Classes:", VOC_CLASSES)
    print("\nPascal VOC 2007 Class Mapping:")
    for name, idx in CLASS_TO_IDX.items():
        print(f"  {idx:2d}  {name}")

    # ------------------------------------------------------------------ #
    # 2. Load datasets
    # ------------------------------------------------------------------ #
    train_dataset, val_dataset = get_datasets(args.root, args.image_size)

    print(f"\nTrain samples : {len(train_dataset)}")
    print(f"Val   samples : {len(val_dataset)}  (used as test set)")

    # ------------------------------------------------------------------ #
    # 3. DataLoaders
    # ------------------------------------------------------------------ #
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    images, masks = next(iter(train_loader))
    print(f"\nImage batch shape : {images.shape}")   # (B, 3, H, W)
    print(f"Mask  batch shape : {masks.shape}")      # (B, 1, H, W)

    # ------------------------------------------------------------------ #
    # 4. Visualise random samples
    # ------------------------------------------------------------------ #
    import random
    indices = random.sample(range(len(train_dataset)), args.num_samples)
    for i, idx in enumerate(indices):
        img, mask = train_dataset[idx]
        classes_in_mask = np.unique(mask_to_class_index(mask).numpy())
        classes_in_mask = classes_in_mask[classes_in_mask != 255]
        class_names = [VOC_CLASSES[c] for c in classes_in_mask]
        print(f"\nSample {idx} – classes present: {class_names}")
        save_path = os.path.join(args.save_dir, f"sample_{idx}.png") if args.save_dir else None
        show_sample(img, mask, title=f"Train sample {idx}", save_path=save_path)

    # ------------------------------------------------------------------ #
    # 5. Optional: class distribution
    # ------------------------------------------------------------------ #
    if args.dist:
        class_distribution(train_loader, "train")


if __name__ == "__main__":
    main()
