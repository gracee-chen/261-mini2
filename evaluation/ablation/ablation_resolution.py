#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study 5: Input Resolution — 128 vs 256 vs 384

Compares three input resolutions on the same U-Net (ResNet-50):
  - 128 x 128  : fastest training, coarsest segmentation
  - 256 x 256  : default resolution used across all main experiments
  - 384 x 384  : finest detail, highest memory and compute cost

Higher resolution preserves more spatial detail (small objects, thin boundaries)
but increases training time quadratically.  This experiment quantifies the
speed / accuracy tradeoff and justifies the choice of 256 as the default.

All other hyperparameters (encoder, loss, augmentation, pretrain) are held
constant.  Only image_size changes between variants.

Usage
-----
    cd /path/to/261-mini2
    python evaluation/ablation/ablation_resolution.py \\
        --voc-root ./VOCtrainval_06-Nov-2007
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

from evaluation.ablation.run_ablation import (
    AblationConfig, run_variant, save_ablation_results,
)


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: input resolution")
    p.add_argument("--voc-root",    required=True)
    p.add_argument("--epochs",      type=int,   default=25)
    p.add_argument("--batch-size",  type=int,   default=8,
                   help="Consider reducing to 4 if 384x384 runs out of memory")
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--output-dir",  default="results/ablation")
    return p.parse_args()


def print_resolution_table(results: list):
    """Extended table that also shows training time per epoch (if recorded)."""
    print(f"\n{'='*68}")
    print("  Ablation 5: Input Resolution")
    print(f"{'='*68}")
    print(f"  {'Variant':<20} {'Resolution':>12} {'mIoU':>8} {'mDice':>8} {'PixAcc':>8}")
    print(f"  {'-'*60}")
    res_map = {"128x128": "128x128", "256x256": "256x256", "384x384": "384x384"}
    for m in results:
        res = m.get("resolution", "—")
        print(f"  {m['name']:<20} {res:>12} {m['mIoU']:>8.4f} "
              f"{m['mDice']:>8.4f} {m['pixel_accuracy']:>8.4f}")
    print(f"{'='*68}\n")


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    shared = dict(
        voc_root=args.voc_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        augment=False,
    )

    resolutions = [128, 256, 384]

    variants = [
        AblationConfig(
            name=f"UNet-{r}x{r}",
            checkpoint_dir=f"checkpoints/ablation/resolution_{r}",
            image_size=r,
            **shared,
        )
        for r in resolutions
    ]

    results = []
    for cfg, res in zip(variants, resolutions):
        t0      = time.time()
        metrics = run_variant(cfg, device)
        elapsed = time.time() - t0
        metrics["resolution"]       = f"{res}x{res}"
        metrics["total_time_sec"]   = round(elapsed, 1)
        results.append(metrics)

    print_resolution_table(results)
    save_ablation_results(
        results,
        os.path.join(args.output_dir, "ablation_resolution.json"),
    )


if __name__ == "__main__":
    main()
