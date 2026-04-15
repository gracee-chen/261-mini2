#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study 3: Loss Function — Cross-Entropy vs Dice vs CE+Dice

Compares three loss configurations on the same U-Net (ResNet-50):
  - CrossEntropy only   (ce_weight=1.0, dice_weight=0.0)
  - Dice only           (ce_weight=0.0, dice_weight=1.0)
  - CE + Dice combined  (ce_weight=0.7, dice_weight=0.3)  <- default

CrossEntropy converges fast but is sensitive to class imbalance.
Dice is more robust to imbalance but has noisier gradients.
The combination typically achieves the best result.

Usage
-----
    cd /path/to/261-mini2
    python evaluation/ablation/ablation_loss.py \\
        --voc-root ./VOCtrainval_06-Nov-2007
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch

from evaluation.ablation.run_ablation import (
    AblationConfig, run_variant, print_ablation_table, save_ablation_results,
)


def parse_args():
    p = argparse.ArgumentParser(description="Ablation: loss function comparison")
    p.add_argument("--voc-root",    required=True)
    p.add_argument("--epochs",      type=int,   default=25)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--image-size",  type=int,   default=256)
    p.add_argument("--num-workers", type=int,   default=2)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--output-dir",  default="results/ablation")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    shared = dict(
        voc_root=args.voc_root, image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
        epochs=args.epochs, lr=args.lr,
        encoder_name="resnet50", encoder_weights="imagenet",
        augment=False,
    )

    variants = [
        AblationConfig(name="CrossEntropy only",
                       checkpoint_dir="checkpoints/ablation/loss_ce",
                       ce_weight=1.0, dice_weight=0.0, **shared),
        AblationConfig(name="Dice only",
                       checkpoint_dir="checkpoints/ablation/loss_dice",
                       ce_weight=0.0, dice_weight=1.0, **shared),
        AblationConfig(name="CE + Dice (1.0 + 1.0)",
                       checkpoint_dir="checkpoints/ablation/loss_combined",
                       ce_weight=1.0, dice_weight=1.0, **shared),
    ]

    results = []
    for cfg in variants:
        metrics = run_variant(cfg, device)
        results.append(metrics)

    print_ablation_table(results, title="Ablation 3: Loss Function")
    save_ablation_results(
        results,
        os.path.join(args.output_dir, "ablation_loss.json"),
    )


if __name__ == "__main__":
    main()
