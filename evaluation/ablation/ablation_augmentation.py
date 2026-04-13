#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study 2: Data Augmentation — with vs. without

Compares two training pipelines on the same U-Net (ResNet-50):
  - No augmentation  : standard resize + ToTensor + Normalize
  - With augmentation: adds random horizontal + vertical flip

Augmentation is applied via AugmentedDataset; the val set is never augmented.

Usage
-----
    cd /path/to/261-mini2
    python evaluation/ablation/ablation_augmentation.py \\
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
    p = argparse.ArgumentParser(description="Ablation: data augmentation")
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
    )

    variants = [
        AblationConfig(name="No Augmentation",
                       checkpoint_dir="checkpoints/ablation/aug_none",
                       augment=False, **shared),
        AblationConfig(name="With Augmentation (flip)",
                       checkpoint_dir="checkpoints/ablation/aug_flip",
                       augment=True, **shared),
    ]

    results = []
    for cfg in variants:
        metrics = run_variant(cfg, device)
        results.append(metrics)

    print_ablation_table(results, title="Ablation 2: Data Augmentation")
    save_ablation_results(
        results,
        os.path.join(args.output_dir, "ablation_augmentation.json"),
    )


if __name__ == "__main__":
    main()
