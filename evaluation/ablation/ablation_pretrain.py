#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study 4: Pre-training — From Scratch vs ImageNet Pretrained Backbone

Compares two weight initialisation strategies on the same U-Net (ResNet-50):
  - From scratch        (encoder_weights=None)
  - ImageNet pretrained (encoder_weights='imagenet')

Transfer learning typically gives a significant boost on small datasets like
VOC 2007.  This experiment quantifies the gain from pre-training.

Note: training from scratch usually requires more epochs to converge.
Increase --epochs (e.g. 50+) for a fair comparison.

Usage
-----
    cd /path/to/261-mini2
    python evaluation/ablation/ablation_pretrain.py \\
        --voc-root ./VOCtrainval_06-Nov-2007 --epochs 40
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
    p = argparse.ArgumentParser(description="Ablation: pretrained vs from-scratch backbone")
    p.add_argument("--voc-root",    required=True)
    p.add_argument("--epochs",      type=int,   default=25,
                   help="From-scratch training typically needs more epochs; "
                        "consider --epochs 40+")
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
        encoder_name="resnet50", augment=False,
    )

    variants = [
        AblationConfig(name="From Scratch",
                       checkpoint_dir="checkpoints/ablation/pretrain_none",
                       encoder_weights=None, **shared),
        AblationConfig(name="ImageNet Pretrained",
                       checkpoint_dir="checkpoints/ablation/pretrain_imagenet",
                       encoder_weights="imagenet", **shared),
    ]

    results = []
    for cfg in variants:
        metrics = run_variant(cfg, device)
        results.append(metrics)

    print_ablation_table(results, title="Ablation 4: Pre-training vs From Scratch")
    save_ablation_results(
        results,
        os.path.join(args.output_dir, "ablation_pretrain.json"),
    )


if __name__ == "__main__":
    main()
