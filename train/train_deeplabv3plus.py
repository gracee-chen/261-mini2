#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DeepLabV3+ (ResNet-50 encoder) on Pascal VOC 2007.

DeepLabV3+ uses Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale
context and a lightweight encoder-decoder structure to recover spatial detail.

Usage
-----
    cd /path/to/261-mini2
    python train/train_deeplabv3plus.py --voc-root ./VOCtrainval_06-Nov-2007

Key hyperparameters (all overridable via CLI):
    --epochs      50
    --batch-size  8
    --lr          1e-4
    --image-size  256
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.voc_dataset import get_dataloaders
from models.deeplabv3plus_seg import build_deeplabv3plus
from train.losses import SegmentationLoss
from train.trainer import Trainer


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLabV3+ on Pascal VOC 2007")
    p.add_argument("--voc-root",        required=True,
                   help="Directory containing VOCtrainval_06-Nov-2007/")
    p.add_argument("--encoder",         default="resnet50",
                   help="Encoder backbone (default: resnet50)")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch-size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--image-size",      type=int,   default=256)
    p.add_argument("--num-workers",     type=int,   default=2)
    p.add_argument("--checkpoint-dir",  default="checkpoints/deeplabv3plus")
    p.add_argument("--ce-weight",       type=float, default=0.7)
    p.add_argument("--dice-weight",     type=float, default=0.3)
    return p.parse_args()


# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device : {device}")
    print(f"Encoder: {args.encoder}")

    # Data
    train_loader, val_loader = get_dataloaders(
        root=args.voc_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    # Model
    model    = build_deeplabv3plus(encoder_name=args.encoder)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # Optimiser & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Loss
    loss_fn = SegmentationLoss(
        num_classes=21,
        ce_weight=args.ce_weight,
        dice_weight=args.dice_weight,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        scheduler=scheduler,
        scheduler_mode="epoch",
    )
    trainer.run()


if __name__ == "__main__":
    main()
