#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune SAM2 for closed-vocabulary semantic segmentation on Pascal VOC 2007.

Strategy
--------
SAM2's powerful Hiera image encoder is used as a frozen backbone.
Only the lightweight fuse + head layers (< 2M params) are trained.
Optional: pass --unfreeze-encoder to fine-tune the full model (much slower).

Prerequisites
-------------
1. Install SAM2:
       pip install git+https://github.com/facebookresearch/sam2.git

2. Download SAM2.1 Base+ checkpoint (~160 MB):
       wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
   (or Tiny/Small for faster experiments)

Usage
-----
    cd /path/to/261-mini2
    python train/train_sam2.py \
        --voc-root  ./VOCtrainval_06-Nov-2007 \
        --sam2-ckpt ./sam2.1_hiera_base_plus.pt

Key hyperparameters:
    --epochs      30          (fewer epochs needed — encoder is pretrained)
    --batch-size  8
    --lr          3e-4        (higher LR safe because encoder is frozen)
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
from models.sam2_seg import build_sam2_seg
from train.losses import SegmentationLoss
from train.trainer import Trainer


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train SAM2 semantic seg on Pascal VOC 2007")
    p.add_argument("--voc-root",          required=True,
                   help="Directory containing VOCtrainval_06-Nov-2007/")
    p.add_argument("--sam2-ckpt",         required=True,
                   help="Path to SAM2 checkpoint (.pt file)")
    p.add_argument("--sam2-cfg",
                   default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                   help="SAM2 config name (relative to sam2 package root)")
    p.add_argument("--unfreeze-encoder",  action="store_true",
                   help="Fine-tune entire SAM2 encoder (expensive, more accurate)")
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--batch-size",        type=int,   default=8)
    p.add_argument("--lr",                type=float, default=3e-4,
                   help="LR for head; encoder uses lr/10 when unfrozen")
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--image-size",        type=int,   default=256)
    p.add_argument("--num-workers",       type=int,   default=2)
    p.add_argument("--checkpoint-dir",    default="checkpoints/sam2")
    p.add_argument("--ce-weight",         type=float, default=0.7)
    p.add_argument("--dice-weight",       type=float, default=0.3)
    return p.parse_args()


# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    freeze_encoder = not args.unfreeze_encoder
    print(f"Device          : {device}")
    print(f"SAM2 checkpoint : {args.sam2_ckpt}")
    print(f"Freeze encoder  : {freeze_encoder}")

    # Data
    train_loader, val_loader = get_dataloaders(
        root=args.voc_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    # Model — SAM2 encoder (frozen) + semantic head
    model = build_sam2_seg(
        checkpoint=args.sam2_ckpt,
        config=args.sam2_cfg,
        num_classes=21,
        freeze_encoder=freeze_encoder,
        device=str(device),
    )

    n_total    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {n_total:,}")
    print(f"Trainable params: {n_trainable:,}")

    # Optimiser — separate LR for encoder (when unfrozen) vs head
    if freeze_encoder:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            [
                {"params": model.image_encoder.parameters(), "lr": args.lr / 10},
                {"params": list(model.fuse.parameters()) + list(model.head.parameters()),
                 "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )

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
