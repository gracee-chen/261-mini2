#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DINOv2 + conv decoder on Pascal VOC 2007 for semantic segmentation.

Model: DINOv2 (ViT-B/14) + 3-layer conv head
Paper: Oquab et al., "DINOv2: Learning Robust Visual Features without
       Supervision", TMLR 2024.  arXiv 2304.07193.

Strategy (two-phase training)
------------------------------
Phase 1  (default): freeze DINOv2 backbone, train only the decoder head.
  - Fast convergence, low memory.
  - Add --unfreeze-backbone to skip directly to full fine-tuning.

Phase 2 (optional): unfreeze entire model and fine-tune end-to-end.
  - Run a second time with --resume and --unfreeze-backbone.
  - Use a lower LR (--lr 1e-5) for stable fine-tuning.

Input size
----------
DINOv2 uses 14×14 patches.  We use 224×224 images (= 16×14 patches) which is
the standard DINOv2 input size.  The dataset's 256×256 transforms are
overridden to 224×224 in this script.

Usage
-----
    cd /path/to/261-mini2

    # Phase 1 – train head only (recommended start)
    python train/train_dinov2.py --voc-root ./VOCtrainval_06-Nov-2007

    # Phase 2 – full fine-tune from Phase-1 checkpoint
    python train/train_dinov2.py --voc-root ./VOCtrainval_06-Nov-2007 \
        --unfreeze-backbone --resume checkpoints/dinov2/best.pth --lr 1e-5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.voc_dataset import get_dataloaders
from models.dinov2_seg import build_dinov2_seg
from train.losses import SegmentationLoss
from train.trainer import Trainer

# DINOv2 ViT-B/14 requires input sizes that are multiples of 14.
# 224 = 16 × 14 is the standard pre-training size.
DINOV2_IMAGE_SIZE = 224


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2 seg on Pascal VOC 2007")
    p.add_argument("--voc-root",           required=True,
                   help="Directory containing VOCtrainval_06-Nov-2007/")
    p.add_argument("--dinov2-model",       default="facebook/dinov2-base",
                   help="HuggingFace model ID  (e.g. facebook/dinov2-large)")
    p.add_argument("--unfreeze-backbone",  action="store_true",
                   help="Fine-tune DINOv2 backbone (Phase 2)")
    p.add_argument("--resume",             default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--epochs",             type=int,   default=30)
    p.add_argument("--batch-size",         type=int,   default=16,
                   help="Larger batches possible since backbone is frozen")
    p.add_argument("--lr",                 type=float, default=1e-3,
                   help="Head LR (Phase 1); use 1e-5 for Phase 2 fine-tuning")
    p.add_argument("--weight-decay",       type=float, default=1e-4)
    p.add_argument("--hidden-dim",         type=int,   default=256,
                   help="Decoder intermediate channel width")
    p.add_argument("--num-workers",        type=int,   default=2)
    p.add_argument("--checkpoint-dir",     default="checkpoints/dinov2")
    p.add_argument("--ce-weight",          type=float, default=0.7)
    p.add_argument("--dice-weight",        type=float, default=0.3)
    return p.parse_args()


# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    freeze_backbone = not args.unfreeze_backbone
    print(f"Device          : {device}")
    print(f"DINOv2 model    : {args.dinov2_model}")
    print(f"Freeze backbone : {freeze_backbone}")
    print(f"Image size      : {DINOV2_IMAGE_SIZE}×{DINOV2_IMAGE_SIZE}")

    # Data (224×224 for DINOv2)
    train_loader, val_loader = get_dataloaders(
        root=args.voc_root,
        batch_size=args.batch_size,
        image_size=DINOV2_IMAGE_SIZE,
        num_workers=args.num_workers,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    # Model
    model = build_dinov2_seg(
        model_name=args.dinov2_model,
        num_classes=21,
        freeze_backbone=freeze_backbone,
        hidden_dim=args.hidden_dim,
    )

    # Optional: resume from checkpoint
    # Phase 2 (--unfreeze-backbone + --resume): load weights only; fresh optimizer.
    # Same-phase continuation (--resume without --unfreeze-backbone): restore
    # weights + optimizer and continue from the saved epoch.
    start_epoch            = 1
    resume_optimizer_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        if not args.unfreeze_backbone:
            # Same phase: pick up where we left off
            start_epoch            = ckpt["epoch"] + 1
            resume_optimizer_state = ckpt.get("optimizer")
        print(f"Resumed from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
        if args.unfreeze_backbone:
            print("  Phase 2: fresh optimizer, backbone unfrozen.")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {n_total:,}")
    print(f"Trainable params: {n_trainable:,}")

    # Optimiser — separate LR groups for backbone vs head (Phase 2)
    if freeze_backbone:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            [
                {"params": model.backbone.parameters(),  "lr": args.lr / 10},
                {"params": model.decoder.parameters(),   "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )

    # Restore optimizer state for same-phase resume
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print(f"  Optimizer state restored.")
        except Exception as exc:
            print(f"  [warn] Could not restore optimizer state ({exc}). Starting fresh.")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Advance scheduler to match already-completed epochs so the LR curve
    # is consistent when resuming the same phase.
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

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
        start_epoch=start_epoch,
        scheduler=scheduler,
        scheduler_mode="epoch",
    )
    trainer.run()


if __name__ == "__main__":
    main()
