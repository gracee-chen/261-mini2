# -*- coding: utf-8 -*-
"""
Shared training + evaluation utilities for ablation experiments.
Called by the individual ablation_*.py scripts.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.voc_dataset import get_datasets
from train.losses import SegmentationLoss
from train.trainer import Trainer
from evaluation.metrics.compute_metrics import run_inference, compute_all_metrics


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #

@dataclass
class AblationConfig:
    """All hyperparameters that describe one ablation variant."""
    name: str                            # display name (used in logs / filenames)
    checkpoint_dir: str                  # where to save best.pth / last.pth

    # Data
    voc_root: str = ""
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2
    augment: bool = False                # apply geometric augmentation

    # Model
    encoder_name: str = "resnet50"
    encoder_weights: Optional[str] = "imagenet"   # None = random init

    # Training
    epochs: int = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # Loss
    ce_weight: float = 0.7
    dice_weight: float = 0.3


# --------------------------------------------------------------------------- #
# Augmented dataset wrapper
# --------------------------------------------------------------------------- #

class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper that applies synchronised geometric augmentation to
    a VOCSegmentation dataset returning tensors.

    Augmentations applied:
      - Random horizontal flip (p=0.5)
      - Random vertical flip   (p=0.5)

    Colour jitter would require PIL images (before Normalize); it is omitted
    here to keep the wrapper simple.
    """

    def __init__(self, base_dataset, augment: bool = True):
        self.base    = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import random
        import torchvision.transforms.functional as TF

        image, mask = self.base[idx]   # (3,H,W) float, (1,H,W) uint8

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

        return image, mask


# --------------------------------------------------------------------------- #
# Core: train one variant and evaluate
# --------------------------------------------------------------------------- #

def run_variant(cfg: AblationConfig, device: torch.device,
                compute_hd95: bool = False) -> dict:
    """
    Train a U-Net variant according to AblationConfig (skip if checkpoint exists)
    and evaluate it on the val set.

    Returns
    -------
    dict  with all evaluation metrics plus a "name" key.
    """
    best_ckpt = Path(cfg.checkpoint_dir) / "best.pth"

    # ---- Data -------------------------------------------------------------- #
    train_ds, val_ds = get_datasets(cfg.voc_root, cfg.image_size)
    if cfg.augment:
        train_ds = AugmentedDataset(train_ds, augment=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True)

    # ---- Model ------------------------------------------------------------- #
    from models.unet_seg import build_unet
    model = build_unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
    )

    # ---- Training (skip if checkpoint exists) ------------------------------ #
    if not best_ckpt.exists():
        print(f"\n[{cfg.name}] Training for {cfg.epochs} epochs ...")
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
        loss_fn   = SegmentationLoss(ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            checkpoint_dir=cfg.checkpoint_dir,
            num_epochs=cfg.epochs,
            scheduler=scheduler,
            scheduler_mode="epoch",
        )
        trainer.run()
    else:
        print(f"\n[{cfg.name}] Checkpoint found, skipping training: {best_ckpt}")

    # ---- Evaluation -------------------------------------------------------- #
    print(f"[{cfg.name}] Evaluating ...")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    preds, targets = run_inference(model, val_loader, device)
    metrics        = compute_all_metrics(preds, targets, compute_hd95=compute_hd95)
    metrics["name"] = cfg.name
    return metrics


# --------------------------------------------------------------------------- #
# Result printing and saving
# --------------------------------------------------------------------------- #

def print_ablation_table(results: list, title: str = "Ablation Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Variant':<25} {'mIoU':>8} {'mDice':>8} {'PixAcc':>8}")
    print(f"  {'-'*52}")
    for m in results:
        print(f"  {m['name']:<25} {m['mIoU']:>8.4f} "
              f"{m['mDice']:>8.4f} {m['pixel_accuracy']:>8.4f}")
    print(f"{'='*60}\n")


def save_ablation_results(results: list, output_path: str):
    import json
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved -> {output_path}")
