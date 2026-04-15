#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained segmentation model on the Pascal VOC 2007 val set.

Metrics computed
----------------
  mDice        : Mean Dice Coefficient across all 21 classes
  mIoU         : Mean Intersection-over-Union
  HD95         : 95th-percentile Hausdorff Distance (requires scipy)
  Pixel Acc    : Fraction of correctly labelled pixels (ignoring label 255)
  Per-class IoU and Accuracy

Usage
-----
    cd /path/to/261-mini2

    # Evaluate a single model
    python evaluation/metrics/compute_metrics.py \\
        --model-type  unet \\
        --checkpoint  checkpoints/unet/best.pth \\
        --voc-root    ./VOCtrainval_06-Nov-2007

    # Evaluate and compare multiple models
    python evaluation/metrics/compute_metrics.py \\
        --model-type  unet deeplabv3plus dinov2 \\
        --checkpoint  checkpoints/unet/best.pth \\
                      checkpoints/deeplabv3plus/best.pth \\
                      checkpoints/dinov2/best.pth \\
        --voc-root    ./VOCtrainval_06-Nov-2007 \\
        --output-dir  results/metrics
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.voc_dataset import VOC_CLASSES, NUM_CLASSES, get_datasets

IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_type: str, checkpoint_path: str, device: torch.device,
               sam2_ckpt: str = None, sam2_cfg: str = None):
    """Build the right model architecture and load checkpoint weights."""
    if model_type == "unet":
        from models.unet_seg import build_unet
        model = build_unet()

    elif model_type == "deeplabv3plus":
        from models.deeplabv3plus_seg import build_deeplabv3plus
        model = build_deeplabv3plus()

    elif model_type == "sam2":
        if sam2_ckpt is None:
            raise ValueError("--sam2-ckpt is required when evaluating SAM2")
        from models.sam2_seg import build_sam2_seg
        model = build_sam2_seg(
            checkpoint=sam2_ckpt,
            config=sam2_cfg or "configs/sam2.1/sam2.1_hiera_b+.yaml",
            freeze_encoder=True,
            device=str(device),
        )

    elif model_type == "dinov2":
        from models.dinov2_seg import build_dinov2_seg
        model = build_dinov2_seg(freeze_backbone=False)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"  Loaded [{model_type}]  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    return model


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(model, loader: DataLoader, device: torch.device):
    """
    Run full-dataset inference.

    Returns
    -------
    all_preds   : list of (H, W) numpy int arrays (class indices 0-20)
    all_targets : list of (H, W) numpy int arrays (255 = ignore)
    """
    all_preds, all_targets = [], []

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.squeeze(1).long()
        masks[masks > 20] = IGNORE_INDEX

        logits = model(images)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode="bilinear", align_corners=False)

        preds  = logits.argmax(dim=1).cpu().numpy()
        masks  = masks.numpy()
        for i in range(preds.shape[0]):
            all_preds.append(preds[i])
            all_targets.append(masks[i])

    return all_preds, all_targets


# --------------------------------------------------------------------------- #
# Metric functions
# --------------------------------------------------------------------------- #

def _pixel_accuracy(preds, targets, ignore=IGNORE_INDEX):
    correct = total = 0
    for p, t in zip(preds, targets):
        valid   = (t != ignore)
        correct += (p[valid] == t[valid]).sum()
        total   += valid.sum()
    return correct / total if total > 0 else 0.0


def _per_class_iou_acc(preds, targets, num_classes=NUM_CLASSES, ignore=IGNORE_INDEX):
    """Return (iou_per_class, acc_per_class) each of shape (num_classes,)."""
    inter   = np.zeros(num_classes, dtype=np.int64)
    union   = np.zeros(num_classes, dtype=np.int64)
    tp      = np.zeros(num_classes, dtype=np.int64)
    support = np.zeros(num_classes, dtype=np.int64)

    for p, t in zip(preds, targets):
        valid = (t != ignore)
        p_v, t_v = p[valid], t[valid]
        for c in range(num_classes):
            pc = (p_v == c)
            tc = (t_v == c)
            inter[c]   += (pc & tc).sum()
            union[c]   += (pc | tc).sum()
            tp[c]      += (pc & tc).sum()
            support[c] += tc.sum()

    iou = np.where(union > 0,   inter / union,   np.nan)
    acc = np.where(support > 0, tp    / support, np.nan)
    return iou, acc


def _dice_per_class(preds, targets, num_classes=NUM_CLASSES, ignore=IGNORE_INDEX):
    """Return per-class Dice computed globally (dataset-level, not per-image averaged)."""
    inter      = np.zeros(num_classes, dtype=np.int64)
    cardinality = np.zeros(num_classes, dtype=np.int64)

    for p, t in zip(preds, targets):
        valid = (t != ignore)
        p_v, t_v = p[valid], t[valid]
        for c in range(num_classes):
            pc = (p_v == c)
            tc = (t_v == c)
            inter[c]      += (pc & tc).sum()
            cardinality[c] += pc.sum() + tc.sum()

    return np.where(cardinality > 0, 2.0 * inter / cardinality, np.nan)


def _hausdorff_95_binary(pred_mask: np.ndarray, gt_mask: np.ndarray,
                         max_pts: int = 800) -> float:
    """95th-percentile Hausdorff distance between two binary masks."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial.distance import cdist

    def boundary(m):
        m = m.astype(bool)
        return m & ~binary_erosion(m)

    pred_pts = np.argwhere(boundary(pred_mask))
    gt_pts   = np.argwhere(boundary(gt_mask))

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan

    if len(pred_pts) > max_pts:
        pred_pts = pred_pts[np.random.choice(len(pred_pts), max_pts, replace=False)]
    if len(gt_pts) > max_pts:
        gt_pts   = gt_pts[np.random.choice(len(gt_pts),   max_pts, replace=False)]

    d_p2g = cdist(pred_pts, gt_pts).min(axis=1)
    d_g2p = cdist(gt_pts, pred_pts).min(axis=1)
    return float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))


def _mean_hd95(preds, targets, num_classes=NUM_CLASSES, ignore=IGNORE_INDEX,
               max_images: int = 100):
    """
    Compute per-class HD95 averaged over images, then average across classes.
    Limited to max_images for speed.
    """
    try:
        import scipy  # noqa
    except ImportError:
        print("  [Warning] scipy not installed — skipping HD95.  pip install scipy")
        return np.nan, np.full(num_classes, np.nan)

    indices      = list(range(min(len(preds), max_images)))
    hd_per_class = [[] for _ in range(num_classes)]

    for i in indices:
        p, t = preds[i], targets[i]
        valid_mask = (t != ignore)
        for c in range(num_classes):
            pred_c = (p == c) & valid_mask
            gt_c   = (t == c) & valid_mask
            if gt_c.sum() == 0:
                continue
            hd = _hausdorff_95_binary(pred_c, gt_c)
            if not np.isnan(hd):
                hd_per_class[c].append(hd)

    hd95_per_class = np.array([np.mean(v) if v else np.nan for v in hd_per_class])
    return float(np.nanmean(hd95_per_class)), hd95_per_class


# --------------------------------------------------------------------------- #
# Unified interface
# --------------------------------------------------------------------------- #

def compute_all_metrics(preds, targets, num_classes=NUM_CLASSES,
                        compute_hd95=True, hd95_max_images=100):
    """
    Compute all metrics from precomputed prediction and target lists.

    Parameters
    ----------
    preds / targets  : list of (H, W) numpy arrays
    compute_hd95     : whether to compute HD95 (requires scipy, slow)
    hd95_max_images  : max images used for HD95

    Returns
    -------
    dict with keys: mDice, mIoU, HD95, pixel_accuracy,
                    iou_per_class, acc_per_class, dice_per_class, hd95_per_class
    """
    pixel_acc              = _pixel_accuracy(preds, targets)
    iou_per_class, acc_per_class = _per_class_iou_acc(preds, targets, num_classes)
    dice_per_class         = _dice_per_class(preds, targets, num_classes)

    miou  = float(np.nanmean(iou_per_class))
    mdice = float(np.nanmean(dice_per_class))

    if compute_hd95:
        print("  Computing HD95 (this may take a few minutes)...")
        mean_hd95, hd95_per_class = _mean_hd95(preds, targets, num_classes,
                                                max_images=hd95_max_images)
    else:
        mean_hd95      = np.nan
        hd95_per_class = np.full(num_classes, np.nan)

    return {
        "mDice":          mdice,
        "mIoU":           miou,
        "HD95":           mean_hd95,
        "pixel_accuracy": float(pixel_acc),
        "iou_per_class":  iou_per_class.tolist(),
        "acc_per_class":  acc_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "hd95_per_class": hd95_per_class.tolist(),
    }


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #

def print_metrics(metrics: dict, model_name: str = ""):
    header = f"Results [{model_name}]" if model_name else "Results"
    print(f"\n{'='*62}")
    print(f"  {header}")
    print(f"{'='*62}")
    print(f"  mDice        : {metrics['mDice']:.4f}")
    print(f"  mIoU         : {metrics['mIoU']:.4f}")
    hd = f"{metrics['HD95']:.2f} px" if not np.isnan(metrics['HD95']) else "N/A"
    print(f"  HD95         : {hd}")
    print(f"  Pixel Acc    : {metrics['pixel_accuracy']:.4f}")
    print(f"\n  {'Class':<15} {'IoU':>8} {'Acc':>8} {'Dice':>8}")
    print(f"  {'-'*42}")
    for i, name in enumerate(VOC_CLASSES):
        iou  = metrics["iou_per_class"][i]
        acc  = metrics["acc_per_class"][i]
        dice = metrics["dice_per_class"][i]
        iou_s  = f"{iou:.4f}"  if not np.isnan(iou)  else "  —   "
        acc_s  = f"{acc:.4f}"  if not np.isnan(acc)  else "  —   "
        dice_s = f"{dice:.4f}" if not np.isnan(dice) else "  —   "
        print(f"  {name:<15} {iou_s:>8} {acc_s:>8} {dice_s:>8}")
    print(f"{'='*62}\n")


def print_comparison(results: dict):
    """Print a summary comparison table for multiple models."""
    print(f"\n{'='*65}")
    print("  Model Comparison")
    print(f"{'='*65}")
    print(f"  {'Model':<18} {'mDice':>8} {'mIoU':>8} {'HD95':>8} {'PixAcc':>8}")
    print(f"  {'-'*55}")
    for name, m in results.items():
        hd = f"{m['HD95']:.2f}" if not np.isnan(m['HD95']) else "  N/A "
        print(f"  {name:<18} {m['mDice']:>8.4f} {m['mIoU']:>8.4f} "
              f"{hd:>8} {m['pixel_accuracy']:>8.4f}")
    print(f"{'='*65}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate semantic segmentation models")
    p.add_argument("--model-type",   nargs="+", required=True,
                   choices=["unet", "deeplabv3plus", "sam2", "dinov2"])
    p.add_argument("--checkpoint",   nargs="+", required=True,
                   help="Checkpoint paths in the same order as --model-type")
    p.add_argument("--voc-root",     required=True)
    p.add_argument("--image-size",   type=int, default=256)
    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--num-workers",  type=int, default=2)
    p.add_argument("--no-hd95",      action="store_true",
                   help="Skip HD95 computation (faster)")
    p.add_argument("--hd95-images",  type=int, default=100)
    p.add_argument("--output-dir",   default="results/metrics")
    p.add_argument("--sam2-ckpt",    default=None)
    p.add_argument("--sam2-cfg",     default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    if len(args.model_type) != len(args.checkpoint):
        raise ValueError("--model-type and --checkpoint must have the same length")

    all_results = {}

    for model_type, ckpt_path in zip(args.model_type, args.checkpoint):
        print(f"\n{'='*60}")
        print(f"  Model     : {model_type}")
        print(f"  Checkpoint: {ckpt_path}")

        # DINOv2 requires 224x224 (multiple of patch size 14);
        # all other models use the value from --image-size.
        image_size = 224 if model_type == "dinov2" else args.image_size
        _, val_ds  = get_datasets(args.voc_root, image_size=image_size)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        print(f"  Val set   : {len(val_ds)} images  ({image_size}x{image_size})")

        model  = load_model(model_type, ckpt_path, device,
                            sam2_ckpt=args.sam2_ckpt, sam2_cfg=args.sam2_cfg)
        preds, targets = run_inference(model, val_loader, device)
        metrics = compute_all_metrics(
            preds, targets,
            compute_hd95=not args.no_hd95,
            hd95_max_images=args.hd95_images,
        )
        print_metrics(metrics, model_name=model_type)
        all_results[model_type] = metrics

        out_path = os.path.join(args.output_dir, f"{model_type}_metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved -> {out_path}")

    if len(all_results) > 1:
        print_comparison(all_results)
        out_path = os.path.join(args.output_dir, "comparison.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Comparison saved -> {out_path}")


if __name__ == "__main__":
    main()
