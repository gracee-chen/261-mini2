#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run DINOv2 semantic segmentation inference on images.

Usage
-----
    cd /path/to/261-mini2

    python inference/infer_dinov2.py \
        --checkpoint checkpoints/dinov2/best.pth \
        --images     path/to/image.jpg

    # VOC val split
    python inference/infer_dinov2.py \
        --checkpoint checkpoints/dinov2/best.pth \
        --voc-root   ./VOCtrainval_06-Nov-2007 \
        --num-samples 8 \
        --output-dir  results/dinov2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from inference.infer_unet import MEAN, STD, predict, visualise
from dataset.voc_dataset import VOC_CLASSES, get_datasets
from models.dinov2_seg import build_dinov2_seg

# DINOv2 uses 224×224 (= 16 × 14 patches)
DINOV2_IMAGE_SIZE  = 224
DINOV2_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_image_dinov2(path: str) -> tuple:
    """Return (tensor (1,3,224,224), original PIL image)."""
    img = Image.open(path).convert("RGB")
    return DINOV2_IMAGE_TRANSFORM(img).unsqueeze(0), img


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 semantic seg inference")
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--dinov2-model",    default="facebook/dinov2-base")
    p.add_argument("--images",          nargs="*")
    p.add_argument("--voc-root",        default=None)
    p.add_argument("--num-samples",     type=int, default=4)
    p.add_argument("--output-dir",      default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Build model and load fine-tuned weights
    model = build_dinov2_seg(
        model_name=args.dinov2_model,
        num_classes=21,
        freeze_backbone=False,   # inference: no need to freeze
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    samples = []

    if args.images:
        for path in args.images:
            tensor, pil = load_image_dinov2(path)
            samples.append((tensor, pil, Path(path).stem))

    elif args.voc_root:
        import random
        _, val_ds = get_datasets(args.voc_root, image_size=DINOV2_IMAGE_SIZE)
        indices   = random.sample(range(len(val_ds)), min(args.num_samples, len(val_ds)))
        for idx in indices:
            img_t, _ = val_ds[idx]
            pil_img  = Image.fromarray(
                ((img_t.permute(1, 2, 0).numpy() *
                  np.array(STD) + np.array(MEAN)).clip(0, 1) * 255
                ).astype(np.uint8)
            )
            samples.append((img_t.unsqueeze(0), pil_img, f"val_{idx}"))

    else:
        print("Provide --images or --voc-root.")
        return

    for tensor, pil, label in samples:
        pred = predict(model, tensor, device)
        classes_present = [VOC_CLASSES[c] for c in np.unique(pred) if c < 21]
        print(f"{label}: {classes_present}")
        save_path = (
            os.path.join(args.output_dir, f"{label}_pred.png")
            if args.output_dir else None
        )
        visualise(pil, pred, title=f"DINOv2 — {label}", save_path=save_path)


if __name__ == "__main__":
    main()
