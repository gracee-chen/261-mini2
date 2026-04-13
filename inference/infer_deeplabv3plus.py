#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run DeepLabV3+ inference on images and visualise / save segmentation masks.

Usage
-----
    cd /path/to/261-mini2

    # Single image
    python inference/infer_deeplabv3plus.py \
        --checkpoint checkpoints/deeplabv3plus/best.pth \
        --images path/to/image.jpg

    # VOC val split (first N)
    python inference/infer_deeplabv3plus.py \
        --checkpoint checkpoints/deeplabv3plus/best.pth \
        --voc-root ./VOCtrainval_06-Nov-2007 \
        --num-samples 8 \
        --output-dir results/deeplabv3plus
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Reuse shared helpers from U-Net inference module
from inference.infer_unet import (
    IMAGE_TRANSFORM, MEAN, STD,
    load_image, predict, visualise,
)
from dataset.voc_dataset import VOC_CLASSES, get_datasets
from models.deeplabv3plus_seg import build_deeplabv3plus


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DeepLabV3+ inference on Pascal VOC images")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--encoder",     default="resnet50")
    p.add_argument("--images",      nargs="*")
    p.add_argument("--voc-root",    default=None)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--output-dir",  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load model
    model = build_deeplabv3plus(encoder_name=args.encoder)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    samples = []

    if args.images:
        for path in args.images:
            tensor, pil = load_image(path)
            samples.append((tensor, pil, Path(path).stem))

    elif args.voc_root:
        import random
        _, val_ds = get_datasets(args.voc_root, image_size=256)
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
        visualise(pil, pred, title=f"DeepLabV3+ — {label}", save_path=save_path)


if __name__ == "__main__":
    main()
