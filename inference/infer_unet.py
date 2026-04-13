#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run U-Net inference on images and visualise / save segmentation masks.

Usage
-----
    cd /path/to/261-mini2

    # Single image
    python inference/infer_unet.py \
        --checkpoint checkpoints/unet/best.pth \
        --images path/to/image.jpg

    # Multiple images → save to output-dir
    python inference/infer_unet.py \
        --checkpoint checkpoints/unet/best.pth \
        --images img1.jpg img2.jpg \
        --output-dir results/unet

    # Run on the VOC val split (first N images)
    python inference/infer_unet.py \
        --checkpoint checkpoints/unet/best.pth \
        --voc-root ./VOCtrainval_06-Nov-2007 \
        --num-samples 8 \
        --output-dir results/unet
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

from dataset.voc_dataset import VOC_CLASSES, get_datasets
from models.unet_seg import build_unet


# --------------------------------------------------------------------------- #
# Shared helpers (used by all inference scripts)
# --------------------------------------------------------------------------- #

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_image(path: str) -> tuple:
    """Return (tensor (1,3,H,W), original PIL image)."""
    img = Image.open(path).convert("RGB")
    return IMAGE_TRANSFORM(img).unsqueeze(0), img


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return class-index map (H, W) as a numpy array."""
    import torch.nn.functional as F
    model.eval()
    x      = image_tensor.to(device)
    logits = model(x)
    if hasattr(logits, "logits"):
        logits = logits.logits
    if logits.shape[-2:] != x.shape[-2:]:
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)
    return pred


def visualise(image_pil: Image.Image, pred_mask: np.ndarray, title: str = "",
              save_path: str = None):
    """Side-by-side plot of original image and predicted segmentation mask."""
    # Denormalise: resize PIL to match prediction size
    img_np = np.array(image_pil.resize(pred_mask.shape[::-1]))   # (H, W, 3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if title:
        fig.suptitle(title, fontsize=11)

    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    seg_map = axes[1].imshow(pred_mask, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    cbar = plt.colorbar(seg_map, ax=axes[1], ticks=range(21))
    cbar.ax.set_yticklabels([f"{i}: {VOC_CLASSES[i]}" for i in range(21)])
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {save_path}")
        plt.close()
    else:
        plt.show()


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="U-Net inference on Pascal VOC images")
    p.add_argument("--checkpoint",  required=True,   help="Path to best.pth / last.pth")
    p.add_argument("--encoder",     default="resnet50")
    p.add_argument("--images",      nargs="*",        help="Image file path(s)")
    p.add_argument("--voc-root",    default=None,     help="Use VOC val split instead")
    p.add_argument("--num-samples", type=int, default=4,
                   help="Number of VOC val samples to visualise (with --voc-root)")
    p.add_argument("--output-dir",  default=None,
                   help="Save visualisations here (if omitted, plt.show() is used)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load model
    model = build_unet(encoder_name=args.encoder)
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.4f})")

    samples = []  # list of (image_tensor, PIL_image, label)

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
        visualise(pil, pred, title=f"U-Net — {label}", save_path=save_path)


if __name__ == "__main__":
    main()
