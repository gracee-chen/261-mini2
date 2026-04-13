# -*- coding: utf-8 -*-
"""
Pascal VOC 2007 Segmentation Dataset wrapper.

Expected dataset directory layout (set VOC_ROOT env var or pass root= arg):
    <root>/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/
        JPEGImages/
        SegmentationClass/
        ImageSets/Segmentation/
            train.txt
            val.txt
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from PIL import Image

# --------------------------------------------------------------------------- #
# Class definitions
# --------------------------------------------------------------------------- #
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

NUM_CLASSES = len(VOC_CLASSES)          # 21
CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}

# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(image_size: int = 256):
    """Return (image_transform, target_transform) for a given image size."""
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),          # (1, H, W) uint8
    ])

    return image_transform, target_transform


# --------------------------------------------------------------------------- #
# Augmented dataset wrapper
# --------------------------------------------------------------------------- #
class AugmentedVOCDataset(torch.utils.data.Dataset):
    """
    Wrapper that applies synchronised geometric + colour augmentation to
    a VOCSegmentation dataset that already returns tensors.

    Geometric (applied to both image and mask):
      - Random horizontal flip  (p=0.5)
      - Random vertical flip    (p=0.5)
      - Random rotation ±15°    (p=0.5)

    Colour (applied to image only, after denorm → re-norm):
      - Random brightness/contrast/saturation jitter
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
            # --- Geometric (synchronised) ---
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle,
                                  interpolation=transforms.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask, angle,
                                  interpolation=transforms.InterpolationMode.NEAREST)

            # --- Colour (image only) ---
            # Denormalise → jitter → re-normalise
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean                     # back to [0, 1]
            image = image.clamp(0, 1)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = (image - mean) / std                   # re-normalise

        return image, mask


# --------------------------------------------------------------------------- #
# Target post-processing
# --------------------------------------------------------------------------- #
def mask_to_class_index(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a VOC segmentation mask (1, H, W) uint8 to a (H, W) long tensor
    where each pixel holds its class index 0-20.
    Pixels with value 255 (boundary / ignore) are mapped to 255 and should
    be ignored when computing loss / metrics.
    """
    mask = mask.squeeze(0).long()   # (H, W)
    # values 0-20 are valid classes; 255 is the ignore label
    mask[mask > 20] = 255
    return mask


# --------------------------------------------------------------------------- #
# Dataset factory
# --------------------------------------------------------------------------- #
def get_datasets(root: str, image_size: int = 256):
    """
    Return (train_dataset, val_dataset) for Pascal VOC 2007.

    Parameters
    ----------
    root : str
        Path that contains the VOCtrainval_06-Nov-2007/ folder.
    image_size : int
        Both images and masks are resized to (image_size x image_size).
    """
    image_tf, target_tf = get_transforms(image_size)

    train_dataset = VOCSegmentation(
        root=root,
        year="2007",
        image_set="train",
        download=False,
        transform=image_tf,
        target_transform=target_tf,
    )

    val_dataset = VOCSegmentation(
        root=root,
        year="2007",
        image_set="val",
        download=False,
        transform=image_tf,
        target_transform=target_tf,
    )

    return train_dataset, val_dataset


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #
def get_dataloaders(
    root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 2,
    augment: bool = False,
):
    """
    Return (train_loader, val_loader).

    Parameters
    ----------
    augment : if True, wrap the training set with AugmentedVOCDataset
              (flip, rotation, colour jitter). Val set is never augmented.
    """
    train_ds, val_ds = get_datasets(root, image_size)

    if augment:
        train_ds = AugmentedVOCDataset(train_ds, augment=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
