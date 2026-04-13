# -*- coding: utf-8 -*-
"""
Segmentation losses for Pascal VOC 2007.

IGNORE_INDEX = 255  (VOC boundary pixels — excluded from loss and metrics)

SegmentationLoss combines:
  - CrossEntropyLoss  (pixel-wise classification; fast, stable)
  - DiceLoss          (overlap-based; mitigates class imbalance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# Dice loss
# --------------------------------------------------------------------------- #

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Pixels with label == IGNORE_INDEX are masked out before computing Dice,
    so boundary pixels never contribute to the gradient.

    Parameters
    ----------
    num_classes : number of classes (21 for VOC)
    smooth      : Laplace smoothing to avoid division by zero
    """

    def __init__(self, num_classes: int = 21, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)  raw (un-softmaxed) scores
        targets : (B, H, W)     integer class labels; 255 = ignore
        """
        probs = F.softmax(logits, dim=1)   # (B, C, H, W)
        B, C, H, W = probs.shape

        # Build validity mask and clamp ignored pixels to class 0
        valid         = (targets != IGNORE_INDEX)                       # (B, H, W)
        targets_safe  = targets.clone()
        targets_safe[~valid] = 0

        # One-hot encode; shape → (B, C, H, W)
        one_hot = (
            F.one_hot(targets_safe, num_classes=C)                      # (B, H, W, C)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Zero out ignored pixels in both probability and target tensors
        valid_4d = valid.unsqueeze(1).expand_as(probs)
        probs    = probs   * valid_4d
        one_hot  = one_hot * valid_4d

        # Per-class Dice, averaged across classes
        dims         = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality  = probs.sum(dims) + one_hot.sum(dims)
        dice         = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


# --------------------------------------------------------------------------- #
# Combined loss
# --------------------------------------------------------------------------- #

class SegmentationLoss(nn.Module):
    """
    Weighted sum of CrossEntropy and Dice losses.

    Default weights (0.7 CE + 0.3 Dice) balance fast convergence from CE
    with the class-imbalance mitigation of Dice.

    Parameters
    ----------
    num_classes : 21 for Pascal VOC
    ce_weight   : weight for CrossEntropyLoss term
    dice_weight : weight for DiceLoss term
    """

    def __init__(
        self,
        num_classes: int = 21,
        ce_weight: float = 0.7,
        dice_weight: float = 0.3,
    ):
        super().__init__()
        self.ce      = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.dice    = DiceLoss(num_classes=num_classes)
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)
        targets : (B, H, W) long, 255 = ignore
        """
        return self.ce_w * self.ce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)
