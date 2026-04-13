# -*- coding: utf-8 -*-
"""
DINOv2 (ViT-B/14) backbone + lightweight conv decoder for semantic segmentation.

Why DINOv2 (2024 SOTA approach)?
---------------------------------
DINOv2 (Oquab et al., TMLR 2024) is a self-supervised ViT trained on a
curated 142M-image dataset via DINO + iBOT objectives.  Its patch features
transfer extremely well to dense prediction tasks:
  - Linear probe on Pascal VOC 2012 : ~83 mIoU (frozen backbone)
  - With fine-tuned head            : comparable to supervised SOTA
The "large frozen backbone + lightweight task head" paradigm dominates
segmentation leaderboards in 2024-2025 (DINOv2, SAM, CLIP-based models).

Architecture
------------
  Backbone : facebook/dinov2-base  (ViT-B/14, 12 layers, 768-d hidden)
  Head     : Conv(768→256,1×1) → BN → ReLU
             Conv(256→256,3×3) → BN → ReLU
             Conv(256→21,1×1)
  Output   : (B, 21, H, W) raw logits  (bilinear upsampled from patch grid)

Input sizing
------------
DINOv2 uses 14×14 patches.  For arbitrary (H, W) we crop to the nearest
multiple of 14 before passing through the backbone, then bilinearly upsample
logits back to (H, W).  The recommended training size is 224×224 (= 16×14).

Reference: Oquab et al., "DINOv2: Learning Robust Visual Features without
Supervision", TMLR 2024.  arXiv 2304.07193.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

NUM_CLASSES = 21
PATCH_SIZE  = 14     # ViT-B/14 and ViT-L/14 both use 14-pixel patches


class DINOv2Seg(nn.Module):
    """
    DINOv2 backbone with a 3-layer conv decoder head.

    Parameters
    ----------
    backbone        : pre-loaded Dinov2Model
    num_classes     : VOC = 21 (background + 20 object classes)
    freeze_backbone : freeze backbone weights (train head only)
    hidden_dim      : intermediate channel width in the decoder
    """

    def __init__(
        self,
        backbone: Dinov2Model,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone        = backbone
        self.freeze_backbone = freeze_backbone
        self.patch_size      = PATCH_SIZE
        embed_dim            = backbone.config.hidden_size   # 768 for ViT-B

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )

    def train(self, mode: bool = True):
        """Keep backbone in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Crop to nearest multiple of patch_size (minimal distortion)
        ph = (H // self.patch_size) * self.patch_size
        pw = (W // self.patch_size) * self.patch_size
        x_in = x[:, :, :ph, :pw] if (ph != H or pw != W) else x

        ctx = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with ctx:
            out = self.backbone(x_in)

        # last_hidden_state: (B, 1 + h_p*w_p, embed_dim)  — index 0 is [CLS]
        h_p = ph // self.patch_size
        w_p = pw // self.patch_size
        patch_tokens = out.last_hidden_state[:, 1:, :]          # (B, h_p*w_p, D)
        patch_tokens = (
            patch_tokens
            .reshape(B, h_p, w_p, -1)
            .permute(0, 3, 1, 2)                                 # (B, D, h_p, w_p)
            .contiguous()
        )

        logits = self.decoder(patch_tokens)
        # Upsample back to original spatial size (H, W)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def build_dinov2_seg(
    model_name: str = "facebook/dinov2-base",
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = False,
    hidden_dim: int = 256,
) -> DINOv2Seg:
    """
    Download DINOv2 from HuggingFace and wrap with a segmentation head.

    Parameters
    ----------
    model_name      : HuggingFace model ID
                      'facebook/dinov2-base'  (ViT-B/14, 86M params)
                      'facebook/dinov2-large' (ViT-L/14, 307M params)
    freeze_backbone : start with frozen backbone (recommended for first
                      few epochs to stabilise the head)
    """
    backbone = Dinov2Model.from_pretrained(model_name)
    return DINOv2Seg(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        hidden_dim=hidden_dim,
    )
