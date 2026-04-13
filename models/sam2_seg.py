# -*- coding: utf-8 -*-
"""
SAM2 image encoder adapted for closed-vocabulary semantic segmentation.

Approach
--------
SAM2 (Ravi et al., Meta 2024) was designed as a prompt-based instance /
video segmenter.  We repurpose its Hiera-based image encoder as a frozen
(or fine-tunable) backbone and attach a lightweight semantic head on top of
the FPN feature maps.

FPN feature ordering in SAM2's backbone_fpn output
  backbone_fpn[0]  → finest resolution  (~H/4 × W/4,  256 ch)
  backbone_fpn[1]  → medium resolution  (~H/8 × W/8,  256 ch)
  backbone_fpn[-1] → coarsest           (~H/32 × W/32, 256 ch)  ← image embed

For semantic segmentation we fuse the two finest levels before prediction.

Setup
-----
1. Install SAM2:
       pip install git+https://github.com/facebookresearch/sam2.git
2. Download SAM2.1 Base+ checkpoint (~160 MB):
       wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

Reference: Ravi et al., "SAM 2: Segment Anything in Images and Videos",
arXiv 2408.00714, 2024.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES       = 21
SAM2_FPN_CHANNELS = 256   # all FPN levels output 256-channel feature maps


class SAM2SemanticSeg(nn.Module):
    """
    SAM2 image encoder + conv semantic segmentation head.

    Parameters
    ----------
    image_encoder  : sam2.modeling.backbones.image_encoder.ImageEncoder
    num_classes    : number of output classes (default 21 for VOC)
    freeze_encoder : freeze encoder weights and train only the head
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        num_classes: int = NUM_CLASSES,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

        # Fuse finest two FPN levels before the prediction head
        self.fuse = nn.Sequential(
            nn.Conv2d(SAM2_FPN_CHANNELS * 2, SAM2_FPN_CHANNELS, 1, bias=False),
            nn.BatchNorm2d(SAM2_FPN_CHANNELS),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(SAM2_FPN_CHANNELS, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def train(self, mode: bool = True):
        """Keep encoder in eval mode when frozen (preserves dropout/drop-path)."""
        super().train(mode)
        if self.freeze_encoder:
            self.image_encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            enc_out = self.image_encoder(x)

        fpn = enc_out["backbone_fpn"]
        # fpn[0] = finest resolution, fpn[-1] = coarsest (image embed)
        feat_fine   = fpn[0]   # (B, 256, ~H/4,  ~W/4)
        feat_medium = fpn[1]   # (B, 256, ~H/8,  ~W/8)

        # Upsample medium to match the finest feature map
        feat_medium = F.interpolate(
            feat_medium,
            size=feat_fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused  = self.fuse(torch.cat([feat_fine, feat_medium], dim=1))
        logits = self.head(fused)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits


def build_sam2_seg(
    checkpoint: str,
    config: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
    num_classes: int = NUM_CLASSES,
    freeze_encoder: bool = True,
    device: str = "cpu",
) -> SAM2SemanticSeg:
    """
    Load a SAM2 checkpoint, extract the image encoder, and wrap it for
    semantic segmentation.

    Parameters
    ----------
    checkpoint : path to sam2.1_hiera_base_plus.pt (or another SAM2 checkpoint)
    config     : SAM2 hydra config name (relative to the sam2 package root)
    freeze_encoder : freeze the SAM2 encoder, train only the semantic head
    device     : 'cuda', 'cpu', or 'mps'
    """
    try:
        from sam2.build_sam import build_sam2
    except ImportError as e:
        raise ImportError(
            "SAM2 is not installed.  Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        ) from e

    sam2 = build_sam2(config, checkpoint, device=device)
    return SAM2SemanticSeg(
        image_encoder=sam2.image_encoder,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    )
