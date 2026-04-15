# -*- coding: utf-8 -*-
"""
SAM2 image encoder adapted for closed-vocabulary semantic segmentation.

Approach
--------
SAM2 (Ravi et al., Meta 2024) was designed as a prompt-based instance /
video segmenter.  We repurpose its Hiera-based image encoder as a frozen
(or fine-tunable) backbone and attach a lightweight semantic head on top of
the FPN feature maps.

FPN feature ordering in SAM2's backbone_fpn output (after scalp=1)
  backbone_fpn[0]  → finest resolution  (~H/4 × W/4,  256 ch)
  backbone_fpn[1]  → medium resolution  (~H/8 × W/8,  256 ch)
  backbone_fpn[2]  → coarsest remaining (~H/16 × W/16, 256 ch)

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
        self._logged = False   # one-shot diagnostic

        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

        # Fuse all available FPN levels (3 after scalp=1) for richer features
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(SAM2_FPN_CHANNELS, SAM2_FPN_CHANNELS, 1, bias=False),
                nn.BatchNorm2d(SAM2_FPN_CHANNELS),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        ])

        self.head = nn.Sequential(
            nn.Conv2d(SAM2_FPN_CHANNELS, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
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

        # One-shot diagnostic: print feature shapes and statistics
        if not self._logged:
            self._logged = True
            print(f"  [SAM2 diag] input={x.shape}, FPN levels={len(fpn)}")
            for i, f in enumerate(fpn):
                print(f"  [SAM2 diag] fpn[{i}]: shape={f.shape}, "
                      f"mean={f.float().mean():.4f}, std={f.float().std():.4f}")

        # Fuse all FPN levels to the finest resolution
        target_size = fpn[0].shape[-2:]   # finest level spatial size
        fused = torch.zeros(B, SAM2_FPN_CHANNELS, *target_size,
                            device=x.device, dtype=fpn[0].dtype)
        for i, lat_conv in enumerate(self.lateral_convs):
            if i < len(fpn):
                feat = lat_conv(fpn[i])
                if feat.shape[-2:] != target_size:
                    feat = F.interpolate(feat, size=target_size,
                                         mode="bilinear", align_corners=False)
                fused = fused + feat

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
