# -*- coding: utf-8 -*-
"""
U-Net with ImageNet-pretrained ResNet-50 encoder.

Architecture
------------
  Encoder : ResNet-50 (pretrained on ImageNet via smp)
  Decoder : symmetric U-Net decoder with skip connections
  Output  : (B, 21, H, W) raw logits

Reference: Ronneberger et al., "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI 2015.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

NUM_CLASSES = 21


def build_unet(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Return a U-Net model ready for Pascal VOC semantic segmentation.

    Parameters
    ----------
    encoder_name    : timm/torchvision encoder backbone (default resnet50)
    encoder_weights : 'imagenet' for pretrained weights, None for random init
    num_classes     : number of output classes (21 for VOC)
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,        # raw logits; CrossEntropyLoss handles softmax
    )
