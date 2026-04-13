# -*- coding: utf-8 -*-
"""
DeepLabV3+ with ImageNet-pretrained ResNet-50 encoder.

Architecture
------------
  Encoder : ResNet-50 with dilated convolutions (output stride 16)
  ASPP    : Atrous Spatial Pyramid Pooling at rates [6, 12, 18]
  Decoder : low-level feature fusion (1×1 conv) + 3×3 conv + bilinear upsample
  Output  : (B, 21, H, W) raw logits

Reference: Chen et al., "Encoder-Decoder with Atrous Separable Convolution
for Semantic Image Segmentation", ECCV 2018.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn

NUM_CLASSES = 21


def build_deeplabv3plus(
    encoder_name: str = "resnet50",
    encoder_weights: str = "imagenet",
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    Return a DeepLabV3+ model ready for Pascal VOC semantic segmentation.

    Parameters
    ----------
    encoder_name    : backbone encoder (default resnet50)
    encoder_weights : 'imagenet' for pretrained weights, None for random init
    num_classes     : number of output classes (21 for VOC)
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
