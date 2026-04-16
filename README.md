# Pascal VOC 2007 Semantic Segmentation Benchmark

CS 261 mini-project — benchmarks 4 segmentation models (U-Net, DeepLabV3+, DINOv2, SAM2) on Pascal VOC 2007.

## Run on Colab

```python
# 1. Clone the repo
!git clone https://github.com/gracee-chen/261-mini2.git
cd 261-mini2

# 2. Install dependencies
!pip install -q segmentation-models-pytorch timm transformers scipy
!pip install -q git+https://github.com/facebookresearch/sam2.git

# 3. Download dataset
# Kaggle: zaraks/pascal-voc-2007

# 4. Train models
!python train/train_unet.py --voc-root ./voc_data --epochs 80 --image-size 512
!python train/train_deeplabv3plus.py --voc-root ./voc_data --epochs 80 --image-size 512
!python train/train_dinov2.py --voc-root ./voc_data --epochs 80
!python train/train_sam2.py --voc-root ./voc_data --sam2-ckpt ./sam2.1_hiera_large.pt --epochs 80

# 5. Evaluate
!python evaluation/metrics/compute_metrics.py --model-type unet deeplabv3plus dinov2 sam2 ...

# 6. Run ablation studies
!python evaluation/ablation/ablation_backbone.py --voc-root ./voc_data
```

## Project Structure

```
models/              Training, evaluation, and model code
  unet_seg.py        U-Net with ResNet-50 encoder
  deeplabv3plus_seg.py  DeepLabV3+ with ASPP (OS=8)
  dinov2_seg.py      DINOv2 ViT-B/14 + conv decoder
  sam2_seg.py        SAM2 Hiera-Large + FPN head
train/               Training scripts for each model
evaluation/          Metrics, confusion matrix, ablation studies, visualization
dataset/             VOC dataset wrapper and augmentation
inference/           Single-image inference scripts
```
