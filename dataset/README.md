# Dataset: Pascal VOC 2007 Segmentation

## 1. Download

Download the dataset from Kaggle:

```
https://www.kaggle.com/datasets/zaraks/pascal-voc-2007
```

Unzip so the layout looks like:

```
261-mini2/
└── VOCtrainval_06-Nov-2007/        ← place here (or anywhere, see §2)
    └── VOCdevkit/
        └── VOC2007/
            ├── JPEGImages/
            ├── SegmentationClass/
            └── ImageSets/
                └── Segmentation/
                    ├── train.txt
                    └── val.txt
```

## 2. Set the root path

All scripts read the root path from the `VOC_ROOT` environment variable
(default: `./VOCtrainval_06-Nov-2007`):

```bash
export VOC_ROOT=/path/to/VOCtrainval_06-Nov-2007
```

Or pass `--root` on the command line.

## 3. Explore the dataset

```bash
cd dataset
python explore.py --root $VOC_ROOT --num-samples 4
```

Add `--dist` to also print the pixel-level class distribution (slower).

## 4. Classes

| Index | Class        | Index | Class       |
|-------|--------------|-------|-------------|
| 0     | background   | 11    | diningtable |
| 1     | aeroplane    | 12    | dog         |
| 2     | bicycle      | 13    | horse       |
| 3     | bird         | 14    | motorbike   |
| 4     | boat         | 15    | person      |
| 5     | bottle       | 16    | pottedplant |
| 6     | bus          | 17    | sheep       |
| 7     | car          | 18    | sofa        |
| 8     | cat          | 19    | train       |
| 9     | chair        | 20    | tvmonitor   |
| 10    | cow          |       |             |

Pixel value **255** is the boundary/ignore label and is excluded from loss and metrics.

## 5. Splits

| Split | Role in this project |
|-------|----------------------|
| train | training             |
| val   | **test set** (per project instructions) |

The original `test` split (no ground-truth masks) is ignored.

## 6. Files

| File | Description |
|------|-------------|
| `voc_dataset.py` | Dataset class, transforms, DataLoader factory |
| `explore.py`     | Visualise samples, print class distribution  |
