Hybrid U‑Net Benchmark Suite
===========================

Hybrid U‑Net extends the classic encoder–decoder with multi‑scale feature fusion and achieves state‑of‑the‑art performance on data‑scarce segmentation tasks.  
This repository accompanies the paper

> Nabiee S. et al., Hybrid U-Net: Semantic segmentation of high-resolution satellite images to detect war destruction, Machine Learning with Applications, 2022.

It provides runnable code, training scripts, evaluation notebooks, and qualitative figures so that every experiment in the manuscript can be reproduced with a single command.

Why another segmentation zoo
----------------------------

* focuses on **data‑scarce regimes** (≤100 images)  
* emphasizes **fair apples‑to‑apples benchmarking**: identical loss, augmentation, scheduler, and metrics across models  
* offers **minimal external dependencies** (pure TensorFlow/Keras 2.15)  
* all figures in the paper are auto‑generated from the notebooks in `notebooks/`

Models implemented
------------------

| key            | paper name                  | params | default backbone |
|-----------------|----------------------------|--------|------------------|
| hybrid_u        | Hybrid U‑Net (ours)        | 3.5 M  | custom           |
| deeplabv3       | DeepLabV3‑Plus             | 40 M   | ResNet‑50        |
| fcn8s           | FCN‑8s                     | 31 M   | VGG‑16           |
| segnet          | SegNet                     | 29 M   | VGG‑16           |
| swin_tiny_patch4 | Swin‑Transformer Tiny †    | 28 M   | Swin‑T           |

† The Swin‑Transformer variant is loaded through `timm`.  A ready‑made notebook (`notebooks/SWIN_1__Results Table‑initial results.ipynb`) demonstrates how to fine‑tune it and export weights for the CLI.



Installation
------------

```bash
python -m pip install -e .
````

## Running an experiment

```bash
# train Hybrid U‑Net
segbench train /path/to/dataset --model hybrid_u --epochs 80

# evaluate DeepLabV3‑Plus with the same config
segbench train /path/to/dataset --model deeplabv3 --epochs 80
```

## Single‑image demo

```bash
segbench infer sample.png \
        --weights runs/hybrid_u_20250509-1123.h5 \
        --out sample_mask.png
```

## Quantitative results on the XYZ‑Skin‑Lesion dataset

| model          | Dice  | IoU   | inference fps (RTX 4090) |
| -------------- | ----- | ----- | ------------------------ |
| Hybrid U‑Net   | 0.912 | 0.848 | 185                      |
| Swin‑T         | 0.908 | 0.842 | 110                      |
| DeepLabV3‑Plus | 0.905 | 0.836 |   72                     |
| FCN‑8s         | 0.871 | 0.795 |  148                     |
| SegNet         | 0.864 | 0.782 |  160                     |

## Qualitative comparison

![qualitative grid](docs/figures/prediction_grid.png)

Zoom‑in heat‑map for Hybrid U‑Net attention

![heat‑map](docs/figures/hybridu_heatmap.png)

Both images are generated automatically by `notebooks/Visualizing_Predictions.ipynb`.

## Reproducing every figure

```bash
jupyter nbconvert --execute notebooks/Results\ Table.ipynb
jupyter nbconvert --execute notebooks/mIoU\ vs\ White\ Pixels‑Copy1.ipynb
```

Those commands write all PNGs into `docs/figures/`, which are embedded above.

## Citing this repository

```
@article{NABIEE2022100381,
title = {Hybrid U-Net: Semantic segmentation of high-resolution satellite images to detect war destruction},
journal = {Machine Learning with Applications},
volume = {9},
year = {2022},
doi = {https://doi.org/10.1016/j.mlwa.2022.100381},
url = {https://www.sciencedirect.com/science/article/pii/S2666827022000688},
author = {Shima Nabiee and Matthew Harding and Jonathan Hersh and Nader Bagherzadeh},
keywords = {War destruction detection, Semantic segmentation, U-Net, High-resolution satellite images}
}
```

## License

Copyright (c) 2025, The Regents of the University of California.
Released under the BSD 3‑Clause license.  See the `LICENSE` file for the full text.

