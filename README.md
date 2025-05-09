Hybrid U‑Net Benchmark Suite
============================

Hybrid U‑Net augments the classical encoder–decoder with multi‑scale fusion. This repository provides fully reproducible code and command‑line tools.

Installation
------------

```bash
python -m pip install -e .
```

Training example
----------------

```bash
segbench train /data/seg --model hybrid_u --epochs 80
```

Inference demo
--------------

```bash
segbench infer sample.png --weights runs/hybrid_u_YYYYMMDD-HHMM.h5
```

Results on the X‑YZ dataset
---------------------------

| model     | dice | IoU |
|-----------|------|-----|
| Hybrid U  | 0.912|0.848|
| DeepLabV3 | 0.905|0.836|
| FCN‑8s    | 0.871|0.795|
| SegNet    | 0.864|0.782|

Citation
--------

```
@article{nabiee2025hybridu,
  title={Hybrid U‑Net: A Scale‑Aggregating Encoder–Decoder for Data‑scarce Segmentation},
  author={Shima Nabiee et al.},
  journal={Pattern Recognition Letters},
  year={2025}
}
```
