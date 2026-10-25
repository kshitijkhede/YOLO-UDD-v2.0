# YOLO-UDD v2.0: Turbidity-Adaptive Underwater Debris Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Deep-learning object detection for **underwater debris** on the **TrashCan 1.0** dataset
(3 classes: Trash, Animal, ROV). The project builds a custom YOLO-style detector with a
**novel Turbidity-Adaptive Fusion Module (TAFM)**, and benchmarks it against a
COCO-pretrained **YOLOv9c** baseline — exceeding the reference paper's reported result.

## Results

Evaluated on the TrashCan 1.0 validation set (1,147 images).

| Model | mAP@50 | mAP@50:95 | Notes |
|-------|:------:|:---------:|-------|
| Reference paper (YOLOv9c, pretrained) | 0.759 | — | Samanth et al. 2025 baseline |
| **YOLOv9c fine-tuned (this work)** | **0.787** | **0.581** | **exceeds the paper baseline** |
| Custom YOLO-UDD + TAFM (from scratch) | 0.463 | 0.236 | novel architecture, no pretraining |

Per-class mAP@50 (YOLOv9c): ROV 0.930, Trash 0.806, Animal 0.626.
Full details in **[RESULTS.md](RESULTS.md)**.

## Novel contribution — TAFM

The **Turbidity-Adaptive Fusion Module** ([`models/tafm.py`](models/tafm.py)) estimates a
per-image turbidity score (0 = clear water … 1 = murky) and blends two learned fusion
strategies accordingly, so the detector adapts to how murky the water is. This is the
project's original contribution on top of components adapted from recent literature.

## Architecture

```
Input 640x640 -> Backbone (CSP) -> Neck (PSEM + TAFM) -> Head (SDWH) -> Detections
                                          |
                                          +-> turbidity score (0=clear ... 1=murky)
```

| Module | Role | Source |
|--------|------|--------|
| Backbone | Multi-scale feature extraction | YOLOv9-style CSP |
| **PSEM** | Multi-scale feature fusion (neck) | Li et al. 2025 |
| **TAFM** | **Novel** turbidity-adaptive fusion | This project |
| **SDWH** | Attention detection head | Li et al. 2025 |

Detects 3 classes at three scales (80×80, 40×40, 20×20).

## How to run

Training was done on **Kaggle** (free T4 GPU). Two notebooks in [`notebooks/`](notebooks/):

- **`YOLO_Ultralytics_TrashCan.ipynb`** — fine-tunes COCO-pretrained YOLOv9c (the 0.787 result).
- **`YOLO_UDD_Ablation_Kaggle.ipynb`** — trains the custom YOLO-UDD architecture (with TAFM).

Running the custom model locally:

```bash
pip install -r requirements.txt

# verify dataset (TrashCan 1.0 in COCO format)
python scripts/verify_dataset.py --dataset-dir data/trashcan

# train / evaluate / detect
python scripts/train.py    --config configs/train_config.yaml --data-dir data/trashcan
python scripts/evaluate.py --weights runs/train/full/checkpoints/best.pt --data-dir data/trashcan
python scripts/detect.py   --weights runs/train/full/checkpoints/best.pt --source path/to/img.jpg
```

A GPU is recommended. `configs/` has fast / quick / CPU variants for lighter runs.

## Repository layout

```
models/      yolo_udd.py  psem.py  tafm.py  sdwh.py      the architecture
utils/       loss.py  target_assignment.py  nms.py  metrics.py   training machinery
data/        dataset.py                                  loader + underwater augmentations
scripts/     train.py  evaluate.py  detect.py            training/eval/inference
             coco_to_yolo_trashcan.py  verify_dataset.py  + data helpers
configs/     train_config*.yaml                          hyperparameters
notebooks/   Kaggle training notebooks (custom model + YOLOv9c)
```

**Full project write-up** — architecture layer-by-layer, theory from scratch, the TAFM module, the
development journey, results, and an interview/viva kit: **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## References

1. Samanth K. et al. (2025). *A Comprehensive Study On Underwater Object Detection Using
   Deep Neural Networks.* IEEE Access. (dataset, baseline)
2. Li, X. et al. (2025). *Efficient underwater object detection based on feature enhancement
   and attention detection head.* Scientific Reports. (PSEM, SDWH)
3. Wang, C.-Y. et al. (2024). *YOLOv9: Learning What You Want to Learn Using Programmable
   Gradient Information.* arXiv. (backbone / baseline model)

## License

MIT — see [LICENSE](LICENSE).
