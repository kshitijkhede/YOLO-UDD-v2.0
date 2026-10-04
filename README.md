# YOLO-UDD v2.0: A Turbidity-Adaptive Architecture for Underwater Debris Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep-learning object detector for **underwater debris detection** on the **TrashCan 1.0**
dataset. It extends a YOLO-style detector with three feature-processing modules — **PSEM**
(feature fusion), **SDWH** (attention detection head), and a **novel TAFM** (Turbidity-
Adaptive Fusion Module) that adapts the network to how murky the water is.

> Read **[`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)** — it explains every part
> and concept of this project end to end (architecture, papers, novelty, metrics, interview
> prep).
>
> Finishing the project? Follow **[`COMPLETION_PLAN.md`](COMPLETION_PLAN.md)** — the
> step-by-step roadmap from "architecture built" to "results + ablation."

---

## Architecture

```
Input 640x640 -> Backbone (YOLOv9-style CSP) -> Neck (PSEM + TAFM) -> Head (SDWH) -> Detections
                                                       |
                                                       +-> turbidity score (0=clear ... 1=murky)
```

| Module | Role | Source |
|--------|------|--------|
| Backbone | Multi-scale feature extraction | YOLOv9 concept |
| **PSEM** | Stronger multi-scale fusion (neck) | Li et al. 2025 |
| **TAFM** | **Novel** turbidity-adaptive fusion | This project |
| **SDWH** | Attention detection head | Li et al. 2025 |

Detects **3 classes**: Trash, Animal, ROV. Predicts at three scales (80x80, 40x40, 20x20).

## Status

Architecture OK, forward pass OK, loss/NMS OK, COCO metric wired into validation OK,
`evaluate.py` and `detect.py` fixed OK, 3-class setting unified OK (see `CHANGELOG.md`,
Phase 1). The pipeline is now trainable end-to-end. **Still to do:** run full training +
an ablation to produce real results. **No fully trained model yet.** See `COMPLETION_PLAN.md`.

## Quick start

```bash
# 1. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (if raw data is in Supervisely format)
python scripts/convert_supervisely_to_coco.py --input-dir <raw> --output-dir data/trashcan
python scripts/verify_dataset.py --dataset-dir data/trashcan

# 3. Smoke-test the model
python -c "import torch; from models.yolo_udd import build_yolo_udd; \
m=build_yolo_udd(num_classes=3).eval(); \
p,t=m(torch.randn(1,3,640,640)); print('OK turbidity=',float(t),'scales=',len(p))"

# 4. Train (read COMPLETION_PLAN.md first: fix class count + metric hookup)
python scripts/train.py --config configs/train_config.yaml --data-dir data/trashcan

# 5. Evaluate / detect
python scripts/evaluate.py --weights runs/train/checkpoints/best.pt --data-dir data/trashcan
python scripts/detect.py   --weights runs/train/checkpoints/best.pt --source path/to/img.jpg
```

GPU recommended. For quick iteration use `configs/train_config_quick.yaml` (20% subset) or
`configs/train_config_fast.yaml`. For GPU-free laptops use `configs/train_config_cpu.yaml`.
Kaggle/Colab notebooks are in `notebooks/`.

## Repository layout

```
models/      yolo_udd.py  psem.py  tafm.py  sdwh.py        the architecture
utils/       loss.py  target_assignment.py  nms.py  metrics.py   training machinery
data/        dataset.py                                    loader + underwater augmentations
scripts/     train.py  evaluate.py  detect.py  + convert/verify/subset helpers
configs/     train_config*.yaml                            hyperparameters (main/fast/quick/cpu)
notebooks/   Kaggle & Colab training notebooks
docs/guides/ Kaggle/Colab how-to guides & legacy notes
```

Full file-by-file map: see `PROJECT_DOCUMENTATION.md` section 19.

## References

1. Samanth K. et al. (2025). *A Comprehensive Study On Underwater Object Detection Using
   Deep Neural Networks.* IEEE Access. (dataset, baseline, training recipe)
2. Li, X. et al. (2025). *Efficient underwater object detection based on feature enhancement
   and attention detection head.* Scientific Reports. (PSEM & SDWH)
3. Wang, C.-Y. et al. (2024). *YOLOv9: Learning What You Want to Learn Using Programmable
   Gradient Information.* arXiv. (backbone concept)

See `PROJECT_DOCUMENTATION.md` sections 16-17 for exactly what was taken from each paper and
an honest novelty analysis.

## License

MIT - see `LICENSE`.
