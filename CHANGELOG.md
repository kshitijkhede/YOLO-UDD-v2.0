# Changelog

## [Phase 1 — training/eval plumbing fixes]

Made the repo trainable end-to-end by closing the gaps documented in
`COMPLETION_PLAN.md` Phase 1. (Code implemented and statically verified;
still to be confirmed by a full GPU training run.)

### Fixed
- **Class-count inconsistency (3 vs 22).**
  - `configs/train_config.yaml`, `_fast.yaml`, `_quick.yaml`: `num_classes` set to
    `3`; `class_names` set to `["trash", "animal", "rov"]`.
  - `data/dataset.py`: added `_name_to_class` + `_build_superclass_map`, which read the
    COCO `categories` block and collapse TrashCan's fine-grained categories
    (trash_*, animal_*, rov, plant, …) into the 3 super-classes `{trash:0, animal:1,
    rov:2}`. Categories outside the 3-class config (e.g. `plant`) are skipped, and the
    class label is resolved *before* the box is added so box/label lists stay aligned.
  - `scripts/start_training.py`: preflight model now built with `num_classes=3`.
- **Validation metric not wired in.** `scripts/train.py` now calls
  `compute_metrics_coco(...)` (real COCO mAP@50 / mAP@50:95 / precision / recall)
  instead of the zero-returning `compute_metrics` stub. Best-checkpoint selection and
  mAP-based early stopping now operate on real numbers.
- **`scripts/evaluate.py` crashed on import.** Added the missing `measure_fps(...)`
  and `MetricsCalculator` to `utils/metrics.py` (and exported them from
  `utils/__init__.py`), which is what `evaluate.py` imports.
- **`scripts/detect.py` returned dummy detections.** Now runs `batched_nms(...)` on the
  model output and converts the kept normalized `(x,y,w,h)` boxes to `(x1,y1,x2,y2)` for
  drawing, so inference produces real boxes + the turbidity overlay.

### Still open (see COMPLETION_PLAN.md Phases 2–4)
- No full training run / results / ablation yet.
- Backbone is a YOLOv9-style CSP net, not true GELAN/PGI.
- PSEM/SDWH are reinterpretations of Li et al.; target assignment is simplified.


## [Phase 3 prep — ablation tooling]

Added on/off switches so the ablation study (baseline -> +PSEM -> +PSEM+SDWH -> full)
can be run without editing code, plus a ready-to-run Kaggle notebook.

### Added
- **Ablation switches** `use_psem`, `use_sdwh`, `use_tafm` threaded through
  `build_yolo_udd` -> `YOLOUD` -> `YOLOUDDNeck` (PSEM/TAFM) and `SDWH` (`use_attention`).
  When a module is off it is replaced by its plain equivalent (PSEM -> 3x3 conv; SDWH
  attention -> identity; TAFM -> passthrough with a zero turbidity score), so the ablation
  isolates exactly that module's contribution.
- **`--ablation {baseline,psem,psem_sdwh,full}`** in `scripts/train.py`. It overrides the
  config flags and writes each run to its own `runs/train/<ablation>/` folder. Defaults
  for the flags also added to `configs/train_config.yaml`.
- **Checkpoint-aware model rebuild** in `scripts/evaluate.py` and `scripts/detect.py`:
  they now read the ablation flags saved in the checkpoint's `config` and rebuild the
  matching architecture, so ablation checkpoints load without shape errors.
- **`notebooks/YOLO_UDD_Ablation_Kaggle.ipynb`** — a clean, Run-All Kaggle notebook:
  GPU check -> clone repo -> install deps -> set dataset path -> 3-epoch sanity check ->
  train all four ablations -> evaluate -> results table (`runs/ablation_results.csv`).
