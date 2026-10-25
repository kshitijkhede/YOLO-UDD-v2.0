# YOLO-UDD v2.0 — Results

## Run 1: Full model, trained FROM SCRATCH (no pretrained weights)

- **Dataset:** TrashCan 1.0 (instance_version), 3-class config {trash, animal, rov}
  - Train: 6,065 images / Val: 1,147 images
- **Model:** YOLO-UDD v2.0 full (custom backbone + PSEM neck + TAFM + SDWH head)
- **Training:** 100 epochs, AdamW lr=0.001 (cosine to 1e-4), batch 8, img 640, from scratch
- **Hardware:** Kaggle Tesla T4

### Final metrics (val set, best checkpoint)

| Metric            | Value  |
|-------------------|--------|
| mAP@50            | 0.4634 |
| mAP@50:95         | 0.2359 |
| mAP@75            | 0.2119 |
| Recall            | 0.862  |
| FPS               | 20.75  |
| Avg turbidity (TAFM) | 0.189 |

> Note: the `precision` printed by the first eval run (~0.03) was an artifact of
> evaluating at conf 0.05 (the low threshold needed for the mAP curve). The eval
> now reports precision/recall at conf 0.25; re-run eval for the corrected pair.

### Context vs. the reference paper

- Paper baseline (YOLOv9c, **COCO-pretrained**): 75.9% mAP@50
- This run (custom architecture, **from scratch**): 46.3% mAP@50

The gap is the expected cost of training from scratch on ~6k images with no
pretrained backbone. This run is the honest from-scratch baseline; the pretrained
path (Option B: Ultralytics YOLOv9c + TAFM) targets a paper-competitive number.

## Run 2: Option B — COCO-pretrained YOLOv9c, fine-tuned (Ultralytics)

- **Model:** YOLOv9c (COCO-pretrained), fine-tuned on TrashCan 3-class
- **Training:** 100 epochs, imgsz 640, batch 16, Ultralytics defaults
- **Hardware:** Kaggle Tesla T4

### Final metrics (val set, best checkpoint)

| Metric      | Value  |
|-------------|--------|
| **mAP@50**  | **0.7874** |
| mAP@50:95   | 0.5808 |
| Precision   | 0.8558 |
| Recall      | 0.7150 |

Per-class mAP@50: ROV 0.930, Trash 0.806, Animal 0.626.

### vs. the paper
- Paper baseline (YOLOv9c, COCO-pretrained): **0.759 mAP@50**
- This run: **0.7874 mAP@50 → beats the paper by +2.8 points.**

## Summary

| Model                                   | mAP@50 | mAP@50:95 |
|-----------------------------------------|--------|-----------|
| Custom YOLO-UDD (from scratch) + TAFM   | 0.463  | 0.236     |
| YOLOv9c (COCO-pretrained, Option B)     | 0.787  | 0.581     |
| Paper baseline (YOLOv9c pretrained)     | 0.759  | —         |

## Pending
- [ ] Phase 2: integrate TAFM into the pretrained YOLOv9c -> show "YOLOv9c + TAFM > YOLOv9c alone (0.787)"
- [ ] (Optional) Ablation runs on the custom model (baseline / +PSEM / +PSEM+SDWH)
