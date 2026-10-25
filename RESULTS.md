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

## Pending
- [ ] Ablation runs (baseline / +PSEM / +PSEM+SDWH) to quantify each module's contribution
- [ ] Option B: fine-tune COCO-pretrained YOLOv9c on TrashCan, then integrate TAFM
