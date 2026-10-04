# YOLO-UDD v2.0 — Complete Project Documentation

> **A Turbidity-Adaptive Architecture for High-Fidelity Underwater Debris Detection**
>
> This single document explains **every part and concept** of the project: the problem,
> the dataset, the full architecture (backbone, PSEM, TAFM, SDWH), the training pipeline,
> the loss, target assignment, NMS, evaluation metrics, the research papers it builds on,
> an honest novelty analysis, the current status with known gaps, and an interview-prep
> section. Read this top to bottom and you will understand the project end to end.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem: Why Underwater Detection Is Hard](#2-the-problem-why-underwater-detection-is-hard)
3. [The Dataset: TrashCan 1.0](#3-the-dataset-trashcan-10)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Concept Primer (read this before the modules)](#5-concept-primer)
6. [Module 1 — The Backbone](#6-module-1--the-backbone)
7. [Module 2 — PSEM (Partial Semantic Encoding Module)](#7-module-2--psem)
8. [Module 3 — TAFM (Turbidity-Adaptive Fusion Module) — the novel part](#8-module-3--tafm--the-novel-part)
9. [Module 4 — SDWH (Split Dimension Weighting Head)](#9-module-4--sdwh)
10. [The Data Pipeline](#10-the-data-pipeline)
11. [Target Assignment](#11-target-assignment)
12. [The Loss Function](#12-the-loss-function)
13. [NMS (Non-Maximum Suppression)](#13-nms)
14. [Evaluation Metrics](#14-evaluation-metrics)
15. [The Training Pipeline](#15-the-training-pipeline)
16. [Research Papers & Provenance](#16-research-papers--provenance)
17. [Novelty Analysis — honest answer](#17-novelty-analysis)
18. [Current Status & Known Gaps](#18-current-status--known-gaps)
19. [Repository Map (file by file)](#19-repository-map)
20. [Glossary](#20-glossary)
21. [Interview Preparation Q&A](#21-interview-preparation-qa)

---

## 1. Executive Summary

**Goal.** Detect and localise underwater debris (trash), marine animals, and ROVs
(Remotely Operated Vehicles) in underwater images, so that autonomous systems (AUVs/ROVs)
could eventually find and clean up marine litter.

**Approach.** Take a single-stage, real-time object detector (YOLO family) and improve it
for the underwater domain by adding three feature-processing modules:

| Module | Role | Source |
|--------|------|--------|
| **Backbone** | Extract multi-scale image features | YOLOv9 concept (Wang et al. 2024) |
| **PSEM** | Better multi-scale feature fusion in the neck | Li et al. 2025 |
| **TAFM** | **Novel** — adapt fusion to water turbidity | This project |
| **SDWH** | Attention detection head that suppresses background | Li et al. 2025 |

**Dataset.** TrashCan 1.0, in the 3-class configuration (Trash, Animal, ROV).

**Target.** Beat the YOLOv9c baseline of ~75.9% mAP@50:95 and reach >82% mAP, with TAFM
providing the key extra gain.

**Status (honest).** The architecture is implemented and the forward pass works. The
training loss and NMS work. But validation metrics, the evaluation script, and the
inference decoder are incomplete, and there is a 3-vs-22 class-count inconsistency to
resolve. No fully trained model or final results exist yet. See
[Section 18](#18-current-status--known-gaps) and the companion `COMPLETION_PLAN.md`.

---

## 2. The Problem: Why Underwater Detection Is Hard

Ocean plastic is a large and growing problem (millions of tons/year). Manual cleanup by
divers is slow, expensive, and limited in reach. Automating detection with cameras on
AUVs/ROVs is the promising path — but underwater imaging degrades the picture in ways
that break ordinary detectors:

- **Light absorption & color cast.** Water absorbs red light first, so images turn
  blue/green with depth. Object colors are distorted.
- **Scattering & turbidity.** Suspended particles ("marine snow") scatter light, causing
  haze, low contrast, and blur. Turbidity varies a lot between scenes.
- **Low light.** Deep/murky water is dark; sensor noise rises.
- **Camouflage.** Debris and animals blend into rocks, sand, and coral.
- **Scale variation.** Objects range from tiny fragments to large wreckage.

A detector that works on clean "land" images (COCO) underperforms here. The project's
thesis is that **explicitly handling turbidity** (via TAFM) plus **stronger feature fusion
and attention** (PSEM + SDWH) recovers the lost accuracy.

---

## 3. The Dataset: TrashCan 1.0

- **Origin:** University of Minnesota; derived from J-EDI (JAMSTEC) underwater video.
- **Size:** ~7,212 annotated underwater images.
- **Annotation format on disk:** COCO-style JSON (`images`, `annotations`, `categories`),
  with bounding boxes as `[x, y, width, height]` in absolute pixels. The original TrashCan
  also ships instance segmentation masks, but this project uses only boxes.
- **Two label configurations** (from the baseline paper):
  - **3-class:** `Trash`, `Animal`, `ROV` — what this project targets.
  - **4-class:** adds `Plant`.
  - **22-class (full "instance" version):** fine-grained (e.g. `trash_bottle`,
    `animal_fish`, `trash_net`, …). The repo's `train_config.yaml` currently lists these
    22 names — this is the source of the class-count inconsistency (see Section 18).
- **Split used:** 70% train / 15% val / 15% test.

The companion script `scripts/convert_supervisely_to_coco.py` exists because some versions
of the raw data come in **Supervisely** format (per-image JSON with polygons/rectangles);
it converts polygons/rectangles to COCO bounding boxes.

---

## 4. High-Level Architecture

```
          ┌─────────────────────────────────────────────────────────────┐
Input     │                                                             │
640×640 ──┤  BACKBONE  ──►  NECK (PSEM-PANet + TAFM)  ──►  HEAD (SDWH)  ──┤──► Detections
 RGB      │  (CSP/YOLOv9)   multi-scale fusion +           attention +   │   (boxes, obj,
          │                 turbidity adaptation           3 det heads   │    class) ×3 scales
          └─────────────────────────────────────────────────────────────┘
                                   │
                                   └──► turbidity score (0=clear … 1=murky)
```

**Three scales.** Like all YOLOs, it predicts at three resolutions so it can catch small,
medium, and large objects:

| Scale | Feature map | Detects |
|-------|-------------|---------|
| P3 | 80×80 | small objects |
| P4 | 40×40 | medium objects |
| P5 | 20×20 | large objects |

**Outputs.** For each scale the head produces three tensors:
- **bbox** `[B, 4, H, W]` — box (x, y, w, h) per grid cell
- **obj** `[B, 1, H, W]` — objectness (is there an object here?)
- **cls** `[B, num_classes, H, W]` — class scores

Plus one **turbidity score** per image from TAFM.

---

## 5. Concept Primer

Read these once; every module below reuses them.

- **Convolution (Conv).** A learnable filter sliding over the image/feature map to detect
  patterns (edges → textures → shapes → semantics as you go deeper).
- **Backbone / Neck / Head.** Standard detector anatomy. Backbone extracts features; neck
  fuses features across scales; head turns features into predictions.
- **Feature map.** A `[channels, height, width]` tensor. Early layers = high-res, low
  semantics; deep layers = low-res, high semantics.
- **Multi-scale / Feature Pyramid.** Using several resolutions so both tiny and huge
  objects are detectable. PANet = Path Aggregation Network, a top-down + bottom-up pyramid.
- **Residual connection.** `output = F(x) + x`. Lets gradients flow and preserves info.
- **Attention.** Learnable weighting that emphasizes important features and suppresses
  the rest. Three flavors used here:
  - **Channel attention** — "which feature *channels* matter?" (e.g. Squeeze-and-Excitation).
  - **Spatial attention** — "which *locations* matter?"
  - **Self-attention (Q/K/V)** — each location attends to all others.
- **Anchor-free detection.** Predict boxes directly from grid cells rather than from
  predefined anchor boxes.
- **IoU (Intersection over Union).** Overlap between predicted and true box, 0–1. The core
  measure for both the loss and the metrics.
- **SiLU / ReLU / Sigmoid.** Activation functions (nonlinearities). Sigmoid squashes to
  (0,1), used for scores and gates.
- **BatchNorm (BN).** Normalizes activations for stable, faster training.

---

## 6. Module 1 — The Backbone

**File:** `models/yolo_udd.py` → `YOLOUDDBackbone`, plus helpers `ConvModule`, `CSPBlock`.

**What it does.** Turns the `640×640×3` image into a pyramid of feature maps.

**Building blocks:**
- `ConvModule` = `Conv2d → BatchNorm → SiLU`. The basic unit.
- `CSPBlock` (Cross-Stage-Partial): splits channels into two paths — one passed straight
  through, one through several conv blocks — then concatenates. CSP improves gradient flow
  and cuts computation (idea from CSPNet, used throughout modern YOLO).

**Flow:**
```
stem:     640→320→160           (two stride-2 ConvModules)
stage1 → downsample1:  P3 = 80×80, 128 ch
stage2 → downsample2:  P4 = 40×40, 256 ch
stage3 → downsample3:  P5 = 20×20, 512 ch
stage4:                P6 = 20×20, 1024 ch
returns [P3, P4, P5, P6]
```

**Honest caveat (important for the viva).** The README calls this "YOLOv9c (GELAN)", but
the code is a **generic hand-built CSP stack**. It is *inspired by* YOLOv9 but does **not**
implement GELAN or PGI (the two actual innovations of the YOLOv9 paper — see Section 16).
If asked "is this really YOLOv9?", the correct answer is: *"It's a YOLOv9-style CSP
backbone; we did not reimplement GELAN/PGI."* See the completion plan for the option to
swap in a true YOLOv9 backbone via Ultralytics.

---

## 7. Module 2 — PSEM

**File:** `models/psem.py`. **Role:** sits in the **neck**, replacing ordinary convs after
feature concatenation, to produce richer fused features. Borrowed from **Li et al. 2025**.

**The concept (what it's supposed to achieve).** When you fuse features from two scales,
a plain conv is a weak mixer. PSEM is a stronger mixer that (a) processes channels in two
parallel branches, (b) re-weights with attention, and (c) keeps a residual so nothing is
lost.

**As implemented in this repo:**
1. **Split** input channels into two halves.
2. **Branch 1:** two 3×3 convs (standard receptive field) + residual.
3. **Branch 2:** two **dilated** 3×3 convs (dilation=2 → wider receptive field, good for
   context) + residual.
4. **Concatenate** the branches.
5. **Channel attention** (SE-style): "which channels matter?"
6. **Spatial attention** (max-pool + avg-pool over channels → 7×7 conv → sigmoid): "which
   locations matter?" — this is the CBAM pattern.
7. **Fusion conv** (1×1) to the target channel count.
8. **Outer residual:** `output = fused + input` (with a 1×1 projection if channel counts
   differ).

**Fidelity note.** The *paper's* PSEM uses a three-CBS branch + a 1×1-conv branch +
**FasterNet partial convolution (PConv)** for a lightweight finish. This repo's PSEM is a
**reinterpretation** (dilated branch + CBAM attention). There is even a `PartialConv`
class defined, but it is just a plain `Conv+BN` and PSEM does not use it. For the thesis,
describe this module as *"a PSEM-inspired dual-branch fusion block"* rather than claiming
an exact reimplementation.

---

## 8. Module 3 — TAFM — the novel part

**File:** `models/tafm.py` → `TAFM`, `MultiScaleTAFM`. **This is the project's own
contribution** and the thing that makes it more than a re-implementation.

**The idea.** Clear water and murky water need different features. In clear water, fine
**color/texture** detail is reliable. In murky water, color is washed out, so **shape and
strong edges** are more trustworthy. TAFM *measures* how murky the water is and *shifts the
feature weighting accordingly* — automatically, per image.

**How it works (step by step):**
1. **Turbidity estimator** — a tiny CNN (two stride-2 convs → global average pool → 1×1
   conv → **Sigmoid**) looks at the (downsampled) input image and outputs a single
   **turbidity score `T ∈ [0, 1]`** (0 = clear, 1 = murky).
2. **Two learned strategies** — `β` (clear-water weights, favor color/texture) and `α`
   (murky-water weights, favor shape/edges), both learnable per-channel parameters.
3. **Adaptive blend** —
   `w_adapt = σ( T·α + (1−T)·β )`.
   When `T→1`, α dominates; when `T→0`, β dominates.
4. **Channel attention** refines semantics further.
5. **Apply + residual** — `out = features · w_adapt · channel_attention + features`.
6. `MultiScaleTAFM` runs one TAFM per pyramid level and returns the **average turbidity
   score** (which is also surfaced as a model output for interpretability/monitoring).

**Why this is interesting.** The turbidity score is a *soft, differentiable control signal*
learned end-to-end from the detection loss alone — there is no turbidity label. The network
discovers what "murky" means by whatever helps it detect better. That gives you a free,
interpretable readout (how murky did the model think each image was?) alongside detection.

**Honesty for the viva (read Section 17 too).** The *mechanism* — a learned turbidity gate
blending two fusion strategies — is this project's design and I'm not aware of this exact
module elsewhere. But "turbidity/condition-aware" and "adaptive fusion" are **established
themes** in underwater vision. So the defensible claim is *"a novel module / novel
combination,"* not *"a brand-new concept no one has explored."* Do a focused literature
check before writing "first ever" anywhere.

---

## 9. Module 4 — SDWH

**File:** `models/sdwh.py`. **Role:** the **detection head** with built-in attention, so
it focuses on foreground (debris/animals) and suppresses background noise. Borrowed from
**Li et al. 2025** ("Split Dimension Weighting Head").

**The idea.** Before predicting boxes, weight the features along three "dimensions" in
sequence so the head concentrates on what matters:

1. **Level-wise attention** — weight the three pyramid scales against each other
   (implemented as learnable softmax weights per level). "Which *scale* is most useful
   for this image?"
2. **Spatial-wise attention** — a self-attention (Q/K/V) block so each location attends to
   all others. "Which *locations* hold objects?"
3. **Channel-wise attention** — Squeeze-and-Excitation (avg+max pool → shared MLP →
   sigmoid). "Which *semantic channels* matter?"

**Then three detection heads per scale:**
- **bbox head** → 4 numbers (x, y, w, h)
- **obj head** → 1 number (objectness, sigmoid)
- **cls head** → `num_classes` scores

**Fidelity note.** The paper's SDWH uses specific operators (HSigmoid + avg-pool gating for
level-wise; stacked 3×3 convs for spatial-wise; FC→ReLU→FC "task-wise" for channel). This
repo uses softmax level weights, full self-attention, and SE — a **looser reinterpretation**
that keeps the same three-stage spirit. Also note the file contains an `SDWHLoss` class
whose `forward()` returns zeros — a dead placeholder; the real loss is in `utils/loss.py`.

---

## 10. The Data Pipeline

**File:** `data/dataset.py` → `TrashCanDataset`, `collate_fn`, `create_dataloaders`.

- **Loads** COCO-format annotations; for each image gathers its boxes and labels.
- **Converts** COCO `[x, y, w, h]` (absolute pixels) → YOLO `[x_center, y_center, w, h]`
  (normalized 0–1), then **clips** everything to [0,1] and drops degenerate boxes. *(This
  clipping is the key fix the merged repo keeps from the `main` branch — without it,
  Albumentations throws "bbox out of range" errors.)*
- **Augments** (train only) with **underwater-specific** Albumentations:
  - geometric: resize→flip→shift/scale/rotate
  - **color jitter** (simulate depth color cast)
  - **blur / motion blur / median blur** (simulate turbidity)
  - brightness/contrast (lighting), Gaussian/ISO noise (sensor noise)
  - RGB shift, occasional channel shuffle
  - then ImageNet `Normalize` + `ToTensorV2`
- **Collate:** images stack into a batch tensor; boxes/labels stay as *lists* of
  variable-length tensors (each image has a different number of objects).
- **`create_dataloaders`** builds train/val/test `DataLoader`s.

---

## 11. Target Assignment

**File:** `utils/target_assignment.py` → `assign_targets_simple`, `build_targets`.

**The problem it solves.** The model outputs a prediction at *every* grid cell of every
scale (e.g. 80×80 + 40×40 + 20×20 cells). Training needs to know, for each cell, *what the
right answer is*: is this cell responsible for an object, and if so which box/class? That
mapping from ground-truth boxes to grid cells is **target assignment**.

**How this repo does it (simplified, anchor-free):**
- Normalize GT boxes to [0,1].
- For each GT box, find the grid cell containing its **center** → mark that cell
  **positive**, and write the box/class/objectness targets there.
- Also mark **neighboring cells** within ~15% distance as positive (with reduced
  objectness), for denser supervision.

**Honest caveat.** This is far cruder than production YOLO assigners (SimOTA, Task-Aligned
Assignment, multi-scale IoU matching). It works, but likely **caps achievable accuracy**.
Improving this is a high-value item in the completion plan.

---

## 12. The Loss Function

**File:** `utils/loss.py` → `EIoULoss`, `YOLOUDDLoss`.

A detector's loss has three parts, combined with weights:

```
total = λ_box · bbox_loss + λ_obj · obj_loss + λ_cls · cls_loss
        (λ_box=5.0,          λ_obj=1.0,         λ_cls=1.0)
```

1. **Box loss — EIoU (Efficient IoU).** Beyond plain IoU, EIoU adds (a) center-distance
   penalty and (b) width/height difference penalties, so boxes converge faster and tighter.
   Applied **only to positive cells**.
2. **Objectness loss — BCE** (binary cross-entropy) over **all** cells, with predictions
   clamped and NaN/Inf-guarded, and positive cells weighted higher than negatives (2.0 vs
   0.5) to counter the huge background imbalance.
3. **Classification loss — BCE-with-logits**, applied **only to positive cells**.

**Note.** This loss is *real and working* — contrary to the old README's "placeholder"
label. It depends on `build_targets` from the assignment module.

---

## 13. NMS

**File:** `utils/nms.py` → `nms`, `batched_nms`, `box_iou`.

**Why.** The model predicts many overlapping boxes for the same object. **Non-Maximum
Suppression** keeps the highest-confidence box and removes others that overlap it beyond an
IoU threshold.

**How `batched_nms` works:**
1. Gather predictions from all three scales.
2. Confidence = `sigmoid(objectness) × sigmoid(class score)`.
3. Filter by a confidence threshold.
4. Run NMS **per class**.
5. Sort by score, cap at `max_det` (e.g. 300) detections per image.

This module is working and is what `detect.py`/`evaluate.py` *should* call to turn raw
predictions into final boxes.

---

## 14. Evaluation Metrics

**File:** `utils/metrics.py`.

- **`compute_metrics_coco(...)`** — a **correct COCO-style** evaluator: for each class and
  each IoU threshold from 0.50 to 0.95 (step 0.05), match detections to ground truth, build
  the precision–recall curve, compute Average Precision (AP), and average to get **mAP@50**,
  **mAP@75**, and **mAP@50:95**, plus overall precision/recall.
- **`compute_metrics(...)`** — a thin wrapper that currently **returns all zeros** (a stub).
  **This is the single most impactful bug:** `scripts/train.py` calls *this* during
  validation, so logged mAP is always 0, which also breaks "best checkpoint" selection and
  mAP-based early stopping.

**Key metrics explained:**
- **Precision** = of the boxes I predicted, how many were right.
- **Recall** = of the real objects, how many I found.
- **mAP@50** = mean AP at IoU≥0.50 (lenient).
- **mAP@50:95** = mean AP averaged over IoU 0.50→0.95 (strict; the headline number in the
  papers and the project target).

---

## 15. The Training Pipeline

**File:** `scripts/train.py` → `Trainer`.

- **Optimizer:** AdamW. **LR schedule:** Cosine Annealing. **Grad clipping:** max-norm 10.
- **Config:** read from `configs/train_config.yaml` (hyperparameters below).
- **Loop per epoch:** forward → `YOLOUDDLoss` → backward → step; log to **TensorBoard**
  (loss components + turbidity); then **validate** (runs NMS, *should* compute mAP).
- **Checkpointing:** saves `latest.pt` every epoch and `best.pt` on mAP improvement;
  supports `--resume`.
- **Early stopping:** after `patience` epochs with no mAP gain.

**Default hyperparameters (`train_config.yaml`):**

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| LR (initial → min) | 0.01 → 0.0001 |
| Scheduler | Cosine Annealing |
| Epochs | 100 |
| Batch size | 16 |
| Image size | 640×640 |
| Weight decay | 0.0005 |
| Early-stop patience | 20 |
| Loss weights | box 5.0, obj 1.0, cls 1.0 |

**Config variants:** `_fast` (50 epochs, batch 24), `_quick` (20% subset, 512px, 20 epochs),
`_cpu` (3-class, 320px, batch 1 — for laptops with no GPU).

**Known pipeline issues:** validate() calls the zero-returning `compute_metrics`;
`evaluate.py` imports functions that don't exist; `detect.py` returns dummy detections.
All three are in the completion plan.

---

## 16. Research Papers & Provenance

The project stands on three papers. Knowing exactly what each contributes — and what the
code did vs. didn't take — is essential for the viva.

### Paper 1 — Samanth K. et al., "A Comprehensive Study On Underwater Object Detection Using Deep Neural Networks" (IEEE Access, 2025)
- **Type:** comparison study, not a new architecture.
- **Gives the project:** the **TrashCan 1.0 dataset**, the **3-class setup**, the training
  recipe (AdamW, cosine, 640, 100 epochs), and the **baseline number 75.9% mAP@50:95**
  (their best YOLOv9c on the 3-class set).

### Paper 2 — Li, Xingkun et al., "Efficient underwater object detection based on feature enhancement and attention detection head" (Scientific Reports, 2025)
- **Type:** new modules for YOLO.
- **Gives the project:** the **PSEM** and **SDWH** ideas.
- **Caveats:** the paper's baselines are **YOLOv5n/v6n/v8n** on **UTDAC2020 and RUOD** (not
  TrashCan); its +2.8% mAP gain is **YOLOv8n on UTDAC2020**. The repo's PSEM/SDWH are
  *reinterpretations*, not exact reimplementations.

### Paper 3 — Wang, Chien-Yao et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information" (arXiv, 2024)
- **Type:** the YOLOv9 detector.
- **Two real innovations:** **PGI** (auxiliary reversible branch fighting the information
  bottleneck) and **GELAN** (generalized ELAN backbone).
- **Caveat:** the repo uses the *name* "YOLOv9c" and the baseline *number*, but implements
  **neither GELAN nor PGI** — the backbone is a plain CSP net.

### Provenance summary

| Project piece | Taken from | Faithful to source? |
|---------------|-----------|---------------------|
| Dataset, baseline, training recipe | Paper 1 | Yes |
| PSEM | Paper 2 | Reinterpreted (not exact) |
| SDWH | Paper 2 | Reinterpreted (not exact) |
| "YOLOv9c" backbone | Paper 3 | Name/number only; GELAN & PGI not implemented |
| **TAFM** | **This project** | **Original** |

---

## 17. Novelty Analysis

**Direct answer to "is this completely new, or just taken from one paper?"**

It is **neither a single-paper copy nor a from-scratch invention.** It is best described as:

> **A novel *combination* of existing components for the underwater domain, plus one new
> module (TAFM).**

Breaking it down:

- **Not from one paper.** It fuses three papers (dataset+baseline, PSEM+SDWH modules,
  YOLOv9 backbone concept) — so it is more than "take one paper and apply it."
- **Not completely new either.** The backbone, PSEM, and SDWH are all borrowed ideas; and
  turbidity-/condition-aware processing is an existing theme in underwater vision.
- **The genuinely new piece is TAFM** — a learned turbidity gate that blends two fusion
  strategies, trained end-to-end with no turbidity labels. That specific module is this
  project's own design.

**So, two defensible novelty claims:**
1. **Module-level novelty:** the TAFM mechanism itself.
2. **System-level novelty:** the *specific integration* of YOLOv9-style backbone + PSEM +
   SDWH + **TAFM**, applied to TrashCan, with turbidity adaptation as the differentiator.

**What would strengthen the novelty claim (and protect you in the viva):**
- Run a **literature check** for "turbidity-adaptive / water-quality-aware feature fusion"
  before claiming "first." If similar work exists, position TAFM as *different because…*
  (e.g. no turbidity labels, end-to-end, per-scale).
- **Ablation study:** train baseline → +PSEM → +PSEM+SDWH → +TAFM, and show TAFM adds a
  measurable gain. An ablation is the strongest possible evidence that your novel module
  actually matters — reviewers and interviewers trust numbers over claims.
- Be precise in wording: say *"we propose TAFM, a turbidity-adaptive fusion module, and
  integrate it with PSEM/SDWH in a YOLO-style detector"* — not *"we invented a new
  detector."*

**Bottom line for the viva:** This is a legitimate **M.Tech-level contribution** — a novel
module + a novel combination + domain application + (once done) an ablation proving the
module helps. It is honest to call it novel; it is *dishonest* to call it a brand-new
architecture or to claim the PSEM/SDWH/YOLOv9 parts as your own.

---

## 18. Current Status & Known Gaps

**Works today:** model builds and forward-passes; loss (EIoU + BCE) works; target
assignment works (crudely); NMS works; COCO metric function exists; data pipeline with
underwater augmentations works; training loop runs and checkpoints.

**Update (Phase 1 done):** gaps 1–4 below (metrics hookup, `evaluate.py`, `detect.py`, and the 3-vs-22 class count) have been fixed — see `CHANGELOG.md`. The pipeline is now trainable end-to-end; items 5–8 remain.

**Gaps to fix (in priority order — detailed in `COMPLETION_PLAN.md`):**

1. **Metrics not wired in.** `train.py` validation calls `compute_metrics` (returns zeros)
   instead of `compute_metrics_coco`. ⇒ mAP always 0, best-checkpoint & early-stop broken.
2. **`evaluate.py` crashes on import.** It imports `measure_fps` and `MetricsCalculator`
   from `utils/metrics.py`, which **don't exist**. Must implement them (or rewrite the
   script around `compute_metrics_coco`).
3. **`detect.py` returns dummy detections.** No real decoding; must wire in `batched_nms`
   and convert grid outputs to image-space boxes.
4. **Class-count inconsistency (3 vs 22).** README/`dataset.py` class_map/SDWH default = 3;
   `train_config.yaml` = 22. Pick one (recommend 3-class to match the baseline paper) and
   make everything agree.
5. **Backbone fidelity.** "YOLOv9c" is actually a plain CSP net — either rename it honestly
   or swap in a real YOLOv9 backbone (e.g. via Ultralytics).
6. **PSEM/SDWH fidelity.** Reinterpretations, not the paper's exact modules — document as
   such or align them.
7. **Target assignment is crude** — upgrade toward SimOTA/TAL for better accuracy.
8. **No trained model / results / ablation yet** — the actual research output.

---

## 19. Repository Map

```
YOLO-UDD-v2.0/
├── README.md                    # Short overview + quick start
├── PROJECT_DOCUMENTATION.md     # ← THIS FILE (full explanation)
├── COMPLETION_PLAN.md           # Step-by-step plan to finish
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT
├── QUICKSTART.md                # Quick start cheatsheet
│
├── models/                      # ── THE ARCHITECTURE ──
│   ├── yolo_udd.py              #   Full model: backbone + neck + head
│   ├── psem.py                 #   PSEM fusion module (neck)
│   ├── tafm.py                 #   TAFM turbidity module (NOVEL)
│   ├── sdwh.py                 #   SDWH attention detection head
│   └── __init__.py
│
├── utils/                       # ── TRAINING MACHINERY ──
│   ├── loss.py                 #   EIoU + composite loss (working)
│   ├── target_assignment.py    #   GT→grid-cell assignment (simplified)
│   ├── nms.py                  #   Non-max suppression (working)
│   ├── metrics.py              #   COCO mAP (coco fn works; wrapper is a stub)
│   └── __init__.py
│
├── data/
│   └── dataset.py              #   TrashCan loader + underwater augmentations
│
├── scripts/                     # ── ENTRY POINTS ──
│   ├── train.py                #   Training loop (Trainer)
│   ├── evaluate.py             #   Test-set eval (BROKEN import — see gaps)
│   ├── detect.py               #   Inference (dummy decode — see gaps)
│   ├── convert_supervisely_to_coco.py  # Supervisely → COCO
│   ├── verify_dataset.py       #   Dataset sanity check
│   ├── create_subset.py        #   Make a small subset for quick tests
│   ├── create_dummy_dataset.py #   Synthetic data for smoke tests
│   ├── start_training.py       #   Interactive pre-flight + launch
│   └── run_kaggle_training.py  #   Robust Kaggle launcher
│
├── configs/
│   ├── train_config.yaml       #   Main config (⚠ currently num_classes=22)
│   ├── train_config_fast.yaml  #   50 epochs / batch 24
│   ├── train_config_quick.yaml #   20% subset / 512px / 20 epochs
│   └── train_config_cpu.yaml   #   3-class / 320px / batch 1 (no GPU)
│
├── notebooks/
│   ├── YOLO_UDD_Kaggle_Training.ipynb
│   └── YOLO_UDD_Colab.ipynb
│
└── docs/guides/                 # Kaggle/Colab how-to guides & legacy status notes
```

---

## 20. Glossary

| Term | Meaning |
|------|---------|
| **AUV / ROV** | Autonomous / Remotely Operated Underwater Vehicle |
| **YOLO** | "You Only Look Once" — fast single-stage detector family |
| **Backbone / Neck / Head** | Feature extractor / multi-scale fuser / predictor |
| **CSP** | Cross-Stage-Partial block (efficient feature extraction) |
| **GELAN / PGI** | YOLOv9's real innovations (not implemented here) |
| **PANet** | Path Aggregation Network — top-down + bottom-up pyramid |
| **PSEM** | Partial Semantic Encoding Module (neck fusion, from Li et al.) |
| **SDWH** | Split Dimension Weighting Head (attention head, from Li et al.) |
| **TAFM** | Turbidity-Adaptive Fusion Module (**this project's novel module**) |
| **Turbidity** | Water cloudiness from suspended particles |
| **Anchor-free** | Predict boxes from grid cells, not predefined anchors |
| **IoU** | Intersection over Union (box overlap, 0–1) |
| **EIoU** | Efficient IoU loss (IoU + center + size penalties) |
| **BCE** | Binary Cross-Entropy loss |
| **NMS** | Non-Maximum Suppression (remove duplicate boxes) |
| **mAP@50 / @50:95** | mean Average Precision at IoU 0.5 / averaged 0.5–0.95 |
| **Precision / Recall** | correctness of predictions / coverage of real objects |
| **Attention (channel/spatial/self)** | learnable feature weighting |
| **Ablation study** | remove one piece at a time to prove its contribution |

---

## 21. Interview Preparation Q&A

**Q1. Explain your project in two minutes.**
Underwater images are degraded by color cast, haze, and turbidity, so standard detectors
underperform. I built YOLO-UDD v2.0, a YOLO-style detector for the TrashCan dataset that
adds three feature modules: PSEM for stronger multi-scale fusion, SDWH for attention in the
detection head, and my own TAFM, which estimates how murky the water is and adapts the
feature weighting — favoring color/texture in clear water and shape/edges in murky water.
The target is to beat the ~76% YOLOv9c baseline and reach >82% mAP, with TAFM providing the
key gain.

**Q2. What is genuinely novel here?**
The TAFM module and the specific integration of these modules for underwater debris. PSEM,
SDWH, and the YOLOv9 backbone are from prior work; TAFM — a learned, label-free turbidity
gate blending two fusion strategies end-to-end — is mine. I'd prove its value with an
ablation study.

**Q3. Why does turbidity matter, and how does TAFM handle it?**
Turbidity scatters light and destroys color/contrast. A tiny CNN outputs a turbidity score
T∈[0,1]; the module blends two learned parameter sets as σ(T·α+(1−T)·β), so murky images
lean on shape/edge features and clear images lean on color/texture. No turbidity labels are
needed — it's learned from the detection loss.

**Q4. Walk me through the data flow for one image.**
640×640 image → CSP backbone → P3/P4/P5/P6 features → PSEM-enhanced PANet fuses them
top-down then bottom-up → TAFM re-weights by turbidity → SDWH applies level/spatial/channel
attention and predicts (box, objectness, class) at three scales → NMS removes duplicates →
final detections.

**Q5. Why single-stage (YOLO) over two-stage (Faster R-CNN)?**
Real-time, onboard deployment on AUVs/ROVs needs speed. The baseline paper also showed
two-stage Detectron2 models were too heavy to train well under the same hardware, while
YOLO gave a better speed/accuracy trade-off.

**Q6. What's your loss?**
Composite: EIoU for box regression (IoU + center-distance + size penalties), BCE for
objectness (with positive/negative weighting for class imbalance), and BCE for
classification — weighted 5 / 1 / 1. Applied after target assignment maps GT boxes to grid
cells.

**Q7. How do you evaluate?**
COCO-style mAP@50 and mAP@50:95 across IoU 0.5–0.95, plus precision/recall and FPS. mAP@50:95
is the strict headline metric.

**Q8. What are the limitations / what's left to do? (Answer this honestly — it builds trust.)**
Training-side plumbing: I need to wire the COCO metric into validation, fix the evaluation
and inference scripts, resolve a 3-vs-22 class-count setting, and ideally upgrade the target
assignment and use a true YOLOv9/GELAN backbone. Then I run the full training + ablation to
get final numbers.

**Q9. Why did you choose TrashCan 1.0?**
It's a real, labeled underwater debris dataset with the 3-class setup matching marine-cleanup
goals, and it's the benchmark used by the baseline paper, so my results are comparable.

**Q10. How would you deploy this on an AUV?**
Export to a lightweight runtime (ONNX/TensorRT), run at reduced precision for real-time FPS
on an embedded GPU, feed the camera stream, and use detections + the turbidity readout to
guide navigation/cleanup. TAFM's score is a useful operational signal (how reliable is
vision right now?).

**Q11. (DL fundamentals they may probe)** Be ready to define: convolution, BatchNorm,
residual connection, the three attention types, IoU/NMS, precision vs recall vs mAP,
anchor-free detection, cosine LR schedule, and why class imbalance needs weighted
objectness loss. All are defined in Sections 5, 12–14 and the Glossary.

---

*End of documentation. See `COMPLETION_PLAN.md` for the step-by-step plan to finish the
project, and `README.md` for the quick overview.*
