# YOLO-UDD v2.0 — Completion Plan

This is the step-by-step roadmap to take the project from "architecture built" to
"trained model + results + ablation, ready for thesis/viva." Work the phases in order —
each phase unblocks the next. Every task has a **why**, a **what to do**, and a **done-when**
acceptance check.

**Legend:** 🔴 blocker (nothing works end-to-end without it) · 🟡 important · 🟢 polish.

---

## Phase 0 — Environment & data (half a day)

**0.1 Set up the environment** 🔴
- `python -m venv venv && source venv/bin/activate`
- `pip install -r requirements.txt`
- GPU strongly recommended (Kaggle T4 / Colab are fine — notebooks are in `notebooks/`).
- **Done when:** `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
  (or you've accepted CPU-only for tiny tests).

**0.2 Get the data into COCO format** 🔴
- Download TrashCan 1.0. If it's in Supervisely format, run
  `python scripts/convert_supervisely_to_coco.py --input-dir <raw> --output-dir data/trashcan`.
- Verify: `python scripts/verify_dataset.py --dataset-dir data/trashcan`.
- **Done when:** verify script shows images + annotations + categories for train/val/test.

**0.3 Smoke-test the forward pass** 🔴
- Run the README's forward-pass snippet.
- **Done when:** it prints a turbidity score and 3 detection scales with no error.

---

## Phase 1 — Fix the training/eval plumbing  ✅ DONE (see CHANGELOG.md)

These are the bugs that currently stop you from getting real numbers. Do them first.

**1.1 Resolve the class-count inconsistency (3 vs 22)** 🔴
- **Why:** `train_config.yaml` says `num_classes: 22`, but `dataset.py`'s `class_map`,
  `SDWH` default, and the README all say 3. They must agree or training is meaningless.
- **What:** Decide **3-class** (recommended — matches the baseline paper and the README).
  Set `num_classes: 3` and a 3-name `class_names` in every config. Make `dataset.py`'s
  category→label mapping collapse the 22 fine-grained TrashCan categories into
  {trash, animal, rov} (e.g. a dict mapping each raw category name to one of the three).
- **Done when:** a batch from the dataloader yields labels only in {0,1,2}, and the model
  built with `num_classes=3` runs the loss without shape errors.

**1.2 Wire the real metric into validation** 🔴
- **Why:** `train.py`'s `validate()` calls `compute_metrics` (returns zeros), so mAP is
  always 0 → best-checkpoint and early-stopping are broken.
- **What:** In `validate()`, after `batched_nms(...)`, call `compute_metrics_coco(...)`
  instead of `compute_metrics`. Make sure the GT you pass is in the **same box format**
  (`x,y,w,h`) and **same normalization** (0–1) as the decoded predictions — convert if
  needed. (Tip: `compute_metrics_coco` expects detections as dicts with `boxes/scores/classes`
  and targets as `(boxes, labels)` tuples — which is already what you build.)
- **Done when:** validation prints non-zero mAP that changes across epochs.

**1.3 Fix `scripts/evaluate.py`** 🟡
- **Why:** it imports `measure_fps` and `MetricsCalculator` from `utils/metrics.py`, which
  don't exist → the script crashes on import.
- **What:** Either (a) implement both in `utils/metrics.py` — a `measure_fps(model, device)`
  that times N forward passes and returns FPS, and a small `MetricsCalculator` class that
  accumulates detections/targets and calls `compute_metrics_coco` in `.compute()`; or
  (b) simpler, rewrite `evaluate.py` to loop the test set → `batched_nms` →
  `compute_metrics_coco`, and add an inline FPS timer.
- **Done when:** `python scripts/evaluate.py --weights best.pt --data-dir data/trashcan`
  prints precision/recall/mAP/FPS and writes `evaluation_results.json`.

**1.4 Fix `scripts/detect.py` decoding** 🟡
- **Why:** `detect()` returns dummy empty detections, so inference draws nothing.
- **What:** Replace the dummy block with: forward → `batched_nms(...)` → convert the kept
  normalized `(x,y,w,h)` boxes to pixel `(x1,y1,x2,y2)` for `draw_detections`. Reuse the
  class names already defined in the `Detector`.
- **Done when:** running detect on a sample image saves an image with boxes + the turbidity
  overlay.

**1.5 Quick overfit sanity check** 🔴
- **Why:** before a long run, prove the model *can* learn.
- **What:** `python scripts/create_subset.py --ratio 0.02` (or use `train_config_quick.yaml`)
  and train a few epochs on a tiny subset.
- **Done when:** training loss drops clearly and mAP on that subset rises above 0 — this
  confirms the whole loop (data→loss→backprop→metric) is sound.

---

## Phase 2 — First real training run (2–4 days of compute)

**2.1 Baseline-ish full run** 🔴
- Use `train_config.yaml` (now 3-class) on a GPU (Kaggle/Colab). 100 epochs, batch 16, 640px.
- Monitor TensorBoard: loss components should fall; mAP should rise; turbidity score should
  look sensible (varies across images).
- **Done when:** you have a `best.pt` and a non-trivial mAP on the val set.

**2.2 Evaluate on the test set** 🔴
- `python scripts/evaluate.py --weights best.pt --data-dir data/trashcan --compare-baseline`.
- **Done when:** you have test precision/recall/mAP@50/mAP@50:95/FPS recorded.

---

## Phase 3 — The research result: ablation study (1 week incl. compute) 🟡

> **Tooling is ready (see CHANGELOG.md).** Use `--ablation {baseline,psem,psem_sdwh,full}`
> with `scripts/train.py`, or just run `notebooks/YOLO_UDD_Ablation_Kaggle.ipynb` end to
> end on Kaggle — it trains all four configs and prints the results table for you.

**This is what turns "a model" into "a contribution."** Train four configurations and
tabulate mAP so you can *prove* each module (especially TAFM) helps.

| Config | What to disable | Purpose |
|--------|-----------------|---------|
| A. Baseline | PSEM→plain conv, no TAFM, plain head | establish your own baseline |
| B. +PSEM | add PSEM only | isolate PSEM gain |
| C. +PSEM+SDWH | add SDWH head | isolate SDWH gain |
| D. Full (+TAFM) | everything | **show TAFM's gain** |

- ✅ Done: on/off flags (`use_psem`, `use_sdwh`, `use_tafm`) exist on the model and config,
  exposed via `--ablation`. No code edits needed to toggle configurations.
- **Done when:** you have a 4-row table of mAP@50 and mAP@50:95, and the Full model (D) beats
  the others — demonstrating TAFM's contribution with numbers.

---

## Phase 4 — Strengthen the contribution (optional but valuable) 🟢

- **4.1 Literature check on novelty.** Search "turbidity-adaptive / water-quality-aware
  feature fusion underwater detection." Position TAFM precisely relative to anything similar.
- **4.2 Honest backbone.** Either rename the backbone from "YOLOv9c" to "YOLOv9-style CSP
  backbone," or swap in a real YOLOv9/GELAN backbone (e.g. Ultralytics) for a true baseline.
- **4.3 Better target assignment.** Upgrade `assign_targets_simple` toward SimOTA / Task-
  Aligned Assignment for an accuracy bump.
- **4.4 Faithful PSEM/SDWH (optional).** If you want to claim exact reimplementation, align
  them with Li et al.'s equations (FasterNet PConv in PSEM; the specific level/spatial/task
  operators in SDWH). Otherwise, clearly label them "inspired by."
- **4.5 Qualitative figures.** Save example detections (clear vs murky) and a turbidity-vs-
  accuracy plot — great for the thesis and the viva.

---

## Phase 5 — Write-up & deliverables 🟢

- Final results table (baseline vs full, + ablation).
- Update `README.md` with real numbers (replace the "target" numbers with achieved ones).
- Thesis sections map almost 1:1 to `PROJECT_DOCUMENTATION.md`.
- Prepare the viva using the Q&A in `PROJECT_DOCUMENTATION.md` §21.

---

## Priority order if you're short on time

1. **1.1 + 1.2** (class count + real metric) — without these, nothing is measurable.
2. **1.5** (overfit check) — proves the loop learns.
3. **2.1 + 2.2** (one full run + test eval) — gives you *a* result.
4. **3** (ablation) — gives you *the* contribution.
5. Everything else is polish.

---

## Risk notes

- **Compute:** a 100-epoch 640px run needs a GPU; budget Kaggle/Colab time, use
  `train_config_fast`/`_quick` for iteration, full config only for final numbers.
- **The >82% target is an estimate**, not a guarantee — it chains numbers across different
  papers/datasets (see PROJECT_DOCUMENTATION §16–17). Report whatever you actually achieve;
  a smaller, honest, ablation-backed gain is worth more than a big unverifiable claim.
- **Reproducibility:** fix the seed (already in config), log the exact config with each run,
  and keep `best.pt` + `evaluation_results.json` per experiment.
