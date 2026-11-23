# Experiments

The experiment design, the nine ablations, and how to reproduce each.

---

## The three main experiments

| # | Model | Trained on | Evaluated on | Purpose |
|---|---|---|---|---|
| 1 | 3-class base | Yellow Sticky Traps only | 3-class split | Establishes the domain is learnable |
| 2 | 3-class base | *(no training - reuses Exp 1)* | Synthetic trap images with *Aspilanta* | Zero-shot diagnostic |
| 3 | 4-class proposed | Original + real + synthetic | 4-class target-domain split | The method |

**Experiment 2 needs explaining.** It runs the three-class model over images
that contain the target species. The model has no output for that class, so
every detection is wrong by construction - it produced 8,299 detections, none
correct. This is not a failed experiment; it is the control that turns
"there is a domain gap" from an assumption into a measurement.

Experiment 1 is also scored on the four-class validation set in the
cross-experiment table, where it scores exactly zero on *Aspilanta*. That is
the expected and correct result.

Reproduce:

```bash
python scripts/train.py --stage all
python scripts/evaluate.py --weights work/runs/detect/solution_4class_detector/weights/best.pt
```

---

## The nine ablations

Each removes or changes exactly one component. Everything else is held at the
proposed values, so a difference in the result is attributable to the change.

### Group 1 - Copy-paste compositing (data-oriented)

| Ablation | Question |
|---|---|
| `real_only` | Is synthetic data needed at all, or do the real images suffice? |
| `no_poisson` | Does Poisson blending matter, or is alpha compositing enough? |
| `no_distractors` | Do the distractor moths improve discrimination? |
| `no_pad_aware` | Does placing insects only on the detected pad matter? |

These require regenerating synthetic data with the relevant switch off.

### Group 2 - Transfer learning (stage 1)

| Ablation | Question |
|---|---|
| `scratch_init` | How much does ImageNet pretraining contribute? |
| `yolov8n_arch` | Is YOLOv8s worth its extra capacity over YOLOv8n? |

Evaluated on the **three-class** validation set, not the four-class one - a
three-class model has no target-class output to score.

### Group 3 - Fine-tuning recipe (stage 2)

| Ablation | Question |
|---|---|
| `frozen_backbone` | Does freezing protect against few-shot overfitting? |
| `fresh_imagenet_init` | Is the staged chain better than fine-tuning straight from ImageNet? |
| `no_copy_paste_aug` | Does YOLO's copy-paste augmentation help the rare class? |

Reproduce:

```bash
python scripts/run_ablations.py                          # all nine
python scripts/run_ablations.py --only frozen_backbone   # one
python scripts/run_ablations.py --evaluate-only          # score existing runs
```

Nine full training runs exceed the session limit on most hosted GPUs, so
completed runs are reused by default. Re-invoke until every row appears in the
master table.

### A confound to know about

The compositing ablations regenerate synthetic data at a reduced image count
(`--variant-images`, default 500) to keep the sweep tractable. Those variants
therefore train on *less* data as well as *different* data, so their numbers
are not directly comparable with the main 2,000-image run. Raise the count to
match if you need a clean comparison; it costs generation time only.

All variants use the same seed, so sampling differences do not masquerade as
compositing differences.

---

## Cross-family baselines

| Baseline | Family | What it tests |
|---|---|---|
| OWL-ViT (text) | Open-vocabulary | Can a foundation model find this from a prompt? |
| OWL-ViT (image-guided) | Open-vocabulary | ...from K real exemplars? |
| TFA | Model-oriented | Is the standard few-shot recipe enough? |
| Prototypical network | Algorithm-oriented | Does a task-trained embedding beat frozen CLIP? |

```bash
python scripts/run_baselines.py --baseline owlvit
python scripts/run_baselines.py --baseline protonet
python scripts/train.py --stage tfa
```

Two design points worth noting.

**Exemplar leakage.** The image-guided OWL-ViT variants must exclude the images
their exemplar crops came from, or the baseline is scored partly on its own
prompts. `collect_exemplars` returns the source filenames for exactly this.

**Real-only query set.** The prototypical-vs-CLIP comparison restricts query
crops to those from real photographs. Synthetic crops descend from the same 44
cutouts the support set is drawn from, so including them would let a
task-trained embedding score highly by recognising cutout provenance rather
than the species.

That comparison is also **threshold-free**. An earlier version scored both
embeddings at a mean-similarity threshold and got recall pinned at 1.000 for
every configuration - the numbers measured the threshold rule, not the
embedding. Average precision, ROC-AUC and best-achievable-F1 measure whether
the embedding separates the classes at all. Under class imbalance, read the
AP lift over prevalence, not the raw AP.

---

## The held-out test split

```bash
python scripts/build_datasets.py --carve-test-split
```

Splits the target-domain validation set 70/30, stratified on real vs synthetic
provenance, seed 1337. Tune on the validation half; report the final number on
the test half once.

This exists because the proposed configuration was selected partly by reading
ablation results off the validation set. That is selection on evaluation data,
and it is stated openly rather than hidden.
