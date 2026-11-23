# Results

All numbers are from the project's evaluation runs. The negative and contradictory results are
here too — that is the point of the document.

---

## Detection

| Model | Validation set | mAP@50 | mAP@50-95 | AP<sub>Asp</sub>@50 | AP<sub>Asp</sub>@50-95 |
|---|---|---|---|---|---|
| Stage 1 (3-class base) | Three-class split | 0.868 | 0.378 | — | — |
| **Stage 2 (proposed)** | Mixed target-domain | **0.981** | **0.827** | **0.981** | **0.827** |
| Stage 2 (proposed) | Real-only subset (n=11) | 0.995 | 0.609 | 0.995 | 0.609 |
| Stage 2-F (frozen, ablation) | Mixed target-domain | 0.453 | 0.258 | 0.906 | 0.516 |

Per-class breakdown of the proposed model, confirming the base classes survive
fine-tuning rather than being forgotten:

| Class | Instances | AP@50 | AP@50-95 |
|---|---|---|---|
| MP (*Macrolophus pygmaeus*) | 237 | 0.849 | 0.346 |
| NT (*Nesidiocoris tenuis*) | 127 | 0.976 | 0.482 |
| WF (Whitefly) | 823 | 0.787 | 0.303 |
| **Aspilanta** | 1,629 | **0.981** | **0.826** |

**Zero-shot diagnostic.** The three-class base detector produced 8,299
detections on trap images containing *Aspilanta*, none of them correct by
construction — the model has no output for the class. The domain gap is
measured, not assumed.

---

## Deployment pipeline

| Pipeline | Precision | Recall | F1 |
|---|---|---|---|
| **Single-pass, no filter** | **0.931** | **0.801** | **0.861** |
| Single-pass + CLIP filter | 0.936 | 0.675 | 0.785 |
| Tiled inference | 0.095 | 0.013 | 0.023 |

Both refinements make things worse, and both are off by default because of it.

**Tiling** fails on domain mismatch. The detector was trained on whole images
resized to 640 px; a native-resolution 640 px crop presents insects at an
apparent scale it has never seen. The failure is total, not marginal.

**Blanket CLIP filtering** buys +0.005 precision for −0.126 recall. The cause
is applying it to every detection: it second-guesses confident, correct
detections whose crops happen to sit slightly away from the prototype in CLIP
space.

### Selective CLIP filtering

Section 11.3 applies CLIP only to detections below a trust threshold, keeping
confident ones unconditionally. False positives concentrate among the marginal
detections, which is where a filter can help without costing recall. Sweep the
trust and similarity thresholds on the validation half:

```bash
python scripts/evaluate.py --weights .../best.pt --pipeline selective_clip
```

---

## Cross-family baselines

| Baseline | Family | Result |
|---|---|---|
| OWL-ViT (text prompts) | Open-vocabulary | F1 = 0.382 |
| OWL-ViT (image-guided) | Open-vocabulary | recall < 0.02 |
| TFA | Model-oriented | AP@50 = 0.388 |
| Prototypical network | Algorithm-oriented | ROC-AUC > 0.996 (crop classification only) |

The prototypical result is strong but answers a different question. It
classifies *crops that have already been found*; it cannot locate them in a
full image. That is precisely why an algorithm-oriented method alone does not
solve this problem, and why the pipeline uses one only as an optional
re-scoring step.

---

## The nine ablations

| Component | Effect on real-image AP@50 | Verdict |
|---|---|---|
| Pad-aware placement | **−0.269** | Strongly supported — the largest compositing effect |
| Poisson blending | −0.039 | Supported |
| ImageNet initialisation | −0.868 (the model does not train at all) | Decisively supported |
| YOLOv8s over YOLOv8n | −0.088 three-class mAP@50 | Supported |
| Synthetic data | −0.015 on real images, −0.901 on the mixed set | Weakly supported — see below |
| YOLO copy-paste augmentation | −0.029, within noise at n=11 | **Not supported** |
| Distractor insects | **+0.026** — removing them *improved* real-image AP | **Refuted** |
| Frozen backbone | Freezing costs −0.075 AP@50, −0.311 AP@50-95 | **Changed the method** |
| Staged transfer chain | Comparable either way | **Contributes nothing measurable** |

Three of nine contradicted the design they were meant to confirm. They are
reported because a result set that only confirms its own hypotheses is the
less informative kind.

### The backbone result changed the method

| Configuration | Mixed target-domain | Real-only (n=11) |
|---|---|---|
| Trainable backbone | AP@50 0.981 / 0.827 | 0.995 / 0.609 |
| Frozen backbone | AP@50 0.906 / 0.516 | 0.941 / 0.471 |

Freezing is standard few-shot practice — with few novel examples a trainable
backbone overfits. But by the time fine-tuning starts the generator has
supplied 2,000 images and 9,211 instances, so the regime the freeze protects
against no longer exists. The proposed method changed to `freeze=0`.

**Stated openly:** this configuration was adopted because an ablation preferred
it, which is selection on evaluation data. It also wins on the real-only
subset, which shares no images or cutout provenance with training — that is
supporting evidence, not a clean independent test. The held-out test split
(Section 11.2) exists for exactly this reason.

### The confound in the synthetic-data result

On the 11 real close-up photographs, a detector trained with **no synthetic
data at all** scores 0.926 AP@50 against 0.941 for the configuration it was run
against — a gap of 0.015.

On the mixed validation set the gap is 0.901, which looks like overwhelming
evidence for synthetic data. It is not. The mixed set is mostly synthetic
images, so a detector trained on synthetic data is being asked to recognise the
generator's own output. Part of that 0.901 measures the value of synthetic
training data; part of it measures self-recognition, and the two cannot be
separated from that number alone.

The 0.015 gap on real images is the honest figure, and n=11 is too small to
call it significant.

**The claim this supports** is therefore qualified: careful synthetic
compositing can substitute for large-scale real data collection, but only part
of the evidence survives closer inspection.

### The distractor result

Removing the distractor moths *improved* real-image AP by 0.026. The
hypothesis — that distractors force the detector to discriminate rather than
just detect "an insect" — is not supported here. Two caveats: the ablation
variants were generated at a reduced image count, and n=11 is small.

---

## A labelling defect, found and fixed

Section 11.1 documents a bug in the dataset-merging code. The generator writes
`1 = eu_moth` distractors as visual clutter, not as ground truth. The original
merge **remapped** those boxes into the four-class scheme instead of dropping
them, where they collided with MP (id 0) or NT (id 1) depending on the map.

Nothing crashed. A remapped distractor label is indistinguishable from a real
one once written; the numbers just quietly shifted. It surfaced only as an
inflated class count in a histogram.

The package carries only the corrected version, and
`scripts/build_datasets.py` prints a class histogram after every build so the
same class of error is visible immediately.

Two tests cover it directly — `test_drops_rather_than_remaps` and
`test_drop_takes_precedence_over_force` in `tests/test_yolo_io.py`.

---

## What to take from all this

The headline number is real: **AP@50 = 0.981** for a species with 44 usable
examples and no additional data collection.

The qualifications are also real. The strongest evidence for the central claim
comes from a validation set that partly measures self-recognition; the honest
real-image gap is 0.015 at n=11. Two design decisions the ablations were meant
to validate were refuted instead. One changed the method.
