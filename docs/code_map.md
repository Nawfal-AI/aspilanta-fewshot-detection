# Code map

Which notebook cell became which module, and what changed on the way.

The source is `notebooks/00-research-notebook-final.ipynb`: 164 cells across 11
sections. Cell indices below are zero-based positions in that notebook.

---

## Section 0-1: Environment and base dataset

| Cells | Went to | Notes |
|---|---|---|
| 2–7 | `config.py`, `scripts/_bootstrap.py` | Kaggle restore cells dropped; the working directory is now a config value, not a `/kaggle/working` literal |
| 10–21 | `data/voc.py`, `data/splits.py`, `scripts/prepare_base_dataset.py` | The notebook defined `convert_voc_to_yolo` **three times** (cells 12, 15, 19) with small differences; reconciled into one |
| 23–24 | `models/train.py` | The auxiliary YOLOv8n run became the `yolov8n_arch` ablation rather than a separate code path |
| 26–27 | `pipeline/detector.py` | `detect_and_count` / `batch_detect_and_count` folded into `AspilantaDetector.detect` and `detect_and_save` |
| 29 | `evaluation/reporting.py` | Training-curve plotting generalised |
| 11 | *(not ported)* | A one-off figure-generation cell for a report figure |

## Section 2: Synthetic generation

| Cells | Went to | Notes |
|---|---|---|
| 36–37 | `config.py` (`GeneratorConfig`) | Pip installs and path literals became config |
| 39 (501 lines) | `synthetic/cutouts.py` | rembg subprocess worker kept verbatim (isolated for a real reason; see below) |
| 41 (777 lines) | `synthetic/generator.py`, `synthetic/compositing.py`, `synthetic/pad.py` | Split by concern: pad detection, per-insect compositing, and the generation loop |
| 41 (`verify`) | `synthetic/qc.py` | |
| 34 | *(not ported)* | Kaggle dataset-restore cell, already commented out in the notebook |

## Section 3: The proposed solution

| Cells | Went to | Notes |
|---|---|---|
| 44 | `pipeline/detector.py` | `AshpilantaDetector` → `AspilantaDetector`; module-level `SOLUTION_*` constants → `DeploymentConfig` |
| 46 | `scripts/detect.py` | The interactive `input()` image picker became CLI arguments |
| 50–51 | `config.py` | ~60 `EXP1_*` / `EXP3_*` / `SOLUTION_*` constants → four dataclasses |
| 53 (267 lines) | `data/yolo_io.py` | |
| 54 | `data/splits.py` | Hard-coded candidate lists → `Paths.*_candidates()` |
| 56–60 | `data/builders.py` | Four near-identical build blocks → four functions sharing `DatasetSources` |
| 62 | `synthetic/qc.py` | |
| 64–65, 67, 69 | `models/train.py` | |
| 71 (306 lines) | `models/clip_filter.py` | |
| 73–79 | `pipeline/detector.py`, `pipeline/tiling.py`, `models/clip_filter.py` | |
| 81 | *(not ported)* | Kaggle archive/backup cell |

## Section 4-5: Experiments and deployment evaluation

| Cells | Went to |
|---|---|
| 84, 86, 88, 90 | `evaluation/reporting.py` |
| 94–95 | `evaluation/metrics.py`, `evaluation/harness.py` |
| 97, 99–110 | `evaluation/harness.py` (`evaluate_bucketed_by_size`, `sweep_confidence`) |

## Section 6: Ablations

| Cells | Went to | Notes |
|---|---|---|
| 113 (527 lines) | *(merged away)* | A second copy of the generator, for variants; now `GeneratorConfig` flags |
| 115, 117, 119 | `ablations/runner.py`, `scripts/run_ablations.py` | Nine inline blocks → nine `AblationSpec` records |
| 122–125 | `ablations/runner.py` | Archive-restore logic → `ARCHIVE_NAME_ALIASES` |
| 127 | *(not ported)* | Marked "removed" in the notebook itself |

## Section 7-9: Baselines

| Cells | Went to |
|---|---|
| 133–137 | `baselines/owlvit.py`, `scripts/run_baselines.py` |
| 142 | `models/train.py` (`tfa_freeze_callback`, `train_tfa_baseline`) |
| 146–147 | `baselines/protonet.py` |

## Section 11: Corrections

| Cells | Went to | Notes |
|---|---|---|
| 154 | `data/yolo_io.py` | The corrected `remap_label_lines` is the **only** version in the package |
| 156 | `data/splits.py` (`carve_test_split`) | |
| 158 | `models/clip_filter.py` (`filter_selective`) | |
| 160, 162 | `scripts/evaluate.py` | |

---

## The five consolidations

Each of these merged code that existed in more than one place. They are listed
because a reader comparing the notebook to the package will notice the
difference and should know it was deliberate.

**1. IoU and matching existed in three copies.** The main harness (cell 94),
an `_s9`-suffixed set for the OWL-ViT section (cell 133), and an inline version
in the selective-CLIP sweep (cell 158). Each section was written to run
standalone, which is reasonable in a notebook. But the three sets feed one
comparison table, and if they ever diverged the table would silently stop
comparing like with like. One implementation now, in `evaluation/metrics.py`.

**2. The generator existed twice.** Cell 41 for the main dataset, cell 113 for
the ablation variants: 777 and 527 lines with substantially overlapping
logic. The ablations exist to isolate the effect of one compositing choice, so
any unintended difference between the two copies would be attributed to the
ablated component. Now one function; the variants are `GeneratorConfig` flags.

**3. Label remapping.** The notebook's original `remap_label_lines` mapped the
generator's distractor class into the four-class scheme, where it collided with
MP or NT depending on the map. Section 11.1 documents this and redefines the
function with `drop_class_ids`. Only the corrected version is in the package -
the defective one is not available to be called by accident.

**4. The CLIP filter's silent no-op.** The notebook version read `clip_model`
and `support_images_all` out of `globals()` and returned its input unchanged
when they were absent, which was the case after any kernel restart. Every
"CLIP-filtered" number produced before the fix was identical to the unfiltered
baseline, because the filter never ran. `ClipPrototypeFilter` owns its state
and raises if it cannot be constructed.

**5. `convert_voc_to_yolo` existed three times.** Cells 12, 15 and 19, with
differences in whether unknown class names were warned about or silently
skipped. Reconciled into one function with a `warn_unknown` flag.

---

## Things kept that look odd but are not

**The rembg subprocess.** `synthetic/cutouts.py` shells out to a worker script
rather than importing rembg. This is not superstition: rembg pulls in
onnxruntime, which grabs the GPU and, in the same process as PyTorch, reliably
exhausts CUDA memory partway through a batch. Isolation also means a segfault
costs one image rather than the run.

**Tiled inference.** Measured at F1 0.023 against 0.861 for single-pass, and
off by default. Kept because the failure is informative: it is a clean
demonstration of train/inference domain mismatch, and because the underlying
motivation is sound for genuinely huge images.

**`freeze=0` as the proposed setting.** This contradicts standard few-shot
practice and contradicted this project's own original design. The ablation
evidence and the caveat about having selected it on evaluation data are in
`docs/results.md` and in the `SolutionConfig` docstring.

---

## Naming

This package writes *Aspilanta* throughout. Parts of
the notebook code use "Ashpilanta", including some archived checkpoint class
names. If you load an older checkpoint its `model.names` may differ from
`ClassScheme.full` while referring to the same class; `AspilantaDetector`
prefers the checkpoint's own names and falls back to the scheme.

Run directory names carry similar history. The ablations were first run when
the *frozen* backbone was the proposed method, so `ablation_unfrozen_backbone`
holds what is now the proposed model and `exp3_4class_synthetic_finetuned`
holds what is now an ablation. `ARCHIVE_NAME_ALIASES` in `ablations/runner.py`
records the mapping, and `AspilantaDetector` warns if it can only find the
frozen checkpoint.
