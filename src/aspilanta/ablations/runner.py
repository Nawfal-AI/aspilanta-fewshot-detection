"""The nine ablations.

Each removes or changes exactly one component and re-runs the affected stage.
Three of the nine contradict the design they were meant to confirm - that is
reported rather than quietly dropped, and one of them (backbone freezing)
changed the proposed method.

Ablations fall into two families because they intervene at different stages,
and this determines how each is scored:

* **fine-tuning-recipe** ablations change stage 2 and are evaluated on the
  four-class target-domain validation set;
* **transfer-learning-recipe** ablations change stage 1 and are evaluated on
  the three-class base validation set - scoring them on the four-class set
  would be meaningless, since a three-class model has no target-class output.

A note on archived run names. These ablations were first run when the *frozen*
backbone was the proposed method, so the archives use the old names:
``ablation_unfrozen_backbone`` holds what is now the proposed model, and
``exp3_4class_synthetic_finetuned`` holds what is now the frozen ablation.
:data:`ARCHIVE_NAME_ALIASES` records the mapping so restored runs are labelled
correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

__all__ = ["AblationSpec", "ABLATIONS", "ARCHIVE_NAME_ALIASES", "run_ablations",
           "evaluate_ablations"]

Family = Literal["finetuning", "transfer"]


@dataclass
class AblationSpec:
    """One ablation: what it removes, how to build it, and why it exists."""

    name: str
    family: Family
    question: str
    run_name: str
    # stage-2 overrides
    freeze: int | None = None
    copy_paste: float | None = None
    init_weights: str | None = None       # None = fine-tune from the base detector
    # dataset overrides
    generator_overrides: dict = field(default_factory=dict)
    include_synthetic: bool = True


ABLATIONS: list[AblationSpec] = [
    # ---- 1. Copy-paste compositing (the data-oriented component) ---------- #
    AblationSpec(
        name="real_only", family="finetuning",
        question="Is synthetic data needed at all, or do the real images suffice?",
        run_name="ablation_real_only", include_synthetic=False),
    AblationSpec(
        name="no_poisson", family="finetuning",
        question="Does Poisson blending matter, or is alpha compositing enough?",
        run_name="ablation_no_poisson",
        generator_overrides={"use_poisson": False}),
    AblationSpec(
        name="no_distractors", family="finetuning",
        question="Do the European moth distractors improve discrimination?",
        run_name="ablation_no_distractors",
        generator_overrides={"use_distractors": False}),
    AblationSpec(
        name="no_pad_aware", family="finetuning",
        question="Does placing insects only on the detected adhesive pad matter?",
        run_name="ablation_no_pad_aware",
        generator_overrides={"use_pad_aware": False}),

    # ---- 2. Transfer learning (stage 1) ---------------------------------- #
    AblationSpec(
        name="scratch_init", family="transfer",
        question="How much does ImageNet pretraining contribute?",
        run_name="ablation_scratch_init",
        init_weights="yolov8s.yaml"),      # .yaml = architecture only, no weights
    AblationSpec(
        name="yolov8n_arch", family="transfer",
        question="Is YOLOv8s worth its extra capacity over YOLOv8n here?",
        run_name="ablation_yolov8n",
        init_weights="yolov8n.pt"),

    # ---- 3. Fine-tuning recipe (stage 2) --------------------------------- #
    AblationSpec(
        name="frozen_backbone", family="finetuning",
        question="Does freezing the backbone protect against few-shot overfitting?",
        run_name="ablation_frozen_backbone", freeze=10),
    AblationSpec(
        name="fresh_imagenet_init", family="finetuning",
        question="Is the staged chain better than fine-tuning straight from ImageNet?",
        run_name="ablation_fresh_imagenet_init", init_weights="yolov8s.pt"),
    AblationSpec(
        name="no_copy_paste_aug", family="finetuning",
        question="Does copy-paste augmentation help the rare class?",
        run_name="ablation_no_copy_paste_aug", copy_paste=0.0),
]

# Historical run directories -> current names. Applied when restoring an
# archive so the master table is labelled by what a run actually is rather
# than by what it was called when the proposed method was different.
ARCHIVE_NAME_ALIASES = {
    "ablation_unfrozen_backbone": "proposed_solution",
    "exp3_4class_synthetic_finetuned": "ablation_frozen_backbone",
}


def run_ablations(specs, build_dataset: Callable, train_finetuning: Callable,
                  train_transfer: Callable, base_yaml, base_weights,
                  skip_existing: bool = True) -> dict[str, Path]:
    """Execute a list of ablations and return ``name -> best.pt``.

    Args:
        build_dataset: ``(spec) -> dataset_yaml_path``; owns generating any
            variant synthetic data the spec calls for.
        train_finetuning: ``(run_name, data_yaml, init_weights, freeze,
            copy_paste) -> Path``.
        train_transfer: ``(run_name, init_weights) -> Path``.
        skip_existing: reuse a completed run rather than retraining it.

    ``skip_existing`` is not a convenience. Nine full training runs exceed the
    twelve-hour session limit on the platform this was developed for, so the
    sweep must be resumable across sessions.
    """
    results: dict[str, Path] = {}

    for spec in specs:
        print(f"\n{'=' * 70}\nABLATION [{spec.family}]: {spec.name}\n"
              f"  {spec.question}\n{'=' * 70}")

        if spec.family == "transfer":
            weights = train_transfer(spec.run_name, spec.init_weights)
        else:
            data_yaml = build_dataset(spec)
            weights = train_finetuning(
                spec.run_name, data_yaml,
                spec.init_weights or base_weights,
                spec.freeze, spec.copy_paste)

        results[spec.name] = weights

    print("\nCompleted ablations:")
    for name, path in results.items():
        print(f"  {name:<24} {path}  exists={Path(path).exists()}")
    return results


def evaluate_ablations(results: dict[str, Path], target_val_yaml, base_val_yaml,
                       specs, project, target_class_id: int = 3, device=None):
    """Score every ablation on the validation set appropriate to its family."""
    import pandas as pd

    from ..evaluation.reporting import get_class_ap, safe_float, validate_model

    by_name = {spec.name: spec for spec in specs}
    rows = []

    for name, weights in results.items():
        spec = by_name.get(name)
        if spec is None or not Path(weights).exists():
            continue

        data_yaml = base_val_yaml if spec.family == "transfer" else target_val_yaml
        metrics = validate_model(weights, data_yaml, f"val_{spec.run_name}",
                                 project, device=device)

        row = {
            "ablation": name,
            "family": spec.family,
            "question": spec.question,
            "eval_set": "base 3-class val" if spec.family == "transfer"
                        else "target-domain val",
            "mAP50": safe_float(metrics.box.map50),
            "mAP50_95": safe_float(metrics.box.map),
        }
        if spec.family == "finetuning":
            row["AP_target_50"] = get_class_ap(metrics, target_class_id, "map50")
            row["AP_target_50_95"] = get_class_ap(metrics, target_class_id, "map")
        rows.append(row)

    return pd.DataFrame(rows)
