"""Builders for the four datasets the experiments consume.

Notebook cells 56 to 60 built these inline with a great deal of repetition.
The four are:

===========================  =============================================
dataset                      purpose
===========================  =============================================
``exp1``  (3 classes)        Stage-1 base detector. No Aspilanta at all.
``exp2``  (1 class)          Zero-shot diagnostic. Synthetic images only,
                             evaluated with the 3-class model to confirm it
                             cannot detect the unseen species.
``exp3``  (4 classes)        The proposed solution's training set: original
                             three classes + real Aspilanta + synthetic.
``target_val`` (4 classes)   The common validation set all three
                             experiments are scored on.
===========================  =============================================

Every synthetic copy passes ``drop_class_ids`` so the generator's eu_moth
distractors are removed rather than remapped — see Section 11.1 and the note
in :func:`aspilanta.data.yolo_io.remap_label_lines`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import ClassScheme, Paths
from .yolo_io import (
    copy_folder_yolo,
    copy_one_image_and_label,
    ensure_yolo_dirs,
    get_split_dirs,
    list_images,
    print_dataset_counts,
    safe_clear_dir,
    write_yaml,
)

__all__ = ["DatasetSources", "build_exp1", "build_exp2_zeroshot",
           "build_exp3", "build_target_domain_val", "build_all"]


@dataclass
class DatasetSources:
    """Resolved locations and image lists the builders operate on.

    Splitting the real and synthetic image pools once, here, means every
    downstream dataset sees the same partition — which is what makes the
    cross-experiment comparison in Section 4.3 fair.
    """

    original_root: Path
    real_aspilanta_root: Path
    synthetic_root: Path

    real_train: list[Path]
    real_val: list[Path]
    synth_train: list[Path]
    synth_val: list[Path]

    @classmethod
    def build(cls, original_root: Path, real_aspilanta_root: Path,
              synthetic_root: Path, real_val_ratio: float = 0.20,
              synth_val_ratio: float = 0.20, seed: int = 42,
              k_real_shots: int | None = None) -> DatasetSources:
        """Partition the real and synthetic pools deterministically.

        ``k_real_shots`` caps how many real Aspilanta images reach training.
        Leave it as None to use every available image; set it to 5 or 10 to
        run a strict few-shot ablation.
        """
        rng = random.Random(seed)

        real_images = sorted(list_images(real_aspilanta_root))
        rng.shuffle(real_images)
        n_real_val = max(1, int(round(len(real_images) * real_val_ratio))) if real_images else 0
        real_val = real_images[:n_real_val]
        real_train = real_images[n_real_val:]
        if k_real_shots is not None:
            real_train = real_train[:k_real_shots]

        synth_train_dir, synth_train_lbl = get_split_dirs(synthetic_root, "train")
        synth_val_dir, _ = get_split_dirs(synthetic_root, "val")
        synth_train = sorted(list_images(synth_train_dir)) if synth_train_dir else []
        synth_val = sorted(list_images(synth_val_dir)) if synth_val_dir else []

        # If the generator produced no val split, carve one from train.
        if synth_train and not synth_val:
            rng.shuffle(synth_train)
            n_val = int(round(len(synth_train) * synth_val_ratio))
            synth_val, synth_train = synth_train[:n_val], synth_train[n_val:]

        print(f"Real Aspilanta   : {len(real_train)} train / {len(real_val)} val")
        print(f"Synthetic        : {len(synth_train)} train / {len(synth_val)} val")

        return cls(original_root=Path(original_root),
                   real_aspilanta_root=Path(real_aspilanta_root),
                   synthetic_root=Path(synthetic_root),
                   real_train=real_train, real_val=real_val,
                   synth_train=synth_train, synth_val=synth_val)


def build_exp1(sources: DatasetSources, paths: Paths,
               classes: ClassScheme) -> Path:
    """Three-class base dataset. Aspilanta is deliberately absent.

    This is the training set for the stage-1 detector and the reference point
    the zero-shot diagnostic needs: a model that has provably never seen the
    target species.
    """
    root = safe_clear_dir(paths.exp1_dataset)
    ensure_yolo_dirs(root)

    train_img, train_lbl = get_split_dirs(sources.original_root, "train")
    val_img, val_lbl = get_split_dirs(sources.original_root, "val")

    copy_folder_yolo(train_img, train_lbl,
                     root / "images/train", root / "labels/train",
                     prefix="orig_train_")
    if val_img and val_lbl:
        copy_folder_yolo(val_img, val_lbl,
                         root / "images/val", root / "labels/val",
                         prefix="orig_val_")

    yaml_path = write_yaml(root, paths.work / "exp1.yaml", classes.base)
    print_dataset_counts("Experiment 1 (3-class base)", root)
    return yaml_path


def build_exp2_zeroshot(sources: DatasetSources, paths: Paths) -> Path:
    """Single-class synthetic evaluation set for the zero-shot diagnostic.

    The three-class model is run over synthetic trap images that do contain
    Aspilanta, to record what it predicts for a species it was never trained
    on. Labels keep the generator's own numbering (0 = Aspilanta) and the YAML
    declares one class, because the model being evaluated has no slot for the
    target class at all — the point is to show the failure, not to score it.

    Ultralytics still insists on a train split, so one placeholder image is
    copied across.
    """
    root = safe_clear_dir(paths.exp2_testset)
    ensure_yolo_dirs(root)

    _, synth_val_lbl = get_split_dirs(sources.synthetic_root, "val")
    for img in sources.synth_val:
        copy_one_image_and_label(
            img, root / "images/val", root / "labels/val",
            prefix="synth_val_",
            label_path=(synth_val_lbl / f"{img.stem}.txt") if synth_val_lbl else None,
            label_roots=[sources.synthetic_root])

    val_images = sorted(list_images(root / "images/val"))
    if val_images:
        placeholder = val_images[0]
        import shutil
        shutil.copy2(placeholder, root / "images/train" / placeholder.name)
        shutil.copy2(root / "labels/val" / f"{placeholder.stem}.txt",
                     root / "labels/train" / f"{placeholder.stem}.txt")

    yaml_path = write_yaml(root, paths.work / "exp2_eval.yaml", ["Aspilanta"])
    print_dataset_counts("Experiment 2 (zero-shot eval set)", root)
    return yaml_path


def build_exp3(sources: DatasetSources, paths: Paths,
               classes: ClassScheme, out_root: Path | None = None,
               yaml_name: str = "exp3.yaml",
               include_synthetic: bool = True,
               synthetic_root_override: Path | None = None,
               synth_train_override: Sequence[Path] | None = None,
               synth_val_override: Sequence[Path] | None = None) -> Path:
    """Four-class merged dataset — what the proposed solution trains on.

    Three sources are merged, each with a different label treatment:

    * original three classes — ids 0/1/2 pass through unchanged;
    * real Aspilanta — ``force_class_id`` overrides whatever the source labels
      say, because those files carry single-class numbering;
    * synthetic — generator id 0 maps to the target id, and id 1 (the eu_moth
      distractors) is dropped.

    The override arguments exist so the ablation runner can swap in a variant
    generator's output, or omit synthetic data entirely, without duplicating
    this function.
    """
    root = safe_clear_dir(out_root or paths.exp3_dataset)
    ensure_yolo_dirs(root)

    train_img, train_lbl = get_split_dirs(sources.original_root, "train")
    val_img, val_lbl = get_split_dirs(sources.original_root, "val")

    copy_folder_yolo(train_img, train_lbl,
                     root / "images/train", root / "labels/train",
                     prefix="orig_train_")
    if val_img and val_lbl:
        copy_folder_yolo(val_img, val_lbl,
                         root / "images/val", root / "labels/val",
                         prefix="orig_val_")

    for img in sources.real_train:
        copy_one_image_and_label(img, root / "images/train", root / "labels/train",
                                 prefix="real_train_",
                                 force_class_id=classes.target_id,
                                 label_roots=[sources.real_aspilanta_root])
    for img in sources.real_val:
        copy_one_image_and_label(img, root / "images/val", root / "labels/val",
                                 prefix="real_val_",
                                 force_class_id=classes.target_id,
                                 label_roots=[sources.real_aspilanta_root])

    if include_synthetic:
        synth_root = synthetic_root_override or sources.synthetic_root
        synth_train = (list(synth_train_override) if synth_train_override is not None
                       else sources.synth_train)
        synth_val = (list(synth_val_override) if synth_val_override is not None
                     else sources.synth_val)
        for img in synth_train:
            copy_one_image_and_label(img, root / "images/train", root / "labels/train",
                                     prefix="synth_train_",
                                     class_id_map=classes.synthetic_class_map,
                                     drop_class_ids=classes.synthetic_drop_ids,
                                     label_roots=[synth_root])
        for img in synth_val:
            copy_one_image_and_label(img, root / "images/val", root / "labels/val",
                                     prefix="synth_val_",
                                     class_id_map=classes.synthetic_class_map,
                                     drop_class_ids=classes.synthetic_drop_ids,
                                     label_roots=[synth_root])

    yaml_path = write_yaml(root, paths.work / yaml_name, classes.full)
    print_dataset_counts("Experiment 3 (4-class merged)", root)
    return yaml_path


def build_target_domain_val(sources: DatasetSources, paths: Paths,
                            classes: ClassScheme) -> Path:
    """The common four-class validation set.

    All three experiments are scored here so the numbers are comparable.
    Experiment 1 is a three-class model evaluated against a four-class set and
    will score exactly zero on Aspilanta — that is the correct and expected
    result, not a bug, and it is the whole point of the zero-shot diagnostic.
    """
    root = safe_clear_dir(paths.target_domain_val)
    ensure_yolo_dirs(root)

    for img in sources.real_val:
        copy_one_image_and_label(img, root / "images/val", root / "labels/val",
                                 prefix="real_val_",
                                 force_class_id=classes.target_id,
                                 label_roots=[sources.real_aspilanta_root])
    for img in sources.synth_val:
        copy_one_image_and_label(img, root / "images/val", root / "labels/val",
                                 prefix="synth_val_",
                                 class_id_map=classes.synthetic_class_map,
                                 drop_class_ids=classes.synthetic_drop_ids,
                                 label_roots=[sources.synthetic_root])

    val_images = sorted(list_images(root / "images/val"))
    if val_images:
        import shutil
        placeholder = val_images[0]
        shutil.copy2(placeholder, root / "images/train" / placeholder.name)
        shutil.copy2(root / "labels/val" / f"{placeholder.stem}.txt",
                     root / "labels/train" / f"{placeholder.stem}.txt")

    yaml_path = write_yaml(root, paths.work / "target_domain_val.yaml", classes.full)
    print_dataset_counts("Target-domain validation", root)
    return yaml_path


def build_all(sources: DatasetSources, paths: Paths,
              classes: ClassScheme) -> dict[str, Path]:
    """Build every dataset and return a mapping of name to YAML path."""
    return {
        "exp1": build_exp1(sources, paths, classes),
        "exp2": build_exp2_zeroshot(sources, paths),
        "exp3": build_exp3(sources, paths, classes),
        "target_val": build_target_domain_val(sources, paths, classes),
    }
