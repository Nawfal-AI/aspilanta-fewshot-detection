"""Dataset discovery and split construction.

Two jobs live here:

* finding datasets whose exact location varies between Kaggle, Colab and a
  local checkout (the notebook did this with long hard-coded candidate lists);
* carving reproducible splits, including the decision-held-out test split
  introduced in Section 11.2.
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

from .yolo_io import is_yolo_root, list_images

__all__ = [
    "find_yolo_dataset",
    "find_real_aspilanta_root",
    "split_train_val_test",
    "stratified_pick",
    "carve_test_split",
]

_DATASET_DIR_NAMES = {
    "yolo_dataset", "dataset", "original_dataset",
    "generated_dataset", "generated_dataset_v7", "synthetic_dataset",
}
_ASPILANTA_KEYWORDS = ("ash", "asph", "ashpilanta", "aspilanta")


def find_yolo_dataset(candidates: Iterable[str | Path], name: str,
                      search_roots: Iterable[str | Path] = ()) -> Path:
    """Return the first candidate that is a usable YOLO dataset root.

    Falls back to scanning ``search_roots`` for directories with recognisable
    dataset names. Raises with the full list of locations checked, which is
    far more useful than a bare "not found" when the dataset happens to be
    mounted one level deeper than expected.
    """
    checked: list[str] = []

    for candidate in candidates:
        candidate = Path(candidate)
        checked.append(str(candidate))
        if candidate.exists() and is_yolo_root(candidate):
            print(f"Found {name}: {candidate}")
            return candidate

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and path.name in _DATASET_DIR_NAMES:
                checked.append(str(path))
                if is_yolo_root(path):
                    print(f"Found {name}: {path}")
                    return path

    raise FileNotFoundError(
        f"Could not find {name}. It must contain images/train and labels/train.\n"
        "Checked:\n" + "\n".join(checked[:80]))


def find_real_aspilanta_root(candidates: Iterable[str | Path],
                             search_roots: Iterable[str | Path] = (),
                             max_images: int = 1000) -> Path:
    """Locate the folder of real Aspilanta photographs.

    The keyword search tolerates the inconsistent spelling used across the
    project's own data uploads ("ashpilanta" vs "aspilanta"). ``max_images``
    guards against matching a huge unrelated directory whose path happens to
    contain one of the keywords.
    """
    checked: list[str] = []

    for candidate in candidates:
        candidate = Path(candidate)
        checked.append(str(candidate))
        images = list_images(candidate)
        if candidate.exists() and images:
            print(f"Found real Aspilanta folder: {candidate} ({len(images)} images)")
            return candidate

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            if any(keyword in str(path).lower() for keyword in _ASPILANTA_KEYWORDS):
                images = list_images(path)
                checked.append(str(path))
                if 1 <= len(images) <= max_images:
                    print(f"Found real Aspilanta folder: {path} ({len(images)} images)")
                    return path

    raise FileNotFoundError(
        "Could not find real Aspilanta images.\n"
        "Set paths.real_aspilanta_yolo in your config.\n"
        "Checked:\n" + "\n".join(checked[:80]))


def split_train_val_test(items: Sequence[Path], train: float = 0.70,
                         val: float = 0.15, seed: int = 42
                         ) -> tuple[list[Path], list[Path], list[Path]]:
    """Deterministic three-way split.

    Equivalent to the notebook's two chained ``train_test_split`` calls with
    ``random_state=42``, but without the scikit-learn dependency and without
    the silent assumption that val and test are the same size.
    """
    items = sorted(items, key=lambda p: p.name)
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)

    n_train = int(round(len(items) * train))
    n_val = int(round(len(items) * val))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def pick(idx: list[int]) -> list[Path]:
        return [items[i] for i in sorted(idx)]

    return pick(train_idx), pick(val_idx), pick(test_idx)


def stratified_pick(items: Sequence[Path], frac: float,
                    rng: random.Random) -> tuple[list[Path], list[Path]]:
    """Split ``items`` into a held-out fraction and the remainder.

    Sorted by name before shuffling so the result depends only on the seed,
    not on filesystem iteration order.
    """
    items = sorted(items, key=lambda p: p.name)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    n = max(1, int(round(len(items) * frac))) if items else 0
    chosen = set(indices[:n])
    return ([items[i] for i in sorted(chosen)],
            [items[i] for i in range(len(items)) if i not in chosen])


def carve_test_split(source_root: str | Path, test_root: str | Path,
                     val_root: str | Path, class_names: Sequence[str],
                     test_fraction: float = 0.30, seed: int = 1337,
                     real_prefix: str = "real_") -> dict:
    """Split a validation set into a tuning half and a held-out test half.

    Section 11.2 of the research notebook. The proposed configuration was chosen partly by
    reading ablation numbers off the validation set, so the validation set can
    no longer support an unbiased final claim. This carves a test split that
    nothing is tuned on.

    The split is stratified on provenance: real and synthetic images are
    sampled separately so both halves keep the same real/synthetic mix. With
    only eleven real photographs in total, an unstratified draw could easily
    put nearly all of them on one side.

    Ultralytics requires a ``train`` key to exist in the dataset YAML even for
    a validation-only run, so a single placeholder image is copied into each
    output's train split.
    """
    source_root = Path(source_root)
    src_img = source_root / "images" / "val"
    src_lbl = source_root / "labels" / "val"

    all_images = sorted(list_images(src_img))
    real = [p for p in all_images if p.stem.lower().startswith(real_prefix)]
    synthetic = [p for p in all_images if not p.stem.lower().startswith(real_prefix)]
    print(f"Source: {len(all_images)} images "
          f"({len(real)} real, {len(synthetic)} synthetic)")

    rng = random.Random(seed)
    test_real, val_real = stratified_pick(real, test_fraction, rng)
    test_synth, val_synth = stratified_pick(synthetic, test_fraction, rng)
    test_set = test_real + test_synth
    val_set = val_real + val_synth

    def materialise(images: Sequence[Path], dst_root: str | Path,
                    split_name: str = "val") -> int:
        dst_root = Path(dst_root)
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_images = dst_root / "images" / split_name
        dst_labels = dst_root / "labels" / split_name
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        (dst_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

        written = 0
        for p in images:
            label = src_lbl / f"{p.stem}.txt"
            if not label.exists():
                continue
            shutil.copy2(p, dst_images / p.name)
            shutil.copy2(label, dst_labels / label.name)
            written += 1

        if images:  # placeholder train image, required by Ultralytics
            first = images[0]
            shutil.copy2(dst_images / first.name,
                         dst_root / "images" / "train" / first.name)
            shutil.copy2(dst_labels / f"{first.stem}.txt",
                         dst_root / "labels" / "train" / f"{first.stem}.txt")
        return written

    from .yolo_io import write_yaml

    n_test = materialise(test_set, test_root)
    n_val = materialise(val_set, val_root)
    test_root, val_root = Path(test_root), Path(val_root)
    yaml_test = write_yaml(test_root, test_root.parent / "target_domain_test.yaml",
                           class_names, quiet=True)
    yaml_val = write_yaml(val_root, val_root.parent / "target_domain_valonly.yaml",
                          class_names, quiet=True)

    print(f"TEST: {n_test} images ({len(test_real)} real, {len(test_synth)} synthetic)")
    print(f"VAL : {n_val} images ({len(val_real)} real, {len(val_synth)} synthetic)")
    print(f"Deterministic (seed={seed}); re-running reproduces it exactly.")
    print("Use the val half for tuning, the test half only for the final number.")

    return {"n_test": n_test, "n_val": n_val,
            "yaml_test": yaml_test, "yaml_val": yaml_val,
            "test_images": test_set, "val_images": val_set}
