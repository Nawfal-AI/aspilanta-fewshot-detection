"""Reading, remapping and copying YOLO-format datasets.

These are the helpers the notebook defined in "CELL 3 — Utility functions" and
then redefined in Section 11.1 with ``drop_class_ids`` support. Only the later,
corrected versions are kept here: the earlier ones remapped the generator's
distractor class into a real class id, which is exactly the labelling defect
Section 11.1 documents and fixes.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = [
    "IMG_EXTS",
    "list_images",
    "safe_clear_dir",
    "ensure_yolo_dirs",
    "get_split_dirs",
    "is_yolo_root",
    "find_label_for_image",
    "read_yolo_lines",
    "remap_label_lines",
    "copy_one_image_and_label",
    "copy_folder_yolo",
    "write_yaml",
    "count_images",
    "count_nonempty_labels",
    "class_histogram",
    "print_dataset_counts",
    "yolo_xywh_to_xyxy",
    "strip_distractor_labels",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def list_images(folder: str | Path) -> list[Path]:
    """Every image under ``folder``, recursively, sorted, case-insensitive."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def safe_clear_dir(path: str | Path) -> Path:
    """Delete ``path`` if it exists and recreate it empty."""
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_yolo_dirs(root: str | Path, splits: Sequence[str] = ("train", "val")) -> Path:
    root = Path(root)
    for split in splits:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    return root


def get_split_dirs(root: str | Path, split: str) -> tuple[Path | None, Path | None]:
    """Locate ``images/<split>`` and ``labels/<split>``.

    Tolerates the ``val`` / ``valid`` / ``validation`` naming variation that
    turns up in datasets exported by different tools.
    """
    root = Path(root)
    names = ["val", "valid", "validation"] if split == "val" else [split]
    for name in names:
        img, lbl = root / "images" / name, root / "labels" / name
        if img.exists() and lbl.exists():
            return img, lbl
    return None, None


def is_yolo_root(root: str | Path) -> bool:
    """True if ``root`` looks like a YOLO dataset with a non-empty train split."""
    img_train, lbl_train = get_split_dirs(root, "train")
    return (img_train is not None and lbl_train is not None
            and len(list_images(img_train)) > 0)


def find_label_for_image(img_path: str | Path,
                         possible_roots: Iterable[str | Path] | None = None) -> Path | None:
    """Find the ``.txt`` label belonging to an image.

    Checks the conventional sibling and ``labels/`` locations first, then
    falls back to a recursive search under any supplied roots.
    """
    img_path = Path(img_path)
    candidates = [
        img_path.with_suffix(".txt"),
        img_path.parent / f"{img_path.stem}.txt",
        img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
        img_path.parent.parent / "labels" / "train" / f"{img_path.stem}.txt",
        img_path.parent.parent / "labels" / "val" / f"{img_path.stem}.txt",
        img_path.parent.parent / "labels" / "valid" / f"{img_path.stem}.txt",
    ]
    for root in possible_roots or []:
        root = Path(root)
        if root.exists():
            candidates.extend(root.rglob(f"{img_path.stem}.txt"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def read_yolo_lines(label_path: str | Path | None) -> list[str]:
    if label_path is None:
        return []
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    return text.splitlines() if text else []


def remap_label_lines(lines: Iterable[str],
                      class_id_map: dict[int, int] | None = None,
                      force_class_id: int | None = None,
                      drop_class_ids: Iterable[int] | None = None) -> list[str]:
    """Translate class ids in YOLO label lines.

    Args:
        lines: raw ``"cls cx cy w h"`` lines.
        class_id_map: per-id translation applied when the id is present.
        force_class_id: overrides every id, used for real Aspilanta images
            whose source labels carry the generator's own numbering.
        drop_class_ids: ids removed entirely rather than remapped.

    ``drop_class_ids`` is the fix from Section 11.1. The generator writes
    ``1 = eu_moth`` distractors as visual clutter, not as ground truth;
    remapping them into the four-class scheme silently collides with MP (id 0)
    or NT (id 1) depending on the map, inflating those classes with boxes that
    were never meant to be positives.
    """
    drop = set(drop_class_ids or ())
    out: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        if cid in drop:
            continue
        if force_class_id is not None:
            parts[0] = str(force_class_id)
        elif class_id_map is not None and cid in class_id_map:
            parts[0] = str(class_id_map[cid])
        out.append(" ".join(parts[:5]))
    return out


def yolo_xywh_to_xyxy(box: Sequence[float], width: int, height: int,
                      padding: float = 0.05) -> tuple[int, int, int, int]:
    """Normalised centre-form box to absolute corners, with optional padding.

    ``padding`` expands the box by a fraction of its own size, which is what
    the CLIP crop extraction uses so the crop keeps a little context around
    the insect instead of clipping its wings.
    """
    xc, yc, bw, bh = box
    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)))


# --------------------------------------------------------------------------- #
# Copying
# --------------------------------------------------------------------------- #
def copy_one_image_and_label(img_path: str | Path,
                             dst_images: str | Path,
                             dst_labels: str | Path,
                             prefix: str = "",
                             label_path: str | Path | None = None,
                             class_id_map: dict[int, int] | None = None,
                             force_class_id: int | None = None,
                             label_roots: Iterable[str | Path] | None = None,
                             drop_class_ids: Iterable[int] | None = None) -> bool:
    """Copy one image and its remapped label into a destination split.

    ``prefix`` keeps provenance visible in the filename (``real_train_``,
    ``synth_val_``, ``orig_train_``). Several later stages rely on it —
    :func:`strip_distractor_labels` identifies synthetic files by prefix, and
    the test-split carve in Section 11.2 stratifies on it.

    Returns False when no label could be found, so callers can count misses.
    """
    img_path = Path(img_path)
    dst_images, dst_labels = Path(dst_images), Path(dst_labels)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if label_path is None:
        label_path = find_label_for_image(img_path, possible_roots=label_roots)

    new_stem = f"{prefix}{img_path.stem}"
    shutil.copy2(img_path, dst_images / f"{new_stem}{img_path.suffix}")
    target_label = dst_labels / f"{new_stem}.txt"

    if label_path is None or not Path(label_path).exists():
        target_label.write_text("")
        return False

    lines = remap_label_lines(read_yolo_lines(label_path),
                              class_id_map=class_id_map,
                              force_class_id=force_class_id,
                              drop_class_ids=drop_class_ids)
    target_label.write_text("\n".join(lines))
    return True


def copy_folder_yolo(src_images: str | Path, src_labels: str | Path,
                     dst_images: str | Path, dst_labels: str | Path,
                     prefix: str = "",
                     class_id_map: dict[int, int] | None = None,
                     force_class_id: int | None = None,
                     drop_class_ids: Iterable[int] | None = None) -> tuple[int, int]:
    """Copy a whole split. Returns ``(copied, missing_labels)``."""
    src_images, src_labels = Path(src_images), Path(src_labels)
    if not src_images.exists():
        print(f"WARNING: missing image folder, skipped: {src_images}")
        return 0, 0

    copied = missing = 0
    for img in sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMG_EXTS):
        ok = copy_one_image_and_label(
            img, dst_images, dst_labels, prefix=prefix,
            label_path=src_labels / f"{img.stem}.txt",
            class_id_map=class_id_map, force_class_id=force_class_id,
            drop_class_ids=drop_class_ids)
        copied += 1
        missing += (not ok)
    if missing:
        print(f"WARNING: {missing}/{copied} images from {src_images} had no label.")
    return copied, missing


def write_yaml(dataset_root: str | Path, yaml_path: str | Path,
               names: Sequence[str], quiet: bool = False) -> Path:
    """Write an Ultralytics dataset YAML pointing at ``dataset_root``."""
    dataset_root = Path(dataset_root).resolve()
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = (f"path: {dataset_root}\n"
               f"train: images/train\n"
               f"val: images/val\n"
               f"nc: {len(names)}\n"
               f"names: {list(names)}\n")
    yaml_path.write_text(content)
    if not quiet:
        print(f"Wrote YAML: {yaml_path}\n{content}")
    return yaml_path


# --------------------------------------------------------------------------- #
# Counting and verification
# --------------------------------------------------------------------------- #
def count_images(folder: str | Path) -> int:
    return len(list_images(folder))


def count_nonempty_labels(folder: str | Path) -> int:
    folder = Path(folder)
    if not folder.exists():
        return 0
    return sum(1 for f in folder.glob("*.txt") if f.read_text().strip())


def class_histogram(label_dir: str | Path,
                    names: Sequence[str] | None = None) -> dict:
    """Count boxes per class id in a label directory.

    Worth running after every dataset build — it is how the Section 11.1
    labelling defect became visible in the first place.
    """
    counts: Counter[int] = Counter()
    for f in Path(label_dir).glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            if line.strip():
                counts[int(float(line.split()[0]))] += 1
    if names is None:
        return dict(sorted(counts.items()))
    return {names[k] if k < len(names) else k: v for k, v in sorted(counts.items())}


def print_dataset_counts(name: str, root: str | Path,
                         splits: Sequence[str] = ("train", "val")) -> None:
    root = Path(root)
    print(f"\n{name}: {root}")
    for split in splits:
        print(f"  {split} images: {count_images(root / 'images' / split)}")
        print(f"  {split} labels with objects: "
              f"{count_nonempty_labels(root / 'labels' / split)}")


def strip_distractor_labels(dataset_root: str | Path,
                            moth_class_ids: Iterable[int] = (1,),
                            synthetic_marker: str = "synth") -> dict[str, int]:
    """Remove distractor boxes from an already-built dataset in place.

    This is the Section 11.1 repair applied without a full rebuild. Synthetic
    files are identified by the ``synth`` prefix set at copy time. The correct
    id to pass differs per split because the original merge mapped the moths
    differently in each: in the training set they became 0 (MP), in the
    validation set they kept id 1 (NT).
    """
    root = Path(dataset_root)
    removed = kept = touched = 0
    for split in ("train", "val", "test"):
        label_dir = root / "labels" / split
        if not label_dir.exists():
            continue
        for f in sorted(label_dir.glob("*.txt")):
            if synthetic_marker not in f.stem.lower():
                continue
            out = []
            for line in f.read_text().strip().splitlines():
                parts = line.split()
                if not parts:
                    continue
                if int(float(parts[0])) in set(moth_class_ids):
                    removed += 1
                else:
                    out.append(line)
                    kept += 1
            f.write_text("\n".join(out))
            touched += 1
    print(f"  {root.name}: {touched} synthetic label files touched, "
          f"{removed} distractor boxes removed, {kept} kept")
    return {"files": touched, "removed": removed, "kept": kept}
