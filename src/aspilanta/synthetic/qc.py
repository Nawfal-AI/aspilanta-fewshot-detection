"""Quality-control grids for cutouts and generated composites.

Synthetic data fails silently. A generator with a broken label transform still
writes 2,000 plausible-looking JPEGs, and the problem only surfaces as an
unexplained accuracy ceiling many GPU-hours later. These grids draw the labels
back onto the images, so a defect is visible in seconds.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

__all__ = ["cutout_grid", "verification_grid"]


def cutout_grid(cutout_paths: Sequence[str | Path], out_path: str | Path,
                n: int = 24, cols: int = 6, title: str = "Cutouts",
                checker: bool = True) -> Path:
    """Render extracted cutouts over a checkerboard so alpha errors show.

    A cutout that kept a rectangle of its original background is invisible on
    a white page and obvious against a checkerboard.
    """
    import matplotlib.pyplot as plt

    paths = list(cutout_paths)[:n]
    if not paths:
        raise ValueError("No cutouts to render.")
    rows = (len(paths) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).flatten()

    for ax, path in zip(axes, paths):
        bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            ax.axis("off")
            continue
        if bgra.shape[2] == 4:
            rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            alpha = bgra[:, :, 3].astype(np.float32)[..., None] / 255
            if checker:
                h, w = rgb.shape[:2]
                size = max(4, min(h, w) // 12)
                ys, xs = np.mgrid[0:h, 0:w]
                board = (((ys // size) + (xs // size)) % 2).astype(np.float32)
                background = (0.75 + 0.15 * board)[..., None].repeat(3, axis=2)
            else:
                background = np.ones_like(rgb)
            ax.imshow(np.clip(rgb * alpha + background * (1 - alpha), 0, 1))
        else:
            ax.imshow(cv2.cvtColor(bgra, cv2.COLOR_BGR2RGB))
        ax.set_title(Path(path).stem[:18], fontsize=6)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Cutout grid: {out_path}")
    return out_path


def verification_grid(dataset_root: str | Path, out_path: str | Path,
                      n: int = 16, cols: int = 4, split: str = "train",
                      seed: int | None = None,
                      class_colors: dict[int, tuple[int, int, int]] | None = None) -> Path:
    """Draw ground-truth boxes onto sampled generated images.

    Per-panel counts are printed in the title, so a systematic mismatch
    between what was placed and what was labelled is immediately apparent.
    """
    import matplotlib.pyplot as plt

    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {img_dir}")

    rng = random.Random(seed)
    sample = rng.sample(images, min(n, len(images)))
    colors = class_colors or {0: (0, 220, 0), 1: (0, 60, 220)}
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, img_path in zip(axes, sample):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        counts: dict[int, int] = {}
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                cv2.rectangle(img,
                              (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                              (int((cx + bw / 2) * w), int((cy + bh / 2) * h)),
                              colors.get(cid, (255, 255, 0)), 2)
                counts[cid] = counts.get(cid, 0) + 1
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title("  ".join(f"c{k}={v}" for k, v in sorted(counts.items())) or "empty",
                     fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"Generated dataset QC - {dataset_root.name} / {split}\n"
                 "green = Aspilanta (0), blue = distractor (1)", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"QC grid: {out_path}")
    return out_path
