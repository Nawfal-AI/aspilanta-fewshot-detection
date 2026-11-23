"""The v7 synthetic dataset generator.

This is the data-oriented component of the method: 44 real Aspilanta cutouts
become 2,000 composite trap images containing 9,211 labelled instances.

The notebook carried two near-identical copies of this code - the main
generator (cell 41) and a "self-contained variant generator" for the ablations
(cell 113). They are one function here, parameterised by
:class:`~aspilanta.config.GeneratorConfig`, so the ablation variants are
config changes rather than a forked 500-line copy that can drift.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from ..config import GeneratorConfig
from .compositing import (
    add_sheen,
    augment_background,
    paste_insect,
    sample_size_px,
    soft_iou,
    validate_label,
)
from .pad import extract_clean_pad

__all__ = ["load_bgra_images", "generate_dataset", "GeneratorResult"]


def load_bgra_images(paths: Sequence[str | Path], label: str = "") -> list[np.ndarray]:
    """Load cutouts as BGRA, promoting any that lost their alpha channel."""
    images = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3:
            continue
        if img.shape[2] == 4:
            images.append(img)
        elif img.shape[2] == 3:
            bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = 255
            images.append(bgra)
    if label:
        print(f"  {label:<25} {len(images)} cutouts loaded")
    return images


class GeneratorResult(dict):
    """Statistics from one generation run, also written to ``stats.json``."""

    @property
    def total(self) -> int:
        return self.get("total", 0)

    @property
    def instances(self) -> int:
        return self.get("n_aspilanta", 0)


def _sample_instance_counts(rng: random.Random, config: GeneratorConfig,
                            has_distractors: bool) -> tuple[int, int]:
    """Choose how many insects go on this card.

    The Aspilanta distribution is deliberately long-tailed rather than
    uniform: mostly sparse cards, sometimes crowded ones, and 10% empty. That
    shape is what a real monitoring season looks like, and training on a
    uniform count distribution biases the detector's implicit prior about how
    many objects an image should contain.
    """
    r = rng.random()
    if r < 0.10:
        n_aspilanta = 0
    elif r < 0.45:
        n_aspilanta = rng.randint(1, 3)
    elif r < 0.80:
        n_aspilanta = rng.randint(4, 7)
    else:
        n_aspilanta = rng.randint(8, config.max_aspilanta)

    if not has_distractors or not config.use_distractors:
        n_distractor = 0
    else:
        n_distractor = rng.choices(
            [0, 1, 2, 3, config.max_distractor],
            weights=[0.50, 0.22, 0.14, 0.09, 0.05])[0]
    return n_aspilanta, n_distractor


def _place_instances(canvas, sources, count, class_id, placed, labels, stats,
                     canvas_w, canvas_h, config, rng, size_scale=(1.0, 1.0),
                     device=None):
    """Try to place ``count`` instances of one class, rejecting bad placements.

    Each instance gets up to ``max_attempts`` tries. An attempt fails if the
    box overlaps an existing one beyond ``max_overlap_iou``, the paste does not
    fit, or the resulting label fails validation. Failing quietly and moving on
    is intentional: a card that ends up with 6 insects instead of the 7 that
    were requested is fine, whereas forcing the 7th produces either an
    occluded box or an off-frame label.
    """
    for _ in range(count):
        source = rng.choice(sources)
        for _ in range(config.max_attempts):
            width = sample_size_px(canvas_w, config.pad_width_mm,
                                   config.insect_mean_mm, config.insect_std_mm,
                                   config.insect_min_mm, config.insect_max_mm)
            if size_scale != (1.0, 1.0):
                width = int(width * rng.uniform(*size_scale))
            width = max(4, min(width, canvas_w // 3))

            margin = max(5, width)
            if canvas_w <= 2 * margin or canvas_h <= 2 * margin:
                break
            px = rng.randint(margin, canvas_w - margin)
            py = rng.randint(margin, canvas_h - margin)

            angle = rng.uniform(0, 360)
            rotation = cv2.getRotationMatrix2D(
                (source.shape[1] // 2, source.shape[0] // 2), angle, 1.0)
            rotated = cv2.warpAffine(source, rotation,
                                     (source.shape[1], source.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))

            height = int(width * rotated.shape[0] / max(rotated.shape[1], 1))
            box = [px, py, px + width, py + height]
            if any(soft_iou(box, other) > config.max_overlap_iou for other in placed):
                continue

            canvas, bbox = paste_insect(
                canvas, rotated, width, (px, py),
                use_color_jitter=config.use_color_jitter,
                use_lighting_match=config.use_lighting_match and config.use_pad_aware,
                use_shadow=config.use_shadow,
                use_poisson=config.use_poisson,
                organic_blur_kernel=config.organic_blur_kernel,
                organic_blur_sigma=config.organic_blur_sigma,
                motion_blur_prob=config.motion_blur_prob,
                device=device)
            if bbox is None:
                continue

            if not validate_label(*bbox, min_size=config.label_min_size,
                                  max_size=config.label_max_size):
                stats["labels_rejected"] += 1
                continue

            placed.append(box)
            labels.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} "
                          f"{bbox[2]:.6f} {bbox[3]:.6f}")
            if class_id == 0:
                stats["n_aspilanta"] += 1
                stats["size_px_list"].append(width)
            else:
                stats["n_distractor"] += 1
            break
    return canvas


def generate_dataset(aspilanta_cutouts: Sequence[str | Path],
                     distractor_cutouts: Sequence[str | Path],
                     backgrounds_dir: str | Path,
                     output_dir: str | Path,
                     config: GeneratorConfig | None = None,
                     progress_every: int = 200) -> GeneratorResult:
    """Generate a synthetic YOLO dataset.

    Args:
        aspilanta_cutouts: paths to BGRA target cutouts.
        distractor_cutouts: paths to BGRA distractor cutouts. Pass an empty
            list, or set ``config.use_distractors = False``, for the
            no-distractors ablation.
        backgrounds_dir: folder of real trap-card photographs.
        output_dir: destination YOLO root.
        config: generator settings; defaults reproduce the project's dataset.

    Returns a :class:`GeneratorResult` of run statistics, also written to
    ``stats.json`` alongside the dataset.
    """
    config = config or GeneratorConfig()
    output_dir = Path(output_dir)
    rng = random.Random(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = None

    print("=" * 68)
    print("ASPILANTA SYNTHETIC DATASET GENERATOR  v7")
    print(f"  poisson={config.use_poisson}  distractors={config.use_distractors}  "
          f"pad_aware={config.use_pad_aware}")
    print(f"  target={config.num_images} images  seed={config.seed}  device={device}")
    print("=" * 68)

    aspilanta = load_bgra_images(aspilanta_cutouts, "Aspilanta")
    distractors = (load_bgra_images(distractor_cutouts, "Distractors")
                   if config.use_distractors else [])
    if not aspilanta:
        raise RuntimeError("No Aspilanta cutouts loaded - run cutout extraction first.")
    if config.use_distractors and not distractors:
        print("WARNING: no distractor cutouts found; distractors will be skipped.")

    background_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        background_paths.extend(Path(backgrounds_dir).rglob(pattern))
        background_paths.extend(Path(backgrounds_dir).rglob(pattern.upper()))
    background_paths = sorted(set(background_paths))
    if not background_paths:
        raise RuntimeError(f"No background images found in {backgrounds_dir}")
    print(f"  {len(background_paths)} backgrounds found\n")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "with_aspilanta": 0, "with_distractor": 0, "empty": 0,
             "n_aspilanta": 0, "n_distractor": 0, "pads_found": 0,
             "pads_fallback": 0, "labels_rejected": 0, "size_px_list": []}

    generated = attempts = 0
    max_attempts = config.num_images * 6

    while generated < config.num_images:
        attempts += 1
        if attempts > max_attempts:
            print("Warning: stopping early (too many rejected attempts).")
            break

        split = "train" if generated < int(config.num_images * config.train_ratio) else "val"

        background = cv2.imread(str(rng.choice(background_paths)))
        if background is None:
            continue
        background = augment_background(background)

        if config.extract_pad and config.use_pad_aware:
            canvas, (_, _, canvas_w, canvas_h), pad_found = extract_clean_pad(
                background, config.pad_min_frac, config.pad_max_frac)
            stats["pads_found" if pad_found else "pads_fallback"] += 1
        else:
            # no_pad_aware ablation: paste anywhere on the raw photograph,
            # including the card border and the surrounding scene.
            canvas = background.copy()
            canvas_h, canvas_w = background.shape[:2]
            stats["pads_fallback"] += 1

        if canvas_w < 50 or canvas_h < 50:
            continue

        if rng.random() < config.sheen_prob:
            canvas = add_sheen(canvas)

        n_aspilanta, n_distractor = _sample_instance_counts(
            rng, config, bool(distractors))

        placed: list[list[int]] = []
        labels: list[str] = []

        # The generator writes its own two-class scheme: 0 = Aspilanta,
        # 1 = distractor. Remapping into the four-class scheme happens later,
        # in aspilanta.data.builders.
        canvas = _place_instances(canvas, aspilanta, n_aspilanta, 0,
                                  placed, labels, stats, canvas_w, canvas_h,
                                  config, rng, device=device)
        if distractors and n_distractor:
            canvas = _place_instances(canvas, distractors, n_distractor, 1,
                                      placed, labels, stats, canvas_w, canvas_h,
                                      config, rng, size_scale=(0.8, 2.5),
                                      device=device)

        allow_empty = (generated % config.empty_image_every == 0)
        if not labels and not allow_empty:
            continue

        name = f"synth_{generated:06d}"
        cv2.imwrite(str(output_dir / "images" / split / f"{name}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality])
        (output_dir / "labels" / split / f"{name}.txt").write_text("\n".join(labels))

        stats["total"] += 1
        stats["with_aspilanta"] += int(any(line.startswith("0 ") for line in labels))
        stats["with_distractor"] += int(any(line.startswith("1 ") for line in labels))
        stats["empty"] += int(not labels)
        generated += 1

        if generated % progress_every == 0:
            mean_px = np.mean(stats["size_px_list"]) if stats["size_px_list"] else 0
            print(f"  [{generated:4d}/{config.num_images}]  "
                  f"Asp={stats['n_aspilanta']}  Dist={stats['n_distractor']}  "
                  f"Pads={stats['pads_found']}  MeanPx={mean_px:.1f}  "
                  f"Rejected={stats['labels_rejected']}")

    _write_manifest(output_dir, background_paths, stats, config)

    summary = GeneratorResult({k: v for k, v in stats.items() if k != "size_px_list"})
    if stats["size_px_list"]:
        sizes = np.asarray(stats["size_px_list"])
        summary["mean_px"] = round(float(sizes.mean()), 2)
        summary["median_px"] = round(float(np.median(sizes)), 2)
        summary["std_px"] = round(float(sizes.std()), 2)

    print(f"\n{'=' * 68}\nCOMPLETE\n{'=' * 68}")
    for key in ("total", "with_aspilanta", "with_distractor", "empty",
                "labels_rejected", "n_aspilanta", "n_distractor", "pads_found"):
        print(f"  {key:<18}: {stats[key]}")
    print(f"\n  Output: {output_dir}")
    return summary


def _write_manifest(output_dir: Path, background_paths: Sequence[Path],
                    stats: dict, config: GeneratorConfig) -> None:
    """Write ``dataset.yaml`` and ``stats.json``.

    The mean instance size is back-converted to millimetres so the dataset
    carries a human-checkable record of its own physical calibration - if that
    number drifts away from ``insect_mean_mm``, pad detection has gone wrong
    on a substantial share of the backgrounds.
    """
    mean_mm = config.insect_mean_mm
    if stats["size_px_list"]:
        try:
            sample = cv2.imread(str(background_paths[0]))
            _, (_, _, pad_w, _), _ = extract_clean_pad(sample)
        except Exception:
            pad_w = 600
        mean_mm = float(np.mean(stats["size_px_list"])) / max(
            pad_w / config.pad_width_mm, 1)

    (output_dir / "dataset.yaml").write_text(
        f"# Aspilanta synthetic detection dataset (generator v7)\n"
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: {config.class_names[0]}\n"
        f"  1: {config.class_names[1]}\n"
        f"# size: truncated Gaussian [{config.insect_min_mm}, "
        f"{config.insect_max_mm}] mm, generated mean ~{mean_mm:.2f} mm\n"
        f"# poisson={config.use_poisson} distractors={config.use_distractors} "
        f"pad_aware={config.use_pad_aware} seed={config.seed}\n")

    payload = {k: v for k, v in stats.items() if k != "size_px_list"}
    if stats["size_px_list"]:
        sizes = np.asarray(stats["size_px_list"])
        payload.update(mean_px=round(float(sizes.mean()), 2),
                       median_px=round(float(np.median(sizes)), 2),
                       std_px=round(float(sizes.std()), 2),
                       mean_mm=round(mean_mm, 2))
    payload["config"] = {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in config.__dict__.items()}
    (output_dir / "stats.json").write_text(json.dumps(payload, indent=2))
