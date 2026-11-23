"""OWL-ViT baseline — the foundation-model comparison.

Answers the obvious objection to this whole project: why build a synthetic
data pipeline when open-vocabulary detectors claim to find arbitrary objects
from a text prompt or a single example?

Two variants are evaluated on the same validation set as the proposed method:

* **zero-shot text** — prompted with generic insect vocabulary, no examples;
* **image-guided few-shot** — prompted with K real Aspilanta exemplar crops.

Both underperform the proposed pipeline by a wide margin. That is not a
criticism of OWL-ViT: a 5 mm moth occupying a few dozen pixels on a cluttered
adhesive card is far outside the object scale and domain these models are
trained for.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

import numpy as np

from ..evaluation.metrics import (
    box_iou_xyxy,
    load_target_gt_xyxy,
    match_detections,
    precision_recall_f1,
)

__all__ = ["OwlVitBaseline", "collect_exemplars", "evaluate_owlvit_variant"]

DEFAULT_MODEL = "google/owlvit-base-patch32"
DEFAULT_TEXT_QUERIES = ["insect", "small insect", "moth"]


def collect_exemplars(dataset_root: str | Path, k: int, target_class_id: int = 3,
                      seed: int = 42, real_prefix: str = "real_",
                      splits: Sequence[str] = ("train", "val")):
    """Collect K real target crops to use as image-guided prompts.

    Returns ``(crops, source_filenames)``. The filenames matter: the
    evaluation loop must exclude the images the exemplars came from, or the
    baseline is scored partly on its own prompts.

    Both splits are searched because the target-domain builder can leave very
    few real images in train — with eleven real photographs in total, insisting
    on train-only exemplars can yield zero.
    """
    import cv2
    from PIL import Image

    dataset_root = Path(dataset_root)
    crops, used = [], []

    for split in splits:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        if not img_dir.exists():
            continue

        candidates = sorted(p for p in img_dir.glob(f"{real_prefix}*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        random.Random(seed).shuffle(candidates)

        for path in candidates:
            if len(crops) >= k:
                break
            img = cv2.imread(str(path))
            if img is None:
                continue
            height, width = img.shape[:2]
            gt = load_target_gt_xyxy(lbl_dir / f"{path.stem}.txt", width, height,
                                     target_class_id)
            if len(gt) == 0:
                continue
            x1, y1, x2, y2 = gt[0].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(Image.fromarray(
                cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
            used.append(path.name)

        if len(crops) >= k:
            break

    print(f"Collected {len(crops)}/{k} exemplar crops from: {used}")
    return crops, set(used)


class OwlVitBaseline:
    """OWL-ViT wrapper handling the transformers API drift.

    ``use_fast=False`` is required, not stylistic: the fast processor on
    recent releases exposes neither ``post_process`` nor
    ``post_process_object_detection``, and the slow one does.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OwlViTProcessor.from_pretrained(model_name, use_fast=False)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"Loaded {model_name} on {self.device}")

    def _postprocess_text(self, outputs, target_sizes, score_thresh: float):
        """Call whichever text post-processing method this version exposes.

        The method has been renamed twice across transformers releases and the
        signatures differ, so each candidate is tried with its expected
        keywords and then without the threshold.
        """
        attempts = [
            ("post_process_grounded_object_detection",
             dict(outputs=outputs, threshold=score_thresh, target_sizes=target_sizes)),
            ("post_process_object_detection",
             dict(outputs=outputs, threshold=score_thresh, target_sizes=target_sizes)),
            ("post_process", dict(outputs=outputs, target_sizes=target_sizes)),
        ]
        for name, kwargs in attempts:
            fn = getattr(self.processor, name, None)
            if fn is None:
                continue
            try:
                return fn(**kwargs)[0]
            except (AttributeError, TypeError):
                try:
                    return fn(outputs=outputs, target_sizes=target_sizes)[0]
                except Exception:
                    continue
        raise RuntimeError(
            "No usable OWL-ViT text post-processing method. Available: "
            + str([m for m in dir(self.processor) if m.startswith("post_process")]))

    def detect_text(self, pil_image, text_queries: Sequence[str] = DEFAULT_TEXT_QUERIES,
                    score_thresh: float = 0.05):
        """Zero-shot detection from text prompts. No target examples involved."""
        import torch

        inputs = self.processor(text=[list(text_queries)], images=pil_image,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self._postprocess_text(outputs, target_sizes, score_thresh)
        return results["boxes"].cpu().numpy(), results["scores"].cpu().numpy()

    def detect_image_guided(self, pil_image, exemplar_crops: Sequence,
                            score_thresh: float = 0.5, nms_thresh: float = 0.3):
        """Few-shot detection from exemplar crops.

        OWL-ViT accepts one query image at a time, so each exemplar is run
        separately and the results are pooled under NMS. That pooling is why
        K=5 is not simply better than K=1: more exemplars means more
        independent chances to fire on the same background clutter.
        """
        import torch

        all_boxes, all_scores = [], []
        for exemplar in exemplar_crops:
            inputs = self.processor(query_images=exemplar, images=pil_image,
                                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.image_guided_detection(**inputs)
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)

            fn = getattr(self.processor, "post_process_image_guided_detection", None)
            if fn is None:
                raise RuntimeError("post_process_image_guided_detection unavailable "
                                   "in this transformers version.")
            try:
                results = fn(outputs=outputs, threshold=score_thresh,
                             nms_threshold=nms_thresh, target_sizes=target_sizes)[0]
            except TypeError:
                results = fn(outputs=outputs, target_sizes=target_sizes)[0]

            all_boxes.append(results["boxes"].cpu().numpy())
            all_scores.append(results["scores"].cpu().numpy())

        if not all_boxes or all(len(b) == 0 for b in all_boxes):
            return np.empty((0, 4)), np.empty((0,))

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        keep, suppressed = [], np.zeros(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            keep.append(i)
            ious = box_iou_xyxy(boxes[i], boxes[i + 1:])
            suppressed[i + 1:][ious >= nms_thresh] = True
        return boxes[keep], scores[keep]


def evaluate_owlvit_variant(name: str, detect_fn: Callable, image_dir: str | Path,
                            label_dir: str | Path, score_thresh: float,
                            target_class_id: int = 3, iou_thresh: float = 0.5,
                            max_images: int | None = None,
                            exclude_files: set[str] | None = None) -> dict:
    """Score one OWL-ViT variant against the same ground truth as the method."""
    import cv2
    from PIL import Image

    image_dir, label_dir = Path(image_dir), Path(label_dir)
    exclude_files = exclude_files or set()

    image_paths = sorted(p for p in (list(image_dir.glob("*.jpg"))
                                     + list(image_dir.glob("*.jpeg"))
                                     + list(image_dir.glob("*.png")))
                         if p.name not in exclude_files)
    if max_images:
        image_paths = image_paths[:max_images]

    print(f"\nEvaluating '{name}' on {len(image_paths)} images...")
    total_tp = total_fp = total_fn = 0
    start = time.time()

    for i, image_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            continue
        height, width = img_bgr.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        gt_boxes = load_target_gt_xyxy(label_dir / f"{image_path.stem}.txt",
                                       width, height, target_class_id)
        boxes, scores = detect_fn(pil_image)
        boxes = boxes[scores >= score_thresh]

        tp, fp, fn = match_detections(boxes, gt_boxes, iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(image_paths)}]")

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    elapsed = time.time() - start
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}  "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}  ({elapsed:.1f}s)")

    return {"variant": name, "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": precision, "recall": recall, "f1": f1, "seconds": elapsed}
