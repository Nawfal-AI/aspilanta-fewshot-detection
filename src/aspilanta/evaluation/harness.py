"""Evaluation harness: run a pipeline over a dataset and score it.

Reports precision, recall, F1, count MAE and timing for the target class only.
Count MAE is included because the deployment task is counting, and a pipeline
can post a respectable F1 while systematically over- or under-counting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np

from .metrics import load_target_gt_xyxy, match_detections, precision_recall_f1

__all__ = ["evaluate_pipeline", "evaluate_bucketed_by_size",
           "sweep_confidence", "PipelineName"]

PipelineName = Literal["single_pass", "single_pass_clip", "selective_clip", "tiled",
                       "tiled_clip"]


def _list_images(image_dir: Path) -> list[Path]:
    return sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
                  + list(image_dir.glob("*.png")))


def _predict(model, image_path: Path, img_bgr, pipeline: PipelineName,
             deploy_conf: float, target_class_id: int, clip_filter,
             trust_threshold: float, tile_kwargs: dict):
    """Dispatch one image through the requested pipeline variant."""
    if pipeline in ("tiled", "tiled_clip"):
        from ..pipeline.tiling import cross_tile_nms, tiled_predict

        boxes, scores, classes, _ = tiled_predict(
            model, image_path, keep_class_id=target_class_id, verbose=False,
            **{k: v for k, v in tile_kwargs.items()
               if k in ("tile", "overlap", "conf", "iou")})
        boxes, scores, classes = cross_tile_nms(
            boxes, scores, classes,
            iou_thresh=tile_kwargs.get("merge_iou", 0.55),
            center_dist_frac=tile_kwargs.get("center_dist_frac", 0.6))
        keep = scores >= deploy_conf
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    else:
        result = model.predict(source=str(image_path), conf=deploy_conf,
                               verbose=False, save=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4))
            scores = np.empty((0,))
            classes = np.empty((0,), dtype=int)
        else:
            all_boxes = result.boxes.xyxy.cpu().numpy()
            all_scores = result.boxes.conf.cpu().numpy()
            all_classes = result.boxes.cls.cpu().numpy().astype(int)
            mask = all_classes == target_class_id
            boxes, scores, classes = all_boxes[mask], all_scores[mask], all_classes[mask]

    if pipeline in ("single_pass_clip", "tiled_clip") and clip_filter is not None:
        boxes, scores, classes, _ = clip_filter.filter(boxes, scores, classes, img_bgr)
    elif pipeline == "selective_clip" and clip_filter is not None:
        boxes, scores, classes, _ = clip_filter.filter_selective(
            boxes, scores, classes, img_bgr, trust_threshold=trust_threshold)

    return boxes, scores, classes


def evaluate_pipeline(model, image_dir: str | Path, label_dir: str | Path,
                      pipeline: PipelineName = "single_pass",
                      target_class_id: int = 3,
                      gt_class_id: int | None = None,
                      iou_thresh: float = 0.5, deploy_conf: float = 0.40,
                      clip_filter=None, trust_threshold: float = 0.60,
                      max_images: int | None = None,
                      tile_kwargs: dict | None = None,
                      verbose: bool = True) -> dict:
    """Score one pipeline variant over a labelled image folder.

    Args:
        target_class_id: the class the *model* emits for the target species.
        gt_class_id: the class id meaning the target in *this label folder*.
            Defaults to ``target_class_id``. Pass 0 when scoring directly
            against raw generator output, whose labels use the generator's own
            numbering.

    The two ids being separate parameters is deliberate. Conflating them
    produces a silent recall of zero — the evaluation reads no ground truth at
    all and reports it as a model failure.
    """
    import cv2

    image_dir, label_dir = Path(image_dir), Path(label_dir)
    gt_class_id = target_class_id if gt_class_id is None else gt_class_id
    tile_kwargs = tile_kwargs or {}

    image_paths = _list_images(image_dir)
    if max_images:
        image_paths = image_paths[:max_images]

    if verbose:
        print(f"Evaluating '{pipeline}' on {len(image_paths)} images "
              f"(target only, ground-truth class id={gt_class_id})...")

    total_tp = total_fp = total_fn = 0
    count_errors: list[int] = []
    times: list[float] = []

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        gt_boxes = load_target_gt_xyxy(label_dir / f"{image_path.stem}.txt",
                                       width, height, gt_class_id)

        start = time.time()
        boxes, scores, classes = _predict(model, image_path, img, pipeline,
                                          deploy_conf, target_class_id,
                                          clip_filter, trust_threshold, tile_kwargs)
        times.append(time.time() - start)

        tp, fp, fn = match_detections(boxes, gt_boxes, iou_thresh=iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        count_errors.append(abs(len(boxes) - len(gt_boxes)))

    precision, recall, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    summary = {
        "pipeline": pipeline,
        "n_images": len(image_paths),
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count_MAE": round(float(np.mean(count_errors)), 4) if count_errors else None,
        "avg_inference_sec": round(float(np.mean(times)), 4) if times else None,
        "total_inference_sec": round(float(np.sum(times)), 2) if times else None,
    }

    if verbose:
        print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
        print(f"  P={summary['precision']:.3f}  R={summary['recall']:.3f}  "
              f"F1={summary['f1']:.3f}  countMAE={summary['count_MAE']}  "
              f"{summary['avg_inference_sec']}s/img")

    return summary


def evaluate_bucketed_by_size(model, image_dir: str | Path, label_dir: str | Path,
                              size_threshold: int = 1500, **kwargs) -> dict:
    """Score small and large images separately.

    Aggregate metrics hide scale effects. If tiling helps anywhere it should
    help here, on the large bucket — and the fact that it does not is a much
    stronger finding than a single aggregate number would give.
    """
    import cv2

    image_dir = Path(image_dir)
    small, large = [], []
    for path in _list_images(image_dir):
        img = cv2.imread(str(path))
        if img is None:
            continue
        (large if max(img.shape[:2]) > size_threshold else small).append(path)

    print(f"Size buckets at {size_threshold}px: "
          f"{len(small)} small, {len(large)} large")

    results = {}
    for name, paths in (("small", small), ("large", large)):
        if not paths:
            continue
        import os
        import tempfile
        # Evaluate the bucket by pointing the harness at a temporary directory
        # of symlinks, so no images are copied and the label folder is shared.
        with tempfile.TemporaryDirectory() as tmp:
            for path in paths:
                os.symlink(path.resolve(), Path(tmp) / path.name)
            results[name] = evaluate_pipeline(model, tmp, label_dir, **kwargs)
            results[name]["bucket"] = name
    return results


def sweep_confidence(model, image_dir: str | Path, label_dir: str | Path,
                     candidates=(0.10, 0.20, 0.25, 0.30, 0.40, 0.50),
                     **kwargs) -> list[dict]:
    """Score the pipeline at several confidence thresholds.

    Run this on the validation half only. The test split carved in Section
    11.2 exists precisely so that a threshold chosen here can be reported
    honestly against data it was not chosen on.
    """
    rows = []
    for conf in candidates:
        row = evaluate_pipeline(model, image_dir, label_dir,
                                deploy_conf=conf, verbose=False, **kwargs)
        row["deploy_conf"] = conf
        rows.append(row)
        print(f"  conf={conf:.2f} -> P={row['precision']:.3f} "
              f"R={row['recall']:.3f} F1={row['f1']:.3f}")
    return rows
