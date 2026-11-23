"""Detection metrics: IoU, greedy matching, and single-class ground truth.

The notebook carried three near-identical copies of these functions — the main
harness, an ``_s9`` suffixed set for the OWL-ViT section, and an inline version
inside the selective-CLIP sweep — each written so its section could run
standalone. Divergence between copies is exactly how a comparison table stops
being a comparison, so there is one implementation here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_target_gt_xyxy", "box_iou_xyxy", "match_detections",
           "precision_recall_f1"]


def load_target_gt_xyxy(label_path: str | Path, img_w: int, img_h: int,
                        target_class_id: int = 3) -> np.ndarray:
    """Load ground-truth boxes for one class from a YOLO label file.

    ``target_class_id`` must match the numbering of the label file being read,
    which is not always the four-class scheme. Raw generator output uses
    ``0 = Aspilanta``; merged datasets use ``3``. Passing the wrong one yields
    an empty ground truth and a recall of zero that looks like a model failure
    rather than a bookkeeping error, so callers should pass it explicitly.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return np.empty((0, 4))

    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if not parts or int(float(parts[0])) != target_class_id:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        boxes.append([(xc - w / 2) * img_w, (yc - h / 2) * img_h,
                      (xc + w / 2) * img_w, (yc + h / 2) * img_h])

    return np.array(boxes) if boxes else np.empty((0, 4))


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box ``[4]`` and an array of boxes ``[N, 4]``."""
    if len(boxes) == 0:
        return np.empty((0,))

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_box = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_boxes = (np.maximum(0, boxes[:, 2] - boxes[:, 0])
                  * np.maximum(0, boxes[:, 3] - boxes[:, 1]))
    union = area_box + area_boxes - intersection
    return np.where(union > 0, intersection / union, 0)


def match_detections(pred_boxes: np.ndarray, gt_boxes: np.ndarray,
                     iou_thresh: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching. Returns ``(TP, FP, FN)``.

    Predictions should arrive sorted by descending confidence, which is what
    the detector returns: greedy matching gives each ground-truth box to the
    highest-scoring prediction that claims it. A second prediction on the same
    insect counts as a false positive, which is the behaviour wanted for a
    counting application — double-counting is a real error, not a near miss.
    """
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = fp = 0

    for box in pred_boxes:
        unmatched = np.where(~matched)[0]
        if len(unmatched) == 0:
            fp += 1
            continue
        ious = box_iou_xyxy(box, gt_boxes[unmatched])
        best = int(np.argmax(ious)) if len(ious) else -1
        if best >= 0 and ious[best] >= iou_thresh:
            matched[unmatched[best]] = True
            tp += 1
        else:
            fp += 1

    return tp, fp, int((~matched).sum())


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Standard triple, with zero returned rather than NaN on empty inputs."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
