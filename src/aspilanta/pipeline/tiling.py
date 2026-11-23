"""Tiled inference for very large field photographs.

Kept, documented, and **off by default**. On the project's own evaluation data
tiling scores F1 0.023 against single-pass 0.861 — a catastrophic loss, not a
marginal one. The cause is domain mismatch: the detector was trained on whole
images resized to 640 px, so a native-resolution 640 px crop presents insects
at an apparent scale it has never seen.

It stays in the codebase because the failure is informative and because the
underlying motivation is sound for a genuinely different input regime — a
40-megapixel photograph in which a 5 mm insect survives a resize to 640 px as
roughly two pixels. If you enable it for such images, re-measure on your own
data first. Do not assume it helps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["make_tiles", "tiled_predict", "cross_tile_nms"]


def make_tiles(img_w: int, img_h: int, tile: int = 640,
               overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Overlapping tile grid covering the whole image.

    Overlap exists so that no insect is lost by falling on a tile boundary.
    The right and bottom edges get an extra tile flush against the far edge
    rather than a partial one, keeping every tile the same size — a partial
    tile would be padded or rescaled by the model and detected at a different
    effective scale from its neighbours.
    """
    stride = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(img_w - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(img_h - tile, 0) + 1, stride)) or [0]
    if xs[-1] + tile < img_w:
        xs.append(max(0, img_w - tile))
    if ys[-1] + tile < img_h:
        ys.append(max(0, img_h - tile))

    seen: set[tuple[int, int, int, int]] = set()
    tiles: list[tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x2, y2 = min(x + tile, img_w), min(y + tile, img_h)
            box = (max(0, x2 - tile), max(0, y2 - tile), x2, y2)
            if box not in seen:
                seen.add(box)
                tiles.append(box)
    return tiles


def tiled_predict(model, image_path: str | Path, tile: int = 640,
                  overlap: float = 0.25, conf: float = 0.25, iou: float = 0.45,
                  keep_class_id: int | None = 3, verbose: bool = True):
    """Run the detector tile by tile and project boxes back to image coordinates.

    Returns ``(boxes_xyxy, scores, classes, image_bgr)``.

    Non-target classes are discarded inside the loop rather than carried
    downstream. The deployment goal is an Aspilanta count; MP/NT/WF are
    detected only because they share the model, and dropping them here keeps
    the cross-tile NMS cost proportional to the boxes that matter.
    """
    import cv2

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]

    tiles = make_tiles(width, height, tile, overlap)
    if verbose:
        print(f"{image_path.name}: {width}x{height}px -> {len(tiles)} tiles "
              f"({tile}px, {int(overlap * 100)}% overlap)")

    all_boxes, all_scores, all_classes = [], [], []
    for (x1, y1, x2, y2) in tiles:
        result = model.predict(source=img[y1:y2, x1:x2], conf=conf, iou=iou,
                               verbose=False, save=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy().copy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        if keep_class_id is not None:
            keep = class_ids == keep_class_id
            if not keep.any():
                continue
            xyxy, scores, class_ids = xyxy[keep], scores[keep], class_ids[keep]

        xyxy[:, [0, 2]] += x1
        xyxy[:, [1, 3]] += y1
        all_boxes.append(xyxy)
        all_scores.append(scores)
        all_classes.append(class_ids)

    if not all_boxes:
        return (np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int), img)

    return (np.concatenate(all_boxes), np.concatenate(all_scores),
            np.concatenate(all_classes), img)


def cross_tile_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                   iou_thresh: float = 0.55, center_dist_frac: float = 0.6):
    """Merge duplicate detections of the same insect from overlapping tiles.

    Two boxes are duplicates if **either** their IoU exceeds ``iou_thresh``
    **or** their centres lie within ``center_dist_frac`` of their mean box
    size. The centre-distance criterion is the addition that matters: adjacent
    tiles often each catch a different partial view of an insect near the seam,
    producing two boxes of the same object whose IoU is genuinely low. Plain
    IoU-based NMS keeps both and inflates the count.
    """
    if len(boxes) == 0:
        return boxes, scores, classes

    order = np.argsort(-scores)
    boxes, scores, classes = boxes[order], scores[order], classes[order]

    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                        (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    sizes = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])

    from ..evaluation.metrics import box_iou_xyxy

    n = len(boxes)
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []

    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, n):
            if suppressed[j] or classes[j] != classes[i]:
                continue
            iou = box_iou_xyxy(boxes[i], boxes[j:j + 1])[0]
            center_distance = float(np.linalg.norm(centers[i] - centers[j]))
            average_size = (sizes[i] + sizes[j]) / 2
            if iou >= iou_thresh or center_distance <= center_dist_frac * average_size:
                suppressed[j] = True

    keep = sorted(keep)
    return boxes[keep], scores[keep], classes[keep]
