"""``AspilantaDetector`` - the deliverable.

One class, one method. Give it an image; get back an annotated image, a table
of detections, and a count.

    from aspilanta import AspilantaDetector

    detector = AspilantaDetector(weights="path/to/best.pt")
    annotated, detections, counts = detector.detect("trap_photo.jpg")
    print(counts["aspilanta_count"])

Everything else in this repository exists to produce the checkpoint this class
loads. The defaults are the measured ones: single-pass inference, no CLIP
filter, Aspilanta reported alone with the other three classes detected
internally as context and filtered from the output.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import ClassScheme, DeploymentConfig

__all__ = ["AspilantaDetector", "PROPOSED_RUN_NAMES", "FROZEN_ABLATION_RUN_NAMES"]


# Checkpoint names in preference order. The proposed model - trainable
# backbone - is archived under several historical names. The frozen model is
# now only an ablation, but its run directory has the most official-sounding
# name of the lot, so it is listed separately and warned about: loading it by
# accident costs AP@50 0.981 -> 0.906 with nothing to indicate anything is wrong.
PROPOSED_RUN_NAMES = (
    "solution_4class_detector",
    "exp3_4class_trainable_backbone",
    "ablation_unfrozen_backbone",
)
FROZEN_ABLATION_RUN_NAMES = (
    "exp3_4class_synthetic_finetuned",
    "ablation_frozen_backbone",
)

_PALETTE = {0: (255, 160, 60), 1: (60, 200, 255), 2: (180, 180, 180),
            3: (40, 220, 60)}


class AspilantaDetector:
    """Detect and count *Aspilanta* on a yellow sticky trap photograph.

    Args:
        weights: path to a trained four-class checkpoint. When omitted, the
            constructor searches ``search_roots``.
        config: inference settings; defaults are the measured ones.
        classes: the label scheme, for naming boxes when the checkpoint
            carries no names.
        target_only: annotate and report the target class only. Other insects
            are still detected - they appear in the returned DataFrame as
            context - but they are not drawn and not counted.
        search_roots: where to look for a checkpoint when ``weights`` is None.
    """

    def __init__(self, weights: str | Path | None = None,
                 config: DeploymentConfig | None = None,
                 classes: ClassScheme | None = None,
                 target_only: bool = True,
                 search_roots: Sequence[str | Path] | None = None,
                 clip_filter=None, verbose: bool = True):
        from ultralytics import YOLO

        self.config = config or DeploymentConfig()
        self.classes = classes or ClassScheme()
        self.target_only = target_only
        self.clip_filter = clip_filter

        weights = Path(weights) if weights else self._find_weights(
            search_roots, verbose=verbose)
        if weights is None or not Path(weights).exists():
            raise FileNotFoundError(
                "No trained four-class checkpoint found.\n"
                "Pass weights=..., or train one with scripts/train_solution.py.")

        self.weights = Path(weights)
        self.model = YOLO(str(self.weights))
        self.names = (
            {int(k): v for k, v in self.model.names.items()}
            if isinstance(self.model.names, dict)
            else dict(enumerate(self.model.names)))

        if verbose:
            mode = "tiled" if self.config.use_tiling else "single-pass"
            if self.config.use_clip_filter:
                mode += " + CLIP filter"
            print(f"Detector ready\n  weights : {self.weights}\n"
                  f"  classes : {self.names}\n"
                  f"  mode    : {mode}  conf={self.config.conf} "
                  f"iou={self.config.iou}")

    # -- weight discovery --------------------------------------------------- #
    @staticmethod
    def _find_weights(search_roots: Sequence[str | Path] | None,
                      verbose: bool = True) -> Path | None:
        from ..models.train import find_weights

        roots = [Path(r) for r in (search_roots or
                                   [Path("work"), Path("."), Path("/kaggle/working"),
                                    Path("/kaggle/input")])]

        found = find_weights(PROPOSED_RUN_NAMES, roots, verbose=verbose)
        if found is not None:
            return found

        found = find_weights(FROZEN_ABLATION_RUN_NAMES, roots, verbose=False)
        if found is not None:
            if verbose:
                print("=" * 70)
                print("WARNING: only the FROZEN-backbone checkpoint was found.")
                print("  That configuration is an ABLATION, not the proposed method.")
                print("  Expect AP@50 around 0.906 rather than 0.981.")
                print(f"  Using: {found}")
                print("=" * 70)
            return found

        for root in roots:                      # last resort
            if root.exists():
                for candidate in sorted(root.rglob("best.pt")):
                    if verbose:
                        print(f"WARNING: using an unidentified checkpoint: {candidate}")
                    return candidate
        return None

    # -- inference ---------------------------------------------------------- #
    def _to_bgr(self, image) -> np.ndarray:
        import cv2
        from PIL import Image

        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            return bgr
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return image

    def _single_pass(self, bgr: np.ndarray) -> np.ndarray:
        """Whole-image predict. Returns ``[N, 6]`` of x1,y1,x2,y2,score,class."""
        result = self.model.predict(bgr, conf=self.config.conf,
                                    iou=self.config.iou, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        return np.concatenate([
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()[:, None],
            result.boxes.cls.cpu().numpy()[:, None],
        ], axis=1).astype(np.float32)

    def _tiled(self, bgr: np.ndarray) -> np.ndarray:
        from .tiling import cross_tile_nms, make_tiles

        detections = []
        for (x1, y1, x2, y2) in make_tiles(bgr.shape[1], bgr.shape[0],
                                           self.config.tile_size,
                                           self.config.tile_overlap):
            tile = bgr[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            found = self._single_pass(tile)
            if len(found):
                found[:, [0, 2]] += x1
                found[:, [1, 3]] += y1
                detections.append(found)

        if not detections:
            return np.zeros((0, 6), dtype=np.float32)

        combined = np.concatenate(detections, axis=0)
        boxes, scores, classes = cross_tile_nms(
            combined[:, :4], combined[:, 4], combined[:, 5].astype(int),
            iou_thresh=self.config.merge_iou,
            center_dist_frac=self.config.center_dist_frac)
        return np.concatenate([boxes, scores[:, None], classes[:, None]],
                              axis=1).astype(np.float32)

    def detect(self, image, conf: float | None = None):
        """Detect on one image.

        Args:
            image: path, PIL image, or BGR ndarray.
            conf: override the confidence floor for this call.

        Returns ``(annotated_bgr, detections_dataframe, counts_dict)``. The
        DataFrame carries every class so downstream code can inspect the
        context; ``counts["aspilanta_count"]`` is the reported number.
        """
        import pandas as pd

        bgr = self._to_bgr(image)
        previous_conf = self.config.conf
        if conf is not None:
            self.config.conf = conf

        try:
            use_tiling = self.config.use_tiling
            if not use_tiling and min(bgr.shape[:2]) > self.config.tiling_size_threshold:
                # Large enough that tiling is at least worth considering, but
                # never automatic - the measured evidence is against it.
                pass
            detections = self._tiled(bgr) if use_tiling else self._single_pass(bgr)

            if self.config.use_clip_filter and self.clip_filter is not None and len(detections):
                target_mask = detections[:, 5].astype(int) == self.classes.target_id
                target, other = detections[target_mask], detections[~target_mask]
                boxes, scores, classes, _ = self.clip_filter.filter_selective(
                    target[:, :4], target[:, 4], target[:, 5].astype(int), bgr,
                    trust_threshold=self.config.selective_trust_threshold)
                filtered = (np.concatenate(
                    [boxes, scores[:, None], classes[:, None]], axis=1)
                    .astype(np.float32) if len(boxes)
                    else np.zeros((0, 6), dtype=np.float32))
                detections = np.concatenate([filtered, other], axis=0)
        finally:
            self.config.conf = previous_conf

        rows = []
        for x1, y1, x2, y2, score, class_id in detections:
            class_id = int(class_id)
            rows.append({
                "class_id": class_id,
                "class": self.names.get(class_id,
                                        self.classes.full[class_id]
                                        if class_id < len(self.classes.full)
                                        else str(class_id)),
                "confidence": round(float(score), 3),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            })
        frame = pd.DataFrame(rows)

        counts = {"aspilanta_count": int(
            (frame["class_id"] == self.classes.target_id).sum()) if len(frame) else 0}
        if len(frame):
            for class_id, group in frame.groupby("class_id"):
                counts[self.names.get(int(class_id), str(class_id))] = {
                    "count": int(len(group)),
                    "mean_confidence": round(float(group["confidence"].mean()), 3),
                }

        return self.annotate(bgr, frame), frame, counts

    # -- output ------------------------------------------------------------- #
    def annotate(self, bgr: np.ndarray, frame) -> np.ndarray:
        """Draw boxes, scaling line width and font to the image size.

        Fixed pixel sizes are unreadable on a 6000 px field photograph and
        cover the insect entirely on a 400 px crop, so both scale with the
        image's shorter side.
        """
        import cv2

        out = bgr.copy()
        thickness = max(2, int(round(min(out.shape[:2]) / 400)))
        font_scale = max(0.5, min(out.shape[:2]) / 1200)

        for _, row in frame.iterrows():
            class_id = int(row["class_id"])
            if self.target_only and class_id != self.classes.target_id:
                continue
            color = _PALETTE.get(class_id, (0, 255, 0))
            cv2.rectangle(out, (row.x1, row.y1), (row.x2, row.y2), color, thickness)

            label = f"{row['class']} {row['confidence']:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(out, (row.x1, max(0, row.y1 - text_h - 6)),
                          (row.x1 + text_w + 4, row.y1), color, -1)
            cv2.putText(out, label, (row.x1 + 2, max(text_h, row.y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1,
                        cv2.LINE_AA)
        return out

    def detect_and_save(self, image, out_dir: str | Path):
        """Detect and write the annotated image, a CSV and a JSON summary."""
        import cv2

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image).stem if isinstance(image, (str, Path)) else "image"

        annotated, frame, counts = self.detect(image)
        image_path = out_dir / f"{stem}_annotated.jpg"
        csv_path = out_dir / f"{stem}_detections.csv"
        json_path = out_dir / f"{stem}_summary.json"

        cv2.imwrite(str(image_path), annotated)
        frame.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(counts, indent=2))

        print(f"Aspilanta detected: {counts['aspilanta_count']}")
        print(f"  {image_path}\n  {csv_path}\n  {json_path}")
        return annotated, frame, counts
