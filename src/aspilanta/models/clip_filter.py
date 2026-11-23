"""Frozen-CLIP few-shot prototype filter — the algorithm-oriented component.

The idea: embed a handful of real Aspilanta crops with a frozen CLIP encoder,
average them into a prototype, and re-score each candidate detection by cosine
similarity to that prototype. Nothing is trained; the few-shot signal enters
entirely through the support set.

Two things about this module are worth reading before using it.

**It is off by default, on evidence.** On ``generated_dataset_v7`` the filter
moves precision 0.931 to 0.936 while recall falls 0.801 to 0.675 — F1 drops
from 0.861 to 0.785. :func:`selective_clip_filter` recovers part of the idea by
consulting CLIP only for detections the detector itself is unsure about.

**The notebook version had a real bug, fixed here.** The original read
``clip_model`` and ``support_images_all`` straight out of ``globals()`` and
silently returned its input unchanged when they were missing — which was the
case after any kernel restart. Every "CLIP-filtered" number reported before
that fix was identical to the unfiltered baseline, because the filter never
ran. The class below owns its state, so it either works or raises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["ClipEmbedder", "ClipPrototypeFilter", "crop_objects_from_dataset",
           "build_support_crops", "selective_clip_filter"]


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class ClipEmbedder:
    """Frozen CLIP image encoder producing L2-normalised embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.projection_dim = self.model.config.projection_dim
        print(f"CLIP ready: {model_name} on {self.device}")

    def embed(self, pil_images: Sequence, batch_size: int = 16):
        """Embed PIL images. Returns an L2-normalised CPU tensor ``[N, D]``.

        The unwrapping below looks paranoid but is load-bearing across
        transformers versions: ``get_image_features`` has variously returned a
        bare tensor, an object with ``image_embeds``, and an object carrying
        unprojected hidden states. Projecting only when the width is wrong
        avoids a matmul shape error on versions that already project.
        """
        import torch
        import torch.nn.functional as F

        if not len(pil_images):
            return torch.empty(0, self.projection_dim)

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch_size):
                batch = list(pil_images[i:i + batch_size])
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                features = self.model.get_image_features(
                    pixel_values=inputs["pixel_values"].to(self.device))

                if not torch.is_tensor(features):
                    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
                        if hasattr(features, attr):
                            candidate = getattr(features, attr)
                            if attr == "last_hidden_state":
                                candidate = candidate[:, 0, :]
                            features = candidate
                            break
                    else:
                        raise TypeError(f"Unexpected CLIP output type: {type(features)}")

                if features.shape[-1] != self.projection_dim:
                    features = self.model.visual_projection(features)

                embeddings.append(F.normalize(features, p=2, dim=-1).cpu())

        return torch.cat(embeddings, dim=0)


# --------------------------------------------------------------------------- #
# Crop extraction
# --------------------------------------------------------------------------- #
def crop_objects_from_dataset(dataset_root: str | Path, split: str = "val",
                              padding: float = 0.10, target_class_id: int = 3):
    """Extract every labelled object as a crop, with a binary target label.

    Returns ``(crops, labels, meta)`` where ``labels`` is 1 for the target
    class and 0 otherwise. ``meta`` records the source filename, which the
    prototypical-network comparison uses to restrict evaluation to crops that
    came from real photographs rather than synthetic composites.
    """
    from PIL import Image

    from ..data.yolo_io import list_images, read_yolo_lines, yolo_xywh_to_xyxy

    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    crops, labels, meta = [], [], []
    for img_path in list_images(img_dir):
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        width, height = image.size

        for line in read_yolo_lines(label_path):
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x1, y1, x2, y2 = yolo_xywh_to_xyxy(
                list(map(float, parts[1:5])), width, height, padding=padding)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(image.crop((x1, y1, x2, y2)))
            labels.append(1 if class_id == target_class_id else 0)
            meta.append({"image": img_path.name, "class_id": class_id})

    return crops, np.asarray(labels), meta


def build_support_crops(image_paths: Sequence[Path], label_roots: Iterable[Path],
                        padding: float = 0.15, limit: int | None = None) -> list:
    """Build the support set from real Aspilanta training images.

    One crop per image, taken from its first box. These are the only real
    examples the filter ever sees, so they must come from the training pool —
    drawing them from validation images would leak.
    """
    from PIL import Image

    from ..data.yolo_io import find_label_for_image, read_yolo_lines, yolo_xywh_to_xyxy

    crops = []
    for img_path in list(image_paths)[:limit] if limit else image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            label_path = find_label_for_image(img_path, possible_roots=label_roots)
            lines = read_yolo_lines(label_path)
            if lines:
                width, height = image.size
                box = list(map(float, lines[0].split()[1:5]))
                x1, y1, x2, y2 = yolo_xywh_to_xyxy(box, width, height, padding=padding)
                crops.append(image.crop((x1, y1, x2, y2)))
            else:
                crops.append(image)
        except Exception:
            continue
    return crops


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #
@dataclass
class _FilterStats:
    checked: int = 0
    dropped: int = 0


class ClipPrototypeFilter:
    """K-shot prototype filter over candidate detections.

    Args:
        embedder: a :class:`ClipEmbedder`.
        support_crops: real target crops forming the support set.
        k: how many support crops to average into the prototype.
        threshold: cosine similarity floor. When None, it is calibrated from
            the support set itself as ``mean - std`` of support-to-prototype
            similarity — deliberately conservative, and reproducible, unlike a
            hand-picked constant.
    """

    def __init__(self, embedder: ClipEmbedder, support_crops: Sequence,
                 k: int = 5, threshold: float | None = None):
        import torch.nn.functional as F

        if not len(support_crops):
            raise ValueError("Cannot build a CLIP prototype from an empty support set.")

        self.embedder = embedder
        self.k = min(k, len(support_crops))
        self.support_crops = list(support_crops)

        support_embeddings = embedder.embed(self.support_crops[:self.k])
        self.prototype = F.normalize(support_embeddings.mean(dim=0, keepdim=True),
                                     p=2, dim=-1)

        if threshold is None:
            all_embeddings = embedder.embed(self.support_crops)
            similarities = (all_embeddings @ self.prototype.T).squeeze(1).numpy()
            threshold = float(similarities.mean() - similarities.std())
            print(f"Calibrated CLIP threshold from {len(self.support_crops)} "
                  f"support crops: {threshold:.4f}")
        self.threshold = threshold
        print(f"CLIP {self.k}-shot prototype built; threshold={self.threshold:.4f}")

    def similarities(self, boxes: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        """Cosine similarity of each box's crop to the prototype."""
        import cv2
        from PIL import Image

        if len(boxes) == 0:
            return np.empty((0,))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            crop = img_rgb[max(0, y1):y2, max(0, x1):x2]
            crops.append(Image.fromarray(
                crop if crop.size else np.zeros((8, 8, 3), dtype=np.uint8)))

        return (self.embedder.embed(crops) @ self.prototype.T).squeeze(1).numpy()

    def filter(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
               img_bgr: np.ndarray, margin: float = 0.0,
               threshold: float | None = None):
        """Drop every detection below the similarity threshold.

        Measured to cost more recall than it buys precision on this data — see
        the module docstring. Prefer :func:`selective_clip_filter`.
        """
        if len(boxes) == 0:
            return boxes, scores, classes, _FilterStats()

        cutoff = (threshold if threshold is not None else self.threshold) + margin
        keep = self.similarities(boxes, img_bgr) >= cutoff
        stats = _FilterStats(checked=len(boxes), dropped=int((~keep).sum()))
        return boxes[keep], scores[keep], classes[keep], stats

    def filter_selective(self, boxes: np.ndarray, scores: np.ndarray,
                         classes: np.ndarray, img_bgr: np.ndarray,
                         trust_threshold: float = 0.60,
                         sim_threshold: float | None = None):
        """Consult CLIP only for detections below ``trust_threshold``.

        Section 11.3. The blanket filter's failure has a clear cause: it
        second-guesses confident, correct detections whose crops happen to sit
        a little away from the prototype in CLIP space. Detections the model is
        already sure about are kept unconditionally, and CLIP arbitrates only
        the marginal ones — which is where false positives actually
        concentrate.
        """
        if len(boxes) == 0:
            return boxes, scores, classes, _FilterStats()

        confident = scores >= trust_threshold
        marginal_idx = np.where(~confident)[0]
        if len(marginal_idx) == 0:
            return boxes, scores, classes, _FilterStats()

        cutoff = sim_threshold if sim_threshold is not None else self.threshold
        similarities = self.similarities(boxes[marginal_idx], img_bgr)

        keep = confident.copy()
        keep[marginal_idx] = similarities >= cutoff
        stats = _FilterStats(checked=int(len(marginal_idx)),
                             dropped=int((~keep).sum()))
        return boxes[keep], scores[keep], classes[keep], stats


def selective_clip_filter(boxes, scores, classes, img_bgr,
                          clip_filter: ClipPrototypeFilter,
                          trust_threshold: float = 0.60,
                          sim_threshold: float | None = None):
    """Functional wrapper around :meth:`ClipPrototypeFilter.filter_selective`."""
    return clip_filter.filter_selective(boxes, scores, classes, img_bgr,
                                        trust_threshold=trust_threshold,
                                        sim_threshold=sim_threshold)
