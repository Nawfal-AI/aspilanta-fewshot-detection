"""The compositing operations that turn a cutout into a plausible detection.

Each function here is one step of the eight-step paste pipeline, and each is
individually switchable because Section 6.2 ablates them. Naive alpha
compositing produces images a detector learns to recognise by their seams
rather than by the insect; these steps exist to remove those shortcuts.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

__all__ = [
    "sample_size_px", "gpu_resize_bgra", "color_jitter_insect",
    "match_local_lighting", "apply_motion_blur", "add_shadow", "add_sheen",
    "augment_background", "soft_iou", "validate_label", "paste_insect",
]


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
def sample_size_px(pad_width_px: int, pad_width_mm: float, mean_mm: float,
                   std_mm: float, min_mm: float, max_mm: float,
                   rng: np.random.Generator | None = None) -> int:
    """Draw one insect width in pixels from a physically calibrated distribution.

    Length is sampled from a Gaussian truncated to ``[min_mm, max_mm]`` and
    then converted through the pad's pixels-per-millimetre factor. Truncation
    matters: an untruncated Gaussian occasionally produces negative or absurd
    sizes, and the small tail is where the hard training examples live, so it
    should be populated deliberately rather than by accident.
    """
    from scipy.stats import truncnorm

    px_per_mm = pad_width_px / pad_width_mm
    a = (min_mm - mean_mm) / std_mm
    b = (max_mm - mean_mm) / std_mm
    mm = truncnorm.rvs(a, b, loc=mean_mm, scale=std_mm,
                       random_state=rng if rng is None else None)
    return max(4, int(round(float(mm) * px_per_mm)))


def gpu_resize_bgra(img: np.ndarray, new_w: int, new_h: int,
                    device=None) -> np.ndarray:
    """Resize a BGRA cutout, using the GPU for the colour channels when available.

    The alpha channel is always resized on CPU with ``INTER_AREA``: bilinear
    interpolation on an alpha matte softens the silhouette edge in a way that
    shows up as a halo after compositing.
    """
    try:
        import torch
        import torch.nn.functional as F

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(rgb.copy()).permute(2, 0, 1)
                  .float().div(255.0).unsqueeze(0).to(device))
        resized = F.interpolate(tensor, size=(new_h, new_w),
                                mode="bilinear", align_corners=False)
        arr = (resized.squeeze(0).permute(1, 2, 0)
               .clamp(0, 1).mul(255).byte().cpu().numpy())
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = cv2.resize(img[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_AREA)

    alpha = cv2.resize(img[:, :, 3], (new_w, new_h), interpolation=cv2.INTER_AREA)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


# --------------------------------------------------------------------------- #
# Per-insect appearance
# --------------------------------------------------------------------------- #
def color_jitter_insect(insect_bgr: np.ndarray) -> np.ndarray:
    """Randomise brightness, contrast and hue per insect.

    Without this, every instance in the dataset descends from one of only 44
    cutouts and shares its exact colour statistics - a cue the detector will
    happily learn instead of the shape.
    """
    out = insect_bgr.astype(np.float32)
    out = np.clip(out * random.uniform(0.80, 1.20) + random.uniform(-15, 15), 0, 255)

    if random.random() < 0.50:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


def match_local_lighting(insect_bgr: np.ndarray, background: np.ndarray,
                         x: int, y: int, w: int, h: int,
                         strength: float = 0.20) -> np.ndarray:
    """Shift the insect's colour toward the local background mean.

    A cutout photographed under one light and pasted onto a card photographed
    under another reads as pasted immediately. Sampling the background in a
    margin around the target position and blending toward that mean removes
    most of the mismatch at a cost of some saturation.
    """
    bg_h, bg_w = background.shape[:2]
    margin = max(10, w // 2)
    patch = background[max(0, y - margin):min(bg_h, y + h + margin),
                       max(0, x - margin):min(bg_w, x + w + margin)]
    if patch.size == 0:
        return insect_bgr

    local_mean = patch.mean(axis=(0, 1))
    blended = insect_bgr.astype(np.float32) * (1 - strength) + local_mean * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def apply_motion_blur(img_bgr: np.ndarray, max_kernel: int = 5) -> np.ndarray:
    """Directional blur, simulating an insect that moved during exposure.

    Applied to a minority of instances. Real trap photographs are taken in the
    field, often handheld, and some of the insects are still alive.
    """
    k = random.choice([3, 5]) if max_kernel >= 5 else 3
    angle = random.uniform(0, 180)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation, (k, k))
    kernel /= kernel.sum() + 1e-9
    return cv2.filter2D(img_bgr, -1, kernel)


def add_shadow(background: np.ndarray, x: int, y: int, w: int, h: int,
               alpha_mask: np.ndarray) -> np.ndarray:
    """Darken a blurred, offset copy of the silhouette beneath the insect.

    The contact shadow is the single cheapest cue that an object is resting on
    a surface rather than floating above it. Offset and strength are
    randomised so the implied light direction varies across the dataset.
    """
    out = background.copy()
    dx = random.randint(1, max(2, w // 8))
    dy = random.randint(1, max(2, h // 8))
    strength = random.uniform(0.06, 0.18)

    shadow = cv2.dilate(alpha_mask, np.ones((5, 5), np.uint8), iterations=1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), float(max(1, w // 6)))
    shadow = shadow.astype(np.float32) / 255.0 * strength

    sx1, sy1 = max(0, x + dx), max(0, y + dy)
    sx2 = min(background.shape[1], x + dx + w)
    sy2 = min(background.shape[0], y + dy + h)
    mx1, my1 = sx1 - (x + dx), sy1 - (y + dy)
    mx2, my2 = mx1 + (sx2 - sx1), my1 + (sy2 - sy1)

    if (sx2 > sx1 and sy2 > sy1
            and mx2 <= shadow.shape[1] and my2 <= shadow.shape[0]):
        roi = out[sy1:sy2, sx1:sx2].astype(np.float32)
        patch = shadow[my1:my2, mx1:mx2]
        if roi.shape[:2] == patch.shape:
            out[sy1:sy2, sx1:sx2] = np.clip(
                roi * (1 - patch[..., np.newaxis]), 0, 255).astype(np.uint8)
    return out


def add_sheen(canvas: np.ndarray) -> np.ndarray:
    """Add a soft specular highlight, imitating glue reflection.

    Adhesive is glossy, and photographs of real traps almost always carry a
    broad soft highlight somewhere on the pad. Plain paper composites do not,
    which is a global cue a detector can exploit.
    """
    h, w = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.ellipse(overlay,
                (random.randint(w // 6, 5 * w // 6), random.randint(h // 6, 5 * h // 6)),
                (random.randint(w // 8, w // 3), random.randint(h // 8, h // 3)),
                random.randint(0, 180), 0, 360, (255, 255, 255), -1)
    alpha = random.uniform(0.03, 0.09)
    return cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)


def augment_background(bg: np.ndarray) -> np.ndarray:
    """Photometric variation on the trap card itself.

    Brightness and contrast always; sensor noise, vignetting and a local
    glare blob with the probabilities below. These correspond to the ways
    field photographs of the same card genuinely differ from each other.
    """
    bg = bg.copy()
    bg = np.clip(bg.astype(np.float32) * random.uniform(0.85, 1.15)
                 + random.uniform(-15, 15), 0, 255).astype(np.uint8)

    if random.random() < 0.40:
        noise = np.random.normal(0, random.uniform(2, 7), bg.shape).astype(np.float32)
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.30:      # vignette
        h, w = bg.shape[:2]
        ys, xs = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        distance = np.sqrt(((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2)
        vignette = 1.0 - np.clip(distance * random.uniform(0.25, 0.55), 0, 0.35)
        bg = np.clip(bg.astype(np.float32) * vignette[..., np.newaxis],
                     0, 255).astype(np.uint8)

    if random.random() < 0.25:      # local glare
        h, w = bg.shape[:2]
        radius = random.randint(15, max(16, min(w, h) // 4))
        mask = np.zeros((h, w), np.float32)
        cv2.circle(mask, (random.randint(0, w), random.randint(0, h)), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, radius // 2))
        bg = np.clip(bg.astype(np.float32)
                     + mask[..., np.newaxis] * 255 * random.uniform(0.04, 0.14),
                     0, 255).astype(np.uint8)
    return bg


# --------------------------------------------------------------------------- #
# Placement bookkeeping
# --------------------------------------------------------------------------- #
def soft_iou(box_a, box_b) -> float:
    """IoU between two ``[x1, y1, x2, y2]`` boxes.

    Used to reject placements that would bury one insect under another. A
    modest overlap allowance is realistic - insects on a crowded trap do touch
    - but heavy occlusion produces boxes whose visible content no longer
    matches their label.
    """
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def validate_label(cx: float, cy: float, bw: float, bh: float,
                   min_size: float = 0.005, max_size: float = 0.90) -> bool:
    """Reject a box before it is written to disk.

    Ultralytics silently tolerates malformed labels, so a box that has drifted
    off-frame or collapsed to a sliver becomes a quiet source of training
    noise. The generator counts rejections and reports them, which is how the
    label-quality figure reported for the project is produced.
    """
    if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0):
        return False
    if not (min_size < bw < max_size and min_size < bh < max_size):
        return False
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    return not (x1 < -0.05 or y1 < -0.05 or x2 > 1.05 or y2 > 1.05)


# --------------------------------------------------------------------------- #
# The paste pipeline
# --------------------------------------------------------------------------- #
def paste_insect(background: np.ndarray, insect_bgra: np.ndarray,
                 target_w: int, pos: tuple[int, int],
                 *, use_color_jitter: bool = True,
                 use_lighting_match: bool = True,
                 use_shadow: bool = True,
                 use_poisson: bool = True,
                 organic_blur_kernel: tuple[int, int] = (3, 3),
                 organic_blur_sigma: float = 0.5,
                 motion_blur_prob: float = 0.15,
                 device=None) -> tuple[np.ndarray, tuple | None]:
    """Composite one insect and return ``(image, normalised_bbox)``.

    Eight steps, in order: resize, colour jitter, local lighting match,
    organic blur, optional motion blur, alpha anti-alias, contact shadow, and
    alpha composite followed by Poisson seamless cloning.

    The Poisson step is the ablated one (Section 6.2, "no_poisson"). It solves
    for pixel values whose gradients match the source inside the mask while
    matching the destination at the boundary, so the paste seam disappears
    even when the two images differ in overall illumination. ``MIXED_CLONE``
    is tried first and ``NORMAL_CLONE`` is the fallback, because MIXED
    preserves background texture through semi-transparent wings but throws on
    some region geometries. Both are skipped below 25 px, where seamlessClone
    is unstable and the seam is invisible anyway.

    Returns ``(background, None)`` when the placement does not fit, so the
    caller can retry at another position.
    """
    bg_h, bg_w = background.shape[:2]
    orig_h, orig_w = insect_bgra.shape[:2]
    scale = target_w / max(orig_w, 1)
    new_w = max(8, int(orig_w * scale))
    new_h = max(8, int(orig_h * scale))
    if new_w >= bg_w or new_h >= bg_h:
        return background, None

    resized = gpu_resize_bgra(insect_bgra, new_w, new_h, device=device)
    insect_bgr = resized[:, :, :3].copy()
    insect_alpha = resized[:, :, 3].copy()

    if use_color_jitter:
        insect_bgr = color_jitter_insect(insect_bgr)

    x, y = pos
    x = max(0, min(x, bg_w - new_w))
    y = max(0, min(y, bg_h - new_h))
    x2, y2 = x + new_w, y + new_h
    if background[y:y2, x:x2].shape[:2] != (new_h, new_w):
        return background, None

    if use_lighting_match:
        insect_bgr = match_local_lighting(insect_bgr, background, x, y, new_w, new_h,
                                          strength=random.uniform(0.10, 0.25))

    insect_bgr = cv2.GaussianBlur(insect_bgr, organic_blur_kernel, organic_blur_sigma)

    if random.random() < motion_blur_prob:
        insect_bgr = apply_motion_blur(insect_bgr, max_kernel=5)

    # Threshold then feather: a hard cut alone leaves a jagged 1 px edge.
    _, insect_alpha = cv2.threshold(insect_alpha, 30, 255, cv2.THRESH_BINARY)
    insect_alpha = cv2.GaussianBlur(insect_alpha, (3, 3), 0.5)

    work = add_shadow(background, x, y, new_w, new_h, insect_alpha) \
        if use_shadow else background.copy()

    alpha_f = insect_alpha.astype(np.float32) / 255.0
    roi = work[y:y2, x:x2].astype(np.float32)
    blended = (insect_bgr.astype(np.float32) * alpha_f[..., np.newaxis]
               + roi * (1 - alpha_f[..., np.newaxis]))
    result = work.copy()
    result[y:y2, x:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    if use_poisson and new_w > 25 and new_h > 25:
        try:
            _, poisson_mask = cv2.threshold(insect_alpha, 50, 255, cv2.THRESH_BINARY)
            if poisson_mask.sum() > 0:
                center = (x + new_w // 2, y + new_h // 2)
                try:
                    result = cv2.seamlessClone(result[y:y2, x:x2], result,
                                               poisson_mask, center, cv2.MIXED_CLONE)
                except cv2.error:
                    result = cv2.seamlessClone(result[y:y2, x:x2], result,
                                               poisson_mask, center, cv2.NORMAL_CLONE)
        except Exception:
            pass    # a failed blend is cosmetic; the alpha composite still stands

    bbox = ((x + new_w / 2) / bg_w, (y + new_h / 2) / bg_h,
            new_w / bg_w, new_h / bg_h)
    return result, bbox
