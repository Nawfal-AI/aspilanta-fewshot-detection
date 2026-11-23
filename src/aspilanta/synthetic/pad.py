"""Sticky-pad detection, cleaning and physical scale calibration.

Insects are composited onto the adhesive pad of a real trap card, not onto
the whole photograph. Getting that region right matters twice over: it keeps
placements inside the surface where insects can physically stick, and it
supplies the pixels-per-millimetre factor that makes the size distribution
physically meaningful rather than arbitrary.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["detect_pad_bbox", "extract_clean_pad", "pixels_per_mm"]

# HSV ranges covering the pad colours that occur in the background photographs:
# yellow (the standard card), white/grey, blue, and green.
_PAD_HSV_RANGES = (
    ((18, 70, 80), (38, 255, 255)),      # yellow
    ((0, 0, 170), (180, 50, 255)),       # white / light grey
    ((90, 50, 60), (135, 255, 255)),     # blue
    ((40, 50, 50), (90, 255, 220)),      # green
)


def detect_pad_bbox(bg: np.ndarray, min_frac: float = 0.05,
                    max_frac: float = 0.95) -> tuple[tuple[int, int, int, int], bool]:
    """Locate the adhesive pad in a trap photograph.

    Returns ``((x, y, w, h), found)``. When detection fails, the box covers the
    whole image and ``found`` is False so the caller can fall back.

    The area sanity check is what stops the detector from latching onto a
    small bright reflection or, at the other extreme, the entire frame when
    the photograph is a flat crop of the pad with no card border visible.
    """
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    height, width = bg.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)
    for low, high in _PAD_HSV_RANGES:
        mask |= cv2.inRange(hsv, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, width, height), False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if not (height * width * min_frac < area < height * width * max_frac):
        return (0, 0, width, height), False

    x, y, w, h = cv2.boundingRect(largest)
    # Inset a little so placements never land on the card border or the
    # detected edge itself, where an insect would look pasted-on.
    inset = max(8, min(w, h) // 25)
    return (x + inset, y + inset, w - 2 * inset, h - 2 * inset), True


def extract_clean_pad(bg: np.ndarray, min_frac: float = 0.05,
                      max_frac: float = 0.95
                      ) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    """Crop the pad and inpaint away any insects already stuck to it.

    Background photographs are real trap cards, so most already carry a few
    insects. Leaving them in would create unlabelled positives — the composite
    would contain insects with no corresponding ground-truth box, teaching the
    detector to suppress exactly what it is meant to find.

    The removal is a blur-difference detector: small dark blobs against a
    smooth pad are found by subtracting the image from its own heavy Gaussian
    blur, filtered by connected-component area (8 to 2000 px, which is the
    range a trapped insect occupies at these resolutions), then inpainted.

    Returns ``(canvas, (x, y, w, h), pad_found)``.
    """
    (px, py, pw, ph), found = detect_pad_bbox(bg, min_frac, max_frac)

    if not found or pw < 50 or ph < 50:
        height, width = bg.shape[:2]
        mx, my = int(width * 0.10), int(height * 0.10)
        crop_w, crop_h = int(width * 0.80), int(height * 0.80)
        return bg[my:my + crop_h, mx:mx + crop_w].copy(), (mx, my, crop_w, crop_h), False

    canvas = bg[py:py + ph, px:px + pw].copy()

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    difference = cv2.subtract(blurred, gray)
    _, blob_mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blob_mask = cv2.morphologyEx(blob_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blob_mask)
    clean = np.zeros_like(blob_mask)
    for label_id in range(1, n_labels):
        if 8 <= stats[label_id, cv2.CC_STAT_AREA] <= 2000:
            clean[labels == label_id] = 255

    if clean.any():
        clean = cv2.dilate(clean, kernel, iterations=2)
        canvas = cv2.inpaint(canvas, clean, 5, cv2.INPAINT_TELEA)

    return canvas, (px, py, pw, ph), True


def pixels_per_mm(pad_width_px: int, pad_width_mm: float) -> float:
    """Scale factor for one trap card.

    A standard yellow sticky trap is 140 mm wide, so the detected pad width in
    pixels fixes the conversion. Every composited insect is sized through this
    factor, which is why a 5 mm moth stays 5 mm whether the source photograph
    was shot at 1000 px or 4000 px across.
    """
    return pad_width_px / pad_width_mm
