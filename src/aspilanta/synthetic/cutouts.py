"""Foreground extraction: raw photographs to alpha-matted cutouts.

Everything the generator composites starts here. The quality of these mattes
sets a ceiling on the quality of the synthetic data: a cutout that keeps a
halo of its original background teaches the detector to look for that halo.

Two extraction paths are provided.

``rembg`` (a U^2-Net segmentation model) gives the better mattes and is used
for the 44 Aspilanta targets. It is run in a **subprocess** rather than
imported. That is not defensiveness for its own sake: rembg pulls in
onnxruntime, which will grab the GPU and, in the same process as PyTorch,
reliably exhausts CUDA memory partway through a batch. Isolating it also means
a segfault costs one image rather than the whole run.

A classical GrabCut + guided-filter chain is the fallback, and the primary
path for the distractor set where volume matters more than perfection.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "SUPPORTED_EXTS", "colour_distance_mask", "guided_filter_alpha",
    "clean_mask", "grabcut_with_hint", "full_extraction",
    "extract_with_rembg", "extract_folder", "write_manifest",
]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Classical extraction chain
# --------------------------------------------------------------------------- #
def colour_distance_mask(img_bgr: np.ndarray, margin_frac: float = 0.08) -> np.ndarray:
    """Rough foreground mask from colour distance to the image border.

    Specimen photographs are shot against a plain surface, so the border
    pixels are a reliable sample of the background. Normalising the distance
    by the border's own per-channel standard deviation makes the 2.5-sigma
    threshold meaningful whether the background is evenly lit or not.
    """
    height, width = img_bgr.shape[:2]
    margin = max(5, int(min(height, width) * margin_frac))
    border = np.vstack([
        img_bgr[:margin, :].reshape(-1, 3),
        img_bgr[-margin:, :].reshape(-1, 3),
        img_bgr[:, :margin].reshape(-1, 3),
        img_bgr[:, -margin:].reshape(-1, 3),
    ]).astype(np.float32)

    bg_mean = border.mean(0)
    bg_std = border.std(0) + 10.0     # floor, so a flat background is not divided by ~0
    distance = np.sqrt(((img_bgr.astype(np.float32) - bg_mean) ** 2
                        / bg_std ** 2).sum(2))
    return ((distance > 2.5) * 255).astype(np.uint8)


def guided_filter_alpha(img_bgr: np.ndarray, rough_mask: np.ndarray,
                        radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Snap a binary mask to image edges, producing a soft alpha matte.

    A guided filter smooths the mask while respecting edges in the guide
    image, which is what recovers antenna and wing-fringe detail that GrabCut
    cuts off square. ``cv2.ximgproc`` lives in opencv-contrib; the bilateral
    filter fallback is noticeably worse but keeps the pipeline running on a
    plain opencv-python install.
    """
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    alpha = rough_mask.astype(np.float32) / 255.0
    try:
        gf = cv2.ximgproc.createGuidedFilter(guide[:, :, np.newaxis], radius, eps)
        refined = np.clip(gf.filter(alpha[:, :, np.newaxis])[:, :, 0], 0, 1)
    except Exception:
        refined = cv2.bilateralFilter(alpha, 9, 0.1, 15)
    return (refined * 255).astype(np.uint8)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Close holes and keep only the largest connected component.

    Insects photographed with spread wings produce masks with interior gaps;
    the flood-fill from the corner closes those. Keeping one component then
    discards dust specks and shadow fragments that survived thresholding.
    """
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    flood = m.copy()
    h, w = m.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m = cv2.bitwise_or(m, cv2.bitwise_not(flood))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n_labels <= 1:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return m
    best = int(np.argmax(areas)) + 1
    out = np.zeros_like(m)
    out[labels == best] = 255
    return out


def grabcut_with_hint(img_bgr: np.ndarray, hint: np.ndarray,
                      iters: int = 10) -> np.ndarray:
    """GrabCut seeded from a colour-distance hint rather than a rectangle.

    Rectangle initialisation assumes the object fills most of the box, which
    is wrong for a thin moth on a wide frame. Seeding probable-foreground from
    the hint, hard-foreground from its erosion, and hard-background from the
    image border gives GrabCut a far better starting partition. Falls back to
    rectangle mode if the mask-mode call throws, which happens when the hint
    is degenerate.
    """
    height, width = img_bgr.shape[:2]
    gc_mask = np.zeros((height, width), np.uint8)
    gc_mask[hint > 0] = cv2.GC_PR_FGD
    gc_mask[hint == 0] = cv2.GC_PR_BGD

    border = max(3, min(height, width) // 20)
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD
    gc_mask[cv2.erode(hint, np.ones((5, 5), np.uint8), iterations=3) > 0] = cv2.GC_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, gc_mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        gc_mask = np.zeros((height, width), np.uint8)
        rect = (border, border, width - 2 * border, height - 2 * border)
        cv2.grabCut(img_bgr, gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)

    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)


def full_extraction(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Complete classical extraction. Returns ``(bgra, coverage_fraction)``.

    Coverage is the fraction of pixels retained, and doubles as the quality
    signal: below 3% the extraction found essentially nothing, above 93% it
    kept the background too. Either extreme triggers a plain rectangle-mode
    GrabCut retry, which is cruder but rarely fails in both directions.
    """
    height, width = img_bgr.shape[:2]

    hint = cv2.dilate(colour_distance_mask(img_bgr), np.ones((5, 5), np.uint8),
                      iterations=2)
    binary = grabcut_with_hint(img_bgr, hint, iters=10)
    alpha = guided_filter_alpha(img_bgr, binary, radius=4, eps=1e-3)
    _, thresholded = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    alpha = guided_filter_alpha(img_bgr, clean_mask(thresholded), radius=3, eps=5e-4)
    coverage = float((alpha > 30).sum() / (height * width))

    if coverage < 0.03 or coverage > 0.93:
        border = max(3, min(height, width) // 20)
        gc_mask = np.zeros((height, width), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        rect = (border, border, width - 2 * border, height - 2 * border)
        cv2.grabCut(img_bgr, gc_mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
        fallback = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                            255, 0).astype(np.uint8)
        alpha = guided_filter_alpha(img_bgr, fallback, radius=4, eps=1e-3)
        coverage = float((alpha > 30).sum() / (height * width))

    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra, coverage


# --------------------------------------------------------------------------- #
# rembg, isolated in a subprocess
# --------------------------------------------------------------------------- #
_REMBG_WORKER = '''
import sys, os, io, gc, json
os.environ["REMBG_PROVIDERS"] = "CPUExecutionProvider"
os.environ["ONNXRUNTIME_PROVIDERS"] = "CPUExecutionProvider"
import cv2, numpy as np
from PIL import Image

input_path, output_path = sys.argv[1], sys.argv[2]
img_bgr = cv2.imread(input_path)
if img_bgr is None:
    sys.exit(1)

coverage, bgra = 0.0, None
try:
    from rembg import remove as rembg_remove, new_session
    session = new_session("u2netp", providers=["CPUExecutionProvider"])
    with open(input_path, "rb") as fh:
        raw = fh.read()
    result = rembg_remove(raw, session=session)
    img_pil = Image.open(io.BytesIO(result)).convert("RGBA")
    bgra = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    alpha = np.array(img_pil.split()[3])
    coverage = float((alpha > 30).sum() / alpha.size)
    del session, raw, result
    gc.collect()
except Exception as exc:
    sys.stderr.write("rembg: %s\\n" % exc)

if bgra is None or coverage < 0.05:
    h, w = img_bgr.shape[:2]
    border = max(3, min(h, w) // 20)
    gc_mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, (border, border, w - 2 * border, h - 2 * border),
                bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    coverage = float((mask > 30).sum() / mask.size)

cv2.imwrite(output_path, bgra)
print(json.dumps({"coverage": round(coverage, 3)}))
'''


def _worker_script(work_dir: Path) -> Path:
    path = Path(work_dir) / "_rembg_worker.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_REMBG_WORKER)
    return path


def extract_with_rembg(src: Path, dst: Path, worker: Path,
                       timeout: int = 120) -> float | None:
    """Run the rembg worker on one image. Returns coverage, or None on failure."""
    env = {**os.environ,
           "REMBG_PROVIDERS": "CPUExecutionProvider",
           "ONNXRUNTIME_PROVIDERS": "CPUExecutionProvider"}
    try:
        result = subprocess.run([sys.executable, str(worker), str(src), str(dst)],
                                capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1]).get("coverage", 0.0)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
def extract_folder(in_dir: str | Path, out_dir: str | Path,
                   work_dir: str | Path, method: str = "rembg",
                   force: bool = False, max_images: int | None = None,
                   min_coverage: float = 0.03) -> list[Path]:
    """Extract every image in a folder to a BGRA cutout PNG.

    Args:
        method: ``"rembg"`` (better mattes, slower) or ``"grabcut"``.
        force: re-extract even when the output already exists.
        min_coverage: discard results that kept less than this fraction of
            pixels — those are failures, not thin insects.

    Already-extracted files are reused unless ``force`` is set, which matters
    on a session-limited platform where a run may be interrupted and resumed.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)
    if max_images:
        sources = sources[:max_images]
    print(f"Extracting {len(sources)} images from {in_dir} using '{method}'")

    worker = _worker_script(work_dir) if method == "rembg" else None
    saved: list[Path] = []

    for i, src in enumerate(sources, 1):
        dst = out_dir / f"{src.stem}_cutout.png"

        if dst.exists() and not force:
            saved.append(dst)
            print(f"  [{i:03d}/{len(sources)}] {src.name[:48]:<48} cached")
            continue

        coverage = None
        if method == "rembg":
            coverage = extract_with_rembg(src, dst, worker)

        if coverage is None:      # rembg unavailable, failed, or method == grabcut
            img = cv2.imread(str(src))
            if img is None:
                print(f"  [{i:03d}/{len(sources)}] {src.name[:48]:<48} unreadable")
                continue
            bgra, coverage = full_extraction(img)
            if coverage > min_coverage:
                cv2.imwrite(str(dst), bgra)

        if dst.exists() and coverage is not None and coverage > min_coverage:
            flag = "ok " if coverage > 0.15 else "low"
            saved.append(dst)
            print(f"  [{i:03d}/{len(sources)}] {src.name[:48]:<48} {flag} cov={coverage:.0%}")
        else:
            print(f"  [{i:03d}/{len(sources)}] {src.name[:48]:<48} FAILED")

        gc.collect()

    print(f"\n  {len(saved)}/{len(sources)} cutouts saved to {out_dir}")
    return saved


def write_manifest(cutouts: Sequence[Path], manifest_path: str | Path) -> Path:
    """Record the cutout paths the generator should load.

    Keeping this list explicit rather than globbing at generation time means a
    dataset can be regenerated from exactly the same cutouts even if the
    extraction folder later gains or loses files.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps([str(p) for p in cutouts], indent=2))
    print(f"Wrote manifest: {manifest_path} ({len(cutouts)} entries)")
    return manifest_path
