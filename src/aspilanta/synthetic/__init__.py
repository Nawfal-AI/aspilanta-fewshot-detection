"""Synthetic data generation: cutout extraction, compositing, the v7 generator."""

from .compositing import paste_insect, sample_size_px, soft_iou, validate_label
from .cutouts import extract_folder, full_extraction, write_manifest
from .generator import generate_dataset, load_bgra_images
from .pad import detect_pad_bbox, extract_clean_pad, pixels_per_mm
from .qc import cutout_grid, verification_grid

__all__ = [
    "paste_insect", "sample_size_px", "soft_iou", "validate_label",
    "extract_folder", "full_extraction", "write_manifest",
    "generate_dataset", "load_bgra_images",
    "detect_pad_bbox", "extract_clean_pad", "pixels_per_mm",
    "cutout_grid", "verification_grid",
]
