"""Model training and the CLIP few-shot filter."""

from .train import (
    find_weights,
    train_base_detector,
    train_solution,
    train_tfa_baseline,
    train_variant,
)

__all__ = ["find_weights", "train_base_detector", "train_solution",
           "train_tfa_baseline", "train_variant",
           "ClipEmbedder", "ClipPrototypeFilter", "crop_objects_from_dataset",
           "build_support_crops"]


def __getattr__(name):
    # transformers is a heavy import and only the CLIP path needs it.
    if name in ("ClipEmbedder", "ClipPrototypeFilter",
                "crop_objects_from_dataset", "build_support_crops"):
        from . import clip_filter
        return getattr(clip_filter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
