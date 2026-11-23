"""Deployment inference: the detector, tiling, and cross-tile NMS."""

from .tiling import cross_tile_nms, make_tiles, tiled_predict

__all__ = ["AspilantaDetector", "cross_tile_nms", "make_tiles", "tiled_predict"]


def __getattr__(name):
    if name == "AspilantaDetector":
        from .detector import AspilantaDetector
        return AspilantaDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
