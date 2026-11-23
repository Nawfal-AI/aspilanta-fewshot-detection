"""Few-shot detection of *Aspilanta* on yellow sticky traps.

MSc research project, Artificial Intelligence, Università della Svizzera
italiana (USI).

The method combines one technique from each of the three few-shot object
detection families:

===================  =========================================================
family               technique
===================  =========================================================
data-oriented        copy-paste compositing (synthetic generator v7)
model-oriented       YOLOv8s staged transfer learning, trainable backbone
algorithm-oriented   frozen CLIP few-shot prototype filter (optional)
===================  =========================================================

Quick start::

    from aspilanta import AspilantaDetector

    detector = AspilantaDetector(weights="runs/detect/solution/weights/best.pt")
    annotated, detections, counts = detector.detect("trap_photo.jpg")
    print(counts["aspilanta_count"])
"""

from .config import (
    ClassScheme,
    DeploymentConfig,
    GeneratorConfig,
    Paths,
    ProjectConfig,
    SolutionConfig,
    TrainingConfig,
    load_config,
)

__version__ = "1.0.0"
__author__ = "Nawfal Abdul Malick"

__all__ = [
    "AspilantaDetector",
    "ProjectConfig", "Paths", "ClassScheme", "TrainingConfig",
    "SolutionConfig", "GeneratorConfig", "DeploymentConfig", "load_config",
    "__version__",
]


def __getattr__(name):
    # AspilantaDetector pulls in ultralytics and torch, which take several
    # seconds to import. Deferring it keeps `import aspilanta` cheap for the
    # data-preparation paths that never touch a model.
    if name == "AspilantaDetector":
        from .pipeline.detector import AspilantaDetector
        return AspilantaDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
