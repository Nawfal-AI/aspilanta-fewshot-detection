"""Central configuration for the Aspilanta few-shot detection project.

In the original research notebook every path was a hard-coded ``/kaggle/working``
or ``/kaggle/input`` literal and every hyperparameter was a bare module-level
constant (``EXP1_EPOCHS``, ``EXP3_FREEZE``, ...). This module collects all of
that into dataclasses so the same code runs on Kaggle, Colab, or a local
machine without editing source.

Every default here reproduces the value used for the results reported in the
project. Override them from a YAML file (see ``configs/``) or from the
environment:

    ASPILANTA_WORK_DIR   working directory for datasets, runs, results
    ASPILANTA_INPUT_DIR  read-only directory holding the raw input datasets
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Paths",
    "ClassScheme",
    "TrainingConfig",
    "SolutionConfig",
    "GeneratorConfig",
    "DeploymentConfig",
    "ProjectConfig",
    "load_config",
]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _default_work_dir() -> Path:
    """Kaggle if we are on Kaggle, otherwise a local ./work directory."""
    env = os.environ.get("ASPILANTA_WORK_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path.cwd() / "work"


def _default_input_dir() -> Path:
    env = os.environ.get("ASPILANTA_INPUT_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/input").exists():
        return Path("/kaggle/input")
    return Path.cwd() / "data"


@dataclass
class Paths:
    """Every filesystem location the project reads or writes.

    ``work`` is writable scratch space; ``inputs`` is where the raw source
    datasets live and is treated as read-only.
    """

    work: Path = field(default_factory=_default_work_dir)
    inputs: Path = field(default_factory=_default_input_dir)

    # -- raw sources (set these to wherever your copies live) --------------- #
    yellow_sticky_traps: Path | None = None   # cloned md-121/yellow-sticky-traps-dataset
    aspilanta_raw: Path | None = None         # the 55 real Aspilanta photographs
    distractor_raw: Path | None = None        # hand-picked European moth photographs
    backgrounds: Path | None = None           # real empty trap-card photographs
    real_aspilanta_yolo: Path | None = None   # real Aspilanta images with YOLO labels

    def __post_init__(self) -> None:
        self.work = Path(self.work)
        self.inputs = Path(self.inputs)
        for name in ("yellow_sticky_traps", "aspilanta_raw", "distractor_raw",
                     "backgrounds", "real_aspilanta_yolo"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value))

    # -- derived working locations ------------------------------------------ #
    @property
    def base_yolo_dataset(self) -> Path:
        """Three-class YOLO conversion of the Yellow Sticky Traps dataset."""
        return self.work / "yolo_dataset"

    @property
    def cutouts(self) -> Path:
        return self.work / "cutouts"

    @property
    def cutouts_aspilanta(self) -> Path:
        return self.cutouts / "aspilanta"

    @property
    def cutouts_distractor(self) -> Path:
        return self.cutouts / "distractor"

    @property
    def synthetic(self) -> Path:
        """Output of the v7 synthetic generator."""
        return self.work / "generated_dataset_v7"

    @property
    def exp1_dataset(self) -> Path:
        return self.work / "exp1_dataset"

    @property
    def exp2_testset(self) -> Path:
        return self.work / "exp2_testset"

    @property
    def exp3_dataset(self) -> Path:
        """The merged four-class dataset the proposed solution trains on."""
        return self.work / "exp3_dataset"

    @property
    def target_domain_val(self) -> Path:
        return self.work / "target_domain_val_dataset"

    @property
    def target_domain_test(self) -> Path:
        """Decision-held-out test split carved in Section 11.2."""
        return self.work / "target_domain_test_dataset"

    @property
    def target_domain_valonly(self) -> Path:
        return self.work / "target_domain_valonly_dataset"

    @property
    def runs(self) -> Path:
        return self.work / "runs" / "detect"

    @property
    def results(self) -> Path:
        return self.work / "thesis_results"

    @property
    def outputs(self) -> Path:
        return self.work / "outputs"

    def ensure(self) -> Paths:
        """Create the writable directories that later stages assume exist."""
        for p in (self.work, self.results, self.outputs, self.cutouts_aspilanta,
                  self.cutouts_distractor):
            p.mkdir(parents=True, exist_ok=True)
        return self

    # -- candidate search lists (mirrors the notebook's fallback behaviour) -- #
    def real_aspilanta_candidates(self) -> list[Path]:
        candidates = [
            self.work / "real_ashpilanta",
            self.work / "asphilanta-1",
            self.inputs / "asphilanta-1",
            self.inputs / "real-ashpilanta",
        ]
        if self.real_aspilanta_yolo is not None:
            candidates.insert(0, self.real_aspilanta_yolo)
        return candidates

    def base_dataset_candidates(self) -> list[Path]:
        return [
            self.base_yolo_dataset,
            self.inputs / "yolo_dataset",
            self.work / "original_dataset",
        ]

    def synthetic_candidates(self) -> list[Path]:
        return [
            self.synthetic,
            self.inputs / "generated_dataset_v7",
            self.work / "generated_dataset",
        ]


# --------------------------------------------------------------------------- #
# Class scheme
# --------------------------------------------------------------------------- #
@dataclass
class ClassScheme:
    """The two label schemes used across the project.

    ``base`` is the three data-rich classes from the Yellow Sticky Traps
    dataset. ``full`` adds Aspilanta as class 3 — this is the scheme the
    proposed detector is trained and evaluated in.

    The generator writes its own two-class scheme internally
    (0 = Aspilanta, 1 = eu_moth distractor); ``generator_aspilanta_id`` and
    ``generator_distractor_id`` record it so label remapping stays explicit.
    """

    base: list[str] = field(default_factory=lambda: ["MP", "NT", "WF"])
    full: list[str] = field(default_factory=lambda: ["MP", "NT", "WF", "Aspilanta"])
    target_id: int = 3
    generator_aspilanta_id: int = 0
    generator_distractor_id: int = 1

    @property
    def n_base(self) -> int:
        return len(self.base)

    @property
    def n_full(self) -> int:
        return len(self.full)

    @property
    def synthetic_class_map(self) -> dict[int, int]:
        """Generator ids -> four-class ids, for the Aspilanta class only."""
        return {self.generator_aspilanta_id: self.target_id}

    @property
    def synthetic_drop_ids(self) -> set[int]:
        """Generator ids removed rather than remapped.

        Section 11.1 of the research notebook documents why: the eu_moth distractors are
        visual clutter, not ground truth. Remapping them collides with MP or NT
        depending on the map, which is the labelling defect that section fixes.
        """
        return {self.generator_distractor_id}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    """Stage-1 recipe: the three-class base detector.

    YOLOv8s rather than YOLOv8n — the architecture ablation in Section 6.6
    measures the difference.
    """

    model: str = "yolov8s.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    lrf: float = 0.01
    warmup_epochs: int = 3
    patience: int = 10
    freeze: int = 0
    workers: int = 2
    run_name: str = "exp1_base_3class"

    # augmentation
    mosaic: float = 1.0
    degrees: float = 10.0
    fliplr: float = 0.5
    flipud: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.5
    translate: float = 0.1

    def ultralytics_kwargs(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs, "imgsz": self.imgsz, "batch": self.batch,
            "workers": self.workers, "patience": self.patience,
            "freeze": self.freeze, "lr0": self.lr0, "lrf": self.lrf,
            "warmup_epochs": self.warmup_epochs, "mosaic": self.mosaic,
            "degrees": self.degrees, "fliplr": self.fliplr, "flipud": self.flipud,
            "hsv_h": self.hsv_h, "hsv_s": self.hsv_s, "hsv_v": self.hsv_v,
            "scale": self.scale, "translate": self.translate,
        }


@dataclass
class SolutionConfig:
    """Stage-2 recipe: the proposed four-class detector.

    ``freeze = 0`` is deliberate and is the one place where this project
    departs from its own earlier design. The Section 6.4 ablation contradicts
    backbone freezing on both validation sets:

        =====================  ========================  ==================
        configuration          mixed target-domain       real-only (11 img)
        =====================  ========================  ==================
        trainable backbone     AP@50 0.981 / 0.827       0.995 / 0.609
        frozen backbone        AP@50 0.906 / 0.516       0.941 / 0.471
        =====================  ========================  ==================

    Freezing defends against few-shot overfitting, but by the time fine-tuning
    starts the generator has already supplied ~2,000 images and 9,211
    instances, so the regime it protects against no longer exists. The frozen
    setting is kept as ``frozen_ablation_freeze`` for Section 6.4.

    Stated openly: this configuration was chosen because an ablation preferred
    it, which is selection on evaluation data. It also wins on the real-only
    subset, which shares no images or cutout provenance with training.
    """

    epochs: int = 80
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.005
    lrf: float = 0.01
    warmup_epochs: int = 5
    patience: int = 15
    freeze: int = 0
    frozen_ablation_freeze: int = 10
    workers: int = 2
    run_name: str = "solution_4class_detector"

    # augmentation, tuned for a rare small-object class
    mosaic: float = 1.0
    copy_paste: float = 0.3
    degrees: float = 15.0
    fliplr: float = 0.5
    flipud: float = 0.2
    scale: float = 0.5
    translate: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # schedule
    cos_lr: bool = True
    close_mosaic: int = 15
    label_smoothing: float = 0.05

    def ultralytics_kwargs(self, freeze: int | None = None,
                           copy_paste: float | None = None) -> dict[str, Any]:
        return {
            "epochs": self.epochs, "imgsz": self.imgsz, "batch": self.batch,
            "workers": self.workers, "patience": self.patience,
            "freeze": self.freeze if freeze is None else freeze,
            "lr0": self.lr0, "lrf": self.lrf,
            "warmup_epochs": self.warmup_epochs, "mosaic": self.mosaic,
            "copy_paste": self.copy_paste if copy_paste is None else copy_paste,
            "degrees": self.degrees, "fliplr": self.fliplr, "flipud": self.flipud,
            "scale": self.scale, "translate": self.translate,
            "hsv_h": self.hsv_h, "hsv_s": self.hsv_s, "hsv_v": self.hsv_v,
            "cos_lr": self.cos_lr, "close_mosaic": self.close_mosaic,
            "label_smoothing": self.label_smoothing,
        }


# --------------------------------------------------------------------------- #
# Synthetic generator
# --------------------------------------------------------------------------- #
@dataclass
class GeneratorConfig:
    """Parameters of the v7 copy-paste compositing generator.

    The size distribution is physically calibrated: insect length is drawn
    from a truncated Gaussian in millimetres and converted to pixels using the
    detected trap-pad width, so a composite is scale-correct regardless of the
    resolution of the background photograph.
    """

    # physical scale
    insect_mean_mm: float = 5.0
    insect_std_mm: float = 0.8
    insect_min_mm: float = 2.5
    insect_max_mm: float = 8.0
    pad_width_mm: float = 140.0

    # dataset size
    num_images: int = 2000
    train_ratio: float = 0.85
    seed: int = 42

    # instance counts per image
    max_aspilanta: int = 10
    max_distractor: int = 5
    max_attempts: int = 40
    max_overlap_iou: float = 0.30
    empty_image_every: int = 10          # ~10% empty traps, matching real cards

    # pad handling
    extract_pad: bool = True
    pad_min_frac: float = 0.05
    pad_max_frac: float = 0.95

    # compositing switches (the ablations in Section 6.2 flip these)
    use_poisson: bool = True
    use_distractors: bool = True
    use_pad_aware: bool = True
    use_shadow: bool = True
    use_color_jitter: bool = True
    use_lighting_match: bool = True

    # appearance
    organic_blur_kernel: tuple[int, int] = (3, 3)
    organic_blur_sigma: float = 0.5
    motion_blur_prob: float = 0.15
    sheen_prob: float = 0.30

    # label validation
    label_min_size: float = 0.005
    label_max_size: float = 0.90

    # output naming
    class_names: tuple[str, str] = ("Aspilanta", "eu_moth")
    jpeg_quality: int = 95


# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #
@dataclass
class DeploymentConfig:
    """Inference-time settings for the deployed detector.

    ``use_tiling`` and ``use_clip_filter`` both default to False, and both
    defaults are measured rather than assumed. On ``generated_dataset_v7``:

    * tiled pipeline      P=0.095  R=0.013  F1=0.023
    * single-pass         P=0.931  R=0.801  F1=0.861
    * single-pass + CLIP  P=0.936  R=0.675  F1=0.785

    Tiling loses badly because tile crops are out of distribution relative to
    how the detector was trained. The CLIP filter buys +0.5pp precision for
    -12.6pp recall. Section 11.3 recovers some of the CLIP idea by applying it
    selectively — only to detections below a trust threshold.
    """

    conf: float = 0.25            # detection confidence floor
    iou: float = 0.45             # per-image NMS
    deploy_conf: float = 0.40     # final reported confidence threshold
    use_tiling: bool = False
    use_clip_filter: bool = False
    target_only: bool = True      # report Aspilanta only; MP/NT/WF are context

    # tiling, used only when explicitly enabled
    tile_size: int = 640
    tile_overlap: float = 0.25
    tile_conf: float = 0.25
    tile_iou: float = 0.45
    merge_iou: float = 0.55
    center_dist_frac: float = 0.6
    tiling_size_threshold: int = 3000   # px; both dimensions must exceed this

    # CLIP filter
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_k_values: tuple[int, ...] = (1, 2, 5, 10, 20)
    clip_default_k: int = 5
    clip_fallback_threshold: float = 0.75
    selective_trust_threshold: float = 0.60   # Section 11.3


# --------------------------------------------------------------------------- #
# Top-level container
# --------------------------------------------------------------------------- #
@dataclass
class ProjectConfig:
    paths: Paths = field(default_factory=Paths)
    classes: ClassScheme = field(default_factory=ClassScheme)
    base_training: TrainingConfig = field(default_factory=TrainingConfig)
    solution: SolutionConfig = field(default_factory=SolutionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    # data splitting
    real_val_ratio: float = 0.20
    synth_val_ratio: float = 0.20
    random_seed: int = 42
    k_real_shots: int | None = None   # None = use every real image available

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def device(self) -> int | str:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def _apply(target: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            raise KeyError(f"Unknown option {key!r} for {type(target).__name__}")
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply(current, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Build a :class:`ProjectConfig`, optionally overlaying a YAML file.

    The YAML mirrors the dataclass structure, so a file only needs to name the
    keys it changes:

    .. code-block:: yaml

        paths:
          work: /data/aspilanta/work
        solution:
          epochs: 120
    """
    config = ProjectConfig()
    if path is None:
        return config

    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    for section, values in data.items():
        if not hasattr(config, section):
            raise KeyError(f"Unknown config section {section!r}")
        current = getattr(config, section)
        if isinstance(values, dict) and hasattr(current, "__dataclass_fields__"):
            _apply(current, values)
        else:
            setattr(config, section, values)

    config.paths.__post_init__()
    return config
