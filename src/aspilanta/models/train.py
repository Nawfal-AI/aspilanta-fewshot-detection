"""Detector training: the two staged runs and the ablation-parameterised form.

The method is a two-stage transfer chain:

1. **Base detector.** YOLOv8s, ImageNet-initialised, trained on the three
   data-rich classes from the Yellow Sticky Traps dataset. Aspilanta is absent.
2. **Solution.** The same network fine-tuned to four classes on the original
   data plus real and synthetic Aspilanta, with the backbone left trainable.

Stage 2 starting from stage 1 rather than from ImageNet is itself an ablation
(Section 6.4, ``fresh_imagenet_init``), as is the trainable backbone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Paths, SolutionConfig, TrainingConfig

__all__ = ["resolve_device", "find_weights", "train_base_detector",
           "train_solution", "train_variant", "tfa_freeze_callback",
           "train_tfa_baseline"]


def resolve_device(device: int | str | None = None) -> int | str:
    if device is not None:
        return device
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def find_weights(run_names: Sequence[str], search_roots: Sequence[Path],
                 subdirs: Sequence[str] = ("runs/detect", "ablations", "runs", ""),
                 verbose: bool = True) -> Path | None:
    """Locate a ``best.pt`` for any of ``run_names``, preferring earlier names.

    Order matters more than it looks. The proposed model is archived under
    several historical names, and the *frozen* configuration — now only an
    ablation — sits under ``exp3_4class_synthetic_finetuned``. Picking that one
    by accident silently costs AP@50 0.981 to 0.906, with no error to notice.
    Pass the proposed names first.
    """
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for sub in subdirs:
            for name in run_names:
                candidate = (root / sub / name / "weights" / "best.pt" if sub
                             else root / name / "weights" / "best.pt")
                if candidate.exists():
                    if verbose:
                        print(f"Found weights: {candidate}")
                    return candidate

    for root in search_roots:      # slower recursive fallback
        root = Path(root)
        if not root.exists():
            continue
        for name in run_names:
            hits = sorted(root.rglob(f"{name}/weights/best.pt"))
            if hits:
                if verbose:
                    print(f"Found weights (recursive): {hits[0]}")
                return hits[0]
    return None


def _train(init_weights: str | Path, data_yaml: str | Path, run_name: str,
           project: Path, kwargs: dict[str, Any], device: int | str,
           callbacks: dict[str, Any] | None = None) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(init_weights))
    for event, fn in (callbacks or {}).items():
        model.add_callback(event, fn)

    model.train(data=str(data_yaml), device=device, project=str(project),
                name=run_name, exist_ok=True, save=True, plots=True, **kwargs)

    best = Path(project) / run_name / "weights" / "best.pt"
    print(f"Finished {run_name} -> {best}  exists={best.exists()}")
    return best


def train_base_detector(data_yaml: str | Path, paths: Paths,
                        config: TrainingConfig | None = None,
                        device: int | str | None = None,
                        init_weights: str | None = None) -> Path:
    """Stage 1: the three-class base detector.

    YOLOv8s rather than YOLOv8n. The size ablation (Section 6.6) measures the
    difference directly; 's' wins on small insects at essentially the same
    inference cost on the target hardware.
    """
    config = config or TrainingConfig()
    device = resolve_device(device)

    print("=" * 70)
    print("STAGE 1 — base three-class detector")
    print(f"  init={init_weights or config.model}  data={data_yaml}")
    print("=" * 70)

    return _train(init_weights or config.model, data_yaml, config.run_name,
                  paths.runs, config.ultralytics_kwargs(), device)


def train_solution(data_yaml: str | Path, init_weights: str | Path, paths: Paths,
                   config: SolutionConfig | None = None,
                   device: int | str | None = None,
                   run_name: str | None = None) -> Path:
    """Stage 2: the proposed four-class detector.

    ``freeze=0`` — the backbone stays trainable. See
    :class:`~aspilanta.config.SolutionConfig` for the ablation evidence behind
    that choice and for the caveat about having selected it on evaluation data.

    ``copy_paste`` augmentation and ``close_mosaic`` both target the same
    problem from opposite ends: the rare class needs extra exposure early, and
    the final epochs need clean un-mosaicked images so the model calibrates on
    what it will actually see at inference.
    """
    config = config or SolutionConfig()
    device = resolve_device(device)

    print("=" * 70)
    print("STAGE 2 — proposed four-class detector")
    print(f"  init={init_weights}")
    print(f"  backbone={'TRAINABLE' if config.freeze == 0 else f'frozen to {config.freeze}'}")
    print(f"  epochs={config.epochs}  lr0={config.lr0}  copy_paste={config.copy_paste}")
    print("=" * 70)

    return _train(init_weights, data_yaml, run_name or config.run_name,
                  paths.runs, config.ultralytics_kwargs(), device)


def train_variant(run_name: str, data_yaml: str | Path, init_weights: str | Path,
                  paths: Paths, config: SolutionConfig | None = None,
                  freeze: int | None = None, copy_paste: float | None = None,
                  device: int | str | None = None) -> Path:
    """One ablation run: the stage-2 recipe with specific settings overridden.

    Everything except the named overrides is held at the proposed values, so a
    difference in the result is attributable to the override alone.
    """
    config = config or SolutionConfig()
    device = resolve_device(device)

    print("=" * 70)
    print(f"ABLATION: {run_name}")
    print(f"  data={data_yaml}  init={init_weights}  "
          f"freeze={config.freeze if freeze is None else freeze}  "
          f"copy_paste={config.copy_paste if copy_paste is None else copy_paste}")
    print("=" * 70)

    return _train(init_weights, data_yaml, run_name, paths.runs,
                  config.ultralytics_kwargs(freeze=freeze, copy_paste=copy_paste),
                  device)


# --------------------------------------------------------------------------- #
# TFA baseline
# --------------------------------------------------------------------------- #
def tfa_freeze_callback(trainer) -> None:
    """Freeze everything except the Detect head's classification branch.

    This is the YOLOv8 equivalent of TFA's two-stage recipe (Wang et al.,
    2020), where only the novel-class classifier is trained on the few-shot
    data. Registered on ``on_train_start`` so it runs *after* Ultralytics'
    own freeze-by-index logic and can override it — index-based freezing
    cannot express "the cls branch of the head but not the box branch".

    ``cv2`` is the box-regression branch of ``Detect``, ``cv3`` the
    classification branch. If a future Ultralytics release renames them this
    raises rather than silently training everything, which would turn the
    baseline into an ordinary fine-tune and quietly invalidate the comparison.
    """
    model = trainer.model
    head = model.model[-1]

    for param in model.parameters():
        param.requires_grad_(False)

    if not hasattr(head, "cv3"):
        raise AttributeError(
            "Detect head has no 'cv3' attribute — Ultralytics internals may have "
            "changed. Inspect trainer.model.model[-1] to find the classification "
            "branch and update this callback.")

    unfrozen = 0
    for param in head.cv3.parameters():
        param.requires_grad_(True)
        unfrozen += param.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"[TFA] Unfrozen (classification head only): {unfrozen:,} / {total:,} "
          f"params ({100 * unfrozen / total:.2f}%)")


def train_tfa_baseline(data_yaml: str | Path, init_weights: str | Path,
                       paths: Paths, config: SolutionConfig | None = None,
                       device: int | str | None = None,
                       run_name: str = "baseline_tfa") -> Path:
    """Model-oriented baseline: TFA-style frozen-detector fine-tuning.

    Trained on the same data with the same schedule as the proposed solution,
    so the only difference is what is allowed to learn. ``freeze=[]`` is passed
    to Ultralytics deliberately — all freezing is handled by the callback.
    """
    config = config or SolutionConfig()
    device = resolve_device(device)

    print("=" * 70)
    print("BASELINE — TFA-style frozen detector (classification head only)")
    print(f"  data={data_yaml}  init={init_weights}")
    print("=" * 70)

    kwargs = config.ultralytics_kwargs()
    kwargs["freeze"] = []
    return _train(init_weights, data_yaml, run_name, paths.runs, kwargs, device,
                  callbacks={"on_train_start": tfa_freeze_callback})
