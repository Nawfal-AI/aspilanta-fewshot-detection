"""Per-class AP extraction, comparison tables and plots.

Ultralytics returns its per-class metrics in a shape that has changed between
releases, so :func:`get_class_ap` is deliberately defensive. Everything else
here turns result dicts into the CSV and Markdown the project's tables are built
from.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["safe_float", "get_class_ap", "validate_model", "comparison_table",
           "save_table", "plot_comparison"]


def safe_float(value: Any) -> float | None:
    """Coerce a metric to float, returning None rather than raising.

    Ultralytics hands back numpy scalars, zero-dimensional arrays and
    occasionally None depending on version and on whether a class had any
    instances in the validation set.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.item())
        except Exception:
            return None


def get_class_ap(metrics, class_id: int, metric: str = "map50") -> float | None:
    """Per-class AP for one class id.

    ``metrics.box.ap_class_index`` maps positions in the per-class arrays back
    to class ids, and it only lists classes that actually appeared. Indexing
    the arrays by class id directly is the obvious mistake and silently
    returns another class's score.
    """
    box = getattr(metrics, "box", None)
    if box is None:
        return None

    indices = getattr(box, "ap_class_index", None)
    if indices is None:
        return None

    try:
        position = list(indices).index(class_id)
    except ValueError:
        return None      # class absent from this validation set

    arrays = {"map50": getattr(box, "ap50", None),
              "map": getattr(box, "ap", None),
              "precision": getattr(box, "p", None),
              "recall": getattr(box, "r", None)}
    array = arrays.get(metric)
    if array is None or position >= len(array):
        return None

    value = array[position]
    if metric == "map" and getattr(value, "ndim", 0) > 0:
        value = value.mean()      # ap is [N, 10] over IoU thresholds
    return safe_float(value)


def validate_model(weights: str | Path, data_yaml: str | Path, run_name: str,
                   project: str | Path, split: str = "val", device=None):
    """Run Ultralytics validation and return the metrics object."""
    from ultralytics import YOLO

    return YOLO(str(weights)).val(
        data=str(data_yaml), split=split, verbose=False, plots=False,
        device=device, project=str(project), name=run_name, exist_ok=True)


def comparison_table(entries: Sequence[tuple[str, Any]], target_class_id: int = 3):
    """Build a DataFrame of mAP and per-class AP from ``(label, metrics)`` pairs."""
    import pandas as pd

    rows = []
    for label, metrics in entries:
        rows.append({
            "variant": label,
            "mAP50": safe_float(metrics.box.map50),
            "mAP50_95": safe_float(metrics.box.map),
            "AP_target_50": get_class_ap(metrics, target_class_id, "map50"),
            "AP_target_50_95": get_class_ap(metrics, target_class_id, "map"),
        })
    return pd.DataFrame(rows)


def save_table(frame, path: str | Path, also_markdown: bool = True) -> Path:
    """Write a DataFrame to CSV, and optionally a paste-ready Markdown twin."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"\n{frame.to_string(index=False)}\nSaved: {path}")

    if also_markdown:
        try:
            md_path = path.with_suffix(".md")
            md_path.write_text(frame.to_markdown(index=False))
            print(f"Saved: {md_path}")
        except ImportError:
            pass      # to_markdown needs tabulate; not worth a hard dependency
    return path


def plot_comparison(frame, out_path: str | Path, x: str = "variant",
                    metrics: Sequence[str] = ("precision", "recall", "f1"),
                    title: str = "Pipeline comparison") -> Path:
    """Grouped bar chart of several metrics across variants."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = frame[x].tolist()
    positions = np.arange(len(labels))
    width = 0.8 / max(len(metrics), 1)

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 5))
    for i, metric in enumerate(metrics):
        if metric not in frame.columns:
            continue
        ax.bar(positions + i * width, frame[metric].fillna(0), width, label=metric)

    ax.set_xticks(positions + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")
    return out_path
