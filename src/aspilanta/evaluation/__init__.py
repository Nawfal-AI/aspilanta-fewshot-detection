"""Evaluation: metrics, the pipeline harness, and result reporting."""

from .harness import evaluate_bucketed_by_size, evaluate_pipeline, sweep_confidence
from .metrics import box_iou_xyxy, load_target_gt_xyxy, match_detections, precision_recall_f1
from .reporting import (
                      comparison_table,
                      get_class_ap,
                      plot_comparison,
                      safe_float,
                      save_table,
                      validate_model,
)

__all__ = [
    "evaluate_bucketed_by_size", "evaluate_pipeline", "sweep_confidence",
    "box_iou_xyxy", "load_target_gt_xyxy", "match_detections", "precision_recall_f1",
    "comparison_table", "get_class_ap", "plot_comparison", "safe_float",
    "save_table", "validate_model",
]
