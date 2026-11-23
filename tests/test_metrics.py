"""Tests for IoU, greedy matching and ground-truth loading."""

import numpy as np
import pytest

from aspilanta.evaluation.metrics import (
    box_iou_xyxy,
    load_target_gt_xyxy,
    match_detections,
    precision_recall_f1,
)


class TestBoxIou:
    def test_identical_boxes(self):
        box = np.array([0, 0, 10, 10])
        assert box_iou_xyxy(box, np.array([[0, 0, 10, 10]]))[0] == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        box = np.array([0, 0, 10, 10])
        assert box_iou_xyxy(box, np.array([[20, 20, 30, 30]]))[0] == 0.0

    def test_half_overlap(self):
        box = np.array([0, 0, 10, 10])
        # Intersection 50, union 150.
        assert box_iou_xyxy(box, np.array([[5, 0, 15, 10]]))[0] == pytest.approx(1 / 3)

    def test_empty_input(self):
        assert len(box_iou_xyxy(np.array([0, 0, 1, 1]), np.empty((0, 4)))) == 0


class TestMatchDetections:
    def test_perfect_match(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        assert match_detections(boxes, boxes.copy()) == (2, 0, 0)

    def test_all_false_positives(self):
        preds = np.array([[0, 0, 10, 10]], dtype=float)
        assert match_detections(preds, np.empty((0, 4))) == (0, 1, 0)

    def test_all_false_negatives(self):
        gt = np.array([[0, 0, 10, 10]], dtype=float)
        assert match_detections(np.empty((0, 4)), gt) == (0, 0, 1)

    def test_duplicate_prediction_counts_as_false_positive(self):
        """Two boxes on one insect is a real error for a counting task."""
        preds = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        gt = np.array([[0, 0, 10, 10]], dtype=float)
        tp, fp, fn = match_detections(preds, gt)
        assert (tp, fp, fn) == (1, 1, 0)

    def test_below_iou_threshold_is_a_miss(self):
        preds = np.array([[0, 0, 10, 10]], dtype=float)
        gt = np.array([[8, 8, 18, 18]], dtype=float)
        assert match_detections(preds, gt, iou_thresh=0.5) == (0, 1, 1)


class TestLoadGroundTruth:
    def test_filters_to_target_class(self, tmp_path):
        label = tmp_path / "a.txt"
        label.write_text("3 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.1 0.1")
        boxes = load_target_gt_xyxy(label, 100, 100, target_class_id=3)
        assert boxes.shape == (1, 4)
        assert boxes[0].tolist() == [40.0, 40.0, 60.0, 60.0]

    def test_generator_numbering_needs_class_id_zero(self, tmp_path):
        """Raw generator labels use 0 for the target, not 3.

        Passing the wrong id yields an empty ground truth and a recall of zero
        that looks like a model failure rather than a bookkeeping error.
        """
        label = tmp_path / "synth.txt"
        label.write_text("0 0.5 0.5 0.2 0.2")
        assert len(load_target_gt_xyxy(label, 100, 100, target_class_id=3)) == 0
        assert len(load_target_gt_xyxy(label, 100, 100, target_class_id=0)) == 1

    def test_missing_file_returns_empty(self, tmp_path):
        assert len(load_target_gt_xyxy(tmp_path / "nope.txt", 10, 10)) == 0


class TestPrecisionRecallF1:
    def test_typical(self):
        p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=2)
        assert p == pytest.approx(0.8)
        assert r == pytest.approx(0.8)
        assert f1 == pytest.approx(0.8)

    def test_all_zero_is_zero_not_nan(self):
        assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
