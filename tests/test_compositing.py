"""Tests for the compositing primitives and label validation."""

import pytest

from aspilanta.synthetic.compositing import soft_iou, validate_label
from aspilanta.synthetic.pad import pixels_per_mm


class TestSoftIou:
    def test_identical(self):
        assert soft_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_disjoint(self):
        assert soft_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert soft_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0

    def test_partial(self):
        assert soft_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


class TestValidateLabel:
    def test_accepts_a_normal_box(self):
        assert validate_label(0.5, 0.5, 0.05, 0.05)

    def test_rejects_centre_outside_the_frame(self):
        assert not validate_label(1.2, 0.5, 0.05, 0.05)
        assert not validate_label(0.5, -0.1, 0.05, 0.05)

    def test_rejects_a_collapsed_box(self):
        """Slivers are training noise Ultralytics accepts without complaint."""
        assert not validate_label(0.5, 0.5, 0.001, 0.05)

    def test_rejects_an_oversized_box(self):
        assert not validate_label(0.5, 0.5, 0.95, 0.95)

    def test_rejects_a_box_that_ran_off_the_edge(self):
        assert not validate_label(0.03, 0.5, 0.2, 0.05)

    def test_tolerates_a_small_overhang(self):
        """Real insects do sit partly off the pad edge."""
        assert validate_label(0.06, 0.5, 0.1, 0.05)


class TestPixelsPerMm:
    def test_standard_trap_card(self):
        # A 140 mm card imaged 1400 px wide gives exactly 10 px/mm.
        assert pixels_per_mm(1400, 140.0) == pytest.approx(10.0)

    def test_scale_is_resolution_independent(self):
        """A 5 mm insect stays 5 mm regardless of background resolution."""
        low = pixels_per_mm(700, 140.0)
        high = pixels_per_mm(2800, 140.0)
        assert 5 * low == pytest.approx(25.0)
        assert 5 * high == pytest.approx(100.0)
        assert (5 * high) / (5 * low) == pytest.approx(2800 / 700)
