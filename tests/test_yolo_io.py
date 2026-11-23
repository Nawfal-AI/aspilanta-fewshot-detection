"""Tests for label remapping and dataset copying.

The distractor-drop tests cover the Section 11.1 defect specifically. That bug
survived a full experimental campaign because a remapped distractor label is
indistinguishable from a real one once written — nothing crashes, the numbers
just quietly shift.
"""

from pathlib import Path

import pytest

from aspilanta.data.yolo_io import (
    class_histogram,
    copy_one_image_and_label,
    read_yolo_lines,
    remap_label_lines,
    strip_distractor_labels,
    write_yaml,
    yolo_xywh_to_xyxy,
)


class TestRemapLabelLines:
    def test_passes_through_unchanged(self):
        lines = ["0 0.5 0.5 0.1 0.1", "2 0.2 0.3 0.05 0.05"]
        assert remap_label_lines(lines) == lines

    def test_applies_class_map(self):
        out = remap_label_lines(["0 0.5 0.5 0.1 0.1"], class_id_map={0: 3})
        assert out == ["3 0.5 0.5 0.1 0.1"]

    def test_force_overrides_every_id(self):
        out = remap_label_lines(["0 0.5 0.5 0.1 0.1", "7 0.2 0.2 0.1 0.1"],
                                force_class_id=3)
        assert [line.split()[0] for line in out] == ["3", "3"]

    def test_drops_rather_than_remaps(self):
        """The Section 11.1 fix: distractors are removed, not renumbered."""
        lines = ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"]
        out = remap_label_lines(lines, class_id_map={0: 3}, drop_class_ids={1})
        assert out == ["3 0.5 0.5 0.1 0.1"]

    def test_drop_takes_precedence_over_force(self):
        """A dropped id must not be resurrected by force_class_id."""
        out = remap_label_lines(["1 0.2 0.2 0.1 0.1"], force_class_id=3,
                                drop_class_ids={1})
        assert out == []

    def test_skips_malformed_lines(self):
        out = remap_label_lines(["0 0.5 0.5", "", "not a label",
                                 "0 0.5 0.5 0.1 0.1"])
        assert out == ["0 0.5 0.5 0.1 0.1"]

    def test_accepts_float_class_ids(self):
        """Some tools write '0.0' rather than '0'."""
        out = remap_label_lines(["0.0 0.5 0.5 0.1 0.1"], class_id_map={0: 3})
        assert out == ["3 0.5 0.5 0.1 0.1"]


class TestYoloXywhToXyxy:
    def test_centre_box_without_padding(self):
        assert yolo_xywh_to_xyxy([0.5, 0.5, 0.5, 0.5], 100, 100, padding=0.0) \
            == (25, 25, 75, 75)

    def test_padding_expands_by_box_fraction(self):
        x1, y1, x2, y2 = yolo_xywh_to_xyxy([0.5, 0.5, 0.5, 0.5], 100, 100,
                                           padding=0.10)
        assert (x1, y1, x2, y2) == (20, 20, 80, 80)

    def test_clamps_to_image_bounds(self):
        x1, y1, x2, y2 = yolo_xywh_to_xyxy([0.5, 0.5, 1.0, 1.0], 100, 100,
                                           padding=0.5)
        assert x1 == 0 and y1 == 0 and x2 == 100 and y2 == 100


class TestCopyAndHistogram:
    @pytest.fixture
    def dataset(self, tmp_path):
        src_img = tmp_path / "src" / "images"
        src_lbl = tmp_path / "src" / "labels"
        src_img.mkdir(parents=True)
        src_lbl.mkdir(parents=True)
        (src_img / "synth_000001.jpg").write_bytes(b"not-a-real-jpeg")
        (src_lbl / "synth_000001.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1")
        return tmp_path

    def test_copy_applies_prefix_and_drop(self, dataset):
        dst_img = dataset / "out" / "images" / "train"
        dst_lbl = dataset / "out" / "labels" / "train"
        ok = copy_one_image_and_label(
            dataset / "src" / "images" / "synth_000001.jpg", dst_img, dst_lbl,
            prefix="synth_train_",
            label_path=dataset / "src" / "labels" / "synth_000001.txt",
            class_id_map={0: 3}, drop_class_ids={1})

        assert ok
        assert (dst_img / "synth_train_synth_000001.jpg").exists()
        lines = read_yolo_lines(dst_lbl / "synth_train_synth_000001.txt")
        assert lines == ["3 0.5 0.5 0.1 0.1"]

    def test_copy_reports_missing_label(self, dataset):
        (dataset / "src" / "images" / "orphan.jpg").write_bytes(b"x")
        ok = copy_one_image_and_label(
            dataset / "src" / "images" / "orphan.jpg",
            dataset / "out" / "images", dataset / "out" / "labels")
        assert ok is False

    def test_class_histogram_names_classes(self, dataset):
        counts = class_histogram(dataset / "src" / "labels",
                                 ["MP", "NT", "WF", "Aspilanta"])
        assert counts == {"MP": 1, "NT": 1}

    def test_strip_distractor_labels(self, dataset):
        root = dataset / "built"
        (root / "labels" / "val").mkdir(parents=True)
        (root / "labels" / "val" / "synth_val_a.txt").write_text(
            "3 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1")
        (root / "labels" / "val" / "real_val_b.txt").write_text(
            "3 0.4 0.4 0.1 0.1")

        result = strip_distractor_labels(root, moth_class_ids=(1,))
        assert result["removed"] == 1
        # Real files carry no synthetic marker and must be left untouched.
        assert (root / "labels" / "val" / "real_val_b.txt").read_text() \
            == "3 0.4 0.4 0.1 0.1"


def test_write_yaml_round_trips(tmp_path):
    import yaml

    path = write_yaml(tmp_path / "ds", tmp_path / "ds.yaml",
                      ["MP", "NT", "WF", "Aspilanta"], quiet=True)
    data = yaml.safe_load(Path(path).read_text())
    assert data["nc"] == 4
    assert data["names"][3] == "Aspilanta"
    assert data["train"] == "images/train"
