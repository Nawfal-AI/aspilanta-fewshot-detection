"""PASCAL VOC parsing and conversion to YOLO format."""

from __future__ import annotations

import pytest

from aspilanta.data.voc import convert_voc_to_yolo, discover_class_names, parse_voc_boxes


def test_parse_reads_boxes_and_size(voc_annotation):
    boxes, names, (width, height) = parse_voc_boxes(voc_annotation)
    assert (width, height) == (1000, 500)
    assert boxes[0] == (100, 50, 200, 150)
    assert names == ["WF", "UNKNOWN"]


def test_conversion_normalises_to_centre_form(voc_annotation, tmp_path):
    out = tmp_path / "card_001.txt"
    count = convert_voc_to_yolo(voc_annotation, ["MP", "NT", "WF"], out)

    assert count == 1        # the UNKNOWN object is skipped, not guessed at
    class_id, xc, yc, w, h = out.read_text().split()
    assert class_id == "2"                       # WF is index 2
    assert float(xc) == pytest.approx(0.15)      # (100+200)/2 / 1000
    assert float(yc) == pytest.approx(0.20)      # (50+150)/2 / 500
    assert float(w) == pytest.approx(0.10)
    assert float(h) == pytest.approx(0.20)


def test_class_order_determines_the_id(voc_annotation, tmp_path):
    out = tmp_path / "reordered.txt"
    convert_voc_to_yolo(voc_annotation, ["WF", "MP", "NT"], out)
    assert out.read_text().split()[0] == "0"


def test_image_with_no_known_classes_writes_an_empty_file(voc_annotation, tmp_path):
    out = tmp_path / "empty.txt"
    assert convert_voc_to_yolo(voc_annotation, ["MP", "NT"], out) == 0
    assert out.read_text() == ""


def test_discover_class_names_returns_sorted_unique(voc_annotation):
    assert discover_class_names(voc_annotation.parent) == ["UNKNOWN", "WF"]
