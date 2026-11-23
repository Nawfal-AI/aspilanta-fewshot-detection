"""Shared fixtures. Nothing here needs torch, ultralytics or a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def yolo_dataset(tmp_path):
    """A minimal but valid YOLO dataset root."""
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    import numpy as np
    cv2 = pytest.importorskip("cv2")

    for split, count in (("train", 3), ("val", 2)):
        for i in range(count):
            stem = f"{split}_{i:03d}"
            cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"),
                        np.full((64, 64, 3), 200, dtype=np.uint8))
            (root / "labels" / split / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1")
    return root


@pytest.fixture
def voc_annotation(tmp_path):
    """One PASCAL VOC XML with two objects, one of an unknown class."""
    path = tmp_path / "card_001.xml"
    path.write_text("""<annotation>
  <size><width>1000</width><height>500</height><depth>3</depth></size>
  <object>
    <name>WF</name>
    <bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox>
  </object>
  <object>
    <name>UNKNOWN</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox>
  </object>
</annotation>""")
    return path
