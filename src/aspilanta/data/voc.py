"""PASCAL VOC to YOLO conversion for the Yellow Sticky Traps dataset.

The dataset ships one XML per trap card with absolute-pixel bounding boxes;
Ultralytics wants normalised centre-form boxes in a plain text file. Notebook
cells 12, 15 and 19 did this three times with slightly different code — this
is the single reconciled version.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

__all__ = ["parse_voc_boxes", "convert_voc_to_yolo", "discover_class_names"]


def parse_voc_boxes(xml_file: str | Path) -> tuple[list[tuple[int, int, int, int]], list[str], tuple[int, int]]:
    """Read one VOC XML. Returns ``(boxes_xyxy, class_names, (width, height))``."""
    root = ET.parse(str(xml_file)).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes, names = [], []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        boxes.append((
            int(float(bbox.find("xmin").text)),
            int(float(bbox.find("ymin").text)),
            int(float(bbox.find("xmax").text)),
            int(float(bbox.find("ymax").text)),
        ))
        names.append(obj.find("name").text)
    return boxes, names, (width, height)


def convert_voc_to_yolo(xml_file: str | Path, classes: Sequence[str],
                        output_file: str | Path,
                        warn_unknown: bool = False) -> int:
    """Convert one VOC XML to a YOLO label file. Returns the number of boxes.

    Objects whose class name is not in ``classes`` are skipped rather than
    guessed at — the source dataset contains occasional stray labels.
    """
    boxes, names, (width, height) = parse_voc_boxes(xml_file)

    lines = []
    for (xmin, ymin, xmax, ymax), name in zip(boxes, names):
        if name not in classes:
            if warn_unknown:
                print(f"Warning: unknown class {name!r} in {xml_file}, skipping.")
            continue
        class_id = classes.index(name)
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} "
                     f"{box_w:.6f} {box_h:.6f}")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))
    return len(lines)


def discover_class_names(annotations_dir: str | Path, limit: int | None = None) -> list[str]:
    """Collect every distinct class name appearing in a folder of VOC XMLs.

    Useful before committing to a class order, since the Yellow Sticky Traps
    annotations use short codes (MP/NT/WF) whose exact spelling has varied
    between releases of the dataset.
    """
    found: set[str] = set()
    for i, xml_path in enumerate(sorted(Path(annotations_dir).glob("*.xml"))):
        if limit is not None and i >= limit:
            break
        _, names, _ = parse_voc_boxes(xml_path)
        found.update(names)
    return sorted(found)
