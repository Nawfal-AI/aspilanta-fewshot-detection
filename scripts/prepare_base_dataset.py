#!/usr/bin/env python3
"""Convert the Yellow Sticky Traps dataset from PASCAL VOC to YOLO format.

    git clone https://github.com/md-121/yellow-sticky-traps-dataset.git
    python scripts/prepare_base_dataset.py --source yellow-sticky-traps-dataset

Produces a three-class YOLO dataset plus its ``dataset.yaml``. Nothing here
involves Aspilanta; this is the data-rich foundation the transfer chain starts
from.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Cloned dataset root (contains images/ and annotations/)")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--inspect", action="store_true",
                        help="Print the class names found and exit without converting")
    args = parser.parse_args()

    config = load(args)
    import shutil

    from aspilanta.data.splits import split_train_val_test
    from aspilanta.data.voc import convert_voc_to_yolo, discover_class_names
    from aspilanta.data.yolo_io import (
        class_histogram,
        ensure_yolo_dirs,
        list_images,
        safe_clear_dir,
        write_yaml,
    )

    annotations = args.source / "annotations"
    images_dir = args.source / "images"
    if not annotations.exists() or not images_dir.exists():
        print(f"Expected images/ and annotations/ under {args.source}")
        return 1

    found = discover_class_names(annotations, limit=200)
    print(f"Class names present in the annotations: {found}")
    print(f"Class order this project uses:           {config.classes.base}")
    if args.inspect:
        return 0

    missing = set(config.classes.base) - set(found)
    if missing:
        print(f"\nWARNING: {sorted(missing)} not seen in the first 200 XML files.")
        print("The dataset's short codes have varied between releases; if these "
              "look wrong, set classes.base in your config to match.")

    out = safe_clear_dir(config.paths.base_yolo_dataset)
    ensure_yolo_dirs(out, splits=("train", "val", "test"))

    images = list_images(images_dir)
    print(f"\nFound {len(images)} images")
    train, val, test = split_train_val_test(images, args.train, args.val,
                                            seed=config.random_seed)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    total = 0
    for split, files in (("train", train), ("val", val), ("test", test)):
        for image in files:
            shutil.copy2(image, out / "images" / split / image.name)
            xml = annotations / f"{image.stem}.xml"
            label = out / "labels" / split / f"{image.stem}.txt"
            if xml.exists():
                total += convert_voc_to_yolo(xml, config.classes.base, label)
            else:
                label.write_text("")

    print(f"\nConverted {total} annotations.")
    for split in ("train", "val", "test"):
        print(f"  {split}: {class_histogram(out / 'labels' / split, config.classes.base)}")

    write_yaml(out, config.paths.work / "dataset.yaml", config.classes.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
