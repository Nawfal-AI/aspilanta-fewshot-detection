#!/usr/bin/env python3
"""Detect and count Aspilanta in one image or a folder of images.

    python scripts/detect.py --weights runs/detect/solution/weights/best.pt \
                             --image trap_photo.jpg --out-dir predictions/

Writes an annotated JPEG, a per-detection CSV and a JSON count summary per
input image.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--weights", type=Path, default=None,
                        help="Trained four-class checkpoint. Searched for if omitted.")
    parser.add_argument("--image", type=Path, required=True,
                        help="Image file, or a directory of images")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write results")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence floor (overrides the config)")
    parser.add_argument("--tiling", action="store_true",
                        help="Enable tiled inference. Measured to hurt badly on "
                             "this data (F1 0.023 vs 0.861) - verify on yours first.")
    parser.add_argument("--all-classes", action="store_true",
                        help="Annotate MP/NT/WF too, not just the target")
    args = parser.parse_args()

    config = load(args)
    from aspilanta.data.yolo_io import list_images
    from aspilanta.pipeline.detector import AspilantaDetector

    if args.conf is not None:
        config.deployment.conf = args.conf
    config.deployment.use_tiling = args.tiling

    detector = AspilantaDetector(
        weights=args.weights, config=config.deployment, classes=config.classes,
        target_only=not args.all_classes,
        search_roots=[config.paths.work, config.paths.inputs, Path(".")])

    out_dir = args.out_dir or (config.paths.results / "predictions")
    images = list_images(args.image) if args.image.is_dir() else [args.image]
    if not images:
        print(f"No images found at {args.image}")
        return 1

    total = 0
    for image in images:
        _, _, counts = detector.detect_and_save(image, out_dir)
        total += counts["aspilanta_count"]

    if len(images) > 1:
        print(f"\n{len(images)} images, {total} Aspilanta detected in total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
