#!/usr/bin/env python3
"""Extract alpha-matted cutouts from raw specimen photographs.

    python scripts/extract_cutouts.py --aspilanta data/raw/aspilanta \
                                      --distractors data/raw/eu_moths

Writes BGRA PNGs plus a JSON manifest per class, and QC grids rendered over a
checkerboard so alpha failures are visible. Inspect those grids before
generating 2,000 images from bad mattes.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--aspilanta", type=Path, default=None,
                        help="Folder of raw target photographs")
    parser.add_argument("--distractors", type=Path, default=None,
                        help="Folder of raw distractor photographs")
    parser.add_argument("--method", choices=("rembg", "grabcut"), default="rembg",
                        help="rembg gives better mattes; grabcut needs no extra deps")
    parser.add_argument("--distractor-method", choices=("rembg", "grabcut"),
                        default="grabcut",
                        help="Distractors are numerous; grabcut is much faster")
    parser.add_argument("--max-distractors", type=int, default=500)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even where a cutout already exists")
    args = parser.parse_args()

    config = load(args)
    from aspilanta.synthetic.cutouts import extract_folder, write_manifest
    from aspilanta.synthetic.qc import cutout_grid

    aspilanta_src = args.aspilanta or config.paths.aspilanta_raw
    distractor_src = args.distractors or config.paths.distractor_raw
    if aspilanta_src is None:
        print("No target source given. Pass --aspilanta or set "
              "paths.aspilanta_raw in your config.")
        return 1

    targets = extract_folder(aspilanta_src, config.paths.cutouts_aspilanta,
                             config.paths.work, method=args.method,
                             force=args.force)
    write_manifest(targets, config.paths.outputs / "aspilanta_cutouts.json")
    if targets:
        cutout_grid(targets, config.paths.outputs / "aspilanta_cutouts_grid.png",
                    n=min(48, len(targets)), title="Aspilanta cutouts")

    if distractor_src is not None:
        distractors = extract_folder(distractor_src, config.paths.cutouts_distractor,
                                     config.paths.work,
                                     method=args.distractor_method,
                                     force=args.force,
                                     max_images=args.max_distractors)
        write_manifest(distractors, config.paths.outputs / "distractor_cutouts.json")
        if distractors:
            cutout_grid(distractors,
                        config.paths.outputs / "distractor_cutouts_grid.png",
                        n=min(48, len(distractors)), title="Distractor cutouts")
    else:
        print("\nNo distractor source given — the generator will run without them, "
              "which is the 'no_distractors' ablation rather than the method.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
