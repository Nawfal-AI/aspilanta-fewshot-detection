#!/usr/bin/env python3
"""Generate the synthetic training dataset by copy-paste compositing.

    python scripts/generate_synthetic.py --backgrounds data/raw/backgrounds

Reads the cutout manifests written by extract_cutouts.py and composites them
onto real trap-card photographs under physically calibrated scale, pad-aware
placement, and Poisson blending.

The ablation switches produce the Section 6.2 variants:

    --no-poisson       alpha compositing only
    --no-distractors   target insects alone
    --no-pad-aware     paste anywhere on the photograph, not just the pad
"""

from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--backgrounds", type=Path, default=None,
                        help="Folder of real empty trap-card photographs")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output dataset root")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-poisson", action="store_true")
    parser.add_argument("--no-distractors", action="store_true")
    parser.add_argument("--no-pad-aware", action="store_true")
    parser.add_argument("--skip-qc", action="store_true")
    args = parser.parse_args()

    config = load(args)
    from aspilanta.synthetic.generator import generate_dataset
    from aspilanta.synthetic.qc import verification_grid

    backgrounds = args.backgrounds or config.paths.backgrounds
    if backgrounds is None:
        print("No backgrounds given. Pass --backgrounds or set paths.backgrounds.")
        return 1

    generator = config.generator
    if args.num_images is not None:
        generator.num_images = args.num_images
    if args.seed is not None:
        generator.seed = args.seed
    generator.use_poisson = not args.no_poisson
    generator.use_distractors = not args.no_distractors
    generator.use_pad_aware = not args.no_pad_aware

    aspilanta_manifest = config.paths.outputs / "aspilanta_cutouts.json"
    distractor_manifest = config.paths.outputs / "distractor_cutouts.json"
    if not aspilanta_manifest.exists():
        print(f"Missing {aspilanta_manifest}. Run scripts/extract_cutouts.py first.")
        return 1

    targets = json.loads(aspilanta_manifest.read_text())
    distractors = (json.loads(distractor_manifest.read_text())
                   if distractor_manifest.exists() else [])

    out = args.out or config.paths.synthetic
    generate_dataset(targets, distractors, backgrounds, out, generator)

    if not args.skip_qc:
        verification_grid(out, Path(out) / "verification_grid.png",
                          n=16, seed=generator.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
