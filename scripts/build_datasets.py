#!/usr/bin/env python3
"""Merge the source datasets into the four experiment datasets.

    python scripts/build_datasets.py --real-aspilanta data/raw/real_aspilanta_yolo

Builds exp1 (3-class base), exp2 (zero-shot eval set), exp3 (4-class merged
training set), and the common target-domain validation set, then prints a
class histogram for each.

Read those histograms. The Section 11.1 labelling defect - generator distractor
boxes being remapped into a real class instead of dropped - showed up as an
inflated MP or NT count and nothing else.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--original", type=Path, default=None,
                        help="Three-class YOLO dataset from prepare_base_dataset.py")
    parser.add_argument("--real-aspilanta", type=Path, default=None,
                        help="Real target images with YOLO labels")
    parser.add_argument("--synthetic", type=Path, default=None,
                        help="Generated dataset from generate_synthetic.py")
    parser.add_argument("--k-shots", type=int, default=None,
                        help="Cap real training images, for a strict few-shot run")
    parser.add_argument("--carve-test-split", action="store_true",
                        help="Also carve the decision-held-out test split (Section 11.2)")
    args = parser.parse_args()

    config = load(args)
    from aspilanta.data.builders import DatasetSources, build_all
    from aspilanta.data.splits import (
        carve_test_split,
        find_real_aspilanta_root,
        find_yolo_dataset,
    )
    from aspilanta.data.yolo_io import class_histogram

    paths = config.paths
    original = args.original or find_yolo_dataset(
        paths.base_dataset_candidates(), "base three-class dataset",
        search_roots=[paths.work, paths.inputs])
    synthetic = args.synthetic or find_yolo_dataset(
        paths.synthetic_candidates(), "synthetic dataset",
        search_roots=[paths.work, paths.inputs])
    real = args.real_aspilanta or find_real_aspilanta_root(
        paths.real_aspilanta_candidates(), search_roots=[paths.work, paths.inputs])

    sources = DatasetSources.build(
        original, real, synthetic,
        real_val_ratio=config.real_val_ratio,
        synth_val_ratio=config.synth_val_ratio,
        seed=config.random_seed,
        k_real_shots=args.k_shots if args.k_shots is not None else config.k_real_shots)

    yamls = build_all(sources, paths, config.classes)

    print("\nClass histograms (verify these before training):")
    for name, root in (("exp1 train", paths.exp1_dataset / "labels/train"),
                       ("exp3 train", paths.exp3_dataset / "labels/train"),
                       ("exp3 val", paths.exp3_dataset / "labels/val"),
                       ("target val", paths.target_domain_val / "labels/val")):
        print(f"  {name:<12} {class_histogram(root, config.classes.full)}")
    print("\n  The target-domain validation set should contain the target class "
          "only.\n  Any MP/NT/WF there means distractor labels leaked through.")

    if args.carve_test_split:
        print()
        carve_test_split(paths.target_domain_val, paths.target_domain_test,
                         paths.target_domain_valonly, config.classes.full,
                         seed=1337)

    print("\nDataset YAMLs:")
    for name, path in yamls.items():
        print(f"  {name:<12} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
