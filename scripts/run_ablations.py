#!/usr/bin/env python3
"""Run and score the nine ablation studies.

    python scripts/run_ablations.py --only real_only,frozen_backbone
    python scripts/run_ablations.py --evaluate-only

Nine full training runs do not fit in one session on most hosted GPUs, so
completed runs are reused by default. Re-invoke until every row appears.

The compositing ablations regenerate synthetic data at a reduced image count
(``--variant-images``) to keep the sweep tractable. That is a confound: those
variants train on less data as well as different data, so their numbers are
not directly comparable with the main run. Raise it to match the main dataset
if you need a clean comparison; it costs generation time only.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--only", default=None,
                        help="Comma-separated ablation names to run")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Score existing runs without training anything")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even where a run already exists")
    parser.add_argument("--variant-images", type=int, default=500,
                        help="Images per compositing-ablation variant")
    args = parser.parse_args()

    config = load(args)
    from aspilanta.ablations.runner import ABLATIONS, evaluate_ablations, run_ablations
    from aspilanta.data.builders import DatasetSources, build_exp3
    from aspilanta.data.splits import find_real_aspilanta_root, find_yolo_dataset
    from aspilanta.data.yolo_io import list_images
    from aspilanta.evaluation.reporting import save_table
    from aspilanta.models.train import train_base_detector, train_variant

    paths = config.paths
    specs = ABLATIONS
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        specs = [s for s in ABLATIONS if s.name in wanted]
        unknown = wanted - {s.name for s in ABLATIONS}
        if unknown:
            print(f"Unknown ablation names: {sorted(unknown)}")
            print(f"Available: {[s.name for s in ABLATIONS]}")
            return 1

    base_weights = paths.runs / config.base_training.run_name / "weights" / "best.pt"
    target_val_yaml = paths.work / "target_domain_val.yaml"
    base_val_yaml = paths.work / "exp1.yaml"

    if args.evaluate_only:
        results = {s.name: paths.runs / s.run_name / "weights" / "best.pt"
                   for s in specs}
    else:
        sources = DatasetSources.build(
            find_yolo_dataset(paths.base_dataset_candidates(), "base dataset",
                              search_roots=[paths.work, paths.inputs]),
            find_real_aspilanta_root(paths.real_aspilanta_candidates(),
                                     search_roots=[paths.work, paths.inputs]),
            find_yolo_dataset(paths.synthetic_candidates(), "synthetic dataset",
                              search_roots=[paths.work, paths.inputs]),
            real_val_ratio=config.real_val_ratio,
            synth_val_ratio=config.synth_val_ratio,
            seed=config.random_seed)

        def build_dataset(spec):
            if not spec.include_synthetic:
                return build_exp3(sources, paths, config.classes,
                                  out_root=paths.work / f"ablation_{spec.name}_dataset",
                                  yaml_name=f"ablation_{spec.name}.yaml",
                                  include_synthetic=False)

            if spec.generator_overrides:
                variant_root = _generate_variant(config, spec, args.variant_images)
                return build_exp3(
                    sources, paths, config.classes,
                    out_root=paths.work / f"ablation_{spec.name}_dataset",
                    yaml_name=f"ablation_{spec.name}.yaml",
                    synthetic_root_override=variant_root,
                    synth_train_override=list_images(variant_root / "images/train"),
                    synth_val_override=list_images(variant_root / "images/val"))

            return paths.work / "exp3.yaml"      # unchanged data; recipe-only ablation

        def train_finetuning(run_name, data_yaml, init_weights, freeze, copy_paste):
            best = paths.runs / run_name / "weights" / "best.pt"
            if best.exists() and not args.force:
                print(f"  reusing existing run: {best}")
                return best
            return train_variant(run_name, data_yaml, init_weights, paths,
                                 config.solution, freeze=freeze,
                                 copy_paste=copy_paste, device=args.device)

        def train_transfer(run_name, init_weights):
            best = paths.runs / run_name / "weights" / "best.pt"
            if best.exists() and not args.force:
                print(f"  reusing existing run: {best}")
                return best
            training = config.base_training
            training.run_name = run_name
            return train_base_detector(base_val_yaml, paths, training,
                                       device=args.device,
                                       init_weights=init_weights)

        results = run_ablations(specs, build_dataset, train_finetuning,
                                train_transfer, base_val_yaml, base_weights)

    table = evaluate_ablations(results, target_val_yaml, base_val_yaml, specs,
                               paths.runs, config.classes.target_id, args.device)
    if len(table):
        save_table(table, paths.results / "ablation_master_table.csv")
    return 0


def _generate_variant(config, spec, num_images: int) -> Path:
    """Generate (or reuse) the synthetic variant a compositing ablation needs."""
    import json

    from aspilanta.synthetic.generator import generate_dataset

    out = config.paths.work / f"generated_{spec.name}"
    if (out / "dataset.yaml").exists():
        print(f"  reusing variant dataset: {out}")
        return out

    generator = config.generator
    for key, value in spec.generator_overrides.items():
        setattr(generator, key, value)
    generator.num_images = num_images
    # Same seed across variants, so sampling differences do not masquerade as
    # compositing differences.
    generator.seed = config.random_seed

    targets = json.loads((config.paths.outputs / "aspilanta_cutouts.json").read_text())
    distractor_manifest = config.paths.outputs / "distractor_cutouts.json"
    distractors = (json.loads(distractor_manifest.read_text())
                   if distractor_manifest.exists() else [])

    generate_dataset(targets, distractors, config.paths.backgrounds, out, generator)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
