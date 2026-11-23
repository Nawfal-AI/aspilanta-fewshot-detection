#!/usr/bin/env python3
"""Evaluate a trained detector.

    python scripts/evaluate.py --weights .../best.pt --pipeline single_pass

Two evaluation modes:

``--mode ap``        Ultralytics validation — mAP and per-class AP. This is the
                     number reported as the headline result.
``--mode pipeline``  End-to-end precision/recall/F1/count-MAE for the deployed
                     pipeline, including tiling and CLIP variants.

Pass ``--gt-class-id 0`` when scoring against raw generator output, whose
labels use the generator's own numbering rather than the merged four-class
scheme. Getting this wrong reports a recall of zero that looks like a model
failure.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mode", choices=("ap", "pipeline", "both"), default="both")
    parser.add_argument("--data-yaml", type=Path, default=None,
                        help="Validation YAML for --mode ap")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--pipeline", default="single_pass",
                        choices=("single_pass", "single_pass_clip", "selective_clip",
                                 "tiled", "tiled_clip"))
    parser.add_argument("--gt-class-id", type=int, default=None,
                        help="Target class id in the label files being read")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sweep-conf", action="store_true",
                        help="Sweep the confidence threshold. Validation half only.")
    args = parser.parse_args()

    config = load(args)
    paths = config.paths
    results = {}

    if args.mode in ("ap", "both"):
        from aspilanta.evaluation.reporting import get_class_ap, safe_float, validate_model

        data_yaml = args.data_yaml or paths.work / "target_domain_val.yaml"
        if not data_yaml.exists():
            print(f"Missing {data_yaml}; skipping AP evaluation.")
        else:
            metrics = validate_model(args.weights, data_yaml,
                                     f"val_{Path(args.weights).parent.parent.name}",
                                     paths.runs, device=args.device)
            results["mAP50"] = safe_float(metrics.box.map50)
            results["mAP50_95"] = safe_float(metrics.box.map)
            results["AP_target_50"] = get_class_ap(metrics, config.classes.target_id,
                                                   "map50")
            results["AP_target_50_95"] = get_class_ap(metrics, config.classes.target_id,
                                                      "map")
            print("\nPer-class AP:")
            for i, name in enumerate(config.classes.full):
                print(f"  {name:<12} AP@50={get_class_ap(metrics, i, 'map50')}")

    if args.mode in ("pipeline", "both"):
        from ultralytics import YOLO

        from aspilanta.evaluation.harness import evaluate_pipeline, sweep_confidence

        image_dir = args.image_dir or paths.target_domain_val / "images/val"
        label_dir = args.label_dir or paths.target_domain_val / "labels/val"
        if not Path(image_dir).exists():
            print(f"Missing {image_dir}; skipping pipeline evaluation.")
        else:
            model = YOLO(str(args.weights))
            clip_filter = None
            if "clip" in args.pipeline:
                clip_filter = _build_clip_filter(config)
                if clip_filter is None:
                    print("Could not build a CLIP filter; falling back to single_pass.")
                    args.pipeline = "single_pass"

            common = dict(target_class_id=config.classes.target_id,
                          gt_class_id=args.gt_class_id,
                          deploy_conf=args.conf or config.deployment.deploy_conf,
                          clip_filter=clip_filter,
                          trust_threshold=config.deployment.selective_trust_threshold,
                          max_images=args.max_images)

            if args.sweep_conf:
                sweep_confidence(model, image_dir, label_dir,
                                 pipeline=args.pipeline,
                                 **{k: v for k, v in common.items()
                                    if k != "deploy_conf"})
            else:
                results.update(evaluate_pipeline(model, image_dir, label_dir,
                                                 pipeline=args.pipeline, **common))

    if results:
        import json
        out = paths.results / f"eval_{Path(args.weights).parent.parent.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved: {out}")
    return 0


def _build_clip_filter(config):
    """Assemble a CLIP prototype filter from real training crops."""
    from aspilanta.data.splits import find_real_aspilanta_root
    from aspilanta.data.yolo_io import list_images
    from aspilanta.models.clip_filter import (
        ClipEmbedder,
        ClipPrototypeFilter,
        build_support_crops,
    )
    try:
        real_root = find_real_aspilanta_root(
            config.paths.real_aspilanta_candidates(),
            search_roots=[config.paths.work, config.paths.inputs])
        crops = build_support_crops(list_images(real_root), [real_root])
        if not crops:
            return None
        embedder = ClipEmbedder(config.deployment.clip_model_name)
        return ClipPrototypeFilter(embedder, crops, k=config.deployment.clip_default_k)
    except Exception as exc:
        print(f"CLIP filter unavailable: {exc}")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
