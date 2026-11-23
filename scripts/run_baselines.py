#!/usr/bin/env python3
"""Run the cross-family baselines.

    python scripts/run_baselines.py --baseline owlvit
    python scripts/run_baselines.py --baseline protonet

``owlvit``    Foundation model: zero-shot text prompts and image-guided
              few-shot detection.
``protonet``  Algorithm-oriented: a task-trained prototypical embedding scored
              against frozen CLIP on identical crops.

The TFA baseline is a training recipe, so it lives in
``scripts/train.py --stage tfa`` rather than here.
"""

from __future__ import annotations

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--baseline", choices=("owlvit", "protonet"), required=True)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap for sanity checks; omit for the full set")
    parser.add_argument("--k-values", default="1,5",
                        help="Support-set sizes to evaluate")
    parser.add_argument("--episodes", type=int, default=300,
                        help="Prototypical training episodes")
    args = parser.parse_args()

    config = load(args)
    k_values = [int(k) for k in args.k_values.split(",")]
    return (_run_owlvit(config, args, k_values) if args.baseline == "owlvit"
            else _run_protonet(config, args, k_values))


def _run_owlvit(config, args, k_values) -> int:
    import pandas as pd

    from aspilanta.baselines.owlvit import (
        DEFAULT_TEXT_QUERIES,
        OwlVitBaseline,
        collect_exemplars,
        evaluate_owlvit_variant,
    )
    from aspilanta.evaluation.reporting import save_table

    paths = config.paths
    image_dir = paths.target_domain_val / "images/val"
    label_dir = paths.target_domain_val / "labels/val"
    if not image_dir.exists():
        print(f"Missing {image_dir}. Run scripts/build_datasets.py first.")
        return 1

    model = OwlVitBaseline()
    rows = [evaluate_owlvit_variant(
        "owlvit_zeroshot_text",
        lambda im: model.detect_text(im, DEFAULT_TEXT_QUERIES, score_thresh=0.05),
        image_dir, label_dir, score_thresh=0.05,
        target_class_id=config.classes.target_id, max_images=args.max_images)]

    max_k = max(k_values)
    exemplars, exemplar_files = collect_exemplars(
        paths.target_domain_val, max_k, config.classes.target_id,
        seed=config.random_seed)

    for k in k_values:
        if k > len(exemplars):
            print(f"Skipping K={k}: only {len(exemplars)} exemplars available.")
            continue
        crops = exemplars[:k]
        rows.append(evaluate_owlvit_variant(
            f"owlvit_imageguided_k{k}",
            lambda im, c=crops: model.detect_image_guided(im, c, score_thresh=0.5),
            image_dir, label_dir, score_thresh=0.5,
            target_class_id=config.classes.target_id,
            max_images=args.max_images,
            # Exclude the exemplars' own source images, or the baseline is
            # partly scored on its own prompts.
            exclude_files=exemplar_files))

    save_table(pd.DataFrame(rows), paths.results / "baseline_owlvit.csv")
    return 0


def _run_protonet(config, args, k_values) -> int:
    import numpy as np

    from aspilanta.baselines.protonet import ProtoTrainer, compare_embeddings
    from aspilanta.data.splits import find_real_aspilanta_root
    from aspilanta.data.yolo_io import list_images
    from aspilanta.evaluation.reporting import save_table
    from aspilanta.models.clip_filter import (
        ClipEmbedder,
        build_support_crops,
        crop_objects_from_dataset,
    )

    paths = config.paths
    merged = paths.exp3_dataset
    if not (merged / "images/train").exists():
        print(f"Missing {merged}. Run scripts/build_datasets.py first.")
        return 1

    train_crops, train_labels, _ = crop_objects_from_dataset(
        merged, "train", target_class_id=config.classes.target_id)
    query_crops, query_labels, query_meta = crop_objects_from_dataset(
        merged, "val", target_class_id=config.classes.target_id)

    # Restrict the query set to crops from real photographs. Synthetic crops
    # descend from the same 44 cutouts the support set is drawn from, so
    # including them would let a task-trained embedding score highly by
    # recognising cutout provenance rather than the species.
    is_real = np.array(["synth" not in m["image"].lower() for m in query_meta])
    query_crops = [c for c, keep in zip(query_crops, is_real) if keep]
    query_labels = np.asarray(query_labels)[is_real]
    print(f"Query crops restricted to real sources: {len(query_crops)}")

    real_root = find_real_aspilanta_root(paths.real_aspilanta_candidates(),
                                         search_roots=[paths.work, paths.inputs])
    support_crops = build_support_crops(list_images(real_root), [real_root])

    trainer = ProtoTrainer(train_crops, train_labels, seed=config.random_seed)
    trainer.train(episodes=args.episodes)
    trainer.save(paths.results / "proto_embed_net.pt")

    embedder = ClipEmbedder(config.deployment.clip_model_name)
    table, prevalence = compare_embeddings(
        support_crops, query_crops, query_labels,
        clip_embed_fn=embedder.embed, proto_embed_fn=trainer.embed,
        k_values=k_values)

    print(f"\nRandom-baseline AP (positive prevalence) = {prevalence:.4f}")
    save_table(table, paths.results / "baseline_protonet_vs_clip.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
