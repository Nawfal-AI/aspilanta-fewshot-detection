#!/usr/bin/env python3
"""Train the detector - either stage, or both in sequence.

    python scripts/train.py --stage base       # 3-class foundation
    python scripts/train.py --stage solution   # 4-class proposed method
    python scripts/train.py --stage all

Stage 2 fine-tunes from stage 1's best checkpoint with the backbone trainable.
That is the proposed configuration and it contradicts the design this project
started from - see docs/results.md for the ablation evidence and the caveat
that it was selected on evaluation data.
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import base_parser, load


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--stage", choices=("base", "solution", "tfa", "all"),
                        default="all")
    parser.add_argument("--exp1-yaml", type=Path, default=None)
    parser.add_argument("--exp3-yaml", type=Path, default=None)
    parser.add_argument("--init-weights", type=Path, default=None,
                        help="Stage-2 starting weights (defaults to the stage-1 best)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--freeze", type=int, default=None,
                        help="Backbone freeze depth. 0 is the proposed setting; "
                             "10 reproduces the frozen ablation.")
    args = parser.parse_args()

    config = load(args)
    from aspilanta.models.train import train_base_detector, train_solution, train_tfa_baseline

    paths = config.paths
    exp1_yaml = args.exp1_yaml or paths.work / "exp1.yaml"
    exp3_yaml = args.exp3_yaml or paths.work / "exp3.yaml"

    if args.epochs is not None:
        config.base_training.epochs = config.solution.epochs = args.epochs
    if args.batch is not None:
        config.base_training.batch = config.solution.batch = args.batch
    if args.freeze is not None:
        config.solution.freeze = args.freeze

    base_weights = None
    if args.stage in ("base", "all"):
        if not exp1_yaml.exists():
            print(f"Missing {exp1_yaml}. Run scripts/build_datasets.py first.")
            return 1
        base_weights = train_base_detector(exp1_yaml, paths, config.base_training,
                                           device=args.device)

    if args.stage in ("solution", "tfa", "all"):
        if not exp3_yaml.exists():
            print(f"Missing {exp3_yaml}. Run scripts/build_datasets.py first.")
            return 1

        init = args.init_weights or base_weights or (
            paths.runs / config.base_training.run_name / "weights" / "best.pt")
        if not Path(init).exists():
            print(f"Stage-2 starting weights not found: {init}\n"
                  "Train the base detector first, or pass --init-weights.")
            return 1

        if args.stage == "tfa":
            train_tfa_baseline(exp3_yaml, init, paths, config.solution,
                               device=args.device)
        else:
            train_solution(exp3_yaml, init, paths, config.solution,
                           device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
