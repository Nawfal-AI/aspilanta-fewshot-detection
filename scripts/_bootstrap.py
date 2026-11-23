"""Shared argument parsing and path bootstrapping for the CLI scripts.

Each script is runnable directly from a checkout (``python scripts/detect.py``)
without installing the package first, which is how they get used on a Kaggle or
Colab session where ``pip install -e .`` is an extra step people skip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config overlaying the defaults")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Writable working directory (overrides the config)")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Read-only directory holding the raw source datasets")
    parser.add_argument("--device", default=None,
                        help="CUDA device index, or 'cpu'")
    return parser


def load(args):
    """Build a ProjectConfig from the parsed arguments."""
    from aspilanta.config import load_config

    config = load_config(args.config)
    if getattr(args, "work_dir", None):
        config.paths.work = Path(args.work_dir)
    if getattr(args, "input_dir", None):
        config.paths.inputs = Path(args.input_dir)
    config.paths.__post_init__()
    config.paths.ensure()
    return config
