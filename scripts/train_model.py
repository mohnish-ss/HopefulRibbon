#!/usr/bin/env python3
"""Train and evaluate the Hopeful Ribbon classification pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from App.ml.training import RANDOM_SEED, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "App" / "data" / "breast-cancer.csv",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts"
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_and_evaluate(args.dataset, args.artifacts_dir, args.random_seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
