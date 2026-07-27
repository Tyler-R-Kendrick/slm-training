"""Run the bounded local SLM-292 semantic-contrast wiring control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm292_semantic_contrast_objective import (
    run_local_control,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run_local_control(args.corpus, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
