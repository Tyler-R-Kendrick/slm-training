#!/usr/bin/env python3
"""Run the documented fixture-only SLM-199 rate-training path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_legal_edit_flow_fixture import main as run_documented_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--fixture-train", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/runs/slm199")
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--exact-samples", type=int, default=256)
    parser.add_argument("--max-wall-minutes", type=float, default=2.8)
    args = parser.parse_args(argv)
    if not 0 < args.max_wall_minutes <= 3:
        parser.error("--max-wall-minutes must be in (0, 3]")
    if args.describe:
        print(
            json.dumps(
                {
                    "schema": "SLM199FixtureTrainDescriptionV1",
                    "default_off": True,
                    "fidelity": "adapted_path_approximation",
                    "checkpoint_policy": (
                        "no checkpoint; writes design JSON/Markdown and "
                        "AgentEvals/AgentV fixture evidence"
                    ),
                },
                indent=2,
            )
        )
        return 0
    return run_documented_fixture(
        [
            "--output-dir",
            str(args.output_dir),
            "--train-steps",
            str(args.steps),
            "--exact-samples",
            str(args.exact_samples),
            "--max-wall-minutes",
            str(args.max_wall_minutes),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
