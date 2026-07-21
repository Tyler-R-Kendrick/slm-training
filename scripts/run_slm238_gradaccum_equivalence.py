#!/usr/bin/env python3
"""Run the SLM-238 (GAE0-01) gradient-accumulation equivalence sweep.

Follows up on the single-run July-15 telemetry probes
(``iter-telemetry-gradaccum2-20260715.md``,
``iter-telemetry-effective-batch-20260715.md``) with a multi-seed, matched
initialization/LR/data comparison of ``grad_accum_steps=2`` (micro-batch 4)
against direct ``batch_size=8`` training on the real model_build train loop,
asking whether the two arms' final training loss stays within a
pre-registered relative tolerance of each other.

Examples:
  python -m scripts.run_slm238_gradaccum_equivalence --mode plan-only
  python -m scripts.run_slm238_gradaccum_equivalence --mode fixture
  python -m scripts.run_slm238_gradaccum_equivalence --mode fixture \
      --steps 40 --n-records 8 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm238_gradaccum_equivalence import (
    CLOSE_RELATIVE_TOLERANCE,
    DEFAULT_N_RECORDS,
    DEFAULT_SEEDS,
    DEFAULT_STEPS,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    GradAccumEquivalenceReport,
    render_markdown,
    run_gradaccum_equivalence_sweep,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm238-gae0-01-gradaccum-equivalence-20260721.json"
_DESIGN_MD = "docs/design/iter-slm238-gae0-01-gradaccum-equivalence-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload(steps: int, n_records: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    return {
        "schema": "GradAccumEquivalenceReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "At matched initialization, learning rate, optimizer, and total record "
            "set, grad_accum_steps=2 (micro-batch 4) reaches a final training loss "
            f"within {CLOSE_RELATIVE_TOLERANCE:.0%} relative difference of direct "
            "batch_size=8 training across multiple seeds."
        ),
        "falsifier": (
            "The accum arm's final loss differs from the direct arm's by more than "
            f"{CLOSE_RELATIVE_TOLERANCE:.0%} relative in a consistent direction across "
            "a majority of seeds, or the accel telemetry fields do not match the "
            "configured accumulation values, or either arm produces a non-finite loss."
        ),
        "steps": steps,
        "n_records": n_records,
        "close_relative_tolerance": CLOSE_RELATIVE_TOLERANCE,
        "seeds": list(seeds),
        "comparisons": [],
        "accum_wins": 0,
        "direct_wins": 0,
        "ties": 0,
        "unstable_seeds": 0,
        "close_seeds": 0,
        "mean_relative_diff": None,
        "mean_delta": None,
        "stdev_delta": None,
        "all_finite": True,
        "all_metadata_ok": True,
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m scripts.run_slm238_gradaccum_equivalence "
            "--mode fixture` to execute."
        ),
        "honest_caveats": [
            "Plan-only: no seed was evaluated.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm238_gradaccum_equivalence",
            "harness.experiments.slm227_muon_convergence",
            "harness.model_build.train",
            "model.twotower",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-238 GAE0-01 gradient-accumulation equivalence sweep",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Optimizer steps per arm (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=DEFAULT_N_RECORDS,
        help=f"Number of synthetic train records; must be even (default: {DEFAULT_N_RECORDS}).",
    )
    parser.add_argument(
        "--close-relative-tolerance",
        type=float,
        default=CLOSE_RELATIVE_TOLERANCE,
        help=f"Relative-difference tolerance for 'close' (default: {CLOSE_RELATIVE_TOLERANCE}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Seeds to sweep (default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    parser.add_argument("--design-json", type=Path, default=None)
    parser.add_argument("--design-md", type=Path, default=None)

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    args.output_dir = args.output_dir or Path(
        f"outputs/runs/slm238-gradaccum-equivalence-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload(args.steps, args.n_records, tuple(args.seeds))
    else:
        report = run_gradaccum_equivalence_sweep(
            steps=args.steps,
            n_records=args.n_records,
            close_relative_tolerance=args.close_relative_tolerance,
            seeds=tuple(args.seeds),
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm238_gradaccum_equivalence_report.json"
    run_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if args.mode == "fixture" and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or root / _DESIGN_JSON
        md_path = args.design_md or root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(
            render_markdown(GradAccumEquivalenceReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
