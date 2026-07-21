#!/usr/bin/env python3
"""Run the SLM-240 (LRS0-01) learning-rate schedule gap probe.

Reading harnesses/model_build/config.py and train_loop.py end to end finds
no warmup/decay knob on ModelBuildConfig and no torch.optim.lr_scheduler
usage anywhere in the train loop -- the optimizer is built once with a
static lr and never adjusted. This script confirms that reading against a
real run by instrumenting the live optimizer.step() calls (a restore-after
monkeypatch spy, not a code change) during the real model_build train loop,
for both supported optimizers (adamw, muon_hybrid) across several seeds, and
also checks whether metrics.jsonl ever logs the applied lr.

Examples:
  python -m scripts.run_slm240_lr_schedule_gap --mode plan-only
  python -m scripts.run_slm240_lr_schedule_gap --mode fixture
  python -m scripts.run_slm240_lr_schedule_gap --mode fixture \
      --steps 20 --n-records 4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm240_lr_schedule_gap import (
    DEFAULT_N_RECORDS,
    DEFAULT_OPTIMIZERS,
    DEFAULT_SEEDS,
    DEFAULT_STEPS,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    LrScheduleGapReport,
    render_markdown,
    run_lr_schedule_gap_probe,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm240-lrs0-01-lr-schedule-gap-20260721.json"
_DESIGN_MD = "docs/design/iter-slm240-lrs0-01-lr-schedule-gap-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload(
    steps: int, n_records: int, optimizers: tuple[str, ...], seeds: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "schema": "LrScheduleGapReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "The real model_build train loop applies a bit-identical, "
            "constant learning rate to every optimizer parameter group for "
            "the entire run, for both adamw and muon_hybrid, with no warmup "
            "and no decay, and metrics.jsonl never records the applied lr."
        ),
        "falsifier": (
            "Any optimizer.step() call reports a parameter-group lr that "
            "differs from its first recorded value, or a first recorded lr "
            "does not match the configured lr/muon_lr/adamw_lr, or any "
            "metrics.jsonl row logs an lr/learning_rate field."
        ),
        "steps": steps,
        "n_records": n_records,
        "optimizers": list(optimizers),
        "seeds": list(seeds),
        "arms": [],
        "all_lr_constant": True,
        "all_lr_matches_config": True,
        "any_metrics_log_lr": False,
        "all_finite": True,
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m scripts."
            "run_slm240_lr_schedule_gap --mode fixture` to execute."
        ),
        "honest_caveats": ["Plan-only: no arm was evaluated."],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm240_lr_schedule_gap",
            "harness.model_build.train",
            "model.twotower",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-240 LRS0-01 learning-rate schedule gap probe",
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
        help=f"Number of synthetic train records (default: {DEFAULT_N_RECORDS}).",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=list(DEFAULT_OPTIMIZERS),
        help=f"Optimizer names to probe (default: {list(DEFAULT_OPTIMIZERS)}).",
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
        f"outputs/runs/slm240-lr-schedule-gap-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload(
            args.steps, args.n_records, tuple(args.optimizers), tuple(args.seeds)
        )
    else:
        report = run_lr_schedule_gap_probe(
            steps=args.steps,
            n_records=args.n_records,
            optimizers=tuple(args.optimizers),
            seeds=tuple(args.seeds),
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm240_lr_schedule_gap_report.json"
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
            render_markdown(LrScheduleGapReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
