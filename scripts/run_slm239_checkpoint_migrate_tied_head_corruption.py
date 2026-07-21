#!/usr/bin/env python3
"""Run the SLM-239 (CPM0-01) checkpoint-migrate output-head corruption probe.

Checks whether the shipped, unmodified ``migrate_twotower_checkpoint``
(``src/slm_training/models/checkpoint_migrate.py``) actually preserves its
promised token-string vocabulary remap for *every* vocab-indexed weight, or
whether the output head's naive whole-tensor copy silently corrupts (tied
case) or misaligns (untied case) the migrated checkpoint whenever the old and
new train-records set happen to produce the same vocabulary size but a
different first-occurrence token order.

Examples:
  python -m scripts.run_slm239_checkpoint_migrate_tied_head_corruption --mode plan-only
  python -m scripts.run_slm239_checkpoint_migrate_tied_head_corruption --mode fixture
  python -m scripts.run_slm239_checkpoint_migrate_tied_head_corruption --mode fixture \
      --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm239_checkpoint_migrate_tied_head_corruption import (
    CORRECT_FRACTION_THRESHOLD,
    DEFAULT_SEEDS,
    DEFAULT_TIE_ARMS,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    CheckpointMigrateCorruptionReport,
    render_markdown,
    run_checkpoint_migrate_corruption_sweep,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm239-cpm0-01-checkpoint-migrate-tied-head-corruption-20260721.json"
_DESIGN_MD = "docs/design/iter-slm239-cpm0-01-checkpoint-migrate-tied-head-corruption-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload(seeds: tuple[int, ...], tie_arms: tuple[bool, ...]) -> dict[str, Any]:
    return {
        "schema": "CheckpointMigrateCorruptionReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "migrate_twotower_checkpoint correctly remaps every vocab-indexed "
            "weight by shared token string when old and new vocab sizes "
            "coincide but token order differs."
        ),
        "falsifier": (
            "The output head (tok.weight under tying, lm_head.weight when "
            "untied) fails to match the token-string-correct source row for "
            "shifted tokens on a majority of seeds."
        ),
        "correct_fraction_threshold": CORRECT_FRACTION_THRESHOLD,
        "seeds": list(seeds),
        "tie_arms": list(tie_arms),
        "results": [],
        "tied_corrupted_count": 0,
        "tied_total_count": 0,
        "untied_corrupted_count": 0,
        "untied_total_count": 0,
        "any_vocab_size_mismatch": False,
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run "
            "`python -m scripts.run_slm239_checkpoint_migrate_tied_head_corruption "
            "--mode fixture` to execute."
        ),
        "honest_caveats": ["Plan-only: no seed was evaluated."],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm239_checkpoint_migrate_tied_head_corruption",
            "model.twotower",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-239 CPM0-01 checkpoint-migrate output-head corruption probe",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Seeds to sweep (default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--correct-fraction-threshold",
        type=float,
        default=CORRECT_FRACTION_THRESHOLD,
        help=f"Below-this-fraction-is-corrupted threshold (default: {CORRECT_FRACTION_THRESHOLD}).",
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
        f"outputs/runs/slm239-checkpoint-migrate-tied-head-corruption-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload(tuple(args.seeds), DEFAULT_TIE_ARMS)
    else:
        report = run_checkpoint_migrate_corruption_sweep(
            seeds=tuple(args.seeds),
            correct_fraction_threshold=args.correct_fraction_threshold,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm239_checkpoint_migrate_tied_head_corruption_report.json"
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
            render_markdown(CheckpointMigrateCorruptionReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
