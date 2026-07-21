#!/usr/bin/env python3
"""Run the SLM-241 (GRT0-01) D2 canonicalizer round-trip / alpha-invariance
stress probe.

Exercises the real, unmodified ``slm_training.dsl.canonicalize.canonicalize``
and ``slm_training.dsl.parser.validate`` against a coverage-guided corpus of
real, grammar-valid OpenUI programs produced by the canonical typed
generator (``slm_training.data.progspec.generate.ProgramGenerator``), asking
whether canonicalize's documented normal-form claim -- idempotent,
always-valid, alpha-invariant under non-root binder renaming -- holds at
generator scale, not just the 3 hand-picked unit-test examples.

Examples:
  python -m scripts.run_slm241_grammar_round_trip_alpha_invariance --mode plan-only
  python -m scripts.run_slm241_grammar_round_trip_alpha_invariance --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm241_grammar_round_trip_alpha_invariance import (
    DEFAULT_COUNT_PER_SEED,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    GrammarRoundTripReport,
    render_markdown,
    run_round_trip_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm241-grt0-01-grammar-round-trip-alpha-invariance-20260721.json"
_DESIGN_MD = "docs/design/iter-slm241-grt0-01-grammar-round-trip-alpha-invariance-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload(seeds: tuple[int, ...], count_per_seed: int) -> dict[str, Any]:
    return {
        "schema": "GrammarRoundTripReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "The D2 canonicalizer is a stable normal form (idempotent, "
            "always-valid, alpha-invariant under non-root binder renaming) "
            "across a coverage-guided generator corpus, not just 3 "
            "hand-picked unit-test examples."
        ),
        "falsifier": (
            "Any generated candidate for which canonicalize(canonicalize(x)) "
            "!= canonicalize(x); canonicalize(x) fails to re-validate; or a "
            "binder-permuted, grammar-valid variant of x canonicalizes to a "
            "different string than x."
        ),
        "seeds": list(seeds),
        "count_per_seed": count_per_seed,
        "rows": [],
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm241_grammar_round_trip_alpha_invariance "
            "--mode fixture` to execute."
        ),
        "honest_caveats": ["Plan-only: no row was scored."],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm241_grammar_round_trip_alpha_invariance",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-241 GRT0-01 canonicalizer round-trip / alpha-invariance probe",
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
        help=f"Generator seeds to sweep (default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--count-per-seed",
        type=int,
        default=DEFAULT_COUNT_PER_SEED,
        help=f"Candidates generated per seed (default: {DEFAULT_COUNT_PER_SEED}).",
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
        f"outputs/runs/slm241-grammar-round-trip-alpha-invariance-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(args.seeds)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload(seeds, args.count_per_seed)
    else:
        report = run_round_trip_fixture(
            seeds=seeds,
            count_per_seed=args.count_per_seed,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm241_grammar_round_trip_alpha_invariance_report.json"
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
            render_markdown(GrammarRoundTripReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
