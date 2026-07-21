#!/usr/bin/env python3
"""Run the SLM-190 (FFE2-02) exact finite-state CTMC reference fixture.

Example:
  python -m scripts.run_exact_flow_fixture --mode describe
  python -m scripts.run_exact_flow_fixture --mode enumerate
  python -m scripts.run_exact_flow_fixture --path
  python -m scripts.run_exact_flow_fixture --sample
  python -m scripts.run_exact_flow_fixture --compare-objectives
  python -m scripts.run_exact_flow_fixture --lumpability
  python -m scripts.run_exact_flow_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm190_exact_flow import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    ExactFlowReport,
    run_exact_flow_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm190-exact-flow-20260721.json"
_DESIGN_MD = "docs/design/iter-slm190-exact-flow-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-190 exact finite-state CTMC reference fixture schema

Domains:
  - toy_layout: bounded OpenUI layout AST with tree-edit actions.
  - choice_sequence: bounded pushdown grammar with two commutative paths.
  - canonical_edit_graph: SLM-188 edit algebra from sketch to target.

ExactFlowReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, cases, objective_rows,
  lumpability_cases, n_domains, disposition, disposition_rationale,
  honest_caveats, version_stamp, timestamp.

ExactFlowCase fields:
  case_id, domain_id, source_fingerprint, target_fingerprint, rate_fn_name,
  time, n_states, n_transitions, n_terminals, mass_conservation_error,
  total_hazard_mean, endpoint_tv_exact_vs_gillespie, illegal_edge_rate_sum,
  multipath_entropy_bits, exact_endpoint_mass, gillespie_terminal_rate.

Lumpability cases report one of:
  lumpable, not_lumpable, unknown_numeric.

Claim class: wiring / fixture only.  No model, GPU, or checkpoint involvement.
"""


def _build_payload(
    mode: str,
    output_dir: Path,
    argv_flags: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "ExactFlowReportV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
            "status": "plan_only",
            "claim_class": "wiring",
            "hypothesis": _HYPOTHESIS,
            "falsifier": _FALSIFIER,
            "cases": [],
            "objective_rows": [],
            "lumpability_cases": [],
            "n_domains": 3,
            "disposition": "inconclusive",
            "disposition_rationale": "Plan-only manifest; run --mode fixture to execute.",
            "honest_caveats": [
                "Plan-only: no exact graphs were enumerated.",
                "Real flow objectives require a trained model and decode path.",
            ],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm190_exact_flow",
                "flow.reference",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_exact_flow_fixture --mode plan-only"
        return payload, command

    report = run_exact_flow_fixture(
        output_dir=output_dir,
        rate_fn_names=tuple(argv_flags.get("rate_fn_names") or ARM_NAMES),
        times=tuple(argv_flags.get("times") or (0.5, 1.0, 2.0)),
        seed=argv_flags.get("seed", 0),
        write_design_docs=True,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_exact_flow_fixture --mode fixture"
    return payload, command


_HYPOTHESIS = (
    "An exact finite-state CTMC reference over compiler-certified legal edits "
    "reveals objective-dependent differences in total hazard, endpoint "
    "distribution, and path statistics, and most natural quotient partitions "
    "over program structure are not strongly lumpable."
)

_FALSIFIER = (
    "On every representative exact domain, normalized next-edit CE plus a fixed "
    "time schedule reproduces the rate-based endpoint distribution and path "
    "statistics within tolerance, and the chosen quotient partitions are "
    "strongly lumpable."
)


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-190 (FFE2-02): exact CTMC reference plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            "**Machine-readable result:** [`iter-slm190-exact-flow-20260721.json`](iter-slm190-exact-flow-20260721.json)",
            "",
            "This is a plan-only manifest. The exact CTMC reference package, "
            "adapters, sampler, lumpability tests, and objective comparisons are "
            "wired; run `--mode fixture` to execute the CPU-only enumeration.",
            "",
            "## Hypothesis",
            "",
            _HYPOTHESIS,
            "",
            "## Falsifier",
            "",
            _FALSIFIER,
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = ExactFlowReport.from_dict(payload)
    from slm_training.harnesses.experiments.slm190_exact_flow import render_markdown

    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-190 FFE2-02 exact finite-state CTMC reference fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture", "describe"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU enumeration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm190-exact-flow-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--rate-fn-names",
        nargs="+",
        default=list(ARM_NAMES),
        help="Which rate parameterizations to exercise (default: all).",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[1.0],
        help="Time values for endpoint integration (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for Gillespie sampling (default: 0).",
    )
    parser.add_argument(
        "--enumerate",
        dest="enumerate_flag",
        action="store_true",
        help="When set with --mode fixture, the run emits state counts (always on).",
    )
    parser.add_argument(
        "--path",
        dest="path_flag",
        action="store_true",
        help="When set, the fixture reports path/trajectory statistics (always on).",
    )
    parser.add_argument(
        "--sample",
        dest="sample_flag",
        action="store_true",
        help="When set, the fixture runs Gillespie sampling (always on).",
    )
    parser.add_argument(
        "--compare-objectives",
        dest="compare_objectives_flag",
        action="store_true",
        help="When set, the fixture compares target parameterizations (always on).",
    )
    parser.add_argument(
        "--lumpability",
        dest="lumpability_flag",
        action="store_true",
        help="When set, the fixture runs lumpability tests (always on).",
    )
    parser.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON.",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.mode == "describe":
        print(_describe_schema())
        return 0

    output_dir = args.output_dir or Path(f"outputs/runs/slm190-exact-flow-{_today_yyyymmdd()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "rate_fn_names": tuple(args.rate_fn_names),
        "times": tuple(args.times),
        "seed": args.seed,
    }
    payload, command = _build_payload(args.mode, output_dir, flags)
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    run_json = output_dir / "slm190_exact_flow_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture" and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or root / _DESIGN_JSON
        md_path = args.design_md or root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report_text, encoding="utf-8")
        command_line = command
        if args.output_dir is not None:
            command_line += f" --output-dir {output_dir}"
        md_path.write_text(_build_markdown(payload, command_line), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
