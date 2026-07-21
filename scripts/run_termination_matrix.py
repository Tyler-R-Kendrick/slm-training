#!/usr/bin/env python3
"""Run the SLM-191 (FFE2-03) termination-policy fixture matrix.

Example:
  python -m scripts.run_termination_matrix --describe
  python -m scripts.run_termination_matrix --exact-fixture
  python -m scripts.run_termination_matrix --exact-fixture --k-values 2 4 8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm191_termination_matrix import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    TerminationManifestV1,
    run_termination_matrix,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm191-termination-matrix-20260721.json"
_DESIGN_MD = "docs/design/iter-slm191-termination-matrix-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-191 termination-policy fixture matrix schema

TerminationPolicy protocol:
  - name: str
  - decide(ctx: TerminationContext) -> TerminationDecision
  - to_dict() -> dict

Reference arms:
  - explicit_stop
  - absorbing_hazard
  - fixed_k
  - fixed_k_plus_selector
  - hybrid_min_progress
  - oracle_length

TerminationManifestV1 fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, arms, cases, target_rows,
  n_cases, n_arms, k_value, n_samples_per_arm, disposition,
  disposition_rationale, honest_caveats, version_stamp, timestamp.

TerminationTargetRowV1 extends FlowTargetRowV1 with:
  terminal_fingerprints, absorption_mass, exact_edit_count_distribution,
  exact_holding_time_mean/p50/p95, oracle_edit_count.

Claim class: wiring / fixture only.  No model, GPU, or checkpoint involvement.
"""


_HYPOTHESIS = (
    "Termination semantics materially change the empirical endpoint distribution, "
    "edit-count distribution, and premature/late-stop rates on exact CTMC fixtures; "
    "a shared TerminationPolicy protocol lets direct-policy and flow samplers be "
    "compared on the same scalar signals."
)

_FALSIFIER = (
    "All six termination arms produce identical endpoint distributions and edit-count "
    "distributions on every exact domain, or the oracle-length arm is not the strongest "
    "baseline, or explicit-stop/absorbing-hazard arms are uncalibrated (Brier/ECE far "
    "above chance) on the known target signal."
)


def _build_payload(
    mode: str,
    output_dir: Path,
    argv_flags: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "TerminationManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
            "status": "plan_only",
            "claim_class": "wiring",
            "hypothesis": _HYPOTHESIS,
            "falsifier": _FALSIFIER,
            "arms": [],
            "cases": [],
            "target_rows": [],
            "n_cases": 0,
            "n_arms": len(ARM_NAMES),
            "k_value": argv_flags.get("k_value", 4),
            "n_samples_per_arm": argv_flags.get("n_samples_per_arm", 100),
            "disposition": "inconclusive",
            "disposition_rationale": "Plan-only manifest; run --exact-fixture to execute.",
            "honest_caveats": [
                "Plan-only: no exact CTMC paths were sampled.",
                "Real termination heads require a trained model and decode path.",
            ],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm191_termination_matrix",
                "flow.termination",
                "flow.reference",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_termination_matrix --plan-only"
        return payload, command

    k_values = argv_flags.get("k_values") or [4]
    k_value = int(k_values[0])
    manifest = run_termination_matrix(
        output_dir=output_dir,
        k_value=k_value,
        n_samples_per_arm=argv_flags.get("n_samples_per_arm", 100),
        horizon=argv_flags.get("horizon", 1.0),
        rate_fn_name=argv_flags.get("rate_fn_name", "uniform_rate"),
        seed=argv_flags.get("seed", 0),
        write_design_docs=argv_flags.get("write_design_docs", True),
    )
    payload = manifest.to_dict()
    command = "python -m scripts.run_termination_matrix --exact-fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-191 (FFE2-03): termination-policy matrix plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            "**Machine-readable result:** [`iter-slm191-termination-matrix-20260721.json`](iter-slm191-termination-matrix-20260721.json)",
            "",
            "This is a plan-only manifest. The TerminationPolicy protocol, six reference "
            "arms, and calibration instrumentation are wired; run `--exact-fixture` to "
            "execute the CPU-only exact CTMC matrix.",
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

    manifest = TerminationManifestV1.from_dict(payload)
    from slm_training.harnesses.experiments.slm191_termination_matrix import render_markdown

    return render_markdown(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-191 FFE2-03 termination-policy fixture matrix",
        exit_on_error=False,
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the schema and arms, then exit.",
    )
    parser.add_argument(
        "--exact-fixture",
        action="store_true",
        help="Run the CPU-only exact-CTMC termination fixture.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write a plan-only manifest without sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm191-termination-matrix-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4],
        help="K values for fixed-K arms (default: 4; first value is used).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        dest="n_samples_per_arm",
        help="Samples per arm per domain (default: 100).",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=1.0,
        help="CTMC integration horizon used for wall-time bounds (default: 1.0).",
    )
    parser.add_argument(
        "--rate-fn",
        choices={"uniform_rate", "distance_rate"},
        default="uniform_rate",
        dest="rate_fn_name",
        help="Rate parameterization for the exact generator (default: uniform_rate).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint path (fixture mode ignores this; reserved for future model runs).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Acknowledge fixture-only wiring evidence before running.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for sampling (default: 0).",
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

    if args.describe:
        print(_describe_schema())
        return 0

    mode = "plan-only" if args.plan_only else "fixture"
    if args.confirm and mode == "fixture":
        print("Confirmed: fixture-only wiring evidence, no model/GPU/ship claim.")

    output_dir = args.output_dir or Path(f"outputs/runs/slm191-termination-matrix-{_today_yyyymmdd()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "k_values": args.k_values,
        "n_samples_per_arm": args.n_samples_per_arm,
        "horizon": args.horizon,
        "rate_fn_name": args.rate_fn_name,
        "seed": args.seed,
        "write_design_docs": args.write_design_docs,
    }
    payload, command = _build_payload(mode, output_dir, flags)
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    run_json = output_dir / "slm191_termination_matrix_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if mode == "fixture" and args.write_design_docs:
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
