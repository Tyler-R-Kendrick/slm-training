#!/usr/bin/env python3
"""Run the SLM-193 (FFE3-02) bit-exact flow-cache fixture.

Example:
  python -m scripts.bench_flow_caches --describe
  python -m scripts.bench_flow_caches --fixture
  python -m scripts.bench_flow_caches --fixture --output-dir outputs/runs/slm193-custom
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm193_flow_caches import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    FlowCacheManifestV1,
    run_flow_cache_fixture,
)
from slm_training.versioning import build_version_stamp


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


_DESIGN_JSON = f"docs/design/iter-slm193-flow-caches-{_today_yyyymmdd()}.json"
_DESIGN_MD = f"docs/design/iter-slm193-flow-caches-{_today_yyyymmdd()}.md"


def _describe_schema() -> str:
    return f"""\
SLM-193 bit-exact flow-cache fixture schema

Matrix set: {MATRIX_SET}
Matrix version: {MATRIX_VERSION}
Experiment ID: {EXPERIMENT_ID}

Reference arms:
{chr(10).join(f'  - {name}' for name in ARM_NAMES)}

FlowCacheManifestV1 fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, disposition, disposition_rationale,
  arms (CacheArmResult), cases (CacheCaseRecord),
  n_cases, n_arms, honest_caveats, version_stamp, timestamp.

CacheArmResult fields:
  arm_name, total_ms, hit_rate, n_entries, bytes_stored, speedup, work_units.

Claim class: wiring / fixture only.  No model, GPU, or checkpoint involvement.
"""


_HYPOTHESIS = (
    "Exact state fingerprints recur across decode attempts and evaluation records, "
    "so bit-exact content-addressed caches reduce deterministic solver/bridge wall "
    "time by at least 2x while preserving identical outputs and certificates."
)

_FALSIFIER = (
    "Cache hit rates stay below 20% on warm repeated requests, or lookup/serialization "
    "overhead offsets the saved work on warm p50/p95, or cached results diverge from "
    "fresh computation."
)


def _build_payload(
    mode: str,
    output_dir: Path,
    argv_flags: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "FlowCacheManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
            "status": "plan_only",
            "claim_class": "wiring",
            "hypothesis": _HYPOTHESIS,
            "falsifier": _FALSIFIER,
            "disposition": "cache_wired",
            "disposition_rationale": "Plan-only manifest; run --fixture to execute.",
            "arms": [],
            "cases": [],
            "n_cases": 0,
            "n_arms": len(ARM_NAMES),
            "honest_caveats": [
                "Plan-only: no cache measurements were taken.",
                "Real restart provenance requires a replay-safe certificate contract.",
            ],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm193_flow_caches",
                "harness.core",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.bench_flow_caches --plan-only"
        return payload, command

    manifest = run_flow_cache_fixture(
        output_dir=output_dir,
        write_design_docs=argv_flags.get("write_design_docs", True),
    )
    payload = manifest.to_dict()
    command = "python -m scripts.bench_flow_caches --fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-193 (FFE3-02): flow-cache plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            f"**Machine-readable result:** [`iter-slm193-flow-caches-{_today_yyyymmdd()}.json`](iter-slm193-flow-caches-{_today_yyyymmdd()}.json)",
            "",
            "This is a plan-only manifest. The cache arms, hit-rate counters, and "
            "invalidation tests are wired; run `--fixture` to execute the CPU-only "
            "fixture matrix.",
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

    manifest = FlowCacheManifestV1.from_dict(payload)
    from slm_training.harnesses.experiments.slm193_flow_caches import render_markdown

    return render_markdown(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-193 FFE3-02 bit-exact flow-cache fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the schema and arms, then exit.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the CPU-only cache fixture.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write a plan-only manifest without executing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm193-flow-caches-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Acknowledge fixture-only wiring evidence before running.",
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

    output_dir = args.output_dir or Path(f"outputs/runs/slm193-flow-caches-{_today_yyyymmdd()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "write_design_docs": args.write_design_docs,
    }
    payload, command = _build_payload(mode, output_dir, flags)
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    run_json = output_dir / "slm193_flow_caches_report.json"
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
