#!/usr/bin/env python3
"""Run the SLM-362 (DSH1-10) matched CAP0 curriculum experiment.

Sharding (hard run cap): each ``--mode fixture --arms ...`` command trains
and evaluates only the listed arms and writes ``shard_<ARM>.json``;
``--mode aggregate`` combines all four shards into the final report + docs.

Example:
  python -m scripts.run_slm362_cap0_experiment --mode plan-only
  python -m scripts.run_slm362_cap0_experiment --mode fixture --arms BASE ATOMIC_ONLY
  python -m scripts.run_slm362_cap0_experiment --mode fixture --arms STAGED_NO_PREFERENCE STAGED_FULL
  python -m scripts.run_slm362_cap0_experiment --mode aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm362_cap0_experiment import (
    ARMS,
    build_preregistration,
    finalize_report,
    load_shards,
    prepare_context,
    render_markdown,
    run_arm_shard,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm362-cap0-experiment-20260725.json"
_DESIGN_MD = "docs/design/iter-slm362-cap0-experiment-20260725.md"
_STAMP_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm362_cap0_experiment",
    "harness.train_data",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stamp() -> dict[str, Any]:
    return build_version_stamp(*_STAMP_COMPONENTS)


def _agentv_diagnostic(run_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """AgentV diagnostic publication; failure is non-fatal."""
    try:
        from slm_training.evals.agentv import publish_agentv_evaluation

        cert = payload["certificate"]
        cases = [
            {
                "id": "cap0-certificate-decision",
                "criteria": (
                    "CERT_CAP0 decision is reported with its stop-rule codes; "
                    "fixture n rejection is an honest outcome"
                ),
                "result": {
                    "status": cert["status"],
                    "rejection_codes": cert["rejection_codes"],
                },
                "assertions": [
                    {
                        "id": "cap0-certificate-decision:status_declared",
                        "actual": cert["status"] in {"issued", "rejected"},
                        "operator": "eq",
                        "expected": True,
                    }
                ],
            }
        ]
        result = publish_agentv_evaluation(
            run_dir,
            name="slm362-cap0-experiment",
            claim=(
                "CAP0 curriculum experiment diagnostic: fixture-scale CERT_CAP0 "
                "decision with paired evidence (fixture n cannot certify)"
            ),
            cases=cases,
            version_stamp=payload.get("version_stamp"),
        )
        # Design records must stay machine-portable: repo-relative paths only.
        root = str(Path(__file__).resolve().parents[1])
        return json.loads(json.dumps(result).replace(root + "/", ""))
    except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


def _prepare(args: argparse.Namespace):
    return prepare_context(
        seeds=args.seeds,
        records_per_arm=args.records_per_arm,
        steps=args.steps,
        batch_size=args.batch_size,
        suite_seed=args.suite_seed,
    )


def _write_final(payload: dict[str, Any], output_dir: Path, command: str) -> Path:
    payload["version_stamp"] = _stamp()
    payload["timestamp"] = _now()
    payload["status"] = "fixture"
    # Bind the real code commit into the certificate artifact.
    commit = payload["version_stamp"].get("code_commit")
    if commit:
        payload["certificate"]["artifact"]["binding"]["code_commit"] = commit
    payload["agentv_diagnostic"] = _agentv_diagnostic(output_dir, payload)

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "slm362_cap0_experiment_report.json"
    report_path.write_text(report_text, encoding="utf-8")

    root = Path(__file__).resolve().parents[1]
    (root / _DESIGN_JSON).write_text(report_text, encoding="utf-8")
    (root / _DESIGN_MD).write_text(
        render_markdown(payload, command=command), encoding="utf-8"
    )
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-362 DSH1-10 matched CAP0 curriculum experiment",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture", "aggregate"},
        default="plan-only",
        help=(
            "plan-only writes the preregistration; fixture trains/evals the "
            "--arms shards; aggregate combines all shards into the report."
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=list(ARMS),
        default=list(ARMS),
        help="Arms to compute in fixture mode (shard to stay under the run cap).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run artifact directory (default: outputs/experiments/slm362-cap0)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--records-per-arm", type=int, default=16)
    parser.add_argument("--suite-seed", type=int, default=7)
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path("outputs/experiments/slm362-cap0")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = f"python -m scripts.run_slm362_cap0_experiment --mode {args.mode}"

    if args.mode == "plan-only":
        prereg = build_preregistration(
            seeds=args.seeds,
            records_per_arm=args.records_per_arm,
            steps=args.steps,
            batch_size=args.batch_size,
            suite_seed=args.suite_seed,
        )
        payload: dict[str, Any] = {
            "schema": "Slm362Cap0ExperimentReportV1",
            "experiment_id": "slm362-cap0-experiment",
            "claim_class": "wiring",
            "status": "plan_only",
            "preregistration": prereg.to_dict(),
            "preregistration_sha256": prereg.sha256,
            "version_stamp": _stamp(),
            "timestamp": _now(),
        }
        report_path = output_dir / "slm362_cap0_experiment_plan.json"
        report_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(str(report_path))
        return 0

    if args.mode == "fixture":
        started = time.perf_counter()
        context, adapter = _prepare(args)
        for arm in args.arms:
            shard = run_arm_shard(context, adapter, arm, output_dir=output_dir)
            print(f"shard: {arm} ({len(shard['seeds'])} seeds)")
        if set(args.arms) == set(ARMS):
            arm_results = load_shards(
                output_dir, prereg_sha256=context.prereg.sha256
            )
            payload = finalize_report(context, arm_results, started=started)
            print(str(_write_final(payload, output_dir, command)))
        else:
            print(f"elapsed_seconds={round(time.perf_counter() - started, 3)}")
        return 0

    # aggregate: all shards must exist under the current preregistration.
    started = time.perf_counter()
    context, _adapter = _prepare(args)
    arm_results = load_shards(output_dir, prereg_sha256=context.prereg.sha256)
    payload = finalize_report(context, arm_results, started=started)
    print(str(_write_final(payload, output_dir, command)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
