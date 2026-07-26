#!/usr/bin/env python3
"""Record SLM-407's local operator-serving efficiency preflight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.operator_systems_benchmark import (
    OperatorServingIdentityV1,
    OperatorServingWorkV1,
    build_operator_serving_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _portable(value: Any, *, output_dir: Path) -> Any:
    """Do not commit ephemeral worktree paths in durable run evidence."""
    if isinstance(value, str):
        prefix = str(output_dir.resolve())
        return (
            "agentv-dir://" + value[len(prefix) :].lstrip("/")
            if value.startswith(prefix)
            else value
        )
    if isinstance(value, list):
        return [_portable(item, output_dir=output_dir) for item in value]
    if isinstance(value, dict):
        return {
            key: _portable(item, output_dir=output_dir)
            for key, item in value.items()
        }
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    """AgentV emits absolute paths; convert them to durable repo URIs."""
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for source, replacement in replacements.items():
            content = content.replace(source, replacement)
        path.write_text(content, encoding="utf-8")


def _markdown(report: dict[str, Any], typed: dict[str, Any]) -> str:
    decision = typed.get("decision", {})
    return "\n".join(
        [
            "# DSH3-32 operator-serving work preflight",
            "",
            f"- Verdict: **{report['verdict']}** — {report['reason']}",
            "- Recipe: local CPU evidence inspection; no remote workload, checkpoint, or human-rating gate.",
            "- Primary metric: target-model forward equivalents, reconciled from raw target and calibrated non-target forwards.",
            f"- Typed-policy quality preflight: `{decision.get('verdict')}` — {decision.get('reason')}",
            "- Measured serving rows: " + str(len(report["raw_rows"])),
            "",
            "This is not a ship or efficiency claim. Every serving arm must share the recorded identity and matched meaningful-parse quality; failures, timeouts, legal-set construction, dry runs, executor, validation, materialization, caching, batching, CPU/device, and wall work stay in the denominator.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-policy-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        help="optional JSON with identity and measured raw_rows from this benchmark contract",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    typed = json.loads(args.typed_policy_report.read_text(encoding="utf-8"))
    typed_decision = typed.get("decision", {})
    identity = OperatorServingIdentityV1(
        target_model="unmeasured",
        target_model_revision="unmeasured",
        device="cpu",
        precision="unmeasured",
        batch_size=1,
        cache_mode="unmeasured",
        context_fingerprint="dsh3-28-local-preflight",
    )
    rows: tuple[OperatorServingWorkV1, ...] = ()
    if args.observations:
        observation = json.loads(args.observations.read_text(encoding="utf-8"))
        identity = OperatorServingIdentityV1.from_dict(observation["identity"])
        rows = tuple(
            OperatorServingWorkV1.from_dict(row)
            for row in observation.get("raw_rows", [])
        )
    report = build_operator_serving_report(identity=identity, rows=rows)
    report["typed_policy_quality_preflight"] = {
        "path": str(args.typed_policy_report),
        "decision": typed_decision,
        "ready_for_efficiency_comparison": typed_decision.get("verdict") == "measured",
    }
    stamp = build_version_stamp(
        "harness.experiments.operator_systems_benchmark",
        "harness.experiments.typed_operator_policy",
    )
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh3-32-operator-systems-benchmark",
        claim="bounded_local_preflight_not_efficiency_or_ship_claim",
        version_stamp=stamp,
        cases=[
            {
                "id": "full-denominator-contract",
                "criteria": "The preflight retains all observed outcomes and its primary numeraire reconciles.",
                "pass": all(
                    summary["primary_numeraire"]["reconciles"]
                    for summary in report["per_arm"].values()
                ),
                "result": report["per_arm"],
            },
            {
                "id": "no-unmatched-efficiency-claim",
                "criteria": "No efficiency claim is made until every required arm is measured at matched quality.",
                "pass": report["verdict"] != "measured" or bool(report["efficiency_claim_supported_arms"]),
                "result": {
                    "verdict": report["verdict"],
                    "missing_arms": report["missing_arms"],
                },
            },
        ],
    )
    _rewrite_agentv_paths(output_dir)
    payload = {
        "schema": "dsh3_32_operator_systems_benchmark_preflight/v1",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": identity.device,
            "backend": "local_operator_serving_accounting",
            "suite_n": len(rows),
            "ship_claim": False,
            "human_rating_gate": False,
        },
        "benchmark": report,
        "agentv": _portable(agentv, output_dir=output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _markdown(report, typed), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
