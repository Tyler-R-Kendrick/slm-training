#!/usr/bin/env python3
"""Publish SLM-411's local exact-bulk crossover preflight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.bulk_operator_matrix import run_local_preflight
from slm_training.harness_core.versioning import build_version_stamp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TYPED_POLICY_REPORT = (
    ROOT
    / "docs/design/dsh3-28-typed-operator-policy-20260726-cap2-v2-fiveheads/report.json"
)


def _portable(value: Any, *, output_dir: Path) -> Any:
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
        return {key: _portable(item, output_dir=output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
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


def _markdown(report: dict[str, Any]) -> str:
    probes = report["legal_set_fixture_probes"]
    rows = "\n".join(
        f"| {row['fanout']} | {row['legal_actions']} | yes | yes | wiring only |"
        for row in probes
    )
    return "\n".join(
        [
            "# DSH5-03 exact bulk crossover preflight",
            "",
            f"- Verdict: **{report['verdict']}** — {report['reason']}",
            "- Recipe: local CPU compiler fixtures at fanout 1/2/4/8; no checkpoint, remote/HF workload, or human-rating gate.",
            "- Primary serving metric remains target-model forward equivalents; it is unmeasured here because every matched policy arm is absent.",
            "",
            "| fanout | legal bulk actions | replay exact | primitive equivalent | claim |",
            "| ---: | ---: | --- | --- | --- |",
            rows,
            "",
            "These are real pack/legal-set/executor/replay checks, but they are compiler-fixture wiring evidence only. The DSH3 typed policy is rejected and this run has no primitive-policy, bulk-policy, bulk-disabled, full-generation, or oracle-sequence serving rows. Therefore it does not claim quality, efficiency, a crossover, a model improvement, or ship readiness.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typed-policy-report", type=Path, default=DEFAULT_TYPED_POLICY_REPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    typed = json.loads(args.typed_policy_report.read_text(encoding="utf-8"))
    report = run_local_preflight(typed.get("decision", {}))
    stamp = build_version_stamp(
        "dsl.operators.legal_set",
        "model.operator_policy_view",
        "model.operator_feature_encoder",
        "evals.bulk_operator_matrix",
        "harness.experiments.operator_systems_benchmark",
    )
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh5-03-bulk-operator-crossover",
        claim="local_exact_bulk_fixture_preflight_not_crossover_or_ship_claim",
        version_stamp=stamp,
        cases=[
            {
                "id": "live-legal-bulk-actions",
                "criteria": "Each fanout exposes a complete live legal bulk action that replays and matches diagnostic primitive lowering.",
                "pass": report["fixture_contract_satisfied"],
                "result": report["legal_set_fixture_probes"],
            },
            {
                "id": "unavailable-is-not-efficiency",
                "criteria": "Missing matched policy arms must remain unavailable rather than become a crossover or efficiency claim.",
                "pass": report["verdict"] == "unavailable" and not report["ship_claim"],
                "result": {"systems": report["systems"], "crossover": report["crossover"]},
            },
            {
                "id": "typed-policy-negative-retained",
                "criteria": "The rejected typed-policy decision remains visible and is not replaced by fixture evidence.",
                "pass": report["typed_policy_decision"].get("verdict") == "reject",
                "result": report["typed_policy_decision"],
            },
        ],
    )
    _rewrite_agentv_paths(output_dir)
    payload = {
        "schema": "dsh5_03_bulk_operator_crossover_preflight/v1",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": "cpu",
            "backend": "local_exact_bulk_fixture",
            "fanouts": [1, 2, 4, 8],
            "ship_claim": False,
            "human_rating_gate": False,
        },
        "crossover": report,
        "agentv": _portable(agentv, output_dir=output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
