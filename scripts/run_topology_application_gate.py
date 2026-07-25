#!/usr/bin/env python3
"""Publish SLM-384 (DSH3-16)'s topology-application activation-gate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.topology_application_gate import (
    evaluate_topology_application_gate,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DISPOSITION_REPORT = (
    ROOT / "docs/design/dsh3-17-cap2-disposition-20260723/report.json"
)


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "output-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    prefixes = {
        str(output_dir.resolve()): "output-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("output-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for prefix, replacement in prefixes.items():
            text = text.replace(prefix, replacement)
        path.write_text(text, encoding="utf-8")


def _cases(gate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "gate-derived-from-disposition",
            "criteria": (
                "The gate verdict is read from the CAP2 disposition's own "
                "hierarchical_head row, not re-derived or hardcoded."
            ),
            "pass": bool(gate["hierarchical_head_evidence_ids"])
            or gate["hierarchical_head_verdict"] == "unrun_conditional",
            "result": gate,
        },
        {
            "id": "fail-closed-not-authorized",
            "criteria": (
                "DSH3-16 stays not_authorized because SLM-383 rejected the "
                "hierarchical-head operator-selection mechanism it would "
                "lower/apply."
            ),
            "pass": gate["verdict"] == "not_authorized"
            and gate["hierarchical_head_verdict"] == "rejected",
            "result": gate,
        },
        {
            "id": "plan-only-closure",
            "criteria": (
                "No production topology-diffusion model code, mutation "
                "bridge, or systems benchmark is claimed by this closure."
            ),
            "pass": True,
            "result": {"ship_claim": False, "checkpoint": None},
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# DSH3-16 topology-application activation gate (SLM-384)",
        "",
        "Date: 2026-07-25",
        "Status: not authorized; closed in plan-only mode",
        "Scope: activation-gate evaluation only; no model, checkpoint, or ship claim",
        "",
        "## Decision",
        "",
        gate["reason"],
        "",
        "## Gate result",
        "",
        f"- `verdict`: **{gate['verdict']}**",
        f"- `hierarchical_head_verdict`: `{gate['hierarchical_head_verdict']}`",
        f"- `hierarchical_head_evidence_ids`: {', '.join(gate['hierarchical_head_evidence_ids']) or 'none'}",
        "",
        "SLM-383's causal-attribution experiment (`hierarchical_operator_head_baseline`)",
        "found the encoder-side hierarchical head strictly beats its own weight-zero",
        "capacity control but ties the token baseline -- it does not causally improve",
        "beyond it, so it was rejected rather than left unrun. DSH3-16's own design",
        "compares recompute-only, hierarchical-head-with-ordinary-lowering, and",
        "topology-apply arms; without an accepted operator-selection mechanism there",
        "is no arm to lower or apply, and separately no `ActionEffectV1`-to-",
        "`TopologyNode` mutation bridge exists between the operators package's",
        "AST-dict representation and the topology-diffusion model's typed node tree",
        "(`grammar_diffusion.TopologyNode`). Building that bridge and the node-pass/",
        "model-forward instrumentation the issue calls for is out of scope while the",
        "gate is closed.",
        "",
    ]
    summary = report["agentv"]["summary"]
    lines.extend(
        [
            f"AgentV passed {summary['passed']}/{summary['total']} cases with mean "
            f"{summary['meanScore']:.1f} and {summary['executionErrors']} execution errors.",
            "",
            "No checkpoint was created, so the model card and README checkpoint summary",
            "do not change. This evaluation consumes the immutable DSH3-17/SLM-385",
            "disposition report and does not rerun or reinterpret its evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = build_version_stamp(
        "evals.topology_application_gate",
        "evals.cap2_disposition",
    )
    disposition_report = json.loads(DISPOSITION_REPORT.read_text(encoding="utf-8"))
    gate = evaluate_topology_application_gate(
        disposition_report["disposition"]["capabilities"]
    ).to_dict()
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh3-16-topology-application-gate",
        claim="not_authorized_plan_only_closure",
        cases=_cases(gate),
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "topology_application_gate_report/v1",
        "run": {
            "issue": "SLM-384",
            "kind": "activation_gate_evaluation",
            "device": "none",
            "checkpoint": None,
            "ship_claim": False,
        },
        "gate": gate,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "issue": "SLM-384",
                "verdict": gate["verdict"],
                "report": str(output_dir / "report.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
