#!/usr/bin/env python3
"""Publish the terminal CAP2 capability disposition and certificate decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.cap2_disposition import build_cap2_disposition
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
CAP2_REPORT = (
    ROOT / "docs/design/dsh3-13-cap2-operator-eval-20260723/report.json"
)
TOKEN_REPORT = (
    ROOT / "docs/design/e803-reserved-operator-baseline-20260723/report.json"
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


def _cases(disposition: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = {
        item["capability"]: item for item in disposition["capabilities"]
    }
    return [
        {
            "id": "complete-capability-ledger",
            "criteria": "Every CAP2 capability has exactly one typed verdict.",
            "pass": len(capabilities) == 7,
            "result": capabilities,
        },
        {
            "id": "exact-evidence-identities",
            "criteria": "Every retained evidence row pins code, data, suite, config, hardware, and result identities.",
            "pass": all(
                all(
                    key in item
                    for key in (
                        "code_identity",
                        "data_identity",
                        "checkpoint_identity",
                        "suite_identity",
                        "config_identity",
                        "hardware_identity",
                        "result_identity",
                    )
                )
                for item in disposition["evidence"]
            ),
            "result": disposition["evidence"],
        },
        {
            "id": "unrun-is-not-benefit",
            "criteria": "Unrun conditional mechanisms never claim an implemented benefit.",
            "pass": all(
                not item["implemented_benefit"]
                for item in disposition["capabilities"]
                if item["verdict"] == "unrun_conditional"
            ),
            "result": disposition["capabilities"],
        },
        {
            "id": "cert-cap2-rejected",
            "criteria": "CERT_CAP2 is rejected when no action form improves CAP2 and CAP1 retention is unavailable.",
            "pass": (
                disposition["cert_cap2"]["issued"] is False
                and disposition["dsh4_action_distillation"]["open"] is False
                and capabilities["discrete_token_action"]["verdict"] == "rejected"
                and capabilities["nl_transform"]["verdict"] == "unavailable"
            ),
            "result": {
                "cert_cap2": disposition["cert_cap2"],
                "dsh4": disposition["dsh4_action_distillation"],
            },
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    disposition = report["disposition"]
    lines = [
        "# DSH3-17 CAP2 capability disposition (SLM-385)",
        "",
        "Date: 2026-07-23",
        "Status: CERT_CAP2 rejected; DSH4 action distillation closed",
        "Scope: terminal evidence disposition; no model, checkpoint, or ship claim",
        "",
        "## Decision",
        "",
        disposition["cert_cap2"]["reason"],
        "The compiler-owned operator contracts remain useful correctness infrastructure,",
        "but no learned operator representation earned a capability certificate.",
        "",
        "## Capability ledger",
        "",
        "| Capability | Verdict | Implemented benefit | Evidence |",
        "| --- | --- | ---: | --- |",
    ]
    for item in disposition["capabilities"]:
        lines.append(
            f"| `{item['capability']}` | `{item['verdict']}` | "
            f"{str(item['implemented_benefit']).lower()} | "
            f"{', '.join(item['evidence_ids']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "Symbolic transformation and bounded merge are `contract_only`: exact fixture",
            "generation/replay exists, but no learned benefit passed. E803 rejects the",
            "discrete token action. NL is unavailable without CERT_CAP1. The hierarchical",
            "head and topology application remain unrun conditionals because their",
            "prerequisite failed. No exact-hardware matched-quality efficiency evidence",
            "exists.",
            "",
            "## Certificate and downstream gate",
            "",
            "- `CERT_CAP2`: **not issued**",
            "- DSH4 action distillation: **closed**",
            "- checkpoint/model-card roster change: **none**",
            "- production/ship claim: **none**",
            "",
            "## Evidence identities",
            "",
        ]
    )
    for item in disposition["evidence"]:
        lines.extend(
            [
                f"### `{item['evidence_id']}`",
                "",
                f"- class: `{item['evidence_class']}`",
                f"- code: `{item['code_identity']}`",
                f"- checkpoint: `{item['checkpoint_identity']}`",
                f"- data: `{json.dumps(item['data_identity'], sort_keys=True)}`",
                f"- suite: `{json.dumps(item['suite_identity'], sort_keys=True)}`",
                f"- config: `{json.dumps(item['config_identity'], sort_keys=True)}`",
                f"- hardware: `{json.dumps(item['hardware_identity'], sort_keys=True)}`",
                "",
            ]
        )
    summary = report["agentv"]["summary"]
    lines.extend(
        [
            "## Integrity result",
            "",
            f"AgentV passed {summary['passed']}/{summary['total']} cases with mean "
            f"{summary['meanScore']:.1f} and {summary['executionErrors']} execution errors.",
            "No experiment was rerun for this disposition; it consumes the immutable",
            "SLM-381 and E803 reports and preserves their positive, negative, unavailable,",
            "and unrun boundaries.",
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
        "evals.cap2_disposition",
        "evals.cap2_operator",
        "harness.experiments.reserved_operator_baseline",
    )
    disposition = build_cap2_disposition(
        cap2_report=json.loads(CAP2_REPORT.read_text(encoding="utf-8")),
        token_report=json.loads(TOKEN_REPORT.read_text(encoding="utf-8")),
        version_stamp=stamp,
    ).to_dict()
    agentv = publish_agentv_evaluation(
        output_dir,
        name="cap2-capability-disposition-v1",
        claim="cert_cap2_rejected_dsh4_closed",
        cases=_cases(disposition),
    )
    _rewrite_agentv_paths(output_dir)
    report = {
        "schema": "cap2_capability_disposition_report/v1",
        "run": {
            "kind": "evidence_disposition",
            "device": "none",
            "steps": 0,
            "checkpoint": None,
            "ship_claim": False,
        },
        "disposition": disposition,
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
                "cert_cap2_issued": False,
                "dsh4_action_distillation_open": False,
                "report": str(output_dir / "report.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
