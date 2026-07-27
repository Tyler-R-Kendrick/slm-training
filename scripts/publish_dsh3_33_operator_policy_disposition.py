#!/usr/bin/env python3
"""Publish SLM-408's rebased CAP2 policy disposition from durable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.cap2_operator_policy_disposition import (
    build_rebased_cap2_operator_policy_disposition,
)
from slm_training.harness_core.versioning import build_version_stamp


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "cap2_fixture": "docs/design/dsh3-28-cap2-operator-v2-20260726/report.json",
    "typed_policy": "docs/design/dsh3-28-typed-operator-policy-20260726-cap2-v2-fiveheads/report.json",
    "coverage": "docs/design/dsh3-29-operator-ecoc-coverage-20260726-local/report.json",
    "negative_ablation": "docs/design/dsh3-30-collapse-negative-ablation-20260726-local/report.json",
    "failure_prediction": "docs/design/dsh3-31-operator-failure-prediction-20260726-local/report.json",
    "systems": "docs/design/dsh3-32-operator-serving-work-20260726-local/report.json",
    "permutation": "docs/design/dsh3-27-operator-permutation-consistency.json",
}


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


def _portable(value: Any, output_dir: Path) -> Any:
    if isinstance(value, str):
        prefix = str(output_dir.resolve())
        return "agentv-dir://" + value[len(prefix) :].lstrip("/") if value.startswith(prefix) else value
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _markdown(disposition: dict[str, Any]) -> str:
    lines = [
        "# DSH3-33 rebased CAP2 operator-policy disposition",
        "",
        "Status: **DSH5 closed** — no learned operator-policy claim is supported.",
        "",
        "| Claim | Verdict | Evidence |",
        "| --- | --- | --- |",
    ]
    for claim in disposition["claims"]:
        lines.append(
            f"| `{claim['claim']}` | `{claim['verdict']}` | "
            f"{', '.join(claim['evidence_ids']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "The frozen DSH3-17 disposition remains historical and unchanged. "
            "The local evidence retains the typed-policy negative stop rule, "
            "the unavailable systems comparison, and fixture-only runtime/"
            "permutation evidence. No checkpoint, model-card update, remote run, "
            "human-rating gate, production change, or DSH5 authorization follows.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=ROOT / default)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        name: json.loads(getattr(args, name).read_text(encoding="utf-8"))
        for name in DEFAULTS
    }
    stamp = build_version_stamp("evals.cap2_operator_policy_disposition")
    disposition = build_rebased_cap2_operator_policy_disposition(
        **reports, version_stamp=stamp
    ).to_dict()
    claims = {item["claim"]: item for item in disposition["claims"]}
    agentv = publish_agentv_evaluation(
        output_dir,
        name="dsh3-33-rebased-cap2-disposition",
        claim="rebased_cap2_disposition_not_ship_or_dsh5_authorization",
        version_stamp=stamp,
        cases=[
            {
                "id": "complete-claim-ledger",
                "criteria": "Every required CAP2 claim has one typed disposition.",
                "pass": len(claims) == 10,
                "result": claims,
            },
            {
                "id": "negative-and-unavailable-retained",
                "criteria": "Negative typed semantics and unavailable systems work remain non-promotional.",
                "pass": claims["held_out_semantics"]["verdict"] == "negative" and claims["systems_efficiency"]["verdict"] == "unavailable",
                "result": {key: claims[key] for key in ("held_out_semantics", "systems_efficiency")},
            },
            {
                "id": "dsh5-remains-closed",
                "criteria": "DSH5 has no allowed learned-policy inventory without supported held-out semantics.",
                "pass": not disposition["dsh5"]["may_start"] and not disposition["dsh5"]["allowed_heads"],
                "result": disposition["dsh5"],
            },
        ],
    )
    _rewrite_agentv_paths(output_dir)
    payload = {
        "schema": "dsh3_33_rebased_cap2_disposition_report/v1",
        "disposition": disposition,
        "agentv": _portable(agentv, output_dir),
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(_markdown(disposition), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
