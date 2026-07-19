#!/usr/bin/env python3
"""Build the SDE3-03 (SLM-177) proxy-metric calibration activation/budget manifest.

This CLI emits the machine-readable manifest that must exist before any cheap
proxy is trained or used to triage experiment rows. It is intentionally
plan-only: no model training, no full-suite invocation changes, and no ship
claim.

Example::

    python -m scripts.build_proxy_metric_calibration_manifest \
        --manifest-id sde3-03-v1 \
        --max-historical-rows 10000 \
        --gate-slm-105 \
        --gate-slm-169 \
        --gate-slm-175 \
        --gate-contract-review \
        --gate-budget \
        --out-json docs/design/iter-proxy-metric-calibration-20260719.json \
        --out-md docs/design/iter-proxy-metric-calibration-20260719.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.proxy_metric_calibration import (
    ActivationGate,
    BudgetCap,
    CalibrationArm,
    ProxyFeatureSet,
    build_proxy_metric_calibration_manifest,
    validate_proxy_metric_calibration_manifest,
)
from slm_training.versioning import build_version_stamp


_DEFAULT_GATES: list[ActivationGate] = [
    ActivationGate(
        gate_id="slm105_binding_aware_metrics",
        depends_on_issue_id="SLM-105",
        required_status="Done",
        available=False,
        evidence="Binding-aware deterministic metrics stable and versioned.",
    ),
    ActivationGate(
        gate_id="slm169_canonical_ast_binding",
        depends_on_issue_id="SLM-169",
        required_status="Done",
        available=False,
        evidence="Canonical AST, codec round-trip, binding integrity gates.",
    ),
    ActivationGate(
        gate_id="slm175_eval_cache",
        depends_on_issue_id="SLM-175",
        required_status="Done",
        available=False,
        evidence="Content-addressed evaluation cache artifacts available.",
    ),
    ActivationGate(
        gate_id="proxy_feature_contract_reviewed",
        depends_on_issue_id="SLM-177",
        required_status="Done",
        available=False,
        evidence="Feature contract reviewed: no forbidden features included.",
    ),
    ActivationGate(
        gate_id="budget_approved",
        depends_on_issue_id="SLM-177",
        required_status="approved",
        available=False,
        evidence="Calibration budget approved.",
    ),
]

_DEFAULT_ARMS: list[CalibrationArm] = [
    CalibrationArm(
        arm_id="rule_baseline",
        arm_kind="rule_baseline",
        eligible=True,
    ),
    CalibrationArm(
        arm_id="regularized_linear",
        arm_kind="regularized_linear",
        eligible=True,
    ),
    CalibrationArm(
        arm_id="bounded_tree",
        arm_kind="bounded_tree",
        eligible=False,
        omission_reason="reserved for ablation if linear models are insufficient",
    ),
    CalibrationArm(
        arm_id="shadow_only",
        arm_kind="shadow_only",
        eligible=True,
    ),
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-id", default="sde3-03-v1")
    parser.add_argument(
        "--gate-slm-105",
        action="store_true",
        help="Mark the SLM-105 binding-aware metrics gate as available.",
    )
    parser.add_argument(
        "--gate-slm-169",
        action="store_true",
        help="Mark the SLM-169 canonical-AST/binding gate as available.",
    )
    parser.add_argument(
        "--gate-slm-175",
        action="store_true",
        help="Mark the SLM-175 eval-cache gate as available.",
    )
    parser.add_argument(
        "--gate-contract-review",
        action="store_true",
        help="Mark the feature-contract-review gate as available.",
    )
    parser.add_argument(
        "--gate-budget",
        action="store_true",
        help="Mark the budget-approved gate as available.",
    )
    parser.add_argument(
        "--max-historical-rows",
        type=int,
        default=0,
        help="Maximum historical rows to use for calibration.",
    )
    parser.add_argument("--max-dollars", type=float, default=0.0)
    parser.add_argument("--gpu-hours", type=float, default=0.0)
    parser.add_argument("--eval-dollars", type=float, default=0.0)
    parser.add_argument("--total-dollars", type=float, default=0.0)
    parser.add_argument(
        "--primary-metric", default="binding_aware_meaningful_program_rate"
    )
    parser.add_argument("--conservative-floor", type=float, default=0.70)
    parser.add_argument("--risk-budget", type=float, default=0.05)
    parser.add_argument(
        "--proxy-eval-mode",
        default="off",
        choices=["off", "shadow", "triage"],
    )
    parser.add_argument("--audit-rate", type=float, default=0.10)
    parser.add_argument("--force-full-every-n", type=int, default=20)
    parser.add_argument(
        "--gate-available",
        dest="gate_available",
        action="append",
        default=[],
        help="Mark a gate_id as available (can be repeated).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Destination JSON manifest path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional destination markdown summary path.",
    )
    parser.add_argument(
        "--note",
        default="SDE3-03 proxy-metric calibration activation/budget manifest (wiring slice).",
    )
    return parser.parse_args(argv)


def _apply_gate_flags(gates: list[ActivationGate], args: argparse.Namespace) -> list[ActivationGate]:
    available_ids = set(args.gate_available or [])
    flag_map = {
        "SLM-105": args.gate_slm_105,
        "SLM-169": args.gate_slm_169,
        "SLM-175": args.gate_slm_175,
        "SLM-177": args.gate_contract_review or args.gate_budget,
    }
    out: list[ActivationGate] = []
    for gate in gates:
        available = gate.gate_id in available_ids or flag_map.get(gate.depends_on_issue_id, False)
        if gate.gate_id == "budget_approved":
            available = available or args.gate_budget
        elif gate.gate_id == "proxy_feature_contract_reviewed":
            available = available or args.gate_contract_review
        out.append(
            ActivationGate(
                gate_id=gate.gate_id,
                depends_on_issue_id=gate.depends_on_issue_id,
                required_status=gate.required_status,
                available=available,
                evidence=gate.evidence,
            )
        )
    return out


def _render_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# SDE3-03 proxy-metric calibration activation/budget manifest",
        "",
        f"- **manifest_id**: {manifest['manifest_id']}",
        f"- **schema_version**: {manifest['schema_version']}",
        f"- **hypothesis_id**: {manifest['hypothesis_id']}",
        f"- **activation_status**: {manifest['activation_status']}",
        f"- **activation_verdict**: {manifest['activation_verdict']}",
        f"- **campaign_verdict**: {manifest['campaign_verdict']}",
        f"- **primary_metric**: {manifest['primary_metric']}",
        f"- **proxy_eval_mode**: {manifest['proxy_eval_mode']}",
        f"- **conservative_floor**: {manifest['conservative_floor']}",
        f"- **risk_budget**: {manifest['risk_budget']}",
        f"- **manifest_hash**: {manifest['manifest_hash']}",
        "",
        "## Activation gates",
        "",
    ]
    for gate in manifest["activation_gates"]:  # type: ignore[index]
        status = "✅ available" if gate["available"] else "❌ not available"
        lines.append(
            f"- **{gate['gate_id']}** ({gate.get('depends_on_issue_id', '-')} -> {gate.get('required_status', '-')}): {status}"
        )
        if gate.get("evidence"):
            lines.append(f"  - evidence: {gate['evidence']}")
    lines.extend(
        [
            "",
            "## Feature contract",
            "",
            f"- schema_version: {manifest['feature_set']['feature_schema_version']}",
            f"- target_primary: {manifest['feature_set']['target_primary']}",
            f"- target_gate: {manifest['feature_set']['target_gate']}",
            f"- features: {', '.join(manifest['feature_set']['feature_names'])}",
            "",
            "## Budget cap",
            "",
            f"- max_historical_rows: {manifest['budget'].get('max_historical_rows')}",
            f"- max_dollars: {manifest['budget'].get('max_dollars')}",
            f"- gpu_hours: {manifest['budget'].get('gpu_hours')}",
            f"- eval_dollars: {manifest['budget'].get('eval_dollars')}",
            f"- total_dollars: {manifest['budget'].get('total_dollars')}",
            "",
            "## Arms",
            "",
        ]
    )
    for arm in manifest["arms"]:  # type: ignore[index]
        eligibility = "eligible" if arm["eligible"] else "omitted"
        lines.append(f"- **{arm['arm_id']}** ({arm['arm_kind']}) — {eligibility}")
        if arm.get("omission_reason"):
            lines.append(f"  - omission_reason: {arm['omission_reason']}")
    lines.extend(
        [
            "",
            "## Honest caveats",
            "",
            "This manifest is a wiring-only artifact. No proxy model has been trained, no full-suite invocation behavior has changed, and no promotion or ship claim has occurred. The default `proxy_eval_mode` is `off`; triage mode must not be enabled until the activation criteria are met.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    gates = _apply_gate_flags(_DEFAULT_GATES, args)

    feature_set = ProxyFeatureSet(
        feature_schema_version="proxy_features/v1",
        feature_names=(
            "parser_valid",
            "schema_valid",
            "binding_aware_meaningful_rate",
            "component_recall",
            "role_recall",
            "minimality_flag",
            "empty_output_flag",
            "first_attempt_action_count",
            "legal_action_margin",
            "entropy",
            "termination_confidence",
            "ast_node_count",
            "binding_graph_edges",
            "latency_ms",
            "output_length",
            "tree_depth",
            "component_count",
        ),
        target_primary=args.primary_metric,
        target_gate="full_gate_pass",
        allowed_sources=(
            "parser",
            "binding_aware_metric",
            "semantic_contract",
            "first_attempt_stats",
            "legal_action_stats",
            "canonical_ast",
            "timing",
            "suite_metadata",
        ),
        forbidden_features=(
            "agentv_score",
            "external_judge_score",
            "full_gate_result",
            "gold_action_trace",
            "checkpoint_id",
            "experiment_name",
            "source_commit",
        ),
    )
    budget = BudgetCap(
        max_historical_rows=args.max_historical_rows,
        max_dollars=args.max_dollars,
        gpu_hours=args.gpu_hours,
        eval_dollars=args.eval_dollars,
        total_dollars=args.total_dollars,
    )

    manifest_obj = build_proxy_metric_calibration_manifest(
        manifest_id=args.manifest_id,
        feature_set=feature_set,
        budget=budget,
        arms=_DEFAULT_ARMS,
        activation_gates=gates,
        primary_metric=args.primary_metric,
        conservative_floor=args.conservative_floor,
        risk_budget=args.risk_budget,
        proxy_eval_mode=args.proxy_eval_mode,  # type: ignore[arg-type]
        audit_rate=args.audit_rate,
        force_full_every_n=args.force_full_every_n,
        note=args.note,
    )
    manifest = manifest_obj.to_dict()

    errors = validate_proxy_metric_calibration_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest validation error: {error}")
        return 1

    payload = {
        "version_stamp": build_version_stamp("harness.experiments"),
        "manifest": manifest,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out_json}")
    print(
        f"activation_status={manifest['activation_status']} "
        f"activation_verdict={manifest['activation_verdict']} "
        f"campaign_verdict={manifest['campaign_verdict']} "
        f"manifest_hash={manifest['manifest_hash']}"
    )

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(manifest), encoding="utf-8")
        print(f"wrote {args.out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
