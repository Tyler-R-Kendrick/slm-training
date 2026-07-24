#!/usr/bin/env python3
"""Describe, screen, and honestly analyze the SLM-200 objective factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm200_flow_objective_attribution import (
    MATRIX_SET,
    OBJECTIVES,
    run_matrix,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DESIGN_JSON = ROOT / "docs/design/iter-slm200-flow-objective-attribution-20260723.json"
DESIGN_MD = ROOT / "docs/design/iter-slm200-flow-objective-attribution-20260723.md"
AGENTV_DIR = ROOT / "docs/design/iter-slm200-flow-objective-attribution-agentv-20260723"


def _cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    arms = report["arms"]
    exact = arms["A9"]["oracle"]
    parity = report["primary_parity"]
    return [
        {
            "id": "factorial-registry",
            "criteria": "The preregistered A0-A9 matrix is explicit and no unavailable arm is silently dropped.",
            "pass": set(arms) == {f"A{index}" for index in range(10)},
            "checks": {
                "all_ten_arms_present": set(arms)
                == {f"A{index}" for index in range(10)},
                "a0_typed_unavailable": arms["A0"]["status"] == "unavailable",
                "a8_shuffled_negative_control": arms["A8"]["spec"][
                    "shuffled_targets"
                ],
                "a9_exact_only": arms["A9"]["spec"]["exact_only"],
            },
            "result": {
                key: value["status"] for key, value in sorted(arms.items())
            },
        },
        {
            "id": "primary-parity",
            "criteria": "A1-A8 share parameter count, row order, candidate corpus, and one-edit fixed-K decoder.",
            "pass": all(
                parity[key]
                for key in (
                    "parameter_count_within_one_percent",
                    "identical_row_order",
                    "identical_decoder",
                    "declared_differences_only",
                )
            ),
            "checks": {
                "parameter_count_within_one_percent": parity[
                    "parameter_count_within_one_percent"
                ],
                "identical_row_order": parity["identical_row_order"],
                "identical_decoder": parity["identical_decoder"],
                "paired_initialization": parity["paired_initialization"],
                "declared_differences_only": parity[
                    "declared_differences_only"
                ],
            },
            "result": parity,
        },
        {
            "id": "exact-oracle",
            "criteria": "A9 reproduces the closed exact generator, endpoint, and event-count fixture within tolerance.",
            "pass": (
                not exact["generator_errors"]
                and exact["illegal_edge_rate_sum"] == 0.0
                and exact["rate_fit"]["max_abs_error"] < 1e-3
                and exact["analytic_endpoint_tv"] < 0.01
                and exact["event_count_tv"] < 0.05
            ),
            "checks": {
                "valid_generator": not exact["generator_errors"],
                "zero_illegal_rate": exact["illegal_edge_rate_sum"] == 0.0,
                "max_rate_error_below_1e3": exact["rate_fit"][
                    "max_abs_error"
                ]
                < 1e-3,
                "endpoint_tv_below_001": exact["analytic_endpoint_tv"] < 0.01,
                "event_count_tv_below_005": exact["event_count_tv"] < 0.05,
            },
            "result": {
                "rate_fit": exact["rate_fit"],
                "analytic_endpoint_tv": exact["analytic_endpoint_tv"],
                "event_count_tv": exact["event_count_tv"],
            },
        },
        {
            "id": "confirmation-firewall",
            "criteria": "The underpowered fixture screen does not touch or masquerade as confirmation.",
            "pass": (
                report["confirmation"]["status"] == "not_touched"
                and not report["confirmation"]["touch_ledger"]
                and not report["analysis"]["flow_win"]
            ),
            "checks": {
                "confirmation_not_touched": report["confirmation"]["status"]
                == "not_touched",
                "empty_touch_ledger": not report["confirmation"]["touch_ledger"],
                "no_flow_win": not report["analysis"]["flow_win"],
                "no_checkpoint": not report["checkpoint"]["written"],
            },
            "result": {
                "analysis": report["analysis"],
                "confirmation": report["confirmation"],
            },
        },
        {
            "id": "honest-disposition",
            "criteria": "The result is classified as an underpowered fixture with no checkpoint or causal objective claim.",
            "pass": report["honest_verdict"]
            == "no_conclusion_underpowered_fixture",
            "checks": {
                "underpowered_fixture_verdict": report["honest_verdict"]
                == "no_conclusion_underpowered_fixture",
                "fixture_claim_class": report["claim_class"] == "wiring",
                "protocol_pins_present": all(
                    report["protocol_pins"].get(key)
                    for key in (
                        "power_protocol_sha256",
                        "resolution_protocol_sha256",
                        "utility_protocol_sha256",
                    )
                ),
            },
            "result": {
                "verdict": report["honest_verdict"],
                "protocol_pins": report["protocol_pins"],
            },
        },
    ]


def _markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    exact = report["arms"]["A9"]["oracle"]
    statuses = ", ".join(
        f"{arm}={payload['status']}" for arm, payload in sorted(report["arms"].items())
    )
    return "\n".join(
        [
            "# SLM-200 (VFA1-02): objective × state-weighting attribution",
            "",
            "**Status:** measured non-publishable fixture screen; confirmation not touched.",
            f"**Verdict:** `{report['honest_verdict']}`.",
            "",
            "## Frozen matrix and parity",
            "",
            f"- Arms: {statuses}.",
            f"- Production parameter counts: `{report['primary_parity']['parameter_counts']}`.",
            f"- Identical train-row order: `{report['primary_parity']['identical_row_order']}`.",
            f"- Identical decoder: `{report['primary_parity']['identical_decoder']}`.",
            f"- Seeds/steps: `{report['recipe']['seeds']}` / `{report['recipe']['steps']}`.",
            "",
            "## Measured fixture results",
            "",
            f"- Development-selected simpler control: `{analysis['strongest_simpler_control']}`.",
            f"- A7/control target-exact rate: `{analysis['a7_target_exact_rate']:.3f}` / `{analysis['control_target_exact_rate']:.3f}`.",
            f"- Paired descriptive delta: `{analysis['paired_delta']:+.3f}`.",
            f"- A9 max rate error: `{exact['rate_fit']['max_abs_error']:.9f}`.",
            f"- A9 analytic endpoint/event-count TV: `{exact['analytic_endpoint_tv']:.9f}` / `{exact['event_count_tv']:.9f}`.",
            "",
            "These numbers are descriptive fixture wiring only. The SLM-196 corpus",
            "contains two independent targets and is explicitly non-publishable; A0",
            "has no hash-pinned identical-state input. SLM-183 measured only 0.11",
            "power at the preregistered 0.08 MDE. Therefore no equivalence, weighting,",
            "hazard, or flow-transport causal attribution is licensed.",
            "",
            "## Confirmation and disposition",
            "",
            "- Confirmation status: `not_touched`; touch ledger is empty.",
            "- Checkpoint: none.",
            "- Claim class: wiring.",
            "- Decision: no conclusion; preserve all objectives as experimental and",
            "  require a publishable bridge corpus, complete A0 input, frozen full",
            "  confirmation suite, and powered checkpoints before a single touch.",
            "",
            f"AgentV: `{report.get('agentv', {}).get('summary', {})}`.",
        ]
    ) + "\n"


def _publish(report: dict[str, Any]) -> dict[str, Any]:
    stamp = build_version_stamp(
        "harness.experiments.slm200_flow_objective_attribution"
    )
    report["version_stamp"] = stamp
    published = publish_agentv_evaluation(
        AGENTV_DIR,
        name="slm200-flow-objective-attribution-fixture",
        claim="fixture_wiring_not_confirmation",
        cases=_cases(report),
        version_stamp=stamp,
    )
    report["agentv"] = published
    DESIGN_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    DESIGN_MD.write_text(_markdown(report), encoding="utf-8")
    return report


def _describe() -> dict[str, Any]:
    return {
        "schema": "FlowObjectiveAttributionPlanV1",
        "matrix_set": MATRIX_SET,
        "arms": [vars(spec) for spec in OBJECTIVES],
        "a0": "required but unavailable until a hash-pinned identical-state corpus exists",
        "modes": ["describe", "plan", "screen", "confirm", "resume", "analyze"],
        "max_wall_minutes": MAX_RUN_MINUTES,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--screen", action="store_true")
    modes.add_argument("--confirm", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-wall-minutes", type=float, default=MAX_RUN_MINUTES)
    args = parser.parse_args(argv)
    if args.describe or args.plan:
        print(json.dumps(_describe(), indent=2, sort_keys=True))
        return 0
    if args.analyze:
        print(DESIGN_JSON.read_text(encoding="utf-8"))
        return 0
    if args.confirm:
        if not DESIGN_JSON.is_file():
            raise SystemExit("confirmation blocked: run the development screen first")
        report = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
        reasons = report["confirmation"]["reasons"]
        raise SystemExit("confirmation blocked without a touch: " + "; ".join(reasons))
    report = run_matrix(
        steps=args.steps, max_wall_minutes=args.max_wall_minutes
    )
    _publish(report)
    print(json.dumps(report["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
