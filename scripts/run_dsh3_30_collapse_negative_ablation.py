#!/usr/bin/env python3
"""Run DSH3-30's small synthetic CONFLICT-vs-DIFFERENT_RESULT hard-negative
ablation experiment.

Builds a modest, hand-built set of synthetic ``OperatorPolicyRowV1`` rows
(mixing DIFFERENT_RESULT, CONFLICT, and no-hard-negative rows -- the same
fixture-construction style as
``tests/test_models/test_operator_hard_negative_training.py``, no real
corpus build), runs all five matched hard-negative arms
(`none`/`conflict_only`/`different_result_only`/`equal_mix`/`curriculum_mix`)
through ``evaluate_hard_negative_arms``, and writes the resulting
``CollapseNegativeAblationReportV1`` plus a markdown summary. This is a
fixture-scale wiring demonstration, never a ship claim: see
``docs/design/dsh3-30-collapse-negative-ablation.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.flow.operator_policy_corpus import (
    OperatorPolicyHardNegativeV1,
    OperatorPolicyRowV1,
)
from slm_training.dsl.operators import (
    HardNegativeOutcome,
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.models.operator_hard_negative_training import (
    CollapseNegativeAblationReportV1,
    evaluate_hard_negative_arms,
)
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    OperatorPolicyInputV1,
)
from slm_training.versioning import build_version_stamp


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _action_view(row: int, operator_id: str, *, cost: float = 1.0) -> OperatorActionViewV1:
    return OperatorActionViewV1(
        row=row,
        operator_id=operator_id,
        operator_version="v1",
        locality="node",
        cost=cost,
        effect_signature=(),
        argument_slots=(),
        verdict=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.COMPLETE,
    )


def _different_result_row(
    suffix: str, *, accepted_cost: float = 1.0, alternate_cost: float = 1.0
) -> OperatorPolicyRowV1:
    """Both the accepted and alternate operators are live, legal candidate
    actions at this step -- the swap replayed successfully but disagreed.

    ``accepted_cost``/``alternate_cost`` are varied across the synthetic set
    (see ``_build_rows``) purely so the deterministic proxy scorer's margin
    loss and ``ranking_correct`` are not uniformly identical across every
    row -- this has no bearing on the ablation's own selection/usability
    mechanism, which never reads a score to decide selection.
    """
    accepted_op = f"openui.set_x_{suffix}"
    alternate_op = f"openui.set_y_{suffix}"
    action_views = (
        _action_view(0, accepted_op, cost=accepted_cost),
        _action_view(1, alternate_op, cost=alternate_cost),
    )
    policy_input = OperatorPolicyInputV1(
        reference_rows=(),
        action_rows=action_views,
        ordinary_action_count=0,
        coverage=LegalSetCoverage.COMPLETE,
    )
    _reference_map, action_map = policy_input.canonical_row_maps()
    policy_input_dict = policy_input.to_dict()
    accepted_action_row = action_map[0]
    alternate_action_row = action_map[1]
    hard_negative = OperatorPolicyHardNegativeV1(
        outcome=HardNegativeOutcome.DIFFERENT_RESULT,
        alternate_operator_id=alternate_op,
        alternate_action_row=alternate_action_row,
        conflict_code=None,
        observed_final_state_digest=_digest(f"observed-{suffix}"),
    )
    return OperatorPolicyRowV1(
        row_id=f"row_dr_{suffix}",
        collapse_id=_digest(f"collapse-dr-{suffix}"),
        turn_id=_digest(f"turn-dr-{suffix}"),
        step_index=0,
        request_id="request-1",
        state_digest=_digest(f"state-dr-{suffix}"),
        ast_digest=_digest(f"ast-dr-{suffix}"),
        branch_digest=_digest(f"branch-dr-{suffix}"),
        reference_table_fingerprint=_digest(f"reftable-dr-{suffix}"),
        legal_set_fingerprint=_digest(f"legalset-dr-{suffix}"),
        policy_input_digest=_fingerprint(policy_input_dict),
        policy_input=policy_input_dict,
        accepted_operator_id=accepted_op,
        accepted_action_row=accepted_action_row,
        accepted_argument_rows=(),
        accepted_application_id=_digest(f"application-dr-{suffix}"),
        hard_negative=hard_negative,
        split="train",
    )


def _conflict_row(suffix: str) -> OperatorPolicyRowV1:
    """The alternate operator is genuinely absent from this step's
    re-enumerated ``action_rows`` -- the swap outright failed to replay, so
    there is no live alternate action row to rank against."""
    accepted_op = f"openui.set_a_{suffix}"
    alternate_op = f"openui.set_b_{suffix}"
    action_views = (_action_view(0, accepted_op),)
    policy_input = OperatorPolicyInputV1(
        reference_rows=(),
        action_rows=action_views,
        ordinary_action_count=0,
        coverage=LegalSetCoverage.COMPLETE,
    )
    _reference_map, action_map = policy_input.canonical_row_maps()
    policy_input_dict = policy_input.to_dict()
    accepted_action_row = action_map[0]
    hard_negative = OperatorPolicyHardNegativeV1(
        outcome=HardNegativeOutcome.CONFLICT,
        alternate_operator_id=alternate_op,
        alternate_action_row=None,
        conflict_code="fixture.b_precondition_failed",
        observed_final_state_digest=None,
    )
    return OperatorPolicyRowV1(
        row_id=f"row_cf_{suffix}",
        collapse_id=_digest(f"collapse-cf-{suffix}"),
        turn_id=_digest(f"turn-cf-{suffix}"),
        step_index=0,
        request_id="request-1",
        state_digest=_digest(f"state-cf-{suffix}"),
        ast_digest=_digest(f"ast-cf-{suffix}"),
        branch_digest=_digest(f"branch-cf-{suffix}"),
        reference_table_fingerprint=_digest(f"reftable-cf-{suffix}"),
        legal_set_fingerprint=_digest(f"legalset-cf-{suffix}"),
        policy_input_digest=_fingerprint(policy_input_dict),
        policy_input=policy_input_dict,
        accepted_operator_id=accepted_op,
        accepted_action_row=accepted_action_row,
        accepted_argument_rows=(),
        accepted_application_id=_digest(f"application-cf-{suffix}"),
        hard_negative=hard_negative,
        split="train",
    )


def _no_negative_row(suffix: str) -> OperatorPolicyRowV1:
    """One accepted-only row with no collapse-derived hard negative at all
    (this step's adjacent pair commuted, or this is the trace's last step)."""
    accepted_op = f"openui.set_z_{suffix}"
    action_views = (_action_view(0, accepted_op),)
    policy_input = OperatorPolicyInputV1(
        reference_rows=(),
        action_rows=action_views,
        ordinary_action_count=0,
        coverage=LegalSetCoverage.COMPLETE,
    )
    _reference_map, action_map = policy_input.canonical_row_maps()
    policy_input_dict = policy_input.to_dict()
    return OperatorPolicyRowV1(
        row_id=f"row_nn_{suffix}",
        collapse_id=_digest(f"collapse-nn-{suffix}"),
        turn_id=_digest(f"turn-nn-{suffix}"),
        step_index=1,
        request_id="request-1",
        state_digest=_digest(f"state-nn-{suffix}"),
        ast_digest=_digest(f"ast-nn-{suffix}"),
        branch_digest=_digest(f"branch-nn-{suffix}"),
        reference_table_fingerprint=_digest(f"reftable-nn-{suffix}"),
        legal_set_fingerprint=_digest(f"legalset-nn-{suffix}"),
        policy_input_digest=_fingerprint(policy_input_dict),
        policy_input=policy_input_dict,
        accepted_operator_id=accepted_op,
        accepted_action_row=action_map[0],
        accepted_argument_rows=(),
        accepted_application_id=_digest(f"application-nn-{suffix}"),
        hard_negative=None,
        split="train",
    )


def _build_rows() -> tuple[OperatorPolicyRowV1, ...]:
    # Costs alternate so the deterministic proxy scorer sees a mix of
    # "accepted action was cheaper" (ranking_correct) and "accepted action
    # was pricier" (a real, nonzero margin loss) rows, instead of every row
    # tying identically.
    cost_pairs = ((0.5, 1.5), (1.5, 0.5), (0.5, 1.5), (1.5, 0.5))
    different_result_rows = tuple(
        _different_result_row(f"dr{i}", accepted_cost=accepted, alternate_cost=alternate)
        for i, (accepted, alternate) in enumerate(cost_pairs)
    )
    conflict_rows = tuple(_conflict_row(f"cf{i}") for i in range(2))
    no_negative_rows = tuple(_no_negative_row(f"nn{i}") for i in range(3))
    return different_result_rows + conflict_rows + no_negative_rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DSH3-30 CONFLICT vs DIFFERENT_RESULT collapse hard-negative ablation (SLM-405)",
        "",
        "Status: fixture-scale synthetic demonstration; not a ship claim",
        "",
        f"## Per-arm breakdown ({report['run']['total_rows']} synthetic rows: "
        f"{report['run']['different_result_rows']} DIFFERENT_RESULT, "
        f"{report['run']['conflict_rows']} CONFLICT, "
        f"{report['run']['no_negative_rows']} no-negative)",
        "",
        "| Arm | Total | Usable | Unusable | Usable rate | Ranking correct | Mean margin loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["ablation"]["arms"]:
        lines.append(
            f"| `{arm['arm']}` | {arm['total_rows']} | {arm['usable_rows']} | "
            f"{arm['unusable_rows']} | {arm['usable_negative_rate']:.3f} | "
            f"{arm['ranking_correct']} | {arm['mean_margin_loss']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Best arm by usable-negative-rate: `{report['ablation']['best_arm_by_usable_negative_rate']}`.",
            "",
            "## Honesty",
            "",
            "This is a hand-built, small synthetic fixture (4 DIFFERENT_RESULT rows, 2 CONFLICT rows, 3 "
            "no-negative rows), not a real operator_policy_corpus sample -- it demonstrates the arm-selection "
            "and usable-negative-rate wiring end to end, not a production readiness claim, and not a real "
            "gradient-trained-model quality claim (the scorer here is a fixed, deterministic, label-free "
            "proxy over declared cost/effect-signature, never a trained model). Real verified-conversation-trace "
            "incidence of CONFLICT versus DIFFERENT_RESULT hard negatives remains unmeasured, per DSH3-24's own "
            "documented scope gap.",
            "",
            f"Decision: `{report['decision']['verdict']}` -- {report['decision']['reason']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows()
    ablation: CollapseNegativeAblationReportV1 = evaluate_hard_negative_arms(rows)
    ablation_payload = ablation.to_dict()

    different_result_only = next(
        arm for arm in ablation_payload["arms"] if arm["arm"] == "different_result_only"
    )
    conflict_only = next(
        arm for arm in ablation_payload["arms"] if arm["arm"] == "conflict_only"
    )

    stamp = build_version_stamp(
        "model.operator_hard_negative_training",
        "model.operator_policy_view",
    )

    if different_result_only["usable_negative_rate"] > conflict_only["usable_negative_rate"]:
        decision = {
            "verdict": "fixture-scale-positive-signal",
            "reason": (
                "different_result_only strictly exceeds conflict_only's usable-negative-rate on this small "
                "synthetic fixture -- fixture-scale positive signal for DSH3-30's hypothesis that "
                "DIFFERENT_RESULT hard negatives carry denser, more sample-efficient ordering supervision than "
                "CONFLICT ones. This is not a claim about downstream trained-model quality (no real gradient "
                "training happened here -- the scorer is a fixed deterministic label-free proxy) and not a "
                "real-corpus-scale incidence measurement (real verified-trace CONFLICT/DIFFERENT_RESULT "
                "incidence remains unmeasured, per DSH3-24's own documented scope gap). It does not by itself "
                "justify wiring hard-negative-aware loss into DSH3-28's typed_operator_policy_loss at production "
                "scale -- that needs the real corpus's actual CONFLICT/DIFFERENT_RESULT incidence measured first."
            ),
        }
    else:
        decision = {
            "verdict": "no-benefit-observed",
            "reason": (
                "different_result_only did not exceed conflict_only's usable-negative-rate on this fixture; "
                "this would contradict DSH3-30's hypothesis and should be investigated before any further "
                "hard-negative-aware training work is proposed."
            ),
        }

    report = {
        "schema": "dsh3_30_collapse_negative_ablation_report/v1",
        "issue": "SLM-405",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_rows": len(rows),
            "different_result_rows": sum(
                1
                for row in rows
                if row.hard_negative is not None
                and row.hard_negative.outcome is HardNegativeOutcome.DIFFERENT_RESULT
            ),
            "conflict_rows": sum(
                1
                for row in rows
                if row.hard_negative is not None
                and row.hard_negative.outcome is HardNegativeOutcome.CONFLICT
            ),
            "no_negative_rows": sum(1 for row in rows if row.hard_negative is None),
            "ship_claim": False,
        },
        "ablation": ablation_payload,
        "decision": decision,
        "version_stamp": stamp,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "report.json"), "decision": decision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
