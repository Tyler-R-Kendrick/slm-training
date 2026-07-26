"""Regression tests for the DSH3-30/SLM-405 CONFLICT-vs-DIFFERENT_RESULT
collapse hard-negative ablation mechanism.

Fixtures hand-build minimal ``OperatorPolicyRowV1`` rows directly from
``OperatorPolicyInputV1``/``OperatorActionViewV1`` (mirroring the
dataclass-construction style ``tests/test_models/test_operator_ecoc.py``
uses for ``OperatorActionViewV1`` fixtures, and the general row-shape from
``tests/test_data/flow/test_operator_policy_corpus.py``) rather than
replaying a real collapse trace end to end -- this keeps the fixtures fast,
deterministic, and independent of the DSL pack's bridge toolchain, while
still exercising the exact real-world CONFLICT/DIFFERENT_RESULT
``alternate_action_row`` semantics DSH3-24 established: for a
``DIFFERENT_RESULT`` row, the alternate operator is included among
``policy_input.action_rows`` (a live, legal candidate the swap could
actually execute against); for a ``CONFLICT`` row, the alternate operator is
genuinely *absent* from ``policy_input.action_rows`` (re-enumeration never
found it live at that state), so ``alternate_action_row`` is ``None`` for a
structural reason, not an arbitrary null.
"""

from __future__ import annotations

import hashlib

import pytest

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
    evaluate_hard_negative_arm,
    evaluate_hard_negative_arms,
    hard_negative_margin_loss,
    select_hard_negative_rows,
)
from slm_training.models.operator_policy_view import OperatorActionViewV1, OperatorPolicyInputV1


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


def _different_result_row(suffix: str) -> OperatorPolicyRowV1:
    """One row where the swap replayed successfully but disagreed: both
    ``openui.set_x`` and ``openui.set_y`` are live, legal candidate actions
    at this step -- exactly the case where a DIFFERENT_RESULT negative has a
    real alternate to rank against."""
    accepted_op = f"openui.set_x_{suffix}"
    alternate_op = f"openui.set_y_{suffix}"
    action_views = (_action_view(0, accepted_op), _action_view(1, alternate_op))
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
    """One row where the swap outright failed to replay: ``openui.set_b`` is
    genuinely absent from this step's re-enumerated ``action_rows`` (only
    ``openui.set_a`` is live), so there is no live alternate action row to
    rank against -- ``alternate_action_row`` is ``None`` for a structural
    reason, matching how the real corpus builder produces a CONFLICT row
    whose swap partner never re-enumerates as legal."""
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


# --- Fixture sanity: alternate_action_row is/isn't None for the same
# structural reason the real corpus builder produces it. -------------------


def test_different_result_row_has_a_live_alternate_action_row() -> None:
    row = _different_result_row("dr0")
    assert row.hard_negative is not None
    assert row.hard_negative.outcome is HardNegativeOutcome.DIFFERENT_RESULT
    assert row.hard_negative.alternate_action_row is not None
    action_rows = row.policy_input["action_rows"]
    assert 0 <= row.hard_negative.alternate_action_row < len(action_rows)
    alternate_action = action_rows[row.hard_negative.alternate_action_row]
    assert alternate_action["operator_id"] == row.hard_negative.alternate_operator_id


def test_conflict_row_has_no_live_alternate_action_row() -> None:
    row = _conflict_row("cf0")
    assert row.hard_negative is not None
    assert row.hard_negative.outcome is HardNegativeOutcome.CONFLICT
    assert row.hard_negative.alternate_action_row is None
    # The alternate operator is genuinely absent from action_rows -- not
    # merely nulled while still present.
    operator_ids = {action["operator_id"] for action in row.policy_input["action_rows"]}
    assert row.hard_negative.alternate_operator_id not in operator_ids


# --- The core DSH3-30 hypothesis-mechanism test ------------------------------


def test_different_result_only_has_strictly_higher_usable_negative_rate_than_conflict_only() -> (
    None
):
    """DIFFERENT_RESULT negatives structurally preserve a live alternate to
    rank against; CONFLICT negatives structurally don't. This is DSH3-30's
    hypothesis ("denser, more sample-efficient ordering supervision from
    DIFFERENT_RESULT") made concrete and testable without any gradient
    training: on identical positive-row/example-count budgets (one negative
    row each), different_result_only's usable_negative_rate is 1.0 and
    conflict_only's is 0.0."""
    rows = (_different_result_row("dr1"), _conflict_row("cf1"))

    different_result_report = evaluate_hard_negative_arm(rows, "different_result_only")
    conflict_report = evaluate_hard_negative_arm(rows, "conflict_only")

    assert different_result_report.total_rows == 1
    assert conflict_report.total_rows == 1
    assert different_result_report.usable_negative_rate == 1.0
    assert conflict_report.usable_negative_rate == 0.0
    assert different_result_report.usable_negative_rate > conflict_report.usable_negative_rate


def test_none_arm_always_selects_zero_rows() -> None:
    rows = (_different_result_row("dr2"), _conflict_row("cf2"))
    assert select_hard_negative_rows(rows, "none") == ()
    assert evaluate_hard_negative_arm(rows, "none").total_rows == 0
    assert evaluate_hard_negative_arm((), "none").total_rows == 0


def test_equal_mix_balances_to_the_minority_negative_type_count() -> None:
    rows = (
        _different_result_row("dr3a"),
        _different_result_row("dr3b"),
        _different_result_row("dr3c"),
        _conflict_row("cf3"),
    )

    selected = select_hard_negative_rows(rows, "equal_mix")
    assert len(selected) == 2  # min(3, 1) of each type
    counts: dict[str, int] = {}
    for row in selected:
        assert row.hard_negative is not None
        counts[row.hard_negative.outcome.value] = (
            counts.get(row.hard_negative.outcome.value, 0) + 1
        )
    assert counts.get("different_result", 0) == 1
    assert counts.get("conflict", 0) == 1

    selected_again = select_hard_negative_rows(rows, "equal_mix")
    assert tuple(row.row_id for row in selected) == tuple(
        row.row_id for row in selected_again
    )


def test_curriculum_mix_orders_different_result_before_conflict() -> None:
    rows = (_conflict_row("cf4"), _different_result_row("dr4"))
    selected = select_hard_negative_rows(rows, "curriculum_mix")
    assert len(selected) == 2
    assert selected[0].hard_negative is not None
    assert selected[0].hard_negative.outcome is HardNegativeOutcome.DIFFERENT_RESULT
    assert selected[1].hard_negative is not None
    assert selected[1].hard_negative.outcome is HardNegativeOutcome.CONFLICT


def test_hard_negative_margin_loss_none_when_no_live_alternate() -> None:
    assert hard_negative_margin_loss(1.0, None) is None


def test_hard_negative_margin_loss_zero_when_margin_is_cleared() -> None:
    assert hard_negative_margin_loss(2.0, 0.5, margin=1.0) == 0.0


def test_hard_negative_margin_loss_positive_when_margin_is_not_cleared() -> None:
    loss = hard_negative_margin_loss(0.5, 0.5, margin=1.0)
    assert loss is not None
    assert loss > 0.0
    assert loss == pytest.approx(1.0)


def test_ablation_report_to_dict_round_trips_and_identifies_best_arm() -> None:
    rows = (_different_result_row("dr5"), _conflict_row("cf5"))
    report = evaluate_hard_negative_arms(rows)
    assert isinstance(report, CollapseNegativeAblationReportV1)
    payload = report.to_dict()
    assert payload["schema"] == "collapse_negative_ablation_report/v1"
    assert {arm["arm"] for arm in payload["arms"]} == {
        "none",
        "conflict_only",
        "different_result_only",
        "equal_mix",
        "curriculum_mix",
    }
    for arm_payload in payload["arms"]:
        assert set(arm_payload) == {
            "schema",
            "arm",
            "total_rows",
            "usable_rows",
            "unusable_rows",
            "negative_type_counts",
            "ranking_correct",
            "mean_margin_loss",
            "usable_negative_rate",
        }
    assert payload["best_arm_by_usable_negative_rate"] == "different_result_only"
