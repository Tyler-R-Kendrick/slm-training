from __future__ import annotations

from slm_training.harnesses.experiments.reserved_operator_baseline import (
    ReservedOperatorBaselineArm,
    build_operator_decisions,
    run_reserved_operator_baseline,
)


def _rows(prefix: str) -> list[dict]:
    rows = []
    for group in range(2):
        for index in range(2):
            application_id = f"{prefix}-application-{group}-{index}"
            rows.append(
                {
                    "target_view": "dual",
                    "outcome": "success",
                    "source_record_id": f"{prefix}-source-{group}",
                    "before_ast": f'root = TextContent(":{prefix}.{group}")',
                    "legal_set_fingerprint": f"{prefix}-legal-{group}",
                    "after_ast": f'root = TextContent(":{prefix}.{group}.{index}")',
                    "legal_action": {
                        "operator_id": f"openui.operator_{index}",
                        "application_id": application_id,
                    },
                    "application": {"application_id": application_id},
                    "answer": {
                        "operator": (
                            f"OPERATOR openui.operator_{index} "
                            f"value=value:req:{prefix}{group}{index}"
                        ),
                        "result_ast": (
                            f'root = TextContent(":{prefix}.{group}.{index}")'
                        ),
                    },
                }
            )
    return rows


def test_canonical_rows_form_ambiguous_matched_decisions() -> None:
    decisions = build_operator_decisions(_rows("train"))
    assert len(decisions) == 4
    assert all(len(decision.candidates) == 2 for decision in decisions)
    assert len({decision.context for decision in decisions}) == 2


def test_matched_token_baseline_rejects_ambiguous_fixture() -> None:
    result = run_reserved_operator_baseline(
        train_rows=_rows("train"),
        held_out_rows=_rows("held"),
        seeds=(7,),
        steps=2,
        learning_rate=0.01,
    )
    assert result["experiment_id"] == "E803"
    assert result["verdict"] == "reject"
    assert result["accepted"] is False
    parameter_counts = {
        run["parameter_count"]
        for values in result["arms"].values()
        for run in values
    }
    assert len(parameter_counts) == 1
    assert all(
        run["false_legal_admissions"] == 0
        for values in result["arms"].values()
        for run in values
    )
    assert set(result["arms"]) == {
        arm.value for arm in ReservedOperatorBaselineArm
    }
    assert (
        result["acceptance"]["held_out_result_ast_improves_across_seeds"]
        is False
    )
