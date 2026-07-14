from __future__ import annotations

from slm_training.evals.task_scoreboard import build_task_scoreboard, score_case

BUTTON = 'root = Stack([cta])\ncta = Button(":cta.label")'
CARD = 'root = Stack([card])\ncard = Card([title])\ntitle = TextContent(":title.text")'


def test_structural_metrics_use_ast_and_real_tree_edit() -> None:
    metrics = score_case(
        {"id": "same", "task": "generation", "gold": CARD, "prediction": CARD}
    )["metrics"]
    assert metrics["language_validity"]["value"] == 1.0
    assert metrics["ast_node_f1"]["value"] == 1.0
    assert metrics["ast_edge_f1"]["value"] == 1.0
    assert metrics["tree_edit_similarity"]["value"] == 1.0
    assert metrics["ref_graph_exact"]["value"] == 1.0
    canonical = metrics["canonical_exact"]
    if canonical["value"] is None:
        assert "lang-core" in canonical["reason"]
    else:
        assert canonical["value"] == 1.0
        assert canonical["status"] == "available"

    changed = score_case(
        {"id": "changed", "task": "generation", "gold": CARD, "prediction": BUTTON}
    )["metrics"]
    assert 0.0 <= changed["tree_edit_similarity"]["value"] < 1.0


def test_invalid_prediction_is_invalid_not_fake_structural_zero() -> None:
    metrics = score_case(
        {"id": "bad", "task": "repair", "gold": BUTTON, "prediction": "broken"}
    )["metrics"]
    assert metrics["language_validity"]["value"] == 0.0
    assert metrics["ast_node_f1"]["value"] is None
    assert metrics["repair_minimality"]["value"] is None


def test_l3_equivalence_excludes_exact_match() -> None:
    result = score_case(
        {
            "id": "l3",
            "task": "generation",
            "gold": CARD,
            "prediction": BUTTON,
            "abstraction_level": "L3",
            "prediction_evidence": {
                "required_facts": ["has_action"],
                "forbidden_facts": ["has_form"],
                "predicted_facts": ["has_action"],
            },
        }
    )
    assert result["metrics"]["equivalence_score"]["value"] == 1.0
    assert result["metrics"]["ast_node_f1"]["value"] < 1.0


def test_l4_l5_require_prediction_side_evidence() -> None:
    board = build_task_scoreboard(
        [
            {
                "id": "l4",
                "task": "behavior",
                "gold": BUTTON,
                "prediction": BUTTON,
                "abstraction_level": "L4",
                "gold_evidence": {"behavior_equivalence": 1.0},
                "prediction_evidence": {
                    "required_facts": ["has_action"],
                    "predicted_facts": ["has_action"],
                },
            },
            {
                "id": "l5",
                "task": "behavior",
                "gold": BUTTON,
                "prediction": BUTTON,
                "abstraction_level": "L5",
                "prediction_evidence": {
                    "required_facts": ["has_action"],
                    "predicted_facts": ["has_action"],
                    "behavior_equivalence": 1.0,
                    "render_equivalence": 0.8,
                },
            },
        ]
    )
    assert board["details"][0]["metrics"]["equivalence_score"]["value"] is None
    assert board["details"][1]["metrics"]["equivalence_score"]["value"] > 0.9
    assert board["complete"] is False


def test_task_evidence_metrics_are_coverage_aware() -> None:
    board = build_task_scoreboard(
        [
            {
                "id": "diff",
                "task": "generation",
                "gold": BUTTON,
                "prediction": BUTTON,
                "prediction_evidence": {
                    "length_accuracy": 0.75,
                    "expand_contract_success": True,
                    "steps_to_first_valid": 3,
                },
            }
        ]
    )
    metrics = board["tasks"]["generation"]["metrics"]
    assert metrics["length_accuracy"]["value"] == 0.75
    assert metrics["expand_contract_success"]["value"] == 1.0
    assert metrics["steps_to_first_valid"]["value"] == 3.0
