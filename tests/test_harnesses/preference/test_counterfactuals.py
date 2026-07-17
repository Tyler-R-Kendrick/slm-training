from __future__ import annotations

from types import SimpleNamespace

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference.counterfactuals import (
    label_pareto_candidates,
    semantic_outcome,
)


def _record() -> ExampleRecord:
    return ExampleRecord(
        id="card",
        prompt="Create a Card with TextContent for the hero title.",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )


def test_semantic_outcome_requires_judge_and_meaningful_program() -> None:
    valid = semantic_outcome(_record(), _record().openui)
    wrong = semantic_outcome(
        _record(), 'root = Button(":hero.title")\n'
    )

    assert valid["verified"] is True
    assert valid["metrics"] == {
        "placeholder_fidelity": 1.0,
        "component_recall": 1.0,
        "structural_similarity": 1.0,
        "reward": valid["metrics"]["reward"],
    }
    assert wrong["verified"] is False
    assert "prompt_component_missing_from_output" in wrong["judge"]["reasons"]


def test_pareto_labels_verified_frontier_and_failed_or_dominated_tokens() -> None:
    candidates = [
        {
            "token_id": 3,
            "verified": True,
            "metrics": {
                "placeholder_fidelity": 1.0,
                "component_recall": 1.0,
                "structural_similarity": 0.9,
                "reward": 0.8,
            },
        },
        {
            "token_id": 4,
            "verified": True,
            "metrics": {
                "placeholder_fidelity": 1.0,
                "component_recall": 0.5,
                "structural_similarity": 0.7,
                "reward": 0.6,
            },
        },
        {
            "token_id": 5,
            "verified": False,
            "metrics": {
                "placeholder_fidelity": 0.0,
                "component_recall": 0.0,
                "structural_similarity": 0.0,
                "reward": 0.0,
            },
        },
    ]

    assert label_pareto_candidates(candidates) == ([3], [4, 5])


def test_pareto_keeps_incomparable_verified_candidates() -> None:
    candidates = [
        {
            "token_id": token,
            "verified": True,
            "metrics": {
                "placeholder_fidelity": placeholder,
                "component_recall": recall,
                "structural_similarity": 0.8,
                "reward": 0.7,
            },
        }
        for token, placeholder, recall in ((3, 1.0, 0.5), (4, 0.5, 1.0))
    ]

    assert label_pareto_candidates(candidates) == ([3, 4], [])


def test_same_state_replay_rejects_sample_decode() -> None:
    from slm_training.harnesses.preference.counterfactuals import (
        mine_semantic_counterfactuals,
    )

    model = SimpleNamespace(config=SimpleNamespace(grammar_sample_decode=True))
    recorder = SimpleNamespace(steps=[])
    with pytest.raises(ValueError, match="deterministic decode"):
        mine_semantic_counterfactuals(model, recorder, _record(), "context")
