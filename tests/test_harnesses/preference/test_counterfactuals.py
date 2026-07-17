from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference.counterfactuals import (
    counterfactual_state_signature,
    gold_counterfactual_commits,
    label_pareto_candidates,
    load_counterfactual_state_targets,
    select_counterfactual_states,
    semantic_outcome,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


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


def test_state_selection_stratifies_parser_roles_before_repeating_kinds() -> None:
    commits = [
        {
            "phase": "compiler_tree",
            "allowed_id_set": [1, 2],
            "pre_canvas": [position, 0],
            "t": position,
            "decision_kind": kind,
        }
        for position, kind in (
            (1, "bind_declaration_root"),
            (2, "component_root"),
            (3, "component_root"),
            (8, "bind_reference_root_children"),
            (9, "grammar_rsqb_root_populated"),
        )
    ]

    selected = select_counterfactual_states(
        commits, max_states=4, seed=7, context_key="record-a"
    )

    assert len(selected) == 4
    assert len({row["decision_kind"] for row in selected}) == 4
    assert all(0 <= row["counterfactual_depth_bucket"] <= 3 for row in selected)


def test_state_selection_is_deterministic_and_ignores_ineligible_commits() -> None:
    commits = [
        {
            "phase": "compiler_tree",
            "allowed_id_set": [1, 2],
            "pre_canvas": [position, 0],
            "t": position,
            "decision_kind": "component_bound",
        }
        for position in range(1, 9)
    ]
    commits.extend(
        [
            {
                "phase": "repair",
                "allowed_id_set": [1, 2],
                "pre_canvas": [99, 0],
                "t": 99,
            },
            {
                "phase": "compiler_tree",
                "allowed_id_set": [1],
                "pre_canvas": [100, 0],
                "t": 100,
            },
        ]
    )

    first = select_counterfactual_states(
        commits, max_states=8, seed=11, context_key="record-b"
    )
    second = select_counterfactual_states(
        commits, max_states=8, seed=11, context_key="record-b"
    )

    assert [row["t"] for row in first] == [row["t"] for row in second]
    assert len(first) == 8
    assert len({row["counterfactual_depth_bucket"] for row in first}) == 4
    assert all(row["t"] < 99 for row in first)


def test_state_selection_diversifies_semantic_signatures_within_kind() -> None:
    commits = [
        {
            "phase": "compiler_tree",
            "allowed_id_set": [1, 2, 3],
            "pre_canvas": [position, 0],
            "t": position,
            "id": selected,
            "decision_kind": "component_bound",
        }
        for position, selected in ((1, 1), (2, 1), (3, 2))
    ]

    selected = select_counterfactual_states(
        commits, max_states=2, seed=7, context_key="record-a"
    )

    assert {row["id"] for row in selected} == {1, 2}


def test_state_selection_can_target_exact_grammar_metadata(tmp_path) -> None:
    commits = [
        {
            "phase": "compiler_tree",
            "allowed_id_set": [1, 2, 3],
            "pre_canvas": [position, 0],
            "t": position,
            "id": selected,
            "decision_kind": kind,
        }
        for position, selected, kind in (
            (1, 1, "component_bound"),
            (2, 2, "component_bound"),
            (3, 3, "bind_reference_bound_children"),
        )
    ]
    target = {
        "decision_kind": "component_bound",
        "legal_token_ids": [3, 2, 1],
        "selected_token_id": 2,
    }
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"targets": [target]}))

    signatures = load_counterfactual_state_targets(path)
    selected = select_counterfactual_states(
        commits,
        max_states=4,
        target_signatures=signatures,
    )

    assert signatures == {counterfactual_state_signature(target)}
    assert [row["t"] for row in selected] == [2]


def test_gold_counterfactual_commits_follow_grammar_decisions() -> None:
    tokenizer = DSLNativeTokenizer.build()
    target = tokenizer.encode(
        'root=Card([title])\ntitle=TextContent(":hero.title")',
        add_special=True,
    )

    commits = gold_counterfactual_commits(
        tokenizer,
        target,
        canvas_length=64,
        slot_contract=[":hero.title"],
    )

    assert commits
    assert {
        "bind_declaration_root",
        "bind_reference_root_children",
        "component_root",
        "component_bound",
    } <= {row["decision_kind"] for row in commits}
    for row in commits:
        position = row["t"]
        assert row["id"] == target[position]
        assert row["id"] in row["allowed_id_set"]
        assert row["pre_canvas"][:position] == target[:position]
        assert row["pre_canvas"][position] == tokenizer.mask_id
        assert row["state_source"] == "gold_ast"


def test_gold_state_mining_uses_gold_only_for_selected_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from slm_training.harnesses.preference import counterfactuals

    record = _record()
    tokenizer = DSLNativeTokenizer.build()
    target = tokenizer.encode(record.openui, add_special=True)

    class Recorder:
        steps: list[dict[str, object]] = []

        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def event(self, kind: str, **payload: object) -> None:
            self.events.append({"kind": kind, **payload})

    class Model:
        config = SimpleNamespace(
            grammar_sample_decode=False,
            max_target_len=64,
            slot_contract_constrained_decode=True,
        )
        device_name = "cpu"
        trace_recorder = None

        def _encode_openui(self, *_args, **_kwargs):
            return target

        def _encode_context(self, _texts):
            return torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.bool)

        def _denoiser_hidden(self, canvas, _ctx, _ctx_pad):
            return torch.zeros(1, canvas.shape[1], 2)

        def _project_candidates(self, _hidden, candidates):
            return torch.arange(len(candidates), dtype=torch.float32)

        def _compiler_ltr_decode_one(self, *_args, **_kwargs):
            return torch.tensor([tokenizer.bos_id, tokenizer.eos_id])

        def _decode_openui(self, *_args, **_kwargs):
            return 'root = Button(":hero.title")\n'

        def _ensure_valid_openui(self, text, *_args, **_kwargs):
            return text

    Model.tokenizer = tokenizer

    def outcome(_record: ExampleRecord, text: str) -> dict[str, object]:
        verified = text == record.openui
        score = 1.0 if verified else 0.0
        return {
            "judge": {"ok": verified},
            "meaningful": verified,
            "meaningful_error": None,
            "verified": verified,
            "metrics": {
                "placeholder_fidelity": score,
                "component_recall": score,
                "structural_similarity": score,
                "reward": score,
            },
        }

    monkeypatch.setattr(counterfactuals, "semantic_outcome", outcome)
    recorder = Recorder()
    stats = counterfactuals.mine_semantic_counterfactuals(
        Model(),
        recorder,
        record,
        "production context without gold",
        max_states=1,
        max_candidates=2,
        state_source="gold_ast",
    )

    assert stats["events"] == 1
    probe = next(row for row in recorder.events if row["kind"] == "counterfactual_probe")
    assert probe["state_source"] == "gold_ast"
    candidates = probe["candidates"]
    selected = next(row for row in candidates if row["selected"] is True)
    alternative = next(row for row in candidates if row["selected"] is False)
    assert selected["text"] == record.openui
    assert selected["completion_source"] == "gold_ast"
    assert alternative["completion_source"] == "policy"
