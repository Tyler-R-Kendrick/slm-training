from __future__ import annotations

import json

import pytest

from scripts.train_preference import main
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    counterfactual_evidence_from_traces,
    decision_event_manifest,
    decision_support_signature,
    decision_signature_support,
    events_from_trace,
    load_decision_events,
    load_trace_rows,
    objective_signature_support,
    split_for_group,
    write_decision_events,
)


def _trace() -> dict:
    return {
        "trajectory_id": "trace-1",
        "meta": {
            "record_id": "record-1",
            "context_text": "Generate a card",
            "policy_checkpoint_sha": "checkpoint-sha",
            "tokenizer_sha": "tokenizer-sha",
            "decode_config_hash": "decode-sha",
            "source_suite": "held_out",
            "seed": 7,
        },
        "steps": [
            {
                "commits": [
                    {
                        "t": 1,
                        "id": 4,
                        "raw_id": 9,
                        "pre_canvas": [1, 2, 2],
                        "allowed_id_set": [3, 4, 5],
                    }
                ]
            }
        ],
        "events": [],
    }


def test_constraint_shadow_round_trip(tmp_path) -> None:
    event = events_from_trace(_trace())[0]
    assert event.good_token_ids == (4,)
    assert event.bad_token_ids == (9,)
    assert event.evidence_kind == "constraint_shadow"
    assert event.split == split_for_group("record-1")

    path = tmp_path / "events.jsonl"
    assert write_decision_events(path, [event]) == 1
    assert load_decision_events(path) == [event]


def test_counterfactual_requires_same_state_proof() -> None:
    trace = _trace()
    trace["steps"] = []
    trace["events"] = [
        {
            "kind": "counterfactual_decision",
            "same_state_verified": False,
            "pre_canvas": [1, 2],
            "position": 1,
            "good_token_ids": [3],
            "bad_token_ids": [4],
            "legal_token_ids": [3],
        }
    ]
    with pytest.raises(ValueError, match="same-state"):
        events_from_trace(trace)


def test_counterfactual_requires_legal_good_and_bad_tokens() -> None:
    trace = _trace()
    trace["steps"] = []
    trace["events"] = [
        {
            "kind": "counterfactual_decision",
            "same_state_verified": True,
            "pre_canvas": [1, 2, 2],
            "position": 1,
            "good_token_ids": [3],
            "bad_token_ids": [4],
            "legal_token_ids": [3],
        }
    ]
    with pytest.raises(ValueError, match="qualified judge probe"):
        events_from_trace(trace)


def test_counterfactual_requires_labels_recomputed_from_judge_probe() -> None:
    trace = _trace()
    trace["steps"] = []
    metrics = {
        "placeholder_fidelity": 1.0,
        "component_recall": 1.0,
        "structural_similarity": 1.0,
        "reward": 1.0,
    }
    probe = {
        "kind": "counterfactual_probe",
        "same_state_verified": True,
        "state_hash": "state",
        "pre_canvas": [1, 2, 2],
        "position": 1,
        "legal_token_ids": [3, 4],
        "good_token_ids": [3],
        "bad_token_ids": [4],
        "qualified": True,
        "verifier": {
            "name": "independent_judge+meaningful_program+pareto_v1",
        },
        "candidates": [
            {"token_id": 3, "verified": True, "metrics": metrics},
            {
                "token_id": 4,
                "verified": False,
                "metrics": {name: 0.0 for name in metrics},
            },
        ],
    }
    decision = {
        "kind": "counterfactual_decision",
        "same_state_verified": True,
        "state_hash": "state",
        "pre_canvas": [1, 2, 2],
        "position": 1,
        "good_token_ids": [3],
        "bad_token_ids": [4],
        "legal_token_ids": [3, 4],
    }
    trace["events"] = [probe, decision]

    event = events_from_trace(trace)[0]
    assert event.evidence_kind == "counterfactual"
    evidence = counterfactual_evidence_from_traces([trace])
    assert len(evidence) == 1
    assert evidence[0]["probe"] == probe
    decision["good_token_ids"] = [4]
    with pytest.raises(ValueError, match="does not match judge probe"):
        events_from_trace(trace)


def test_event_rejects_cross_group_split() -> None:
    event = events_from_trace(_trace())[0]
    data = event.to_dict()
    data["split"] = "train" if event.split == "held_out" else "held_out"
    with pytest.raises(ValueError, match="group_id"):
        DecisionEventV1.from_dict(data)


def test_build_local_events_cli(tmp_path, capsys) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(_trace()) + "\n", encoding="utf-8")
    out = tmp_path / "events.jsonl"
    source = tmp_path / "source-manifest.json"
    source.write_text('{"content_fingerprint":"source-sha"}')
    source_two = tmp_path / "source-manifest-two.json"
    source_two.write_text('{"content_fingerprint":"source-sha-two"}')
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.jsonl"
    assert main(
        [
            "build-local-events", "--traces", str(traces), "--out", str(out),
            "--manifest-out", str(manifest), "--dataset-id", "events-v1",
            "--evidence-out", str(evidence),
            "--source-record-manifest", str(source),
            "--source-record-manifest", str(source_two),
        ]
    ) == 0
    assert len(load_decision_events(out)) == 1
    data = json.loads(manifest.read_text())
    assert data["record_count"] == 1
    assert data["source_record_fingerprint"] not in {
        "source-sha",
        "source-sha-two",
    }
    assert data["source_record_fingerprints"] == ["source-sha", "source-sha-two"]
    assert data["policy_checkpoint_sha"] == "checkpoint-sha"
    assert data["judge_evidence_count"] == 0
    assert evidence.read_text() == ""
    assert json.loads(capsys.readouterr().out)["events"] == 1


def test_build_local_events_can_require_counterfactual_evidence(
    tmp_path, capsys
) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(_trace()) + "\n", encoding="utf-8")
    out = tmp_path / "events.jsonl"
    assert main(
        [
            "build-local-events",
            "--traces",
            str(traces),
            "--out",
            str(out),
            "--evidence-kind",
            "counterfactual",
        ]
    ) == 0
    assert load_decision_events(out) == []
    assert json.loads(capsys.readouterr().out)["events"] == 0


def test_load_trace_rows_reads_nested_shards(tmp_path) -> None:
    for index in range(2):
        shard = tmp_path / f"shard-{index}"
        shard.mkdir()
        row = _trace()
        row["trajectory_id"] = f"trace-{index}"
        (shard / "traces.jsonl").write_text(json.dumps(row) + "\n")

    rows = load_trace_rows(tmp_path)

    assert [row["trajectory_id"] for row in rows] == ["trace-0", "trace-1"]


def test_manifest_rejects_mixed_policy_identities() -> None:
    first = events_from_trace(_trace())[0]
    data = first.to_dict()
    data["event_id"] = "other"
    data["policy_checkpoint_sha"] = "other-checkpoint"
    second = DecisionEventV1.from_dict(data)
    with pytest.raises(ValueError, match="mixes policy identities"):
        decision_event_manifest([first, second], dataset_id="mixed")


def test_signature_support_reports_sparse_held_out_semantics() -> None:
    base = events_from_trace(_trace())[0]
    train_group = "train"
    while split_for_group(train_group) != "train":
        train_group += "x"
    train_data = base.to_dict()
    train_data.update(event_id="train-event", group_id=train_group, split="train")
    train = DecisionEventV1.from_dict(train_data)
    held_group = "held"
    while split_for_group(held_group) != "held_out":
        held_group += "x"
    held_data = train.to_dict()
    held_data.update(
        event_id="held-event",
        group_id=held_group,
        split="held_out",
        good_token_ids=[7],
        bad_token_ids=[8],
        legal_token_ids=[7, 8],
    )
    held = DecisionEventV1.from_dict(held_data)

    report = decision_signature_support([train, held], min_train_support=1)

    assert report["held_out_coverage"]["passed"] is False
    assert len(report["held_out_coverage"]["uncovered"]) == 1
    with pytest.raises(ValueError, match="lacks train support"):
        decision_event_manifest(
            [train, held],
            dataset_id="sparse",
            require_signature_support=True,
        )


def test_support_signature_ignores_sampled_negative_variation() -> None:
    event = events_from_trace(_trace())[0]
    data = event.to_dict()
    extra_token = max(event.legal_token_ids) + 100
    data.update(
        event_id="different-negatives",
        bad_token_ids=[extra_token],
        legal_token_ids=sorted(set(event.legal_token_ids) | {extra_token}),
    )
    changed_legal = DecisionEventV1.from_dict(data)
    data["legal_token_ids"] = list(event.legal_token_ids)
    data["bad_token_ids"] = [event.legal_token_ids[-1]]
    changed_bad = DecisionEventV1.from_dict(data)

    assert decision_support_signature(changed_bad) == decision_support_signature(event)
    assert decision_support_signature(changed_legal) != decision_support_signature(event)


def _objective_event(group: str, split: str, bad: int) -> DecisionEventV1:
    return DecisionEventV1(
        event_id=f"e-{group}-{bad}",
        group_id=group,
        context_text="root=Stack([",
        canvas_ids=(1, 2, 3),
        position=1,
        good_token_ids=(4,),
        bad_token_ids=(bad,),
        legal_token_ids=(4, 9, 10),
        evidence_kind="counterfactual",
        evidence_confidence=0.9,
        decision_kind="component",
        split=split,
        policy_checkpoint_sha="pcs",
        tokenizer_sha="tsha",
        decode_config_hash="dch",
        seed=0,
        trajectory_id="traj",
    )


def _grow_group(prefix: str, split: str) -> str:
    group = prefix
    while split_for_group(group) != split:
        group += "x"
    return group


def test_objective_signature_support_catches_e284_pattern() -> None:
    train = _objective_event(_grow_group("obj-train", "train"), "train", 9)
    held = _objective_event(_grow_group("obj-held", "held_out"), "held_out", 10)
    # The state-support signature (good-only) matches, so state support passes...
    assert decision_signature_support([train, held])["held_out_coverage"]["passed"] is True
    # ...but the objective signature (good + bad) differs, so objective support fails.
    objective = objective_signature_support([train, held])
    assert objective["held_out_coverage"]["passed"] is False
    assert len(objective["held_out_coverage"]["uncovered"]) == 1


def test_train_local_from_paths_refuses_objective_unsupported_corpus(tmp_path) -> None:
    from pathlib import Path

    from slm_training.harnesses.preference.local_train import train_local_from_paths

    events_path = tmp_path / "events.jsonl"
    write_decision_events(
        events_path,
        [
            _objective_event(_grow_group("obj-train", "train"), "train", 9),
            _objective_event(_grow_group("obj-held", "held_out"), "held_out", 10),
        ],
    )
    # The admission refusal fires before the checkpoint is loaded, so the missing
    # checkpoint is never reached.
    with pytest.raises(ValueError, match="objective signature"):
        train_local_from_paths(
            Path("nonexistent.pt"),
            events_path,
            out_dir=tmp_path / "out",
            objective="ce_margin",
            require_objective_support=True,
        )


# ── V2 tests (SLM-116) ───────────────────────────────────────────────────

from slm_training.harnesses.preference.local_decisions import (
    ActionOutcomeV2,
    DecisionEventV2,
    DecisionStateV2,
    guard_semantic_view,
    materialize_constraint_shadow,
    materialize_objective_pareto,
    materialize_objective_set_partition,
    materialize_objective_single_best_worst,
    materialize_objective_threshold,
    materialize_v1_from_v2,
    merge_action_evidence,
    migrate_v1_to_v2,
    write_decision_events_v2,
    load_decision_events_v2,
    load_decision_events_v1_or_v2,
    decision_event_manifest_v2,
)


def _v2_state(**overrides) -> DecisionStateV2:
    defaults = {
        "state_id": "",
        "group_id": "record-1",
        "architecture": "twotower",
        "context_text": "Generate a card",
        "context_ids": None,
        "canvas_ids": (1, 2, 2),
        "decision_position": 1,
        "generation_step": 3,
        "legal_action_ids": (3, 4, 5),
        "decision_kind": "component",
        "abstract_state_role": "root_child",
        "grammar_state_hash": "grammar-sha",
        "policy_checkpoint_sha": "checkpoint-sha",
        "tokenizer_sha": "tokenizer-sha",
        "decode_config_hash": "decode-sha",
        "verifier_bundle_hash": "verifier-sha",
        "split": "train",
    }
    defaults.update(overrides)
    # split must agree with group_id
    defaults["split"] = split_for_group(defaults["group_id"])
    return DecisionStateV2(**defaults)


def _v2_outcome(state_id: str, action_id: int, **overrides) -> ActionOutcomeV2:
    defaults = {
        "state_id": state_id,
        "action_id": action_id,
        "legal": True,
        "rollout_policy_sha": "checkpoint-sha",
        "continuation_seeds": (7,),
        "outcome_hashes": ("hash-1",),
        "verifier_vectors": ({"G0": "pass", "G1": "pass"},),
        "reward_vectors": ({"reward": 1.0},),
        "mean_value": 1.0,
        "confidence_interval": (0.9, 1.0),
        "evidence_ids": ("ev-1",),
        "evidence_confidence": 1.0,
    }
    defaults.update(overrides)
    return ActionOutcomeV2(**defaults)


def _v2_event(state: DecisionStateV2 | None = None, outcomes=None, evidence_kind="counterfactual") -> DecisionEventV2:
    state = state or _v2_state()
    if outcomes is None:
        outcomes = (
            _v2_outcome(state.state_id, 4, reward_vectors=({"reward": 1.0},)),
            _v2_outcome(state.state_id, 5, reward_vectors=({"reward": 0.0},)),
        )
    return DecisionEventV2(state=state, outcomes=outcomes, evidence_kind=evidence_kind)


def test_v2_state_id_order_independent() -> None:
    state = _v2_state()
    outcomes_a = (
        _v2_outcome(state.state_id, 4),
        _v2_outcome(state.state_id, 5),
    )
    outcomes_b = (
        _v2_outcome(state.state_id, 5),
        _v2_outcome(state.state_id, 4),
    )
    event_a = _v2_event(state, outcomes_a)
    event_b = _v2_event(state, outcomes_b)
    assert event_a.state.state_id == event_b.state.state_id


def test_v2_merge_same_state() -> None:
    state = _v2_state()
    event1 = _v2_event(
        state,
        outcomes=(_v2_outcome(state.state_id, 4, evidence_ids=("ev-a",)),),
    )
    event2 = _v2_event(
        state,
        outcomes=(_v2_outcome(state.state_id, 5, evidence_ids=("ev-b",)),),
    )
    merged = merge_action_evidence([event1, event2])
    assert len(merged) == 1
    assert {outcome.action_id for outcome in merged[0].outcomes} == {4, 5}


def test_v2_state_metadata_conflict() -> None:
    state = _v2_state()
    event_a = _v2_event(state, evidence_kind="counterfactual")
    event_b = _v2_event(state, evidence_kind="constraint_shadow")
    with pytest.raises(ValueError, match="conflicting evidence_kind"):
        merge_action_evidence([event_a, event_b])


def test_v2_action_outside_legal_set() -> None:
    state = _v2_state(legal_action_ids=(3, 4, 5))
    bad_outcome = _v2_outcome(state.state_id, 99)
    with pytest.raises(ValueError, match="not in the state's legal set"):
        _v2_event(state, outcomes=(bad_outcome,))


def test_v2_unknown_fields_rejected() -> None:
    data = _v2_event().to_dict()
    data["unknown_field"] = "x"
    with pytest.raises(ValueError, match="unknown decision event v2"):
        DecisionEventV2.from_dict(data)


def test_v2_materializer_id_changes_with_config() -> None:
    event = _v2_event()
    view_a = materialize_objective_pareto(event, metric_thresholds={"reward": 0.0})
    view_b = materialize_objective_pareto(event, metric_thresholds={"reward": 0.5})
    assert view_a.materializer_config_hash != view_b.materializer_config_hash
    # Recomputing with the same config yields the same hash.
    view_a2 = materialize_objective_pareto(event, metric_thresholds={"reward": 0.0})
    assert view_a.materializer_config_hash == view_a2.materializer_config_hash


def test_v2_pareto_materializer() -> None:
    state = _v2_state(legal_action_ids=(3, 4, 5, 6))
    outcomes = (
        _v2_outcome(state.state_id, 3, reward_vectors=({"reward": 1.0, "fidelity": 0.9},)),
        _v2_outcome(state.state_id, 4, reward_vectors=({"reward": 0.3, "fidelity": 0.9},)),
        _v2_outcome(state.state_id, 5, reward_vectors=({"reward": 0.0, "fidelity": 0.0},)),
        # 6 has no reward evidence -> ambiguous/unobserved
    )
    event = _v2_event(state, outcomes=outcomes)
    view = materialize_objective_pareto(
        event, metric_thresholds={"reward": 0.5, "fidelity": 0.5}
    )
    assert view.good_action_ids == (3,)
    assert 5 in view.bad_action_ids
    assert 6 in view.ambiguous_action_ids or 6 in view.unobserved_action_ids


def test_v2_constraint_shadow_non_trainable() -> None:
    state = _v2_state(legal_action_ids=(3, 4, 5))
    outcomes = (
        ActionOutcomeV2(
            state_id=state.state_id,
            action_id=4,
            legal=True,
            rollout_policy_sha="checkpoint-sha",
            continuation_seeds=(),
            outcome_hashes=(),
            verifier_vectors=(),
            reward_vectors=(),
            mean_value=None,
            confidence_interval=None,
            evidence_ids=(),
            evidence_confidence=1.0,
        ),
        ActionOutcomeV2(
            state_id=state.state_id,
            action_id=9,
            legal=False,
            rollout_policy_sha="checkpoint-sha",
            continuation_seeds=(),
            outcome_hashes=(),
            verifier_vectors=(),
            reward_vectors=(),
            mean_value=None,
            confidence_interval=None,
            evidence_ids=(),
            evidence_confidence=1.0,
        ),
    )
    event = DecisionEventV2(
        state=state, outcomes=outcomes, evidence_kind="constraint_shadow"
    )
    view = materialize_constraint_shadow(event)
    with pytest.raises(ValueError, match="diagnostic-only"):
        guard_semantic_view(view)
    with pytest.raises(ValueError, match="diagnostic-only"):
        materialize_v1_from_v2(event, view)


def test_v1_migration_preserves_content_fingerprint(tmp_path) -> None:
    v1_event = events_from_trace(_trace())[0]
    path = tmp_path / "events.jsonl"
    write_decision_events(path, [v1_event])
    loaded = load_decision_events(path)
    assert loaded == [v1_event]

    v2_event = migrate_v1_to_v2(v1_event)
    assert v2_event.state.group_id == v1_event.group_id
    assert v2_event.evidence_kind == "constraint_shadow"
    # Incomplete evidence: no rollout seeds/hashes/verifier vectors.
    assert all(not outcome.continuation_seeds for outcome in v2_event.outcomes)
    # Migration is deterministic/idempotent at the V1 level.
    v2_event_2 = migrate_v1_to_v2(v1_event)
    assert v2_event.state.state_id == v2_event_2.state.state_id
    assert v2_event.outcomes == v2_event_2.outcomes


def test_v2_manifest_fingerprints(tmp_path) -> None:
    event = _v2_event()
    manifest = decision_event_manifest_v2([event], dataset_id="v2-test")
    assert manifest["schema_version"] == 2
    assert "state_fingerprint" in manifest
    assert "evidence_fingerprint" in manifest
    assert "objective_fingerprint" in manifest
    # Changing evidence metadata changes the evidence fingerprint but not the
    # state fingerprint (objective view is unchanged because rewards are equal).
    event2 = _v2_event(
        event.state,
        outcomes=(
            _v2_outcome(event.state.state_id, 4, evidence_ids=("different",)),
            _v2_outcome(event.state.state_id, 5),
        ),
    )
    manifest2 = decision_event_manifest_v2([event2], dataset_id="v2-test")
    assert manifest2["state_fingerprint"] == manifest["state_fingerprint"]
    assert manifest2["evidence_fingerprint"] != manifest["evidence_fingerprint"]
    # Changing the reward evidence changes the objective fingerprint too.
    event3 = _v2_event(
        event.state,
        outcomes=(
            _v2_outcome(
                event.state.state_id,
                4,
                mean_value=None,
                reward_vectors=({"reward": 1.0},),
            ),
            _v2_outcome(
                event.state.state_id,
                5,
                mean_value=None,
                reward_vectors=({"reward": 0.0},),
            ),
        ),
    )
    manifest3 = decision_event_manifest_v2([event3], dataset_id="v2-test")
    assert manifest3["objective_fingerprint"] != manifest["objective_fingerprint"]


def test_v2_round_trip(tmp_path) -> None:
    state = _v2_state()
    verifier_vectors = (
        {"G0": "pass", "G1": "pass", "G2": "fail", "G3": "pass",
         "G4": "pass", "G5": "pass", "G6": "pass", "G7": "pass",
         "G8": "pass", "G9": "pass", "G10": "pass", "G11": "pass", "G12": "pass"},
    )
    outcomes = (
        _v2_outcome(
            state.state_id,
            4,
            verifier_vectors=verifier_vectors,
            reward_vectors=({"reward": 1.0},),
        ),
    )
    event = _v2_event(state, outcomes=outcomes)
    path = tmp_path / "events-v2.jsonl"
    write_decision_events_v2(path, [event])
    loaded = load_decision_events_v2(path)
    assert len(loaded) == 1
    assert loaded[0].state.state_id == event.state.state_id
    assert loaded[0].outcomes[0].verifier_vectors == verifier_vectors


def test_v2_load_dispatch(tmp_path) -> None:
    v1_event = events_from_trace(_trace())[0]
    v1_path = tmp_path / "v1.jsonl"
    write_decision_events(v1_path, [v1_event])
    v1_loaded = load_decision_events_v1_or_v2(v1_path)
    assert isinstance(v1_loaded[0], DecisionEventV1)

    v2_event = _v2_event()
    v2_path = tmp_path / "v2.jsonl"
    write_decision_events_v2(v2_path, [v2_event])
    v2_loaded = load_decision_events_v1_or_v2(v2_path)
    assert isinstance(v2_loaded[0], DecisionEventV2)


def test_v2_causal_prefix_and_twotower_canvas_fixture() -> None:
    causal_state = _v2_state(
        architecture="causal",
        context_text=" causal prefix",
        context_ids=(10, 20, 30),
        canvas_ids=None,
        decision_position=3,
    )
    twotower_state = _v2_state(
        architecture="twotower",
        context_text=" twotower prefix",
        canvas_ids=(1, 2, 3),
        decision_position=1,
    )
    # Both produce stable, distinct state IDs from their respective identities.
    assert causal_state.state_id != twotower_state.state_id
    # Reconstructing the same fields yields the same ID.
    causal_state_2 = _v2_state(
        architecture="causal",
        context_text=" causal prefix",
        context_ids=(10, 20, 30),
        canvas_ids=None,
        decision_position=3,
    )
    assert causal_state.state_id == causal_state_2.state_id


def test_v2_threshold_materializer() -> None:
    state = _v2_state(legal_action_ids=(3, 4))
    outcomes = (
        _v2_outcome(state.state_id, 3, mean_value=0.8, confidence_interval=(0.7, 0.9)),
        _v2_outcome(state.state_id, 4, mean_value=0.3, confidence_interval=(0.2, 0.4)),
    )
    event = _v2_event(state, outcomes=outcomes)
    view = materialize_objective_threshold(
        event, threshold=0.5, min_confidence_lower=0.6
    )
    assert view.good_action_ids == (3,)
    assert view.bad_action_ids == (4,)


def test_v2_single_best_worst_materializer() -> None:
    state = _v2_state(legal_action_ids=(3, 4, 5))
    outcomes = (
        _v2_outcome(state.state_id, 3, mean_value=0.9),
        _v2_outcome(state.state_id, 4, mean_value=0.5),
        _v2_outcome(state.state_id, 5, mean_value=0.1),
    )
    event = _v2_event(state, outcomes=outcomes)
    view = materialize_objective_single_best_worst(event)
    assert view.good_action_ids == (3,)
    assert view.bad_action_ids == (5,)
    assert view.ambiguous_action_ids == (4,)


def test_v2_set_partition_materializer() -> None:
    state = _v2_state(legal_action_ids=(3, 4, 5))
    outcomes = (
        _v2_outcome(
            state.state_id, 3, reward_vectors=({"reward": 1.0, "fidelity": 1.0},)
        ),
        _v2_outcome(
            state.state_id, 4, reward_vectors=({"reward": 0.6, "fidelity": 0.6},)
        ),
        _v2_outcome(
            state.state_id, 5, reward_vectors=({"reward": 0.1, "fidelity": 0.1},)
        ),
    )
    event = _v2_event(state, outcomes=outcomes)
    view = materialize_objective_set_partition(event)
    assert view.good_action_ids == (3,)
    assert 5 in view.bad_action_ids
