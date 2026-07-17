from __future__ import annotations

import json

import pytest

from scripts.train_preference import main
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    counterfactual_evidence_from_traces,
    decision_event_manifest,
    events_from_trace,
    load_decision_events,
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
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.jsonl"
    assert main(
        [
            "build-local-events", "--traces", str(traces), "--out", str(out),
            "--manifest-out", str(manifest), "--dataset-id", "events-v1",
            "--evidence-out", str(evidence),
            "--source-record-manifest", str(source),
        ]
    ) == 0
    assert len(load_decision_events(out)) == 1
    data = json.loads(manifest.read_text())
    assert data["record_count"] == 1
    assert data["source_record_fingerprint"] == "source-sha"
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


def test_manifest_rejects_mixed_policy_identities() -> None:
    first = events_from_trace(_trace())[0]
    data = first.to_dict()
    data["event_id"] = "other"
    data["policy_checkpoint_sha"] = "other-checkpoint"
    second = DecisionEventV1.from_dict(data)
    with pytest.raises(ValueError, match="mixes policy identities"):
        decision_event_manifest([first, second], dataset_id="mixed")
