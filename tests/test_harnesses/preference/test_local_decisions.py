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
