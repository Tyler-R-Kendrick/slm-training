"""Tests for SLM-390 quarantined trace-to-eval candidate lifecycle."""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.experiments import slm390_trace_evals as m
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

NOW = "2026-07-25T00:00:00Z"


def _trace(**overrides):
    trace = {
        "trace_id": "trace-001",
        "operation": "train_model",
        "error_type": "ShapeMismatch",
        "message": "decode failed at step 12 for span ab12cd34ef56",
        "expected_output": "layout(<n> children)",
        "observed_output": "layout(<n> child)",
        "task_prompt": "produce the layout for the failed decode",
        "attributes": {"tool.name": "openui-compiler", "tool.version": "1.2.3"},
    }
    trace.update(overrides)
    return trace


def _eval_split_trace(**overrides):
    """A clean trace whose derived eval family lands outside the train split."""
    policy = RootFamilySplitPolicyV1()
    for nonce in range(200):
        trace = _trace(operation=f"train_model_{nonce}", **overrides)
        candidate = m.ingest_trace(trace, now=NOW)
        family = candidate.proposed["task"]["root_family_id"]
        if policy.assign(family) != "train":
            return trace
    raise AssertionError("no eval-split family found")


def _reviewed(trace=None, **kwargs):
    candidate = m.ingest_trace(trace or _eval_split_trace(), now=NOW)
    assert candidate.status == m.STATUS_QUARANTINED
    assert not candidate.rejection_reasons
    return m.mark_reviewed(
        candidate,
        {
            "reviewer_id": "domain-reviewer-7",
            "checklist": {item: True for item in m.REQUIRED_REVIEW_CHECKLIST},
        },
        now=NOW,
    )


def _frozen(tmp_path, trace=None):
    reviewed = _reviewed(trace)
    candidate, frozen = m.freeze_candidate(
        reviewed, training_root_family_ids=("train-family-a",), output_dir=tmp_path, now=NOW
    )
    return candidate, frozen


# --------------------------------------------------------------------------
# Ingest: redaction, clustering, proposals, stop rules
# --------------------------------------------------------------------------


def test_ingest_always_quarantined_no_automatic_transition():
    candidate = m.ingest_trace(_eval_split_trace(), now=NOW)
    assert candidate.status == m.STATUS_QUARANTINED
    assert candidate.review_evidence == {}
    assert candidate.rejection_reasons == ()


def test_redaction_canaries_and_patterns():
    canary = "user-jane-doe-canary"
    trace = _trace(
        message=(
            f"failed on /home/alice/repos/x by {canary} at build.example.com "
            "with hf_abcdefghijklmno and api_key=supersecretvalue"
        )
    )
    candidate = m.ingest_trace(trace, known_canaries=(canary,), now=NOW)
    blob = json.dumps(candidate.proposed) + json.dumps(candidate.redaction_report)
    assert canary not in blob
    assert "/home/alice" not in blob
    assert "build.example.com" not in blob
    assert "hf_abcdefghijklmno" not in blob
    assert "supersecretvalue" not in blob
    assert m.STOP_DEIDENTIFICATION_FAILED not in candidate.rejection_reasons
    assert candidate.redaction_report["canaries"] == 1
    assert candidate.redaction_report["secrets"] >= 2
    assert candidate.redaction_report["paths"] >= 1
    assert candidate.redaction_report["hosts"] >= 1


def test_stop_rule_unredactable_canary_stays_quarantined():
    canary = "canary-in-structural-field"
    trace = _trace(raw_core_dump={"pointer": canary})
    candidate = m.ingest_trace(trace, known_canaries=(canary,), now=NOW)
    assert candidate.status == m.STATUS_QUARANTINED
    assert m.STOP_DEIDENTIFICATION_FAILED in candidate.rejection_reasons


def test_stop_rule_secret_outside_redactable_fields():
    trace = _trace(run_config={"auth": "hf_abcdefghijklmno"})
    candidate = m.ingest_trace(trace, now=NOW)
    assert m.STOP_DEIDENTIFICATION_FAILED in candidate.rejection_reasons


def test_stop_rule_environment_not_reproducible():
    trace = _trace(attributes={"tool.name": "openui-compiler"})  # missing version
    candidate = m.ingest_trace(trace, now=NOW)
    assert candidate.status == m.STATUS_QUARANTINED
    assert m.STOP_ENVIRONMENT_NOT_REPRODUCIBLE in candidate.rejection_reasons


def test_stop_rule_not_independently_verifiable():
    trace = _trace(expected_output=None)
    trace.pop("expected_output")
    candidate = m.ingest_trace(trace, now=NOW)
    assert m.STOP_NOT_INDEPENDENTLY_VERIFIABLE in candidate.rejection_reasons


def test_stop_rule_candidate_cannot_be_reviewed():
    trace = _trace(attributes={})
    candidate = m.ingest_trace(trace, now=NOW)
    with pytest.raises(m.LifecycleError, match="stop-rule"):
        m.mark_reviewed(
            candidate,
            {"reviewer_id": "r", "checklist": {i: True for i in m.REQUIRED_REVIEW_CHECKLIST}},
        )


def test_clustering_groups_recurring_failures():
    a = m.ingest_trace(_trace(message="decode failed at step 12 for span ab12cd34ef56"), now=NOW)
    b = m.ingest_trace(_trace(message="decode failed at step 99 for span 0000ffff1111"), now=NOW)
    c = m.ingest_trace(_trace(message="totally different failure"), now=NOW)
    assert a.cluster_id == b.cluster_id
    assert a.cluster_id != c.cluster_id


def test_proposed_artifacts_present():
    candidate = m.ingest_trace(_eval_split_trace(), now=NOW)
    for key in ("capability", "task", "environment", "verifier"):
        assert key in candidate.proposed
    assert candidate.proposed["environment"]["pins"]["tool.version"] == "1.2.3"


# --------------------------------------------------------------------------
# Lifecycle transitions
# --------------------------------------------------------------------------


def test_review_requires_reviewer_and_full_checklist():
    candidate = m.ingest_trace(_eval_split_trace(), now=NOW)
    with pytest.raises(m.LifecycleError, match="reviewer_id"):
        m.mark_reviewed(candidate, {"checklist": {i: True for i in m.REQUIRED_REVIEW_CHECKLIST}})
    incomplete = {i: True for i in m.REQUIRED_REVIEW_CHECKLIST}
    incomplete["verifier_trajectory_reviewed"] = False
    with pytest.raises(m.LifecycleError, match="checklist"):
        m.mark_reviewed(candidate, {"reviewer_id": "r", "checklist": incomplete})


def test_illegal_transitions_rejected(tmp_path):
    candidate = m.ingest_trace(_eval_split_trace(), now=NOW)
    with pytest.raises(m.LifecycleError):
        m.freeze_candidate(candidate, training_root_family_ids=(), output_dir=tmp_path)
    with pytest.raises(m.LifecycleError):
        m.promote_to_training(candidate, _stub_frozen(tmp_path), existing_training_family_ids=())
    ledger = m.ConfirmationLedger()
    with pytest.raises(m.LifecycleError):
        m.record_confirmation_use(candidate, _stub_frozen(tmp_path), ledger)


def _stub_frozen(tmp_path):
    reviewed = _reviewed()
    _, frozen = m.freeze_candidate(
        reviewed, training_root_family_ids=(), output_dir=tmp_path, now=NOW
    )
    return frozen


def test_full_happy_path_lifecycle(tmp_path):
    reviewed = _reviewed()
    candidate, frozen = m.freeze_candidate(
        reviewed, training_root_family_ids=("train-family-a",), output_dir=tmp_path, now=NOW
    )
    assert candidate.status == m.STATUS_FROZEN
    assert m.verify_frozen_eval(tmp_path / m._FROZEN_FILE_NAME)

    ledger = m.ConfirmationLedger()
    confirmed = m.record_confirmation_use(candidate, frozen, ledger)
    assert confirmed.status == m.STATUS_CONFIRMATION_USED
    # One-touch only: a second confirmation touch on the same frozen eval raises.
    with pytest.raises(m.LifecycleError, match="already consumed"):
        m.record_confirmation_use(candidate, frozen, ledger)


def test_freeze_rejected_when_family_overlaps_training(tmp_path):
    reviewed = _reviewed()
    family = reviewed.proposed["task"]["root_family_id"]
    with pytest.raises(m.LifecycleError, match="duplicates a training root family"):
        m.freeze_candidate(reviewed, training_root_family_ids=(family,), output_dir=tmp_path)


def test_freeze_rejected_for_train_split_family(tmp_path):
    policy = RootFamilySplitPolicyV1()
    trace = None
    for nonce in range(500):
        t = _trace(operation=f"train_split_probe_{nonce}")
        c = m.ingest_trace(t, now=NOW)
        if policy.assign(c.proposed["task"]["root_family_id"]) == "train":
            trace = t
            break
    assert trace is not None
    reviewed = _reviewed(trace)
    with pytest.raises(m.LifecycleError, match="training split"):
        m.freeze_candidate(reviewed, training_root_family_ids=(), output_dir=tmp_path)


def test_freeze_rejected_on_exposed_answer(tmp_path):
    trace = _eval_split_trace(task_prompt="the answer is layout(<n> children)")
    reviewed = _reviewed(trace)
    with pytest.raises(m.LifecycleError, match="exposed answer"):
        m.freeze_candidate(reviewed, training_root_family_ids=(), output_dir=tmp_path)


def test_freeze_rejected_on_trivial_verifier(tmp_path):
    trace = _eval_split_trace(observed_output="layout(<n> children)")
    candidate = m.ingest_trace(trace, now=NOW)
    assert m.STOP_NOT_INDEPENDENTLY_VERIFIABLE not in candidate.rejection_reasons
    reviewed = m.mark_reviewed(
        candidate,
        {"reviewer_id": "r", "checklist": {i: True for i in m.REQUIRED_REVIEW_CHECKLIST}},
    )
    with pytest.raises(m.LifecycleError, match="negative validation"):
        m.freeze_candidate(reviewed, training_root_family_ids=(), output_dir=tmp_path)


def test_environment_replay_deterministic():
    candidate = m.ingest_trace(_eval_split_trace(), now=NOW)
    env = candidate.proposed["environment"]
    assert m.verify_environment_replay(env)
    assert m.verify_environment_replay(env)  # replay twice, identical
    assert not m.verify_environment_replay({"pins": {"tool.name": "x"}})


def test_frozen_eval_immutable_and_tamper_evident(tmp_path):
    _candidate, frozen = _frozen(tmp_path)
    path = tmp_path / m._FROZEN_FILE_NAME
    assert m.verify_frozen_eval(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"]["task"]["prompt"] = "tampered prompt"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert not m.verify_frozen_eval(path)

    # Re-freeze after tampering must fail closed.
    reviewed = _reviewed()
    with pytest.raises((FileExistsError, RuntimeError)):
        m.freeze_candidate(reviewed, training_root_family_ids=(), output_dir=tmp_path, now=NOW)


def test_promotion_creates_derivation_and_dataset_version_without_mutation(tmp_path):
    candidate, frozen = _frozen(tmp_path)
    before = (tmp_path / m._FROZEN_FILE_NAME).read_bytes()
    promoted, activity = m.promote_to_training(
        candidate, frozen, existing_training_family_ids=("train-family-a",), now=NOW
    )
    after = (tmp_path / m._FROZEN_FILE_NAME).read_bytes()

    assert promoted.status == m.STATUS_PROMOTED
    assert before == after
    assert activity.source_frozen_sha256 == frozen.manifest_sha256
    assert activity.dataset_version
    assert activity.training_root_family_id != frozen.root_family_id
    policy = RootFamilySplitPolicyV1()
    assert policy.assign(activity.training_root_family_id) == "train"
    assert policy.assign(frozen.root_family_id) != "train"
    # Frozen eval root family remains disjoint from the new training family.
    assert activity.source_root_family_id == frozen.root_family_id
    assert activity.training_root_family_id.startswith(frozen.root_family_id + ".trainderiv")


def test_determinism_same_trace_same_candidate():
    a = m.ingest_trace(_eval_split_trace(), now=NOW)
    b = m.ingest_trace(_eval_split_trace(), now=NOW)
    assert a.to_dict() == b.to_dict()
