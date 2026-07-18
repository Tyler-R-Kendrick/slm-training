"""LDI3-04 immutable remine->intervene->regenerate campaign (SLM-132).

Torch-free: the fixture backend drives the whole lifecycle without a model. Covers the
fail-closed config, the content-addressed stage DAG (resume / immutability / duplicate
safety / invalidation), admission-gated training, failure-signature migration
categories, deterministic stop rules, parent-restore honesty, and the one-iteration
wiring-only smoke.
"""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.preference.remine_campaign import (
    CAMPAIGN_TAG,
    FailureSignature,
    RemineCampaignConfig,
    RemineConfigError,
    describe_campaign,
    evaluate_stop_rules,
    migrate_signatures,
    run_campaign,
)
from slm_training.harnesses.preference.remine_campaign import CampaignState


def _config(**overrides) -> RemineCampaignConfig:
    base = dict(
        campaign_id="ldi-remine-test",
        created_at="2026-07-18T00:00:00Z",
        base_checkpoint_sha="ckpt0",
        tokenizer_sha="tok0",
        prompt_group_ids=["g1", "g2"],
        adapter_spec={"rank": 4},
        max_iterations=2,
        min_new_evidence=1,
    )
    base.update(overrides)
    return RemineCampaignConfig.from_mapping(base)


# --------------------------------------------------------------------------- #
# Config schema / fingerprint / fail-closed
# --------------------------------------------------------------------------- #
def test_unknown_config_field_fails_closed():
    with pytest.raises(RemineConfigError):
        RemineCampaignConfig.from_mapping({**_config().to_dict(), "bogus": 1})


def test_config_validation_and_fingerprint_deterministic():
    with pytest.raises(RemineConfigError):
        _config(actuator_backend="sae")  # out-of-contract actuator
    with pytest.raises(RemineConfigError):
        _config(max_iterations=9)  # a fourth+ iteration needs explicit justification
    with pytest.raises(RemineConfigError):
        _config(prompt_group_ids=[])  # frozen prompt set must be non-empty
    assert _config().fingerprint() == _config().fingerprint()
    assert _config(decode_config_hash="a").fingerprint() != _config(decode_config_hash="b").fingerprint()


# --------------------------------------------------------------------------- #
# Fixture one-iteration smoke + immutable artifacts
# --------------------------------------------------------------------------- #
def test_fixture_smoke_publishes_artifacts_and_manifest(tmp_path):
    result = run_campaign(_config(), root=tmp_path)
    assert result.status == "wiring_only"
    assert result.iterations[0].iteration == 0
    assert result.iterations[0].authorization == "train_authorized"
    assert result.stages_run > 0 and result.stages_reused == 0
    root = tmp_path / "ldi-remine-test"
    assert (root / "campaign.json").exists()
    assert (root / "events.jsonl").exists()
    assert any((root / "artifacts").rglob("*.json"))
    # the campaign manifest artifact is present
    manifests = list((root / "artifacts" / "remine_campaign_manifest").glob("*.json"))
    assert len(manifests) == 1


def test_resume_reuses_every_stage_and_matches_manifest(tmp_path):
    r1 = run_campaign(_config(), root=tmp_path)
    r2 = run_campaign(_config(), root=tmp_path)  # interrupted-then-resumed == uninterrupted
    assert r2.stages_run == 0
    assert r2.stages_reused == r1.stages_run
    assert r1.manifest() == r2.manifest()
    # duplicate-safe: resume did not append a second manifest artifact
    manifests = list((tmp_path / "ldi-remine-test" / "artifacts" / "remine_campaign_manifest").glob("*.json"))
    assert len(manifests) == 1


def test_changed_upstream_config_invalidates_downstream_reuse(tmp_path):
    run_campaign(_config(), root=tmp_path)
    # decode_config_hash is not part of the CampaignSpec, so the immutable root is
    # reusable, but it changes the config fingerprint -> every stage marker misses.
    changed = run_campaign(_config(decode_config_hash="v2"), root=tmp_path)
    assert changed.stages_reused == 0
    assert changed.stages_run > 0


# --------------------------------------------------------------------------- #
# Admission-gated training / parent restore
# --------------------------------------------------------------------------- #
class _NoFailureBackend:
    """Parent generations are clean -> no admissible repair target."""

    def generate(self, config, *, corpus, adapter_id):
        from slm_training.harnesses.preference.remine_campaign import GeneratedProgram

        return [
            GeneratedProgram(f"{g}:ok", g, corpus, f"trace:{g}", (), None)
            for g in config.prompt_group_ids
        ]

    def train(self, config, *, evidence, parent_adapter_id):  # pragma: no cover - never called
        raise AssertionError("training must be skipped when admission finds no failure")


def test_training_skipped_when_admission_finds_no_safe_direction(tmp_path):
    result = run_campaign(_config(), backend=_NoFailureBackend(), root=tmp_path)
    assert result.iterations[0].authorization == "no_safe_direction"
    assert len(result.iterations) == 1  # no training iteration
    assert result.iterations[0].stop.reason == "no_safe_direction"


# --------------------------------------------------------------------------- #
# Failure-signature migration categories
# --------------------------------------------------------------------------- #
def test_migration_classifies_every_category():
    parent = [
        FailureSignature("g1", "motif", "repaired_one", supported=True),
        FailureSignature("g1", "motif", "persist_one", supported=True),
        FailureSignature("g1", "gate", "g_parent", supported=True),
        FailureSignature("g2", "motif", "vanish_unsupported", supported=False),
    ]
    child = [
        FailureSignature("g1", "motif", "persist_one", supported=True),  # persisted
        FailureSignature("g1", "gate", "g_parent", supported=True),  # persisted
        FailureSignature("g1", "motif", "brand_new", supported=True),  # newly_exposed
        FailureSignature("g3", "gate", "g_regress", supported=True),  # regressed (no parent gate on g3)
    ]
    m = migrate_signatures(parent, child)
    c = m.counts()
    assert c == {"repaired": 1, "persisted": 2, "regressed": 1, "newly_exposed": 1, "unresolved": 1}
    # An unsupported disappearance is unresolved, never repair.
    assert ("g2", "motif", "vanish_unsupported") in m.unresolved
    assert ("g1", "motif", "repaired_one") in m.repaired


def test_aggregate_disappearance_without_support_is_not_repair():
    parent = [FailureSignature("g1", "motif", "x", supported=False)]
    m = migrate_signatures(parent, [])
    assert m.counts()["repaired"] == 0
    assert m.counts()["unresolved"] == 1


# --------------------------------------------------------------------------- #
# Deterministic stop rules
# --------------------------------------------------------------------------- #
def _state(**overrides) -> CampaignState:
    base = dict(
        iteration=1,
        max_iterations=2,
        authorization="train_authorized",
        protected_gate_regressed=False,
        end_to_end_improved=True,
        locality_within_budget=True,
        new_qualified_evidence=5,
        min_new_evidence=1,
        migration=None,
        replication_ok=True,
        budget_exhausted=False,
    )
    base.update(overrides)
    return CampaignState(**base)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"authorization": "no_safe_direction"}, "no_safe_direction"),
        ({"budget_exhausted": True}, "budget_exhausted"),
        ({"protected_gate_regressed": True}, "protected_gate_regressed"),
        ({"locality_within_budget": False}, "locality_or_latency_over_budget"),
        ({"new_qualified_evidence": 0}, "new_evidence_below_threshold"),
        ({"end_to_end_improved": False}, "no_meaningful_end_to_end_improvement"),
        ({"replication_ok": False}, "positive_result_failed_replication"),
        ({"iteration": 2, "max_iterations": 2}, "max_iterations_reached"),
    ],
)
def test_stop_rules_are_deterministic(overrides, reason):
    assert evaluate_stop_rules(_state(**overrides)).stop is True
    assert evaluate_stop_rules(_state(**overrides)).reason == reason


def test_stop_rules_continue_when_all_clear():
    assert evaluate_stop_rules(_state()).stop is False


def test_max_two_trained_iterations_by_default(tmp_path):
    result = run_campaign(_config(max_iterations=2), root=tmp_path)
    trained = [it for it in result.iterations if it.adapter is not None]
    assert len(trained) <= 2


# --------------------------------------------------------------------------- #
# No automatic adapter merge / stacking; explicit lineage
# --------------------------------------------------------------------------- #
def test_adapters_keep_explicit_parent_lineage_and_are_not_merged(tmp_path):
    result = run_campaign(_config(), root=tmp_path)
    trained = [it for it in result.iterations if it.adapter is not None]
    assert trained[0].adapter.parent_adapter_id is None
    if len(trained) > 1:
        # second iteration's adapter names the first as parent (lineage), fresh id.
        assert trained[1].adapter.parent_adapter_id == trained[0].adapter.adapter_id
        assert trained[1].adapter.adapter_id != trained[0].adapter.adapter_id


def test_describe_runs_nothing_and_is_json_serializable():
    d = describe_campaign(_config())
    assert d["tag"] == CAMPAIGN_TAG
    assert len(d["stages_iter0"]) == 8 and len(d["stages_itern"]) == 8
    assert len(d["arms"]) == 4
    json.dumps(d)  # fully serializable, no model/data loaded
