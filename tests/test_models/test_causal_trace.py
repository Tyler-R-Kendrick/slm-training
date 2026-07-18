"""Torch-free tests for exact causal decision-state capture (LDI1-01 / SLM-119).

A tiny deterministic decode over fixture logits/legal-sets exercises the whole
capture→emit→persist path without torch, a real tokenizer, or the grammar:

* constraint shadows fire only when the raw winner is illegal and the selection legal;
* shadow evidence is non-trainable and refused by semantic admission;
* forced (single-legal) steps are deductions, not decisions;
* bounded selection policies retain the right subset without changing the emission;
* integer prefix ids — never decoded text — are the state authority;
* traces round-trip through the shared TraceStore and fail closed on identity mismatch.
"""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.harnesses.preference.decision_events_v2 import admit_semantic_corpus
from slm_training.models.causal_trace import (
    CausalTraceIdentity,
    CausalTraceWriter,
    GeneratedOutcome,
    TracePolicy,
    TraceSelection,
    capture_raw_steps,
    emit_causal_decision,
    fold_policy_identity,
    legal_set_reference,
    load_causal_decision_states,
)

# vocab 6, eos=0. A four-step decode over exact prefixes:
#   ()        raw argmax 5 (illegal), legal {2,3}, select 3   -> constraint shadow
#   (3,)      legal {4} only, select 4                        -> forced deduction
#   (3,4)     raw argmax 2 (legal), legal {2,4}, select 2     -> ordinary decision
#   (3,4,2)   legal {0} (eos now valid), select 0             -> stop
_LOGITS = {
    (): [0.0, 0.1, 0.5, 0.9, 0.2, 3.0],
    (3,): [0.0, 0.1, 0.2, 0.3, 5.0, 0.4],
    (3, 4): [0.0, 0.1, 2.0, 0.2, 1.0, 0.3],
    (3, 4, 2): [4.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}
_LEGAL = {
    (): (2, 3),
    (3,): (4,),
    (3, 4): (2, 4),
    (3, 4, 2): (0,),
}


def _forward(prefix: tuple[int, ...]) -> list[float]:
    return _LOGITS[prefix]


def _allowed(prefix: tuple[int, ...]) -> tuple[int, ...]:
    return _LEGAL[prefix]


def _capture(policy: TracePolicy | None = None, role_of=None):
    return capture_raw_steps(
        forward_logits=_forward,
        allowed_ids=_allowed,
        eos_id=0,
        max_new_tokens=16,
        policy=policy,
        role_of=role_of,
    )


def _identity(adapter: str = "adapterA") -> CausalTraceIdentity:
    return CausalTraceIdentity(
        group_id="grp",
        context_text="root=Stack([",
        policy_checkpoint_sha=fold_policy_identity("base-ckpt", adapter),
        tokenizer_sha="tok-sha",
        decode_config_hash="dch",
        base_model_revision="rev",
        adapter_identity=adapter,
    )


def test_decode_emits_full_generation_and_stop_reason() -> None:
    result = _capture()
    assert result.generated_token_ids == (3, 4, 2, 0)
    assert result.stop_reason == "eos"
    assert result.constraint_shadow_count == 1
    assert len(result.observations) == 4


def test_constraint_shadow_only_when_raw_illegal_and_selection_legal() -> None:
    obs = _capture().observations
    shadow, forced_step, ordinary, eos_step = obs
    assert shadow.constraint_shadow is True
    assert shadow.raw_argmax_id == 5 and shadow.selected_token_id == 3
    assert shadow.forced is False
    # The ordinary step's raw winner is legal, so no shadow is recorded.
    assert ordinary.constraint_shadow is False
    assert ordinary.raw_argmax_id == 2 == ordinary.selected_token_id
    # Forced (single-legal) steps are deductions, never shadows.
    assert forced_step.forced is True and forced_step.constraint_shadow is False
    assert eos_step.selected_token_id == 0


def test_decision_index_counts_only_non_forced_steps() -> None:
    obs = _capture().observations
    # shadow(non-forced)=0, forced keeps 1, ordinary(non-forced)=1, eos forced keeps 2.
    assert [o.decision_index for o in obs] == [0, 1, 1, 2]
    assert [o.generated_ordinal for o in obs] == [0, 1, 2, 3]


def test_eos_selected_only_after_the_prefix_validates() -> None:
    # EOS (0) is absent from every legal set until the final validating prefix.
    for prefix, legal in _LEGAL.items():
        if prefix != (3, 4, 2):
            assert 0 not in legal
    assert _capture().generated_token_ids[-1] == 0


def test_integer_prefix_ids_are_the_state_authority() -> None:
    shadow = _capture().observations[0]
    state, _outcomes, _view = emit_causal_decision(shadow, _identity())
    # The exact integer prefix — not decoded text — identifies the causal state.
    assert state.architecture == "causal"
    assert state.context_ids == shadow.prefix_token_ids == ()
    assert state.legal_action_ids == (2, 3)
    assert state.decision_position == 0


def test_shadow_view_is_non_trainable_and_refused_by_admission() -> None:
    shadow = _capture().observations[0]
    state, outcomes, view = emit_causal_decision(shadow, _identity())
    assert state.decision_kind == "constraint_shadow"
    assert view is not None and view.trainable is False
    assert len(outcomes) == 1 and outcomes[0].legal is True
    # A legality-only shadow view cannot supervise a semantic objective.
    with pytest.raises(ValueError, match="non-trainable"):
        admit_semantic_corpus([(state, view)], materializer_id=view.materializer_id)


def test_ordinary_step_emits_replayable_state_without_invented_outcomes() -> None:
    ordinary = _capture().observations[2]
    state, outcomes, view = emit_causal_decision(ordinary, _identity())
    assert state.decision_kind == "causal_decision"
    assert outcomes == () and view is None


def test_adapter_identity_is_part_of_state_identity() -> None:
    shadow = _capture().observations[0]
    on, _o, _v = emit_causal_decision(shadow, _identity("adapterA"))
    off, _o2, _v2 = emit_causal_decision(shadow, _identity(""))
    assert on.state_id != off.state_id


@pytest.mark.parametrize(
    "policy,expected_ordinals",
    [
        (TracePolicy(selection=TraceSelection.EVERY), [0, 1, 2, 3]),
        (TracePolicy(selection=TraceSelection.CONSTRAINT_SHADOW_ONLY), [0]),
        (TracePolicy(selection=TraceSelection.MARGIN_THRESHOLD, margin_threshold=0.3), [0]),
        (TracePolicy(selection=TraceSelection.SAMPLED_POSITIONS, sampled_positions=(2,)), [2]),
    ],
)
def test_selection_policies_retain_the_right_subset(policy, expected_ordinals) -> None:
    result = _capture(policy=policy)
    assert [o.generated_ordinal for o in result.observations] == expected_ordinals
    # Selection never changes what is emitted.
    assert result.generated_token_ids == (3, 4, 2, 0)


def test_named_role_policy_uses_the_role_callback() -> None:
    role_of = {(): "root"}.get
    result = _capture(
        policy=TracePolicy(selection=TraceSelection.NAMED_ROLES, named_roles=("root",)),
        role_of=role_of,
    )
    assert [o.generated_ordinal for o in result.observations] == [0]
    assert result.observations[0].grammar_role == "root"


def test_legal_set_reference_is_order_independent_content_address() -> None:
    assert legal_set_reference((2, 3)) == legal_set_reference((3, 2, 3))
    assert legal_set_reference((2, 3)) != legal_set_reference((2, 4))


def test_trace_round_trips_through_shared_store(tmp_path) -> None:
    identity = _identity()
    store = TraceStore(tmp_path / "traces", run_id="ldi1-fixture")
    writer = CausalTraceWriter(store, identity)
    writer.record_all(_capture())

    assert len(store) == 4
    rows = list(store.iter_kind("causal_decision"))
    assert len(rows) == 4
    assert all(row["run_id"] == "ldi1-fixture" for row in rows)

    states = load_causal_decision_states(
        store,
        expected_checkpoint_sha=identity.policy_checkpoint_sha,
        expected_tokenizer_sha=identity.tokenizer_sha,
    )
    assert [s.architecture for s in states] == ["causal"] * 4
    assert states[0].context_ids == ()

    manifest = writer.manifest()
    assert manifest["state_count"] == 4
    assert manifest["constraint_shadow_count"] == 1
    assert manifest["duplicate_set_reuse"] == (
        manifest["state_count"] - manifest["unique_legal_sets"]
    )
    assert manifest["bytes_per_state"] > 0

    manifest_path = tmp_path / "causal_manifest.json"
    writer.write_manifest(manifest_path)
    assert json.loads(manifest_path.read_text())["state_count"] == 4


def test_load_fails_closed_on_identity_mismatch(tmp_path) -> None:
    identity = _identity()
    store = TraceStore(tmp_path / "traces", run_id="ldi1-fixture")
    CausalTraceWriter(store, identity).record_all(_capture())

    with pytest.raises(ValueError, match="policy checkpoint"):
        load_causal_decision_states(
            store,
            expected_checkpoint_sha="wrong",
            expected_tokenizer_sha=identity.tokenizer_sha,
        )
    with pytest.raises(ValueError, match="tokenizer"):
        load_causal_decision_states(
            store,
            expected_checkpoint_sha=identity.policy_checkpoint_sha,
            expected_tokenizer_sha="wrong",
        )


def test_generated_outcome_returns_pre_judge_candidate() -> None:
    outcome = GeneratedOutcome(
        action_id=3,
        continuation_seed=7,
        finish_reason="eos",
        raw_program="root=Stack([Text()])",
        canonical_program="root=Stack([Text()])",
    )
    candidate = outcome.to_candidate()
    # The plug-in never judges: no verified/metrics keys leak into the candidate.
    assert candidate["token_id"] == 3
    assert candidate["finalization_changed"] is False
    assert "verified" not in candidate and "metrics" not in candidate
