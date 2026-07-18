"""The committed LDI1-01 fixture trace is load-bearing evidence (SLM-119).

Asserts the fixture-grade artifact shows exact prefix replay, raw/legal distribution
telemetry, at least one non-admittable constraint shadow, and a forced legal
counterfactual outcome — with no semantic label or model-quality claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.parser import validate
from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.harnesses.preference.decision_events_v2 import (
    admit_semantic_corpus,
    materialize_constraint_shadow,
)
from slm_training.models.causal_trace import fold_policy_identity, load_causal_decision_states

_FIXTURE = Path(__file__).parent / "fixtures" / "ldi1"


def test_fixture_states_replay_from_exact_integer_prefixes() -> None:
    store = TraceStore(_FIXTURE)  # read-only: construction does not write
    states = load_causal_decision_states(
        store,
        expected_checkpoint_sha=fold_policy_identity("base-ckpt-sha", "adapter-A"),
        expected_tokenizer_sha="tokenizer-sha-fixture",
    )
    assert [s.architecture for s in states] == ["causal"] * 4
    # Integer prefixes are the authority; the first decision is at the empty suffix.
    assert states[0].context_ids == ()
    assert states[0].decision_position == 0


def test_fixture_rows_carry_raw_and_legal_telemetry() -> None:
    rows = list(TraceStore(_FIXTURE).iter_kind("causal_decision"))
    assert len(rows) == 4
    for row in rows:
        raw = row["raw_observation"]
        assert raw["raw_topk"] and len(raw["raw_topk"][0]) == 3  # (id, logit, logprob)
        assert raw["legal_topk"]
        assert raw["legal_set_reference"]


def test_fixture_has_a_non_admittable_constraint_shadow() -> None:
    store = TraceStore(_FIXTURE)
    shadow_rows = [
        row for row in store.iter_kind("causal_decision") if row["constraint_shadow"]
    ]
    assert len(shadow_rows) >= 1
    states = load_causal_decision_states(
        store,
        expected_checkpoint_sha=fold_policy_identity("base-ckpt-sha", "adapter-A"),
        expected_tokenizer_sha="tokenizer-sha-fixture",
    )
    shadow_state = next(s for s in states if s.decision_kind == "constraint_shadow")
    view = materialize_constraint_shadow(shadow_state, ())
    assert view.trainable is False
    with pytest.raises(ValueError, match="non-trainable"):
        admit_semantic_corpus([(shadow_state, view)], materializer_id=view.materializer_id)


def test_fixture_manifest_and_forced_outcome_are_honest() -> None:
    manifest = json.loads((_FIXTURE / "causal_trace_manifest.json").read_text())
    assert manifest["state_count"] == 4
    assert manifest["constraint_shadow_count"] == 1
    assert manifest["bytes_per_state"] > 0

    outcome = json.loads((_FIXTURE / "forced_counterfactual_outcome.json").read_text())
    # A forced legal action replayed to a canonical valid OpenUI program; no judge label.
    assert outcome["action_id"] == 3
    assert validate(outcome["canonical_program"]).serialized == outcome["canonical_program"]
    assert "verified" not in outcome and "metrics" not in outcome
