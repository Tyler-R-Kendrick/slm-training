"""Regression tests for the LDI1-01 causal decision-state trace fixture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def ldi1_fixture_module():
    """Load the fixture script as a module so its internals can be tested."""
    script = Path("scripts/run_ldi1_01_causal_trace_fixture.py")
    spec = importlib.util.spec_from_file_location("ldi1_01_fixture", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ldi1_01_fixture"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture_summary(ldi1_fixture_module, tmp_path):
    """Run the fixture against a temporary output directory and return summary."""
    out_root = tmp_path / "out"
    docs_json = tmp_path / "docs.json"
    ldi1_fixture_module.OUTPUT_ROOT = out_root
    ldi1_fixture_module.DOCS_JSON = docs_json
    return ldi1_fixture_module._run_fixture()


def test_fixture_produces_observations_and_replays(fixture_summary: dict) -> None:
    """The fixture emits at least one constrained decision, one shadow, and one replay."""
    assert len(fixture_summary["observations"]) == 3
    assert fixture_summary["constraint_shadow_count"] >= 1
    assert len(fixture_summary["forced_action_replays"]) >= 1
    assert fixture_summary["replay_errors"] == []
    assert fixture_summary["loaded_state_count"] == 3


def test_first_observation_is_constraint_shadow(fixture_summary: dict) -> None:
    """The first decision shows the raw argmax outside the legal set and a legal selection."""
    obs = fixture_summary["observations"][0]
    assert obs["raw_argmax_id"] == 4
    assert obs["selected_token_id"] == 2
    assert obs["legal_token_ids"] == [2, 3]
    assert obs["constraint_shadow"] is True
    assert obs["forced"] is False


def test_decision_state_carries_exact_prefix(fixture_summary: dict) -> None:
    """Each emitted DecisionStateV2 stores the exact prefix token ids as state authority."""
    for obs, event in zip(
        fixture_summary["observations"], fixture_summary["decision_event_v2_rows"]
    ):
        assert event["state"]["context_ids"] == obs["prefix_token_ids"]


def test_forced_action_replay_uses_stored_prefix(fixture_summary: dict) -> None:
    """A forced replay starts from the stored prefix and follows the grammar."""
    replay = fixture_summary["forced_action_replays"][0]
    assert replay["forced_action_id"] in {3}  # legal alternative at first decision
    raw = replay["replay"]["raw_program"]
    tokens = [int(t) for t in raw.split()]
    # First replay token must be the forced action.
    assert tokens[0] == replay["forced_action_id"]
    # Replay must end with EOS (token 0) because the synthetic grammar always can.
    assert tokens[-1] == 0


def test_manifest_matches_loaded_states(fixture_summary: dict) -> None:
    """The persisted manifest and the fail-closed loader agree on state count."""
    manifest = fixture_summary["manifest"]
    assert manifest["state_count"] == fixture_summary["loaded_state_count"]
    assert manifest["constraint_shadow_count"] == fixture_summary["constraint_shadow_count"]


def test_capture_raw_steps_replay_determinism(ldi1_fixture_module) -> None:
    """Re-running forward_logits on a stored prefix reproduces the same argmax."""
    mod = ldi1_fixture_module
    result = mod.capture_raw_steps(
        forward_logits=mod._forward_logits,
        allowed_ids=mod._allowed_ids,
        eos_id=0,
        max_new_tokens=5,
        initial_prefix=mod.PROMPT,
        role_of=mod._role_of,
    )
    for obs in result.observations:
        replay_logits = mod._forward_logits(obs.prefix_token_ids)
        replay_argmax = max(
            range(len(replay_logits)), key=lambda i: (replay_logits[i], -i)
        )
        assert replay_argmax == obs.raw_argmax_id
