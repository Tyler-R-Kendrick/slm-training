"""Red-team fixtures: shortcuts, inventory leaks, and hash stability."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.staged import Capability
from slm_training.harnesses.synthesis_plan import (
    SynthesisPlanV1,
    load_synthesis_plan,
    tiny_corpus_generation_policy,
)
from slm_training.harnesses.train_data.feedback import action_receipt
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from slm_training.data.progspec.generate import canonical_root_identity


PLAN_PATH = Path("src/slm_training/resources/synthesis_plans/dsh0_cap0_fixture.json")


def test_legacy_v1_bytes_do_not_grow_a_corpus_generation_field() -> None:
    payload = SynthesisPlanV1.load(PLAN_PATH).to_dict()
    assert "corpus_generation" not in payload
    assert payload["schema_version"] == "synthesis_plan/v1"
    loaded = load_synthesis_plan(PLAN_PATH)
    assert type(loaded).__name__ == "SynthesisPlanV1"


def test_alpha_rename_is_not_a_new_root() -> None:
    left = 'root = Stack([a], "column")\na = TextContent(":slot_0")'
    right = 'root = Stack([b], "column")\nb = TextContent(":slot_0")'
    assert canonical_root_identity(left) == canonical_root_identity(right)


def test_data_only_receipt_rejects_prose_ack() -> None:
    with pytest.raises(ValueError, match="cannot close"):
        action_receipt(
            prior_feedback_hash="a" * 64,
            action_code="cue_only_shortcut",
            synthesis_plan_hash="b" * 64,
            dataset_manifest_hash="c" * 64,
            before={},
            after={},
            disposition="acknowledged",
        )


def test_root_family_cannot_split_across_holdout() -> None:
    policy = RootFamilySplitPolicyV1()
    split = policy.assign("family-x")
    with pytest.raises(ValueError, match="belongs to"):
        policy.require_inherited(
            root_family_id="family-x",
            split_group_id="family-x",
            split="test" if split != "test" else "train",
        )


def test_cap1_policy_forbids_grammar_surface() -> None:
    policy = tiny_corpus_generation_policy(capability=Capability.CAP1_SEMANTICS)
    assert policy.prompts.surface.value == "simplified_nl"
    assert "openui" in policy.prompts.hard_forbidden_cues
