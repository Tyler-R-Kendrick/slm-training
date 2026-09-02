"""The S5 decode-only authority-ladder manifest stays a valid, locked campaign.

The manifest is preregistration, so this test guards the two things a later
edit could silently break: that it still validates against the contract, and
that its recorded ``CampaignLockV1`` digest still matches its content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import (
    SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
    CampaignLockV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "experiments"
    / "decode_only_authority_ladder"
    / "campaign.v1.json"
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def manifest(payload: dict) -> ExperimentCampaignV1:
    return ExperimentCampaignV1.model_validate(payload["manifest"])


def test_manifest_validates_and_digest_is_locked(
    payload: dict, manifest: ExperimentCampaignV1
) -> None:
    assert campaign_manifest_sha256(manifest) == payload["manifest_sha256"]
    CampaignLockV1(manifest_sha256=payload["manifest_sha256"], manifest=manifest)


def test_multi_candidate_campaign_locks_the_only_allowed_selection_rule(
    manifest: ExperimentCampaignV1,
) -> None:
    candidates = [arm for arm in manifest.arms if arm.role == "candidate"]
    assert len(candidates) > 1
    assert manifest.selection_rule == SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST


def test_admit_probes_off_is_a_mechanism_off_negative_control_never_a_candidate(
    manifest: ExperimentCampaignV1,
) -> None:
    arm_id = "m0_admit_probes_off"
    assert arm_id in manifest.mechanism_off_arm_ids
    assert manifest.negative_controls == (arm_id,)
    (arm,) = [item for item in manifest.arms if item.arm_id == arm_id]
    assert arm.role == "control"


def test_the_two_decode_paths_stay_disjoint_ladders(
    payload: dict, manifest: ExperimentCampaignV1
) -> None:
    arm_ids = {arm.arm_id for arm in manifest.arms}
    ladders = payload["ladders"]
    declared = set()
    for ladder in ladders.values():
        members = {ladder["control"], *ladder["candidates"]}
        if "mechanism_off" in ladder:
            members.add(ladder["mechanism_off"])
        assert not (members & declared), "an arm may belong to only one ladder"
        declared |= members
    assert declared == arm_ids


def test_kill_criteria_are_executable_and_name_the_measured_counters(
    manifest: ExperimentCampaignV1,
) -> None:
    criteria = "\n".join(manifest.executable_kill_criteria)
    for token in (
        "kill:gate_regression",
        "kill:m1_joint_rejection",
        "kill:timeout",
        "block_joint_rejections",
    ):
        assert token in criteria


def test_arm_configs_hash_to_the_declared_arm_digests(
    payload: dict, manifest: ExperimentCampaignV1
) -> None:
    import hashlib

    from slm_training.lineage.records import canonical_json

    for arm in manifest.arms:
        config = payload["arm_configs"][arm.arm_id]
        digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
        assert digest == arm.config_sha256
