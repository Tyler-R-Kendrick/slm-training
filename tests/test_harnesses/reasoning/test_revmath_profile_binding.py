"""HARN-10 / SLM-548: revmath profile campaign binding + evidence lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.reasoning.revmath.profile_binding import (
    RevmathCampaignBindingError,
    RevmathProfilePlanV1,
    assert_tasks_match_lock,
    materialize_fixture_campaign,
    replay_profile_run_from_store,
    require_locked_campaign,
    run_revmath_profile,
)
from slm_training.harnesses.reasoning.revmath.schemas import parse_revmath_task
from slm_training.models.decode_stats import iter_decode_stats_records
from slm_training.runtime.telemetry.replay_bundle import ReplayBundleV1

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "src" / "slm_training" / "resources" / "revmath" / "fixtures"


def _task(name: str = "ablation_necessary.task.json"):
    return parse_revmath_task(json.loads((FIXTURES / name).read_text(encoding="utf-8")))


def test_unlocked_execution_rejected(tmp_path: Path) -> None:
    task = _task()
    store_root = tmp_path / "stores"
    # Materialize+lock first under one root, then require_locked on empty root.
    with pytest.raises(RevmathCampaignBindingError, match="unlocked"):
        require_locked_campaign(
            __import__(
                "slm_training.autoresearch.storage", fromlist=["CampaignStore"]
            ).CampaignStore(task.campaign.campaign_id, store_root / "empty"),
            task.campaign.experiment_id,
        )
    del store_root


def test_manifest_mismatch_rejected(tmp_path: Path) -> None:
    task = _task()
    plan = RevmathProfilePlanV1(
        campaign_id=task.campaign.campaign_id,
        experiment_id="exp.rm.hermetic.profile",
    )
    manifest = materialize_fixture_campaign(plan, [task])
    # Strict path: task still carries placeholder digest → reject.
    with pytest.raises(RevmathCampaignBindingError, match="mismatch"):
        run_revmath_profile(
            [task],
            store_root=tmp_path / "strict",
            manifest=manifest,
            rebind_tasks=False,
            hermetic=True,
        )


def test_materialize_lock_run_persist_and_replay(tmp_path: Path) -> None:
    task = _task()
    plan = RevmathProfilePlanV1(
        campaign_id=task.campaign.campaign_id,
        experiment_id="exp.rm.hermetic.profile",
    )
    store_root = tmp_path / "campaigns"
    profile = run_revmath_profile(
        [task],
        store_root=store_root,
        plan=plan,
        hermetic=True,
    )
    assert profile.profile_id == "reasoning/revmath"
    assert profile.manifest_sha256
    assert len(profile.runs) == 1
    assert profile.tasks[0].campaign.campaign_manifest_sha256 == profile.manifest_sha256
    assert profile.disposition["evidence_class"] == "fixture"
    assert profile.disposition["disposition"] == "retain_diagnostic"

    store_path = Path(profile.store_root)
    assert (store_path / "decode_stats.jsonl").is_file()
    records = list(iter_decode_stats_records(store_path / "decode_stats.jsonl"))
    assert len(records) == 1
    assert (store_path / "replay_bundles.jsonl").is_file()
    assert list((store_path / "artifacts" / "mechanism_dispositions").glob("*.json"))

    # Locked reload agrees with bound tasks.
    from slm_training.autoresearch.storage import CampaignStore

    lock = require_locked_campaign(
        CampaignStore(profile.campaign_id, store_root), profile.experiment_id
    )
    assert_tasks_match_lock(profile.tasks, lock)

    replayed = replay_profile_run_from_store(
        store_root,
        profile.campaign_id,
        experiment_id=profile.experiment_id,
    )
    assert replayed["manifest_sha256"] == profile.manifest_sha256
    bundle = ReplayBundleV1.from_dict(replayed["runs"][0]["replay_bundle"])
    assert bundle.bundle_hash == profile.runs[0].replay_bundle.bundle_hash


def test_formal_obligations_without_experiment_spec_rejected(tmp_path: Path) -> None:
    task = _task()
    plan = RevmathProfilePlanV1(
        campaign_id=task.campaign.campaign_id,
        experiment_id="exp.rm.hermetic.formal",
    )
    manifest = materialize_fixture_campaign(plan, [task])
    # Inject a fake formal obligation without matching preflight artifacts.
    from slm_training.autoresearch.schemas import FormalObligationV1

    poisoned = manifest.model_copy(
        update={
            "formal_obligations": (
                FormalObligationV1(
                    obligation_id="formal-" + "a" * 16,
                    template_id="metrics.structural_similarity_monotone",
                    policy="advisory",
                    preflight_sha256="0" * 64,
                ),
            ),
            "artifact_requirements": tuple(manifest.artifact_requirements)
            + (
                __import__(
                    "slm_training.autoresearch.experiment_campaign",
                    fromlist=["ArtifactRequirementV1"],
                ).ArtifactRequirementV1(kind="formal_preflight"),
            ),
        }
    )
    with pytest.raises(RevmathCampaignBindingError, match="formal_obligations"):
        run_revmath_profile(
            [task],
            store_root=tmp_path / "formal",
            manifest=poisoned,
            rebind_tasks=True,
            hermetic=True,
            experiment_spec=None,
        )


def test_cli_materialize(tmp_path: Path) -> None:
    from scripts import run_revmath_profile as cli

    out = tmp_path / "summary.json"
    store = tmp_path / "store"
    rc = cli.main(
        [
            "--task",
            str(FIXTURES / "ablation_necessary.task.json"),
            "--store-root",
            str(store),
            "--materialize-fixture",
            "--experiment-id",
            "exp.rm.hermetic.profile",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "revmath_profile_run/v1"
    assert payload["manifest_sha256"]
