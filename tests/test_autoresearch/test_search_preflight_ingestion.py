"""Actual compiler/preflight -> actual loss producer -> existing search ledger.

Public same-model diagnostic null; no independence, training-gain or ship claim.
"""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from scripts.autotrain_nll import compute_nll
from scripts.train_model import resolve_config
from slm_training.autoresearch import engine
from slm_training.autoresearch.evidence_ledger import pick_evidence_ranked_slug
from slm_training.autoresearch.schemas import ExperimentSpec
from slm_training.autoresearch.preflight.compiled_treatment import (
    begin_attempt,
    lock_driver_designs,
    persist_pair,
)
from slm_training.autoresearch.search.ingestion import (
    ingest_loss_reports,
    locked_search_context,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.schema import load_jsonl, write_jsonl
from slm_training.harness_core.checkpoint_bundle import stage_checkpoint_bundle
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.model_build.factory import build_model
from tests.test_autoresearch.test_compiled_treatment import (
    arms as fixture_arms,
    train_dir as fixture_train_dir,
)
from tests.test_autoresearch.test_experiment_campaign import _manifest
from tests.test_autoresearch.test_harness import campaign

arms = fixture_arms
train_dir = fixture_train_dir


@pytest.fixture
def production_inputs(arms, tmp_path, monkeypatch):
    control, candidate = arms
    eval_root = tmp_path / "data/eval/eval-fixture"
    suite = eval_root / "suites/smoke/records.jsonl"
    rows = load_jsonl(suite)
    # Explicit public fixture root labels; never inferred from arbitrary IDs.
    rows = [
        replace(row, meta={"root_family_id": f"fixture-{i}"})
        for i, row in enumerate(rows)
    ]
    write_jsonl(suite, rows)
    commands = engine.compile_commands(campaign(), control, output_root=tmp_path)
    train = next(cmd for cmd in commands if "scripts.train_model" in cmd)
    cfg = resolve_config(train[3:])
    model = build_model(cfg, load_jsonl(cfg.train_dir / "records.jsonl"))
    checkpoint = tmp_path / "ancestor.pt"
    model.save(checkpoint)
    bundle_root = tmp_path / "bundles"
    sha = stage_checkpoint_bundle(bundle_root, checkpoint, {"claim": "public_fixture"})
    checkpoint = bundle_root / "bundles" / sha / "last.pt"
    arms = tuple(
        arm.model_copy(
            update={
                "knobs": arm.knobs.model_copy(
                    update={
                        "initialize_from": str(checkpoint),
                        "eval_limit": 1,
                    }
                )
            }
        )
        for arm in arms
    )
    store = CampaignStore(campaign().campaign_id, tmp_path / "campaigns")
    store.initialize(campaign())
    by_id = {
        arm.experiment_id: store.write_artifact("experiments", arm) for arm in arms
    }
    template = _manifest(
        campaign_id=store.campaign_id,
        experiment_id=candidate.experiment_id,
        claim_class="wiring",
        arms=[
            {"arm_id": role, "role": role, "config_sha256": "a" * 64}
            for role in ("control", "candidate")
        ],
        mechanism_off_arm_ids=[],
        seeds=[7],
        endpoints=[
            {
                "endpoint_id": "meaning",
                "metric": "smoke.eval_nll",
                "direction": "decrease",
                "role": "primary",
                "minimum_effect": 0.01,
            }
        ],
    )
    return store, by_id, template, eval_root, model


def _lock(inputs):
    store, by_id, template, _, _ = inputs
    pair = lock_driver_designs(
        store,
        by_id,
        "control",
        ["candidate"],
        endpoint={"kind": "denoising_loss"},
        manifest_templates={"candidate": template},
        search_slugs={"candidate": "lr-fixture"},
    )["candidate"]
    return pair, store.load_experiment_campaign("candidate").manifest


def test_supported_mechanism_producer_to_search_consumer(
    production_inputs, monkeypatch
):
    store, _, template, eval_root, model = production_inputs
    with monkeypatch.context() as scoped:
        # Preflight may build shapes, but never score or forward the model.
        def forbidden(*args, **kwargs):
            pytest.fail("preflight executed model scoring")

        scoped.setattr(type(model), "forward", forbidden)
        pair, manifest = _lock(production_inputs)
    measurement = pair["design"]["bindings"]["endpoint"]["loss_measurement"]
    assert len(measurement["selection"]["selected_record_ids"]) == 3
    assert (
        len(
            pair["design"]["bindings"]["endpoint"]["selection"]["smoke"]["decode_cases"]
        )
        == 1
    )
    assert manifest.endpoints == template.endpoints
    assert manifest.promotion_gates == template.promotion_gates
    assert [arm.config_sha256 for arm in manifest.arms] == pair["arm_config_sha256s"]
    path = persist_pair(store, pair)
    identity, effects = locked_search_context(
        store,
        manifest,
        design_sha256=path.stem,
        artifact_digest=content_sha,
    )
    assert not effects
    attempts = [begin_attempt(store, pair, role) for role in ("control", "candidate")]
    report = compute_nll(eval_root, None, model, None)
    assert report["selection"] == measurement["selection"]
    assert report["estimator_id"] == measurement["estimator_id"]
    assert {row["seed"] for row in report["per_record"]} == {
        pair["randomness"]["loss_mask_seed"]
    }
    effect = ingest_loss_reports(
        store,
        manifest,
        artifact_digest=content_sha,
        lineage={
            "design_sha256": path.stem,
            "control_attempt_id": attempts[0]["attempt_id"],
            "candidate_attempt_id": attempts[1]["attempt_id"],
        },
        control=report,
        candidate=report,
    )
    assert effect.identity == identity and effect.complete and effect.benefit == 0
    assert effect.slug == "lr-fixture" and len(effect.paired_case_ids) == 3
    assert locked_search_context(
        store,
        manifest,
        design_sha256=path.stem,
        artifact_digest=content_sha,
    )[1] == [effect]
    kinds = [event["event_type"] for event in store.verify_event_chain()]
    assert (
        kinds.index("experiment_campaign_locked")
        < kinds.index("experiment_design_locked")
        < kinds.index("experiment_attempt_started")
    )
    assert kinds.count("search_effect_recorded") == 1
    for arm_id in pair["arm_ids"]:
        assert store.load_experiment_campaign(arm_id).manifest.experiment_id == arm_id
    # Next campaign derives a current identity from its own pre-execution lock.
    # History comes from all authorized loop stores, not a remembered latest row.
    next_campaign = campaign().model_copy(update={"campaign_id": "next-cycle"})
    future = CampaignStore(next_campaign.campaign_id, store.root.parent)
    future.initialize(next_campaign)
    by_id = {
        key: future.write_artifact(
            "experiments",
            ExperimentSpec.model_validate_json(value.read_text()).model_copy(
                update={"campaign_id": future.campaign_id}
            ),
        )
        for key, value in production_inputs[1].items()
    }
    future_inputs = (
        future,
        by_id,
        template.model_copy(update={"campaign_id": future.campaign_id}),
        eval_root,
        model,
    )
    future_pair, future_manifest = _lock(future_inputs)
    future_path = persist_pair(future, future_pair)
    future_identity, history = locked_search_context(
        future,
        future_manifest,
        design_sha256=future_path.stem,
        artifact_digest=content_sha,
        history_stores=[store, store, future],
    )
    assert future_identity == identity and history == [effect]
    assert (
        pick_evidence_ranked_slug(
            ["lr-fixture", "untried"],
            {},
            search_identity=future_identity,
            search_effects=history,
            rotation_order=["lr-fixture", "untried"],
        )
        == "untried"
    )


def test_old_campaign_lock_cannot_be_retrofitted(production_inputs):
    store, _, template, _, _ = production_inputs
    original = store.lock_experiment_campaign(template)
    with pytest.raises(FileExistsError, match="already locked with different content"):
        _lock(production_inputs)
    assert store.load_experiment_campaign(template.experiment_id) == original
    assert not any(
        e["event_type"] == "experiment_design_locked"
        for e in store.verify_event_chain()
    )


def test_attempt_requires_exact_persisted_design(production_inputs):
    pair, _ = _lock(production_inputs)
    store = production_inputs[0]
    mutated = deepcopy(pair)
    mutated["search_slug"] = "not-the-locked-arm"
    with pytest.raises(ValueError, match="requires_design_lock"):
        begin_attempt(store, mutated, "control")
    other = CampaignStore(store.campaign_id, store.root / "unlocked")
    with pytest.raises((ValueError, FileNotFoundError)):
        begin_attempt(other, pair, "control")


def test_changed_current_selection_does_not_read_old_credit(production_inputs):
    pair, _ = _lock(production_inputs)
    store, by_id, template, eval_root, _ = production_inputs
    suite = eval_root / "suites/smoke/records.jsonl"
    rows = load_jsonl(suite)
    write_jsonl(suite, list(reversed(rows)))
    with pytest.raises(FileExistsError, match="already locked with different content"):
        _lock(production_inputs)
    saved = json.loads(persist_pair(store, pair).read_text())
    assert saved["design"]["bindings"]["endpoint"]["loss_measurement"]["selection"][
        "selected_record_ids"
    ] == [row.id for row in rows]
