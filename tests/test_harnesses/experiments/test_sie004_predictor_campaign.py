"""SIE-004 (SLM-482): fixture-scale EXP-SR-2 predictor campaign tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.sie004_predictor_campaign import (
    ARM_IDS,
    CATALOGUE_ID,
    FACTOR_FAMILY,
    PRIMARY_METRIC,
    Sie004CampaignV1,
    evaluate_predictions,
    legal_support_parity,
    load_authorized_factors,
    load_corpus_examples,
    plan_only_preview,
    retrieval_predict,
    run_campaign,
    split_by_source_program,
)
from slm_training.models.prompt_factor_predictor import PromptFactorExampleV1


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_examples"):
        Sie004CampaignV1(max_examples=5)
    with pytest.raises(ValueError, match="train_fraction"):
        Sie004CampaignV1(train_fraction=0.95)
    with pytest.raises(ValueError, match="fixture"):
        Sie004CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Sie004CampaignV1(max_wall_minutes=1000.0)
    with pytest.raises(ValueError, match="seeds"):
        Sie004CampaignV1(seeds=())


def test_manifest_binds_catalogue_identity_and_arms() -> None:
    campaign = Sie004CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].minimum_effect == catalogue.endpoints[0].minimum_effect
    assert manifest.seeds == (0, 1, 2)
    roles = {arm.arm_id: arm.role for arm in manifest.arms}
    assert roles["learned_predictor"] == "candidate"
    assert roles["gold_oracle"] == "control"
    assert roles["none"] == "control"


def test_plan_only_preview_exposes_catalogue_and_fixture_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["fixture_arms"] == list(ARM_IDS)
    assert preview["factor_family"] == FACTOR_FAMILY
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False


def test_load_authorized_factors_from_sie002_durable_doc() -> None:
    auth = load_authorized_factors()
    assert auth["loaded"] is True
    assert set(auth["authorized_factors"]) == {"roles", "bindings"}
    assert "style_layout" not in auth["authorized_program_families"]
    assert "binder_reference" in auth["authorized_program_families"]


def test_split_by_source_program_never_leaks() -> None:
    examples = [
        PromptFactorExampleV1(source_program_id="a", prompt="column stack", label="stack_column"),
        PromptFactorExampleV1(source_program_id="a", prompt="vertical column", label="stack_column"),
        PromptFactorExampleV1(source_program_id="b", prompt="row toolbar", label="stack_row"),
        PromptFactorExampleV1(source_program_id="c", prompt="horizontal row", label="stack_row"),
        PromptFactorExampleV1(source_program_id="d", prompt="column layout", label="stack_column"),
    ]
    train, held = split_by_source_program(examples, seed=0, train_fraction=0.6)
    assert {e.source_program_id for e in train} & {e.source_program_id for e in held} == set()


def test_retrieval_is_train_only() -> None:
    train = [
        PromptFactorExampleV1(
            source_program_id="t1", prompt="vertical column stack", label="stack_column"
        ),
        PromptFactorExampleV1(
            source_program_id="t2", prompt="horizontal row toolbar", label="stack_row"
        ),
    ]
    assert retrieval_predict(train, "please use a column stack") == "stack_column"
    assert retrieval_predict(train, "zzzz unknown tokens") is None


def test_evaluate_predictions_scores_f1_and_abstention() -> None:
    metrics = evaluate_predictions(
        ["stack_column", "stack_row", "stack_column"],
        ["stack_column", None, "stack_row"],
    )
    assert metrics["n_accepted"] == 2
    assert 0.0 <= metrics["requirement_predictor_f1"] <= 1.0
    assert metrics["abstention_rate"] == pytest.approx(1 / 3)


def test_legal_support_parity_exact() -> None:
    parity = legal_support_parity(authorized_program_families=("binder_reference",))
    assert parity["legal_support_parity_exact"] is True
    assert parity["hard_prune_predictor_off"] is False
    assert parity["hard_prune_predictor_on"] is False


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    examples = load_corpus_examples(max_examples=80)
    assert len(examples) >= 20

    campaign = Sie004CampaignV1(max_examples=80, seeds=(0, 1))
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["legal_support_parity"]["legal_support_parity_exact"] is True
    assert 0.0 <= result["requirement_predictor_f1"] <= 1.0
    assert set(result["aggregate_arms"]) == set(ARM_IDS)
    assert result["aggregate_arms"]["gold_oracle"]["requirement_predictor_f1_mean"] == 1.0
    assert result["aggregate_arms"]["none"]["requirement_predictor_f1_mean"] == 0.0
    # SIE-002 did not authorize style_layout → zero fact emission under that gate.
    for run in result["seed_runs"]:
        for arm, stats in run["arms"].items():
            assert stats["requirement_facts_emitted_under_sie002_auth"] == 0, arm

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
