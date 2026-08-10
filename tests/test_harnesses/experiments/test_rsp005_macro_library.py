"""RSP-005 (SLM-480): EXP-SR-9 macro library campaign tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp005_macro_library import (
    ARM_IDS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    Rsp005CampaignV1,
    build_macro_library_pack,
    load_library_corpus,
    load_prospective_corpus,
    plan_only_preview,
    run_campaign,
    verify_expansion_equivalence,
)


def _require_openui_bridge() -> None:
    from slm_training.dsl import lang_core
    from slm_training.dsl.canonicalize import canonicalize

    if not lang_core.bridge_available():
        pytest.skip("OpenUI bridge dependencies are unavailable")
    try:
        canonicalize(
            'root = Stack([x], "column")\nx = TextContent(":x")',
            validate=False,
        )
    except RuntimeError:
        pytest.skip("OpenUI bridge is not usable in this environment")


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_macros"):
        Rsp005CampaignV1(max_macros=0)
    with pytest.raises(ValueError, match="fixture"):
        Rsp005CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp005CampaignV1(max_wall_minutes=1000.0)


def test_manifest_binds_catalogue_identity() -> None:
    campaign = Rsp005CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].metric == catalogue.endpoints[0].metric
    assert manifest.rollback_gates[0].threshold == catalogue.rollback_gates[0].threshold


def test_plan_only_preview_exposes_catalogue_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False
    assert set(preview["fixture_arms"]) == set(ARM_IDS)


def test_library_and_prospective_corpora_are_disjoint() -> None:
    library = load_library_corpus()
    prospective = load_prospective_corpus()
    assert library
    assert prospective
    assert not set(library) & set(prospective)


def test_expansion_equivalence_holds_for_mdl_library() -> None:
    _require_openui_bridge()
    library = load_library_corpus()
    prospective = load_prospective_corpus()
    pack = build_macro_library_pack("learned_mdl", library, max_macros=8)
    assert pack.n_macros >= 1
    check = verify_expansion_equivalence(prospective, pack)
    assert check["semantics_preserved"] is True
    assert check[PRIMARY_METRIC] >= 0.0


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    _require_openui_bridge()
    campaign = Rsp005CampaignV1(max_macros=8)
    result = run_campaign(campaign, root=tmp_path)

    assert result["semantics_preserved"] is True
    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert set(result["arm_results"]) == set(ARM_IDS)
    assert result["control_rate"] == 0.0
    assert "recommendation" in result

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
