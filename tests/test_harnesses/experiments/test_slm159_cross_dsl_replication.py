"""Tests for SLM-159 (SPV4-01) cross-DSL replication fixture harness."""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.backends.graphql_js import bridge_available
from slm_training.harnesses.experiments.slm159_cross_dsl_replication import (
    CommonConfig,
    CrossDslManifest,
    PackArm,
    assess_pack_readiness,
    build_graphql_seed,
    build_manifest,
    extract_graphql_plan,
    run_fixture_campaign,
    validate_manifest,
)

needs_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="graphql bridge unavailable (cd src/apps/graphql_bridge && npm ci)",
)

GOOD_QUERY = "query { posts(limit: 3) { title author { name } } }"
VARIABLE_QUERY = "query($id: ID!) { post(id: $id) { title body } }"


@pytest.fixture
def manifest() -> CrossDslManifest:
    return build_manifest()


def test_build_manifest_has_required_arms(manifest: CrossDslManifest) -> None:
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert "G1_graphql" in arm_ids
    assert "S1_second_pack" in arm_ids


def test_validate_manifest_passes_for_default(manifest: CrossDslManifest) -> None:
    assert validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_arm_ids(manifest: CrossDslManifest) -> None:
    arms = list(manifest.arms)
    duplicated = arms + [arms[0]]
    bad = CrossDslManifest(arms=tuple(duplicated))
    errors = validate_manifest(bad)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_promotable_blocked(manifest: CrossDslManifest) -> None:
    arms = list(manifest.arms)
    blocked = PackArm(
        **{
            **arms[0].__dict__,
            "blocked": True,
            "promotable": True,
            "blocker": "test blocker",
        }
    )
    new_arms = [blocked if arm.arm_id == blocked.arm_id else arm for arm in arms]
    bad = CrossDslManifest(arms=tuple(new_arms))
    errors = validate_manifest(bad)
    assert any(blocked.arm_id in e for e in errors)


def test_common_config_roundtrip() -> None:
    cfg = CommonConfig()
    data = cfg.to_dict()
    restored = CommonConfig.from_dict(data)
    assert restored.n_graphql_records == cfg.n_graphql_records
    assert restored.seeds == cfg.seeds


def test_manifest_to_dict_roundtrip(manifest: CrossDslManifest) -> None:
    data = manifest.to_dict()
    restored = CrossDslManifest.from_dict(data)
    assert restored.matrix_set == manifest.matrix_set
    assert {a.arm_id for a in restored.arms} == {a.arm_id for a in manifest.arms}


@needs_bridge
def test_extract_graphql_plan_maps_factors() -> None:
    plan = extract_graphql_plan(GOOD_QUERY)
    assert plan.identity.pack_id == "graphql"
    assert plan.identity.provenance == "gold"
    assert plan.archetype.id == "query"
    role_ids = {slot.role_id for slot in plan.role_slots}
    assert "posts" in role_ids
    assert "0/title" in role_ids or any("title" in rid for rid in role_ids)
    assert plan.topology.parent_relation_candidates is not None
    assert len(plan.topology.parent_relation_candidates) > 0


@needs_bridge
def test_extract_graphql_plan_captures_variables() -> None:
    plan = extract_graphql_plan(VARIABLE_QUERY)
    symbol_ids = {sym.symbol_id for sym in plan.symbols}
    assert "$id" in symbol_ids
    assert plan.bindings
    binding = next(b for b in plan.bindings)
    assert "$id" in binding.candidate_symbols


@needs_bridge
def test_build_graphql_seed_roundtrip() -> None:
    plan = extract_graphql_plan(GOOD_QUERY)
    seed_result = build_graphql_seed(plan)
    assert seed_result.ok
    assert seed_result.seed is not None
    from slm_training.dsl.pack import get_pack

    pack = get_pack("graphql")
    original = pack.require("canonicalize")(GOOD_QUERY)
    reproduced = pack.require("canonicalize")(seed_result.seed)
    assert reproduced == original


@needs_bridge
def test_build_graphql_seed_with_variables() -> None:
    plan = extract_graphql_plan(VARIABLE_QUERY)
    seed_result = build_graphql_seed(plan)
    assert seed_result.ok
    assert seed_result.seed is not None
    assert "$id" in seed_result.seed


def test_graphql_readiness_passes() -> None:
    report = assess_pack_readiness("graphql")
    assert report.pack_available
    if bridge_available():
        assert report.parser_available
        assert report.oracle_available
        assert report.generator_available
        assert report.canonicalizer_available
        assert report.placeholder_policy_available


def test_second_pack_readiness_blocked() -> None:
    for candidate in ("design-patterns", "nomenclature", "ontology"):
        report = assess_pack_readiness(candidate)
        assert not report.pack_available
        assert not report.readiness_pass
        assert "not registered" in report.blocker


def test_graphql_seed_fails_without_bridge(monkeypatch) -> None:
    """Seed builder should fail gracefully when the pack backend is unavailable."""
    import slm_training.harnesses.experiments.slm159_cross_dsl_replication as mod

    plan = extract_graphql_plan(GOOD_QUERY)
    from slm_training.dsl.pack import get_pack

    pack = get_pack("graphql")

    def _fake_available() -> bool:
        return False

    monkeypatch.setattr(pack.backend, "available", _fake_available)
    builder = mod.GraphQLPlanSeedBuilder(pack)
    result = builder.build(plan)
    assert not result.ok


@needs_bridge
def test_run_fixture_campaign_produces_rows(manifest: CrossDslManifest) -> None:
    cfg = CommonConfig(
        n_graphql_records=4, graphql_depth=1, seeds=(0,), max_records_per_root=8
    )
    report = run_fixture_campaign(
        manifest=CrossDslManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_slm159",
    )
    assert report.status == "fixture"
    assert report.rows
    assert report.version_stamp
    assert report.readiness_reports
    graphql_report = next(r for r in report.readiness_reports if r.pack_id == "graphql")
    assert graphql_report.pack_available
    blocked_rows = [r for r in report.rows if r.arm_id == "S1_second_pack"]
    assert blocked_rows
    assert not any(r.promotable for r in blocked_rows)


@needs_bridge
def test_run_fixture_campaign_graphql_metrics_positive(manifest: CrossDslManifest) -> None:
    cfg = CommonConfig(
        n_graphql_records=4, graphql_depth=1, seeds=(0,), max_records_per_root=8
    )
    report = run_fixture_campaign(
        manifest=CrossDslManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_slm159_metrics",
    )
    graphql_rows = [r for r in report.rows if r.arm_id == "G1_graphql"]
    assert graphql_rows
    for row in graphql_rows:
        assert row.n_records > 0
        assert row.extraction_coverage == pytest.approx(1.0, abs=1e-9)
        assert row.seed_validity > 0.0
        assert row.round_trip_equal > 0.0
