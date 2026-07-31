"""Tests for production-core vs ecosystem-library tier classification."""

from __future__ import annotations

import json

import pytest

from slm_training.dsl.ecosystem_tier import (
    CORE_COMPONENT_NAMES,
    CORE_FORMAL_MODULES,
    CORE_LOCAL_OPERATOR_IDS,
    ECOSYSTEM_FORMAL_MODULES,
    REGISTRY_PATH,
    EcosystemTier,
    EcosystemTierError,
    assert_core_independent_of_library,
    check_registry,
    classify_component,
    classify_formal_module,
    classify_generation_metric,
    classify_operator,
    formal_module_partition,
    inventory_snapshot,
    manifest,
    split_generation_metrics,
)


def test_core_local_operators_are_production_core() -> None:
    for op_id in CORE_LOCAL_OPERATOR_IDS:
        assert classify_operator(op_id) is EcosystemTier.PRODUCTION_CORE


def test_topology_and_history_operators_are_ecosystem() -> None:
    assert classify_operator("openui.wrap_node") is EcosystemTier.ECOSYSTEM_LIBRARY
    assert (
        classify_operator("conversation.undo") is EcosystemTier.ECOSYSTEM_LIBRARY
    )


def test_component_tier_split() -> None:
    assert classify_component("Stack") is EcosystemTier.PRODUCTION_CORE
    assert classify_component("Modal") is EcosystemTier.ECOSYSTEM_LIBRARY
    assert classify_component("Tabs") is EcosystemTier.ECOSYSTEM_LIBRARY


def test_formal_modules_partition_disjoint() -> None:
    part = formal_module_partition()
    core = set(part["production_core"])
    eco = set(part["ecosystem_library"])
    assert core.isdisjoint(eco)
    assert set(CORE_FORMAL_MODULES) == core
    assert set(ECOSYSTEM_FORMAL_MODULES) == eco
    for module in CORE_FORMAL_MODULES:
        assert classify_formal_module(module) is EcosystemTier.PRODUCTION_CORE
    for module in ECOSYSTEM_FORMAL_MODULES:
        assert classify_formal_module(module) is EcosystemTier.ECOSYSTEM_LIBRARY


def test_unknown_formal_module_fails_closed() -> None:
    with pytest.raises(EcosystemTierError):
        classify_formal_module("LeverProofLean.DoesNotExist")


def test_generation_metrics_split() -> None:
    split = split_generation_metrics(
        {
            "parse_rate": 0.9,
            "meaningful_program_rate": 0.8,
            "component_type_recall": 0.4,
            "structural_similarity": 0.5,
            "custom_debug": 1,
        }
    )
    assert split["production_core"]["parse_rate"] == 0.9
    assert "component_type_recall" in split["ecosystem_library"]
    assert "custom_debug" in split["unassigned"]
    assert classify_generation_metric("parse_rate") is EcosystemTier.PRODUCTION_CORE
    assert (
        classify_generation_metric("component_type_recall")
        is EcosystemTier.ECOSYSTEM_LIBRARY
    )


def test_inventory_snapshot_separates_counts() -> None:
    snap = inventory_snapshot(extra_roles=("primary_cta",))
    assert snap["schema"] == "ecosystem_tier/v1"
    assert snap["counts"]["operators_core"] == len(CORE_LOCAL_OPERATOR_IDS)
    assert snap["counts"]["operators_ecosystem"] >= 1
    assert "primary_cta" in snap["roles"]["ecosystem_library"]
    assert snap["counts"]["roles_ecosystem"] >= 1
    # Core bootstrap names are stable even when preferred inventory is larger.
    assert CORE_COMPONENT_NAMES
    assert snap["fingerprint"]


def test_core_independent_of_library_growth() -> None:
    result = assert_core_independent_of_library(
        core_formal_status="proved",
        ecosystem_library_size=40,
        previous_ecosystem_library_size=20,
    )
    assert result["core_formal_ok"] is True
    assert result["library_grew"] is True
    assert result["separation_holds"] is True


def test_committed_registry_matches_live_manifest() -> None:
    assert REGISTRY_PATH.is_file(), "run write_registry or measure script first"
    check_registry()
    live = manifest()
    committed = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert committed["fingerprint"] == live["fingerprint"]
