"""LOT0-01 (SLM-248): LotusOpenUIFidelityContractV1 / LotusTransferAuthorizationV1.

Docs/spec-only issue: no model code, no training, no corpus build. These tests
only assert the committed JSON contract is well-formed, internally consistent,
and that its embedded hash is a genuine, reproducible digest of its own
content -- not that any mechanism has been implemented or run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "docs" / "design" / "lotus-openui-fidelity-contract-v1.json"
SOURCES_PATH = (
    REPO_ROOT
    / "src"
    / "slm_training"
    / "resources"
    / "autoresearch"
    / "lotus-openui-sources.json"
)

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_id",
    "linear_issue",
    "created_at",
    "version_stamp",
    "canonical_audit_reference",
    "primary_sources",
    "evidence_nuances",
    "mechanism_fidelity_table",
    "non_duplication_map",
    "required_differentiators",
    "claim_classes",
    "preregistration",
    "comparator_hashes",
    "contract_hash",
    "authorization",
}

REQUIRED_MECHANISM_ROW_KEYS = {
    "mechanism",
    "paper_code_evidence",
    "proposed_openui_owner",
    "fidelity",
    "required_control",
    "forbidden_conflation",
}

ALLOWED_FIDELITY_LABELS = {
    "Faithful",
    "Adapted",
    "Surrogate",
    "Adjacent",
    "Rejected",
    "Faithful/Adapted",
}

ALLOWED_VERDICTS = {
    "authorize_bounded_implementation",
    "needs_target_trace_contract",
    "duplicate_of_existing_mechanism",
    "infeasible_under_current_scale",
    "semantic_floor_blocked",
    "reject_transfer",
}


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _recompute_contract_hash(contract: dict) -> str:
    core = {
        k: v for k, v in contract.items() if k not in ("contract_hash", "authorization")
    }
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


@pytest.fixture(scope="module")
def contract() -> dict:
    assert CONTRACT_PATH.exists(), f"missing contract JSON: {CONTRACT_PATH}"
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_schema_round_trip_and_required_top_level_keys(contract: dict) -> None:
    # Round-trip: re-serialize and re-parse to confirm the file is valid, stable JSON.
    reparsed = json.loads(json.dumps(contract))
    assert reparsed == contract
    missing = REQUIRED_TOP_LEVEL_KEYS - contract.keys()
    assert not missing, f"contract missing required top-level keys: {sorted(missing)}"
    assert contract["schema_version"] == "lotus_openui_fidelity_contract/v1"
    assert contract["linear_issue"] == "SLM-248"


def test_mechanism_table_has_required_baseline_rows(contract: dict) -> None:
    rows = contract["mechanism_fidelity_table"]
    assert isinstance(rows, list) and len(rows) >= 8
    expected_mechanisms = {
        "pretrained causal LM backbone",
        "K reasoning blocks × c positions",
        "whole-backbone recurrent passes",
        "original embedding injection",
        "explicit-to-latent curriculum",
        "post-loop direct shared-head supervision",
        "final answer loss",
        "fixed R",
    }
    seen_mechanisms = {row.get("mechanism") for row in rows}
    missing = expected_mechanisms - seen_mechanisms
    assert not missing, f"mechanism table missing baseline rows: {sorted(missing)}"


def test_every_mechanism_row_has_source_and_owner(contract: dict) -> None:
    for row in contract["mechanism_fidelity_table"]:
        missing_keys = REQUIRED_MECHANISM_ROW_KEYS - row.keys()
        assert not missing_keys, f"row {row.get('mechanism')!r} missing keys: {missing_keys}"
        for key in REQUIRED_MECHANISM_ROW_KEYS:
            value = row[key]
            assert isinstance(value, str) and value.strip(), (
                f"row {row.get('mechanism')!r} field {key!r} must be a non-empty string, got {value!r}"
            )
        assert row["fidelity"] in ALLOWED_FIDELITY_LABELS, (
            f"row {row.get('mechanism')!r} has unrecognized fidelity label {row['fidelity']!r}"
        )


def test_no_duplicate_mechanism_rows_claim_same_owner_and_fidelity(contract: dict) -> None:
    rows = contract["mechanism_fidelity_table"]
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["proposed_openui_owner"], row["fidelity"])
        if key in seen:
            pytest.fail(
                "duplicate (proposed_openui_owner, fidelity) pair claimed by "
                f"both {seen[key]!r} and {row['mechanism']!r}: {key}"
            )
        seen[key] = row["mechanism"]


def test_non_duplication_map_covers_the_five_required_comparators(contract: dict) -> None:
    rows = contract["non_duplication_map"]
    assert isinstance(rows, list) and len(rows) == 5
    required_fields = {"comparator", "owner_code", "distinction", "duplication_verdict"}
    for row in rows:
        missing = required_fields - row.keys()
        assert not missing, f"non-duplication row missing fields: {missing}"
        for field in required_fields:
            assert isinstance(row[field], str) and row[field].strip()
    comparator_text = " ".join(row["comparator"].lower() for row in rows)
    for expected_fragment in (
        "rsc internal slots",
        "semanticplanv1",
        "causal adapters/ftpo",
        "explicit compiler traces",
        "flow/search",
    ):
        assert expected_fragment in comparator_text, (
            f"expected non-duplication comparator mentioning {expected_fragment!r}"
        )


def test_required_differentiators_and_claim_classes_present(contract: dict) -> None:
    differentiators = contract["required_differentiators"]
    assert isinstance(differentiators, list) and len(differentiators) >= 5
    assert all(isinstance(item, str) and item.strip() for item in differentiators)

    claim_classes = set(contract["claim_classes"])
    expected_claim_classes = {
        "wiring",
        "faithful_mechanism_fixture",
        "optimization_diagnostic",
        "semantic_quality",
        "causal_latent_use",
        "latency_frontier",
        "efficiency_frontier",
        "adoption",
    }
    assert claim_classes == expected_claim_classes


def test_preregistration_margins_and_primary_endpoint_are_non_empty(contract: dict) -> None:
    prereg = contract["preregistration"]
    assert isinstance(prereg["primary_semantic_endpoint"], str)
    assert prereg["primary_semantic_endpoint"].strip()

    frozen_suite = prereg["frozen_suite"]
    assert isinstance(frozen_suite, list) and len(frozen_suite) >= 1
    assert all(isinstance(s, str) and s.strip() for s in frozen_suite)

    margins = prereg["margins"]
    assert isinstance(margins, dict) and margins
    for key, value in margins.items():
        assert isinstance(value, str) and value.strip(), f"empty margin text for {key!r}"

    for required_key in (
        "minimum_seeds",
        "checkpoint_selection_rules",
        "no_hidden_target_contract",
        "early_stop_conditions",
        "project_close_conditions",
    ):
        assert required_key in prereg, f"preregistration missing {required_key!r}"

    seeds = prereg["minimum_seeds"]
    assert seeds["screening"] == 3
    assert seeds["adoption"] == 5

    assert isinstance(prereg["early_stop_conditions"], list) and prereg["early_stop_conditions"]
    assert (
        isinstance(prereg["project_close_conditions"], list)
        and prereg["project_close_conditions"]
    )


def test_authorization_verdict_is_allowed_and_not_forced_to_authorize(contract: dict) -> None:
    authorization = contract["authorization"]
    assert authorization["verdict"] in ALLOWED_VERDICTS
    assert isinstance(authorization["verdict_rationale"], str)
    assert authorization["verdict_rationale"].strip()
    # This issue's honest analysis of the actual repo code landed on a
    # non-"authorize" verdict; assert that literally so a future edit that
    # silently flips it to look more "productive" fails loudly.
    assert authorization["verdict"] == "needs_target_trace_contract"
    for key in ("allowed_lot1_work", "blocked_work_and_claims", "comparator_hashes"):
        assert key in authorization
        assert isinstance(authorization[key], list) and authorization[key]


def test_contract_hash_matches_fresh_recompute(contract: dict) -> None:
    recomputed = _recompute_contract_hash(contract)
    assert len(contract["contract_hash"]) == 64
    assert contract["contract_hash"] == recomputed
    assert contract["authorization"]["contract_hash"] == recomputed


def test_comparator_hashes_are_well_formed_path_plus_sha256(contract: dict) -> None:
    rows = contract["comparator_hashes"]
    assert isinstance(rows, list) and len(rows) >= 5
    for row in rows:
        assert set(row.keys()) == {"path", "sha256"}
        assert isinstance(row["path"], str) and row["path"].strip()
        digest = row["sha256"]
        assert isinstance(digest, str) and len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex


def test_primary_sources_have_well_formed_urls_or_uris(contract: dict) -> None:
    sources = contract["primary_sources"]
    assert isinstance(sources, list) and len(sources) >= 1
    for source in sources:
        for field in ("source_id", "kind", "title", "uri", "access"):
            assert isinstance(source[field], str) and source[field].strip()
        parsed = urlparse(source["uri"])
        assert parsed.scheme, f"source {source['source_id']!r} has no URI scheme: {source['uri']!r}"


def test_evidence_nuances_are_tagged_and_non_empty(contract: dict) -> None:
    nuances = contract["evidence_nuances"]
    assert isinstance(nuances, list) and len(nuances) >= 8
    allowed_classes = {"primary_fetched", "secondary_preregistered"}
    for nuance in nuances:
        assert isinstance(nuance["fact"], str) and nuance["fact"].strip()
        assert nuance["evidence_class"] in allowed_classes


def test_source_manifest_entry_exists_and_matches_convention() -> None:
    assert SOURCES_PATH.exists(), f"missing LOTUS source manifest: {SOURCES_PATH}"
    manifest = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert isinstance(manifest["source_scope"], str) and manifest["source_scope"].strip()
    sources = manifest["sources"]
    assert isinstance(sources, list) and len(sources) >= 1
    seen_ids: set[str] = set()
    for source in sources:
        assert source["source_id"] not in seen_ids, f"duplicate source_id {source['source_id']!r}"
        seen_ids.add(source["source_id"])
        assert isinstance(source["title"], str) and source["title"].strip()
        assert isinstance(source["uri"], str) and source["uri"].strip()
        assert "metadata" in source and isinstance(source["metadata"], dict)


def test_no_authorize_claim_without_semantic_quality_language(contract: dict) -> None:
    """Acceptance criterion: no model or quality claim is made by this issue."""
    blocked_text = " ".join(contract["authorization"]["blocked_work_and_claims"]).lower()
    assert "no model" in blocked_text or "no training" in blocked_text
    assert "gpu" in blocked_text
