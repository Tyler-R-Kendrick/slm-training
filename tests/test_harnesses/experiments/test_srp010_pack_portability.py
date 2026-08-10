"""Tests for SLM-474 (SRP-010) symbolic-pack portability fixture harness."""

from __future__ import annotations

from dataclasses import replace

import pytest

from slm_training.dsl.pack import PackSlotUnavailable, get_pack
from slm_training.dsl.symbolic_regression_pack import SYMBOLIC_REGRESSION_PACK_ID
from slm_training.harnesses.experiments.srp010_pack_portability import (
    CLAIM_CLASS,
    RELATED_CATALOGUE_ID,
    UnsupportedHook,
    assess_symbolic_pack_portability,
    audit_modules_for_openui_imports,
    build_tiny_sr_problem,
    build_vsp_envelope,
    require_pack_id,
    require_pack_slot,
    run_fixture_campaign,
)


def test_default_isolation_keeps_openui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_GRAMMAR_DSL", raising=False)
    import slm_training.models.grammar as grammar

    if hasattr(grammar, "_ACTIVE_DSL_CACHE"):
        monkeypatch.setattr(grammar, "_ACTIVE_DSL_CACHE", None, raising=False)
    report = assess_symbolic_pack_portability(run_id="test_default")
    assert report.default_isolation.env_unset
    assert report.default_isolation.openui_is_default
    assert report.default_isolation.unresolved_pack_id == "openui"
    assert report.default_isolation.auto_pack_id == "openui"
    assert report.default_isolation.default_alias_pack_id == "openui"
    assert get_pack().pack_id == "openui"
    assert get_pack(SYMBOLIC_REGRESSION_PACK_ID).pack_id == SYMBOLIC_REGRESSION_PACK_ID


def test_e2e_fixture_smoke() -> None:
    report = run_fixture_campaign(run_id="test_e2e")
    assert report.claim_class == CLAIM_CLASS
    assert report.related_catalogue_id == RELATED_CATALOGUE_ID
    assert report.pack_id == SYMBOLIC_REGRESSION_PACK_ID
    assert report.pack_available
    assert report.oracle_ok
    assert report.expression == "v0"
    assert report.canonical_expression == "v0"
    assert report.problem_fingerprint
    assert report.vsp_digest
    assert report.evidence.get("fit_status")
    assert report.sygus_capability.get("conformance") == "inspired_interoperability"
    assert report.version_stamp
    assert report.status in {"fixture", "fixture_failed"}
    assert report.forks
    assert any(f.fork_id == "shadow_dsl_packs_registry" for f in report.forks)
    assert any(f.fork_id == "openui_pinned_language_contract" for f in report.forks)
    assert any(f.fork_id == "sygus_inspired_not_conformance" for f in report.forks)
    enum_status = next(
        h.status for h in report.unsupported_hooks if h.hook_id == "symbolic_expr_enumerate"
    )
    if enum_status == "available":
        assert report.enumeration_exercise.get("exercised") is True
    corpus_status = next(
        h.status for h in report.unsupported_hooks if h.hook_id == "symbolic_expr_corpus"
    )
    if corpus_status == "available":
        assert report.corpus_exercise.get("exercised") is True


def test_import_audit_finds_no_openui() -> None:
    findings = audit_modules_for_openui_imports()
    assert findings
    for finding in findings:
        assert finding.offenders == (), finding


def test_mismatch_pack_id_fails_closed() -> None:
    problem = build_tiny_sr_problem()
    with pytest.raises(ValueError, match="unsupported pack_id"):
        replace(problem, pack_id="openui")
    with pytest.raises(ValueError, match="pack/contract mismatch"):
        require_pack_id(problem, "openui")


def test_missing_require_slot_fails_closed() -> None:
    pack = get_pack(SYMBOLIC_REGRESSION_PACK_ID)
    with pytest.raises(PackSlotUnavailable, match="corpus_generator"):
        require_pack_slot(pack, "corpus_generator")
    with pytest.raises(PackSlotUnavailable, match="completion_artifact"):
        require_pack_slot(pack, "completion_artifact")


def test_honest_unsupported_hooks_do_not_fail_fixture() -> None:
    report = assess_symbolic_pack_portability(run_id="test_hooks")
    hook_ids = {h.hook_id for h in report.unsupported_hooks}
    assert "symbolic_expr_corpus" in hook_ids
    assert "symbolic_expr_enumerate" in hook_ids
    assert "slot.corpus_generator" in hook_ids
    assert "slot.completion_artifact" in hook_ids
    # Pack slots stay intentionally empty even when optional modules import.
    slot_hooks = {
        h.hook_id: h.status
        for h in report.unsupported_hooks
        if h.hook_id.startswith("slot.")
    }
    assert slot_hooks["slot.corpus_generator"] == "unsupported"
    assert slot_hooks["slot.completion_artifact"] == "unsupported"
    unsupported = [h for h in report.unsupported_hooks if h.status == "unsupported"]
    assert unsupported
    assert all(isinstance(h, UnsupportedHook) for h in unsupported)
    if report.default_isolation.openui_is_default:
        assert report.status == "fixture"


def test_optional_modules_exercised_when_importable() -> None:
    report = assess_symbolic_pack_portability(run_id="test_exercise")
    by_id = {h.hook_id: h for h in report.unsupported_hooks}

    enum_hook = by_id["symbolic_expr_enumerate"]
    if enum_hook.status == "available":
        assert report.enumeration_exercise.get("exercised") is True
        assert report.enumeration_exercise.get("states_generated", 0) > 0
        assert report.enumeration_exercise.get("unique_canonical_states", 0) > 0
        assert report.enumeration_exercise.get("optimality_claim") in {
            "exhaustive_under_declared_bounds",
            "unknown_bound_truncated",
        }
        assert "exercised" in enum_hook.detail
    else:
        assert report.enumeration_exercise == {}

    corpus_hook = by_id["symbolic_expr_corpus"]
    if corpus_hook.status == "available":
        assert report.corpus_exercise.get("exercised") is True
        assert report.corpus_exercise.get("num_problems", 0) > 0
        assert "corpus_fingerprint" in report.corpus_exercise
        assert report.corpus_exercise.get("leakage_is_clean") is True
        assert "exercised" in corpus_hook.detail
    else:
        assert report.corpus_exercise == {}


def test_vsp_envelope_matches_problem() -> None:
    problem = build_tiny_sr_problem()
    envelope = build_vsp_envelope(problem)
    assert envelope.pack_identity.pack_id == SYMBOLIC_REGRESSION_PACK_ID
    assert envelope.runtime_symbols[0].surface == "x"
    assert (
        envelope.provenance["symbolic_regression_problem_fingerprint"]
        == problem.fingerprint
    )


def test_report_roundtrip() -> None:
    report = assess_symbolic_pack_portability(run_id="test_roundtrip")
    restored = type(report).from_dict(report.to_dict())
    assert restored.run_id == report.run_id
    assert restored.problem_fingerprint == report.problem_fingerprint
    assert len(restored.forks) == len(report.forks)
    assert len(restored.unsupported_hooks) == len(report.unsupported_hooks)
    assert restored.enumeration_exercise == report.enumeration_exercise
    assert restored.corpus_exercise == report.corpus_exercise
