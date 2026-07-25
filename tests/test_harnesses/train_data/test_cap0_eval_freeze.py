"""DSH1-09 (SLM-361): CAP0 eval-suite freeze + anti-collapse gates."""

from __future__ import annotations

import dataclasses

import pytest

from slm_training.dsl.grammar_capabilities import GrammarCapabilityAdapterV1
from slm_training.dsl.pack import get_pack
from slm_training.harnesses.train_data.cap0_eval_freeze import (
    CERT_CAP0,
    PREREGISTERED_N,
    SUBSUITES,
    Cap0GateV1,
    Cap0MetricsV1,
    RejectionCodeV1,
    SuiteIntegrityError,
    audit_suite,
    build_cap0_eval_suite,
    empty_trivial_baseline_metrics,
    evaluate_gate,
    identity_baseline_metrics,
    stop_rule_violations,
    suite_valid,
    unchanged_input_baseline_metrics,
    verify_suite_integrity,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1

ANSWERS = (
    'item = TextContent(":w.text")\nroot = Stack([item], "column")',
    'item = TextContent(":w.text")\n$s = item\nroot = Stack([item], "column")',
    'hero = TextContent(":smoke.hero.title")\nroot = Stack([hero], "column")',
    'root = Stack([hero, cta], "column")\n'
    'hero = TextContent(":smoke.hero.title")\n'
    'cta = Button(":cta.label", "primary")',
    "root = Separator()",
)

_POLICY = RootFamilySplitPolicyV1()
EVAL_FAMILIES = tuple(
    family
    for family in (f"fam-{index}" for index in range(60))
    if _POLICY.assign(family) == "test"
)[: len(ANSWERS)]
TRAIN_FAMILY = next(
    family
    for family in (f"fam-{index}" for index in range(60))
    if _POLICY.assign(family) == "train"
)


@pytest.fixture(scope="module")
def adapter() -> GrammarCapabilityAdapterV1:
    return GrammarCapabilityAdapterV1(get_pack("openui"))


@pytest.fixture(scope="module")
def suite(adapter: GrammarCapabilityAdapterV1):
    return build_cap0_eval_suite(
        list(zip(ANSWERS, EVAL_FAMILIES)), adapter=adapter, seed=7, dsl="openui"
    )


def _patched(suite, row, **changes):
    patched = dataclasses.replace(row, **changes)
    return dataclasses.replace(
        suite, rows=tuple(patched if r is row else r for r in suite.rows)
    )


# --------------------------------------------------------------------------- #
# Subsuites present, immutable, disjoint
# --------------------------------------------------------------------------- #


def test_all_six_subsuites_present(suite) -> None:
    assert set(SUBSUITES) == {
        "atomic_coverage",
        "expanded_canonicalization",
        "compositional_held_out",
        "prefix_completion",
        "marker_permutation",
        "anti_identity",
    }
    counts = {name: len(suite.rows_for(name)) for name in SUBSUITES}
    assert all(count > 0 for count in counts.values()), counts
    assert suite.manifest["rows_by_subsuite"] == counts
    assert suite.manifest["rows_total"] == len(suite.rows)


def test_suite_integrity_and_tamper_detection(suite) -> None:
    assert verify_suite_integrity(suite) is True
    row = suite.rows[0]
    tampered_row = dataclasses.replace(
        row, payload={**row.payload, "injected": "evil"}
    )
    bad = dataclasses.replace(
        suite, rows=tuple(tampered_row if r is row else r for r in suite.rows)
    )
    with pytest.raises(SuiteIntegrityError):
        verify_suite_integrity(bad)
    tampered_manifest = dataclasses.replace(
        suite, manifest={**suite.manifest, "rows_total": 999}
    )
    with pytest.raises(SuiteIntegrityError):
        verify_suite_integrity(tampered_manifest)


def test_train_eval_root_families_disjoint(suite, adapter) -> None:
    assert suite.root_families if hasattr(suite, "root_families") else True
    families = {row.root_family_id for row in suite.rows}
    assert families
    for family in families:
        assert _POLICY.assign(family) in suite.eval_splits
    # Train-family answers never enter the build.
    mixed = [(ANSWERS[0], TRAIN_FAMILY), *zip(ANSWERS, EVAL_FAMILIES)]
    rebuilt = build_cap0_eval_suite(mixed, adapter=adapter, seed=7, dsl="openui")
    assert TRAIN_FAMILY not in {row.root_family_id for row in rebuilt.rows}
    # A leaked train-family row is a zero-tolerance split-leakage artifact.
    leaked = _patched(suite, suite.rows[0], root_family_id=TRAIN_FAMILY)
    codes = {f.code for f in audit_suite(leaked, adapter=adapter, dsl="openui")}
    assert RejectionCodeV1.SPLIT_LEAKAGE in codes
    assert RejectionCodeV1.SPLIT_LEAKAGE in {
        f.code for f in stop_rule_violations(leaked, adapter=adapter)
    }


# --------------------------------------------------------------------------- #
# Zero-tolerance invariants: every kind produces a rejection artifact
# --------------------------------------------------------------------------- #


def test_audit_clean_suite_has_no_findings(suite, adapter) -> None:
    assert audit_suite(suite, adapter=adapter, dsl="openui") == []
    assert stop_rule_violations(suite, adapter=adapter) == []
    assert suite_valid(suite, adapter=adapter, dsl="openui") is True


def test_free_form_target_rejected(suite, adapter) -> None:
    row = suite.rows[0]
    bad = _patched(suite, row, target="just some free-form text")
    findings = audit_suite(bad, adapter=adapter, dsl="openui")
    assert any(f.code is RejectionCodeV1.FREE_FORM_TARGET for f in findings)


def test_undeclared_marker_rejected(suite, adapter) -> None:
    row = next(r for r in suite.rows if r.provenance["declared_markers"])
    bad = _patched(
        suite, row, provenance={**row.provenance, "declared_markers": []}
    )
    findings = audit_suite(bad, adapter=adapter, dsl="openui")
    assert any(f.code is RejectionCodeV1.UNDECLARED_MARKER for f in findings)


def test_parse_static_mismatch_rejected(suite, adapter) -> None:
    row = next(r for r in suite.rows if r.payload.get("canonical"))
    bad = _patched(
        suite, row, payload={**row.payload, "canonical": "TextConten("}
    )
    findings = audit_suite(bad, adapter=adapter, dsl="openui")
    assert any(f.code is RejectionCodeV1.PARSE_STATIC_MISMATCH for f in findings)


def test_canonical_mismatch_rejected(suite, adapter) -> None:
    row = next(
        r for r in suite.rows if r.target.get("canonical_fingerprint")
    )
    bad = _patched(
        suite,
        row,
        target={**row.target, "canonical_fingerprint": "0" * 64},
    )
    findings = audit_suite(bad, adapter=adapter, dsl="openui")
    assert any(f.code is RejectionCodeV1.CANONICAL_MISMATCH for f in findings)


def test_unresolved_scope_rejected(suite, adapter) -> None:
    row = next(r for r in suite.rows if r.target.get("kind") == "reject_near_miss")
    bad = _patched(
        suite,
        row,
        payload={**row.payload, "candidate": 'root = Stack([missing], "column")'},
    )
    findings = audit_suite(bad, adapter=adapter, dsl="openui")
    assert any(f.code is RejectionCodeV1.UNRESOLVED_SCOPE for f in findings)


def test_missing_provenance_rejected(suite, adapter) -> None:
    row = suite.rows[0]
    provenance = {k: v for k, v in row.provenance.items() if k != "source_sha256"}
    bad = _patched(suite, row, provenance=provenance)
    findings = audit_suite(bad, adapter=adapter, dsl="openui")
    assert any(f.code is RejectionCodeV1.MISSING_PROVENANCE for f in findings)


# --------------------------------------------------------------------------- #
# Stop rules
# --------------------------------------------------------------------------- #


def test_unproven_prefix_invalidates(suite, adapter) -> None:
    row = suite.rows_for("prefix_completion")[0]
    bad = _patched(
        suite, row, provenance={**row.provenance, "prefix_certificate": None}
    )
    findings = stop_rule_violations(bad, adapter=adapter)
    assert any(f.code is RejectionCodeV1.UNPROVEN_PREFIX for f in findings)
    # suite_valid is fail-closed: tampering trips integrity before any audit.
    with pytest.raises(SuiteIntegrityError):
        suite_valid(bad, adapter=adapter, dsl="openui")


def test_incomplete_shortest_reward_invalidates(suite, adapter) -> None:
    row = suite.rows_for("prefix_completion")[0]
    accepted = [dict(item) for item in row.target["accepted"]]
    # An "accepted" item that is itself an incomplete cut: a metric that
    # rewards the shortest incomplete output.
    full = accepted[0]["source"]
    accepted[0]["source"] = full[: max(1, len(full) // 2)]
    bad = _patched(suite, row, target={**row.target, "accepted": accepted})
    findings = stop_rule_violations(bad, adapter=adapter)
    assert any(
        f.code is RejectionCodeV1.INCOMPLETE_SHORTEST_REWARD for f in findings
    )


# --------------------------------------------------------------------------- #
# Gates: baselines fail, fixture n cannot certify
# --------------------------------------------------------------------------- #


def test_collapse_baselines_fail_gate(suite) -> None:
    gate = Cap0GateV1()
    for metrics in (
        identity_baseline_metrics(suite, dsl="openui"),
        unchanged_input_baseline_metrics(suite, dsl="openui"),
        empty_trivial_baseline_metrics(suite),
    ):
        report = evaluate_gate(metrics, gate=gate)
        assert not report.passed
        assert report.certificate is None
        assert report.status != "certified"


def test_fixture_n_reports_insufficient_n(suite) -> None:
    perfect = Cap0MetricsV1(
        canonical_equivalence=1.0,
        accepted_set_mass=1.0,
        production_coverage=1.0,
        structure=1.0,
        binding_aware=1.0,
        empty_trivial_rate=0.0,
        diversity=1.0,
        identity_baseline_score=0.0,
        n_by_subsuite={name: len(suite.rows_for(name)) for name in SUBSUITES},
    )
    report = evaluate_gate(perfect)
    assert all(report.checks.values())  # thresholds all met ...
    assert report.status == "insufficient_n"  # ... but fixture n cannot certify
    assert report.certificate is None
    assert not report.passed
    assert set(report.insufficient_n) == set(PREREGISTERED_N)
    for name, detail in report.insufficient_n.items():
        assert detail["actual"] < detail["required"]


def test_preregistered_n_certifies_with_wilson_bounds(suite) -> None:
    perfect = Cap0MetricsV1(
        canonical_equivalence=1.0,
        accepted_set_mass=1.0,
        production_coverage=1.0,
        structure=1.0,
        binding_aware=1.0,
        empty_trivial_rate=0.0,
        diversity=1.0,
        identity_baseline_score=0.0,
        n_by_subsuite=dict(PREREGISTERED_N),
    )
    report = evaluate_gate(perfect)
    assert report.status == "certified"
    assert report.certificate == CERT_CAP0
    assert report.passed
    assert report.insufficient_n == {}
    for bound in report.confidence_bounds.values():
        assert bound["n"] > 0
        assert 0.0 <= bound["low"] <= bound["high"] <= 1.0
        assert bound["confidence_level"] == Cap0GateV1().confidence_level


def test_gate_versions_thresholds_and_retention() -> None:
    gate = Cap0GateV1()
    assert gate.gate_version == "cap0_gate/v1"
    assert gate.retention_policy == "cap0_gate_retention/v1"
    assert gate.preregistered_n == PREREGISTERED_N


def test_determinism(adapter: GrammarCapabilityAdapterV1) -> None:
    pairs = list(zip(ANSWERS, EVAL_FAMILIES))
    first = build_cap0_eval_suite(pairs, adapter=adapter, seed=7, dsl="openui")
    second = build_cap0_eval_suite(pairs, adapter=adapter, seed=7, dsl="openui")
    assert first == second
    assert first.to_dict() == second.to_dict()
