"""SLM-391 (DSH4-06) portfolio disposition tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments import slm391_portfolio_disposition as d
from scripts import run_slm391_portfolio_disposition as pub

REPO_ROOT = Path(__file__).resolve().parents[3]
NOW = "2026-07-25T00:00:00Z"

# Small fixture budget so the matched CAP1 experiment stays inside the run cap.
CAP1_KWARGS = {"cap1_steps": 1, "cap1_records_per_arm": 4, "cap1_seeds": (0,)}


@pytest.fixture(scope="module")
def disposition() -> d.StagedDslHarnessDispositionV1:
    return d.build_disposition(repo_root=REPO_ROOT, now=NOW, **CAP1_KWARGS)


def _claim(disposition: d.StagedDslHarnessDispositionV1, claim_id: str) -> d.ClaimDispositionV1:
    return next(c for c in disposition.claims if c.claim_id == claim_id)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_claim_registry_completeness() -> None:
    registry = d.claim_registry()
    assert tuple(spec.claim_id for spec in registry) == d.CLAIM_IDS
    assert set(d.CLAIM_IDS) == {
        "cap0_grammar",
        "cap1_semantics",
        "cap2_operators",
        "distillation",
        "efficiency",
        "trace_evals",
    }
    for spec in registry:
        assert spec.scope in {
            d.SCOPE_HARNESS_CONTRACT,
            d.SCOPE_MODEL_CAPABILITY,
            d.SCOPE_SHIP,
        }
        assert spec.stop_rule


def test_verdict_vocabulary_fixed() -> None:
    assert d.VERDICTS == ("supported", "rejected", "invalid", "unknown")


def test_outcome_vocabulary_fixed() -> None:
    for outcome in (
        "positive",
        "negative",
        "invalid",
        "timeout",
        "no_effect",
        "contaminated",
        "prediction_identical",
    ):
        assert outcome in d.OUTCOME_CLASSES


# --------------------------------------------------------------------------
# Artifact binding
# --------------------------------------------------------------------------


def test_bind_artifacts_present_and_absent() -> None:
    registry = {spec.claim_id: spec for spec in d.claim_registry()}
    cap1 = d.bind_artifacts(registry["cap1_semantics"], REPO_ROOT)
    present = {b.path: b for b in cap1 if b.present}
    assert all(b.sha256 for b in present.values())
    assert any(
        b.path == "docs/design/iter-slm368-cap1-experiment-20260725.json"
        and b.identities.get("schema")
        for b in present.values()
    )
    cap0 = d.bind_artifacts(registry["cap0_grammar"], REPO_ROOT)
    assert any(not b.present for b in cap0)
    missing = [b for b in cap0 if b.required and not b.present]
    assert missing  # SLM-361/362 absent on this lineage


# --------------------------------------------------------------------------
# Verdicts + guards
# --------------------------------------------------------------------------


def test_expected_verdicts(disposition: d.StagedDslHarnessDispositionV1) -> None:
    assert _claim(disposition, "cap1_semantics").verdict == d.VERDICT_REJECTED
    assert _claim(disposition, "cap2_operators").verdict == d.VERDICT_SUPPORTED
    assert _claim(disposition, "trace_evals").verdict == d.VERDICT_SUPPORTED
    assert _claim(disposition, "cap0_grammar").verdict == d.VERDICT_UNKNOWN
    assert _claim(disposition, "distillation").verdict == d.VERDICT_UNKNOWN
    assert _claim(disposition, "efficiency").verdict == d.VERDICT_UNKNOWN


def test_every_claim_names_identities_or_unknown(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    for claim in disposition.claims:
        assert claim.identities.get("code"), claim.claim_id
        assert claim.identities.get("hardware"), claim.claim_id
        if claim.verdict == d.VERDICT_SUPPORTED:
            assert claim.identities.get("suite"), claim.claim_id
            assert claim.identities.get("config"), claim.claim_id
        if claim.verdict != d.VERDICT_UNKNOWN:
            continue
        # Unknown claims must reference the missing/incomparable artifacts.
        assert any(not b.present for b in claim.artifacts) or claim.claim_id == "efficiency"


def test_unknown_never_normalized(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    for claim in disposition.claims:
        assert d.guard_unknown_not_normalized(claim)
        if claim.verdict == d.VERDICT_UNKNOWN:
            assert not claim.supported_bounds, claim.claim_id


def test_negative_outcomes_first_class(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    cap1 = _claim(disposition, "cap1_semantics")
    outcomes = {row.system: row.outcome_class for row in cap1.evidence_rows}
    assert outcomes["abstain_control"] == d.OUTCOME_NEGATIVE
    assert outcomes["constant_prediction_control"] == d.OUTCOME_PREDICTION_IDENTICAL
    assert outcomes["CERT_CAP1"] == d.OUTCOME_NEGATIVE
    cert = next(row for row in cap1.evidence_rows if row.system == "CERT_CAP1")
    assert cert.metrics["rejection_codes"]  # stop codes preserved verbatim
    trace = _claim(disposition, "trace_evals")
    invalid = next(row for row in trace.evidence_rows if row.outcome_class == d.OUTCOME_INVALID)
    assert invalid.metrics["status"] == "QUARANTINED"
    assert invalid.metrics["rejection_reasons"]


def test_overlap_detection_row(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    cap1 = _claim(disposition, "cap1_semantics")
    audit = next(
        row for row in cap1.evidence_rows if row.system == "train_eval_overlap_audit"
    )
    assert audit.metrics["suite_integrity_ok"] is True
    if audit.metrics["contamination_findings"]:
        assert audit.outcome_class == d.OUTCOME_CONTAMINATED
    else:
        assert audit.outcome_class == d.OUTCOME_NEGATIVE


def test_no_fixture_only_capability_support(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    for claim in disposition.claims:
        assert d.guard_no_fixture_only_capability_support(claim)
        if claim.scope in d.FIXTURE_FORBIDDEN_SCOPES:
            assert claim.verdict != d.VERDICT_SUPPORTED, claim.claim_id


def test_efficiency_requires_workload_identity(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    efficiency = _claim(disposition, "efficiency")
    assert d.guard_efficiency_requires_workload_identity(efficiency)
    assert efficiency.verdict == d.VERDICT_UNKNOWN
    row = next(r for r in efficiency.evidence_rows if r.workload_identity)
    identity = row.workload_identity
    for key in (
        "workload",
        "hardware",
        "batch_size",
        "model_calls",
        "node_passes",
        "output_chars",
        "fallback_count",
        "timeout_count",
    ):
        assert identity.get(key) is not None, key
    # No supported efficiency claim without a comparative baseline.
    assert row.metrics.get("comparative_baseline") is None


def test_guards_reject_bad_claims() -> None:
    row = d.EvidenceRowV1(
        row_id="r",
        claim_id="cap1_semantics",
        suite="s",
        system="sys",
        outcome_class=d.OUTCOME_POSITIVE,
        fixture_only=True,
        metrics={},
        workload_identity=None,
        notes="",
    )
    bad = d.ClaimDispositionV1(
        claim_id="cap1_semantics",
        capability="c",
        scope=d.SCOPE_MODEL_CAPABILITY,
        verdict=d.VERDICT_SUPPORTED,
        supported_bounds="b",
        rationale="r",
        identities={"code": "x", "suite": "y", "config": "z", "hardware": {"a": 1}},
        artifacts=(),
        evidence_rows=(row,),
    )
    assert not d.guard_no_fixture_only_capability_support(bad)
    missing_identity = d.ClaimDispositionV1(
        claim_id="cap2_operators",
        capability="c",
        scope=d.SCOPE_HARNESS_CONTRACT,
        verdict=d.VERDICT_SUPPORTED,
        supported_bounds="b",
        rationale="r",
        identities={"code": "x"},
        artifacts=(),
        evidence_rows=(),
    )
    assert not d.guard_supported_requires_identity(missing_identity)
    normalized_unknown = d.ClaimDispositionV1(
        claim_id="cap0_grammar",
        capability="c",
        scope=d.SCOPE_MODEL_CAPABILITY,
        verdict=d.VERDICT_UNKNOWN,
        supported_bounds="normalized by assumption",
        rationale="r",
        identities={"code": "x"},
        artifacts=(),
        evidence_rows=(),
    )
    assert not d.guard_unknown_not_normalized(normalized_unknown)
    with pytest.raises(d.DispositionError):
        d.enforce_guards([bad])


def test_follow_ups_deduplicated_and_grounded(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    ids = [f["id"] for f in disposition.follow_ups]
    assert len(ids) == len(set(ids))
    assert "cap0-port-frozen-suite" in ids
    assert "dsh4-distillation-evidence" in ids
    assert "efficiency-normalized-baseline" in ids
    for follow in disposition.follow_ups:
        assert follow["grounding"]


def test_version_stamp(disposition: d.StagedDslHarnessDispositionV1) -> None:
    stamp = disposition.version_stamp
    assert stamp["components"].get(
        "harness.experiments.slm391_portfolio_disposition"
    )


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_determinism_without_training() -> None:
    first = d.build_disposition(repo_root=REPO_ROOT, run_suite_checks=False, now=NOW)
    second = d.build_disposition(repo_root=REPO_ROOT, run_suite_checks=False, now=NOW)
    # The version stamp carries its own wall-clock timestamp by design.
    assert {**first.to_dict(), "version_stamp": {}} == {
        **second.to_dict(),
        "version_stamp": {},
    }


def test_suite_rows_deterministic(tmp_path: Path) -> None:
    a = [r.to_dict() for r in d.run_two_pack_gate_checks()]
    b = [r.to_dict() for r in d.run_two_pack_gate_checks()]
    assert a == b
    c = [r.to_dict() for r in d.run_trace_eval_lifecycle(tmp_path / "t1")]
    e = [r.to_dict() for r in d.run_trace_eval_lifecycle(tmp_path / "t2")]
    # Frozen manifest paths differ by directory only; metrics are stable.
    assert [r["metrics"] for r in c] == [r["metrics"] for r in e]


# --------------------------------------------------------------------------
# Publisher doc updates (append-only)
# --------------------------------------------------------------------------


def test_quality_matrix_append_only(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    existing = (REPO_ROOT / "docs/design/quality-experiment-matrix.md").read_text(
        encoding="utf-8"
    )
    section = pub.quality_matrix_section(disposition)
    updated = pub.append_quality_matrix_section(existing, section)
    assert updated.startswith(existing)
    assert pub._MATRIX_HEADING in updated
    # Idempotent: a second append does not duplicate the section.
    assert pub.append_quality_matrix_section(updated, section) == updated


def test_lineage_entry_append_only(
    disposition: d.StagedDslHarnessDispositionV1,
) -> None:
    existing = (REPO_ROOT / "docs/design/research-lineage.md").read_text(encoding="utf-8")
    entry = pub.lineage_entry(disposition)
    updated = pub.append_lineage_entry(existing, entry)
    head, sep, tail = existing.partition(pub._LINEAGE_ANCHOR)
    assert sep  # honesty-rules footer exists
    assert updated.startswith(head.rstrip("\n"))
    assert updated.endswith(sep + tail)
    assert pub._LINEAGE_HEADING in updated
    assert pub.append_lineage_entry(updated, entry) == updated


def test_snippets_print_only(
    disposition: d.StagedDslHarnessDispositionV1, capsys
) -> None:
    model = pub.model_card_snippet(disposition)
    data = pub.dataset_card_snippet(disposition)
    assert "no checkpoint" in model.lower()
    assert "cap1_two_pack_freeze/v1" in data
    # The publisher never edits MODEL_CARD.md itself.
    assert "MODEL_CARD" not in model


# --------------------------------------------------------------------------
# CLI end to end (fixture scale)
# --------------------------------------------------------------------------


def test_cli_end_to_end(tmp_path: Path) -> None:
    exit_code = pub.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--cap1-steps",
            "1",
            "--cap1-records-per-arm",
            "4",
            "--cap1-seeds",
            "0",
            "--skip-docs",
        ]
    )
    assert exit_code == 0
    payload = json.loads((tmp_path / "slm391_portfolio_disposition.json").read_text())
    assert payload["schema"] == "StagedDslHarnessDispositionV1"
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    verdicts = {c["claim_id"]: c["verdict"] for c in payload["claims"]}
    assert verdicts["cap1_semantics"] == "rejected"
    assert verdicts["cap2_operators"] == "supported"
    assert verdicts["trace_evals"] == "supported"
    assert verdicts["cap0_grammar"] == "unknown"
    assert verdicts["distillation"] == "unknown"
    assert verdicts["efficiency"] == "unknown"
    assert payload["agentv_diagnostic"]
