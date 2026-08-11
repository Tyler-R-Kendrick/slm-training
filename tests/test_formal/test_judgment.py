"""KERN-01 / SLM-516: total solver judgment Lean↔Python parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.schemas import FormalPreflightV1
from slm_training.dsl.solver.state import (
    DomainValue,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import SupportCertificate, SupportQuery
from slm_training.formal.refutation_authority import (
    BindingIdsV1,
    CheckedRefutationEvidenceV1,
    make_exact_replay_evidence,
)
from slm_training.formal.judgment import (
    ClassificationInputsV1,
    JudgmentOutcome,
    SolverJudgmentV1,
    UnknownReason,
    WitnessPayloadV1,
    classify,
    classify_outcome,
    evaluate_truth_case,
    from_exact_closure_query,
    from_formal_preflight_status,
    from_support_certificate,
)


_BINDING = BindingIdsV1(
    state_id="judgment-test-state",
    problem_id="judgment-test-problem",
    source_id="vss",
    tool_id="python_replay",
)


def _refutation_evidence(digest: str = "checked-refutation-digest") -> CheckedRefutationEvidenceV1:
    return make_exact_replay_evidence(_BINDING, evidence_digest=digest)

_TRUTH_TABLE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "judgment_truth_table.v1.json"
)


def _truth_cases() -> list[dict]:
    payload = json.loads(_TRUTH_TABLE.read_text(encoding="utf-8"))
    assert payload["schema"] == "solver_judgment_truth_table/v1"
    return list(payload["cases"])


@pytest.mark.parametrize("case", _truth_cases(), ids=lambda item: item["id"])
def test_frozen_truth_table(case: dict) -> None:
    got = evaluate_truth_case(case)
    for key, expected in case["expect"].items():
        assert got[key] == expected, (case["id"], key, got[key], expected)


def test_failure_flags_never_refuted_exhaustive() -> None:
    flags = (
        "timed_out",
        "skipped_replay",
        "unsupported_capability",
        "incomplete_coverage",
        "missing_tool",
    )
    base = {
        "vss_verdict": "unsupported",
        "exhausted": True,
        "replay_checked": True,
        "replay_ok": True,
    }
    for flag in flags:
        row = dict(base)
        row[flag] = True
        got = evaluate_truth_case({"inputs": row, "checked": True})
        assert got["outcome"] != "refuted", flag


def test_solver_judgment_forbids_bool_coercion() -> None:
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    with pytest.raises(TypeError, match="forbids truthy/falsy"):
        bool(judgment)


def test_round_trip_persisted_schema() -> None:
    original = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            exhausted=True,
            replay_checked=True,
            replay_ok=True,
            refutation_evidence=_refutation_evidence(),
            expected_binding=_BINDING,
        ),
        checked=True,
    )
    rebuilt = SolverJudgmentV1.from_dict(original.to_dict())
    assert rebuilt.to_dict() == original.to_dict()


def test_mutation_rejects_status_payload_mismatch() -> None:
    good = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    bad = good.to_dict()
    bad["outcome"] = "refuted"
    with pytest.raises(ValueError, match="status tag / payload mismatch"):
        SolverJudgmentV1.from_dict(bad)


def test_only_checked_witnessed_and_refuted_authorize_semantic() -> None:
    witnessed = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    refuted = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            exhausted=True,
            replay_checked=True,
            replay_ok=True,
            refutation_evidence=_refutation_evidence(),
            expected_binding=_BINDING,
        ),
        checked=True,
    )
    unknown = classify(ClassificationInputsV1(timed_out=True), checked=True)
    assert witnessed.authorizes_semantic_conclusion()
    assert refuted.authorizes_semantic_conclusion()
    assert not unknown.authorizes_semantic_conclusion()
    unchecked = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=False,
    )
    assert not unchecked.authorizes_semantic_conclusion()


def test_timeout_skipped_replay_never_authorize_removal() -> None:
    for extra in ({"timed_out": True}, {"skipped_replay": True}):
        judgment = classify(
            ClassificationInputsV1(
                vss_verdict="unsupported",
                exhausted=True,
                replay_checked=True,
                replay_ok=True,
                **extra,
            ),
            checked=True,
        )
        assert judgment.outcome is JudgmentOutcome.UNKNOWN
        assert not judgment.authorizes_removal()


def _bounds() -> SolverBounds:
    return SolverBounds(
        max_tokens=100,
        max_nodes=100,
        max_depth=8,
        max_backtracks=100,
        max_verifier_calls=100,
    )


def _cert(verdict: SupportVerdict, *, exhausted: bool, witness: str | None) -> SupportCertificate:
    query = SupportQuery(
        state_fingerprint="0" * 64,
        hole_id=HoleId(namespace="test", path=("h0",), kind="hole"),
        candidate=DomainValue.create("token", {"ids": [1]}),
    )
    return SupportCertificate(
        schema_version=1,
        query=query,
        verdict=verdict,
        problem_id="p",
        pack_id="pack",
        constraint_version="v1",
        bounds=_bounds(),
        search_order="canonical-domain-value-v1",
        explored_state_fingerprints=("0" * 64,),
        coverage_observations=("complete",),
        verifier_profile="fixture",
        witness_source="fixture" if witness else None,
        witness_digest=witness,
        exhausted=exhausted,
        stop_reason=None,
    )


def test_support_certificate_adapter_lossless() -> None:
    supported = _cert(SupportVerdict.SUPPORTED, exhausted=False, witness="9" * 64)
    judgment = from_support_certificate(
        supported, replay_checked=True, replay_ok=True, checked=True
    )
    assert judgment.outcome is JudgmentOutcome.WITNESSED
    assert isinstance(judgment.payload, WitnessPayloadV1)
    assert judgment.payload.witness_digest == "9" * 64
    round_trip = SolverJudgmentV1.from_dict(judgment.to_dict())
    assert round_trip == judgment


def test_unsupported_with_timeout_stays_unknown_not_refuted() -> None:
    cert = _cert(SupportVerdict.UNSUPPORTED, exhausted=True, witness=None)
    cert = SupportCertificate(
        schema_version=cert.schema_version,
        query=cert.query,
        verdict=cert.verdict,
        problem_id=cert.problem_id,
        pack_id=cert.pack_id,
        constraint_version=cert.constraint_version,
        bounds=cert.bounds,
        search_order=cert.search_order,
        explored_state_fingerprints=cert.explored_state_fingerprints,
        coverage_observations=cert.coverage_observations,
        verifier_profile=cert.verifier_profile,
        exhausted=cert.exhausted,
        stop_reason="budget_timeout",
    )
    judgment = from_support_certificate(
        cert, replay_checked=True, replay_ok=True, checked=True
    )
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.TIMEOUT
    assert not judgment.authorizes_removal()


def test_skipped_replay_adapter_never_refuted() -> None:
    cert = _cert(SupportVerdict.UNSUPPORTED, exhausted=True, witness=None)
    judgment = from_support_certificate(
        cert,
        replay_checked=False,
        replay_ok=False,
        checked=True,
        skipped_replay=True,
    )
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert judgment.payload is UnknownReason.SKIPPED_REPLAY
    assert not judgment.authorizes_removal()


def test_exact_closure_adapter() -> None:
    refuted = from_exact_closure_query(
        verdict="unsupported",
        replay_ok=True,
        replay_checked=True,
        checked=True,
        exhausted=True,
        refutation_evidence=_refutation_evidence(),
        expected_binding=_BINDING,
    )
    assert refuted.outcome is JudgmentOutcome.REFUTED
    skipped = from_exact_closure_query(
        verdict="unsupported",
        replay_ok=True,
        replay_checked=True,
        checked=True,
        exhausted=True,
        skipped_replay=True,
        refutation_evidence=_refutation_evidence(),
        expected_binding=_BINDING,
    )
    assert skipped.outcome is JudgmentOutcome.UNKNOWN


def test_formal_preflight_adapter_keeps_lifecycle_separate() -> None:
    witnessed = from_formal_preflight_status("proved", counterexample=None, checked=True)
    refuted = from_formal_preflight_status(
        "refuted", counterexample={"digest": "a" * 64}, checked=True
    )
    unknown = from_formal_preflight_status("timed_out", counterexample=None, checked=False)
    assert witnessed.outcome is JudgmentOutcome.WITNESSED
    assert refuted.outcome is JudgmentOutcome.REFUTED
    assert unknown.outcome is JudgmentOutcome.UNKNOWN
    assert FormalPreflightV1.model_fields["status"].annotation is not None


def test_classify_outcome_is_total() -> None:
    sample = ClassificationInputsV1()
    outcome = classify_outcome(sample)
    assert outcome.value in {"witnessed", "refuted", "unknown", "invalid"}


def test_invalid_reason_on_supported_without_witness() -> None:
    got = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=False,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    assert got.outcome is JudgmentOutcome.UNKNOWN
    assert got.payload is UnknownReason.INCONCLUSIVE
