"""EVID-09 / SLM-542: checked refutation authority mutation suite."""

from __future__ import annotations

import pytest

from slm_training.dsl.solver.state import (
    DomainValue,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import SupportCertificate, SupportQuery
from slm_training.formal.authority import (
    CheckerResultRefV1,
)
from slm_training.formal.closure_stabilization import QueryResultV1, close_pass_results
from slm_training.formal.judgment import (
    ClassificationInputsV1,
    JudgmentOutcome,
    classify,
    from_exact_closure_query,
    from_support_certificate,
)
from slm_training.formal.objects import export_closure_law
from slm_training.formal.refutation_authority import (
    BindingIdsV1,
    CheckedRefutationEvidenceV1,
    LegacyExhaustionFlagV1,
    evidence_authorizes_removal,
    legacy_exhaustion_authorizes_removal,
    make_checked_certificate_evidence,
    make_exact_replay_evidence,
)

_BINDING = BindingIdsV1(
    state_id="evid09-state",
    problem_id="evid09-problem",
    source_id="vss",
    tool_id="python_replay",
)


def _evidence(*, digest: str = "evid09-digest", kind: str = "exact_replay"):
    if kind == "exact_replay":
        return make_exact_replay_evidence(_BINDING, evidence_digest=digest)
    return make_checked_certificate_evidence(_BINDING, evidence_digest=digest)


def test_legacy_exhaustion_flags_never_authorize() -> None:
    flag = LegacyExhaustionFlagV1(
        exhausted=True, replay_ok=True, coverage_complete=True
    )
    assert legacy_exhaustion_authorizes_removal(flag) is False


def test_forged_exhausted_coverage_replay_flags_cannot_refute() -> None:
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            exhausted=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert not judgment.authorizes_removal()


def test_adapter_failure_modes_strip_evidence_and_block_removal() -> None:
    extras = (
        {"skipped_replay": True},
        {"timed_out": True},
        {"missing_tool": True},
        {"incomplete_coverage": True},
        {"unsupported_capability": True},
        {"malformed_input": True},
    )
    for extra in extras:
        judgment = from_exact_closure_query(
            verdict="unsupported",
            replay_ok=True,
            replay_checked=True,
            checked=True,
            exhausted=True,
            refutation_evidence=_evidence(),
            expected_binding=_BINDING,
            **extra,
        )
        assert judgment.outcome in {JudgmentOutcome.UNKNOWN, JudgmentOutcome.INVALID}
        assert not judgment.authorizes_removal()


def test_stale_problem_id_rejects_removal() -> None:
    stale = BindingIdsV1(
        state_id=_BINDING.state_id,
        problem_id="stale-problem",
        source_id=_BINDING.source_id,
        tool_id=_BINDING.tool_id,
    )
    evidence = make_exact_replay_evidence(stale, evidence_digest="digest")
    assert evidence_authorizes_removal(evidence, _BINDING) is False
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            exhausted=True,
            replay_checked=True,
            replay_ok=True,
            refutation_evidence=evidence,
            expected_binding=_BINDING,
        ),
        checked=True,
    )
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert not judgment.authorizes_removal()


def test_skipped_checker_blocks_checked_refutation_authority() -> None:
    from slm_training.formal.authority import from_formal_object_v1

    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="unsupported",
            refutation_evidence=_evidence(),
            expected_binding=_BINDING,
        ),
        checked=True,
    )
    obj = export_closure_law("removable_only_replay_valid_unsupported")
    with pytest.raises(ValueError, match="checked semantic"):
        from_formal_object_v1(
            obj,
            judgment=judgment,
            checker_results=(
                CheckerResultRefV1(backend="python_structural", status="pass"),
                CheckerResultRefV1(backend="python_replay", status="skipped"),
            ),
        )


def test_exact_replay_and_checked_certificate_authorize_removal() -> None:
    for kind in ("exact_replay", "checked_certificate"):
        judgment = classify(
            ClassificationInputsV1(
                vss_verdict="unsupported",
                refutation_evidence=_evidence(kind=kind),
                expected_binding=_BINDING,
            ),
            checked=True,
        )
        assert judgment.outcome is JudgmentOutcome.REFUTED
        assert judgment.authorizes_removal()
        assert judgment.payload.source in {
            "exact_semantic_replay",
            "checked_certificate",
        }


def test_query_result_telemetry_replay_ok_alone_cannot_remove() -> None:
    live = [0, 1, 2]
    forged = QueryResultV1(
        hole=0,
        candidate=1,
        verdict="unsupported",
        replay_ok=True,
    )
    assert close_pass_results(live, [forged]) == live


def test_query_result_with_bound_evidence_removes() -> None:
    live = [0, 1, 2]
    ok = QueryResultV1(
        hole=0,
        candidate=1,
        verdict="unsupported",
        replay_ok=True,
        refutation_evidence=_evidence(),
        expected_binding=_BINDING,
    )
    assert close_pass_results(live, [ok]) == [0, 2]


def test_support_certificate_forged_exhaustion_without_evidence_unknown() -> None:
    query = SupportQuery(
        state_fingerprint="0" * 64,
        hole_id=HoleId(namespace="test", path=("h0",), kind="hole"),
        candidate=DomainValue.create("token", {"ids": [1]}),
    )
    cert = SupportCertificate(
        schema_version=1,
        query=query,
        verdict=SupportVerdict.UNSUPPORTED,
        problem_id="p",
        pack_id="pack",
        constraint_version="v1",
        bounds=SolverBounds(
            max_tokens=100,
            max_nodes=100,
            max_depth=8,
            max_backtracks=100,
            max_verifier_calls=100,
        ),
        search_order="canonical-domain-value-v1",
        explored_state_fingerprints=("0" * 64,),
        coverage_observations=("complete",),
        verifier_profile="fixture",
        exhausted=True,
        stop_reason=None,
    )
    judgment = from_support_certificate(
        cert, replay_checked=True, replay_ok=True, checked=True
    )
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    assert not judgment.authorizes_removal()

    authorized = from_support_certificate(
        cert,
        replay_checked=True,
        replay_ok=True,
        checked=True,
        refutation_evidence=_evidence(digest=cert.digest),
        expected_binding=_BINDING,
    )
    assert authorized.outcome is JudgmentOutcome.REFUTED
    assert authorized.authorizes_removal()


def test_certificate_missing_capability_invalid_for_checked_path() -> None:
    evidence = CheckedRefutationEvidenceV1(
        kind="checked_certificate",
        binding=_BINDING,
        evidence_digest="digest",
        capability_present=False,
        trust_checked=True,
    )
    assert evidence.well_formed() is False
    assert evidence_authorizes_removal(evidence, _BINDING) is False


def test_malformed_evidence_dict_rejected() -> None:
    with pytest.raises(ValueError):
        CheckedRefutationEvidenceV1.from_dict(
            {
                "schema": "checked_refutation_evidence/v1",
                "kind": "nope",
                "binding": _BINDING.to_dict(),
                "evidence_digest": "x",
                "capability_present": True,
                "trust_checked": True,
            }
        )
