"""Multi-prover formal object export and loop-closure tests."""

from __future__ import annotations

from slm_training.dsl.solver.state import (
    DomainValue,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import SupportCertificate, SupportQuery
from slm_training.formal.checkers import (
    CHECKER_LEAN_KERNEL,
    CHECKER_PYTHON_REFERENCE,
    CHECKER_PYTHON_STRUCTURAL,
    check_python_reference,
    check_python_structural,
    run_checkers,
)
from slm_training.formal.loop import (
    close_formal_loop,
    loop_requires_multi_backend,
)
from slm_training.formal.objects import (
    FORMAL_OBJECT_SCHEMA,
    FormalObjectKind,
    FormalObjectV1,
    export_closure_law,
    export_lean_claim,
    export_support_certificate,
    lean_claim_catalog,
)


def _bounds() -> SolverBounds:
    return SolverBounds(
        max_tokens=100,
        max_nodes=100,
        max_depth=8,
        max_backtracks=100,
        max_verifier_calls=100,
    )


def _fp(n: int = 0) -> str:
    return f"{n:064x}"


def _unsupported_cert() -> SupportCertificate:
    query = SupportQuery(
        state_fingerprint=_fp(1),
        hole_id=HoleId(namespace="test", path=("h0",), kind="hole"),
        candidate=DomainValue.create("token", {"ids": [1]}),
    )
    return SupportCertificate(
        schema_version=1,
        query=query,
        verdict=SupportVerdict.UNSUPPORTED,
        problem_id="p",
        pack_id="pack",
        constraint_version="v1",
        bounds=_bounds(),
        search_order="canonical-domain-value-v1",
        explored_state_fingerprints=(_fp(1),),
        coverage_observations=("complete",),
        verifier_profile="fixture",
        exhausted=True,
        stop_reason=None,
    )


def _supported_cert() -> SupportCertificate:
    query = SupportQuery(
        state_fingerprint=_fp(2),
        hole_id=HoleId(namespace="test", path=("h0",), kind="hole"),
        candidate=DomainValue.create("token", {"ids": [2]}),
    )
    return SupportCertificate(
        schema_version=1,
        query=query,
        verdict=SupportVerdict.SUPPORTED,
        problem_id="p",
        pack_id="pack",
        constraint_version="v1",
        bounds=_bounds(),
        search_order="canonical-domain-value-v1",
        explored_state_fingerprints=(_fp(2),),
        coverage_observations=("complete",),
        verifier_profile="fixture",
        witness_source="fixture",
        witness_digest=_fp(9),
        exhausted=False,
    )


def test_export_support_certificate_round_trip() -> None:
    cert = _unsupported_cert()
    obj = export_support_certificate(cert)
    assert obj.schema == FORMAL_OBJECT_SCHEMA
    assert obj.kind is FormalObjectKind.SUPPORT_CERTIFICATE
    assert CHECKER_PYTHON_STRUCTURAL in obj.required_checkers
    assert CHECKER_PYTHON_REFERENCE in obj.required_checkers
    rebuilt = FormalObjectV1.from_dict(obj.to_dict())
    assert rebuilt.content_digest == obj.content_digest


def test_dishonest_unsupported_fails_structural() -> None:
    cert = _unsupported_cert()
    # Mutate to dishonest: not exhausted.
    bad = SupportCertificate(
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
        exhausted=False,
        stop_reason=None,
    )
    obj = export_support_certificate(bad)
    result = check_python_structural(obj)
    assert result.ok is False
    assert "exhausted" in result.detail


def test_supported_and_unsupported_pass_dual_backends() -> None:
    for cert in (_unsupported_cert(), _supported_cert()):
        obj = export_support_certificate(cert)
        results = run_checkers(obj)
        assert all(r.ok for r in results if not r.skipped)
        backends = {r.backend for r in results if r.ok and not r.skipped}
        assert CHECKER_PYTHON_STRUCTURAL in backends
        assert CHECKER_PYTHON_REFERENCE in backends


def test_lean_claim_catalog_exports_and_checks() -> None:
    catalog = lean_claim_catalog()
    assert "Forest.close_monotone" in catalog
    objects = [export_lean_claim(tid) for tid in catalog]
    report = close_formal_loop(objects, min_backends=2, enable_lean=False)
    assert report.closed, report.summary
    assert not report.single_kernel_reliance


def test_closure_laws_close_loop() -> None:
    objects = [
        export_closure_law("removable_only_replay_valid_unsupported"),
        export_closure_law("close_pass_only_shrinks"),
        export_closure_law("fixed_point_when_no_removals"),
    ]
    report = close_formal_loop(objects, min_backends=2)
    assert report.closed


def test_single_lean_kernel_not_enough() -> None:
    from slm_training.formal.checkers import CheckerResult

    results = [
        CheckerResult(
            checker_id=CHECKER_LEAN_KERNEL,
            backend=CHECKER_LEAN_KERNEL,
            ok=True,
            detail="only lean",
        )
    ]
    accepted, reason, _ok_b, _, _ = loop_requires_multi_backend(results, min_backends=2)
    assert accepted is False
    assert "single-kernel" in reason or "need 2" in reason


def test_loop_rejects_empty() -> None:
    report = close_formal_loop([], min_backends=2)
    assert report.closed is False


def test_reference_backend_independent() -> None:
    obj = export_lean_claim("ExactClosure.closePass_subset")
    a = check_python_structural(obj)
    b = check_python_reference(obj)
    assert a.ok and b.ok
    assert a.backend != b.backend
