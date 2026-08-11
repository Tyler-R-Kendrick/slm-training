"""EVID-10 / SLM-553: verified encoding adapters + mutation suite."""

from __future__ import annotations

import pytest

from slm_training.formal.authority import (
    CheckerCapabilityRefV1,
    CheckerResultRefV1,
    build_formal_authority_v2,
)
from slm_training.formal.encoding_adapter import (
    ENCODER_VERSION,
    PRODUCTION_CERTIFICATE_FORMATS,
    THEOREM_FQN,
    THEOREM_TYPE_BINDING,
    BoundedProblemV1,
    CnfRefEncodingAdapter,
    LiteralV1,
    VerifiedEncodingEvidenceV2,
    attach_encoding_evidence_fields,
    encoded_satisfiable,
    encoder_source_hash,
    encoding_authorizes_semantic_result,
    exists_satisfying_assignment,
    fixture_problems,
    load_fixture_payload,
    mint_checked_certificate_via_encoding,
    mutate_encoded_flip_first_literal,
    satisfies,
)
from slm_training.formal.judgment import (
    JudgmentOutcome,
    RefutationPayloadV1,
    SolverJudgmentV1,
)
from slm_training.formal.objects import lean_claim_catalog
from slm_training.formal.refutation_authority import BindingIdsV1, evidence_authorizes_removal

_BINDING = BindingIdsV1(
    state_id="evid10-state",
    problem_id="evid10-problem",
    source_id="encoding_bridge",
    tool_id="python_encoding_ref",
)


def _adapter() -> CnfRefEncodingAdapter:
    return CnfRefEncodingAdapter()


def test_fixture_schema_and_cases_load() -> None:
    payload = load_fixture_payload()
    assert payload["evid"] == "EVID-10"
    problems = fixture_problems()
    assert "unsat_contradiction" in problems
    assert "unsupported_cardinality" in problems


def test_supported_fixtures_match_sat_equivalence() -> None:
    adapter = _adapter()
    payload = load_fixture_payload()
    for case in payload["cases"]:
        if case["expect_status"] != "supported":
            continue
        if case.get("kind") == "encoding_mutation":
            continue
        problem = BoundedProblemV1.from_dict(case["problem"])
        result = adapter.encode(problem)
        assert result.status == "supported"
        assert result.encoded is not None
        sat_enc = encoded_satisfiable(result.encoded)
        sat_prob = exists_satisfying_assignment(problem)
        assert sat_enc == sat_prob == bool(case["expect_satisfiable"])


def test_unsupported_and_unknown_never_encode() -> None:
    adapter = _adapter()
    problems = fixture_problems()
    bad = adapter.encode(problems["unsupported_cardinality"])
    assert bad.status == "unsupported"
    assert bad.encoded is None
    unknown = adapter.encode(problems["unknown_missing_feature_tags"])
    assert unknown.status == "unknown"
    assert unknown.encoded is None


def test_decode_witness_identity_roundtrip() -> None:
    adapter = _adapter()
    problem = fixture_problems()["sat_unit_positive"]
    mapped = adapter.decode_witness(problem, {"x": 1})
    assert mapped == {"x": 1}
    assert satisfies(problem, mapped)
    assert adapter.decode_witness(problem, {"x": 0}) is None


def test_encoding_evidence_v2_records_required_fields() -> None:
    adapter = _adapter()
    problem = fixture_problems()["unsat_contradiction"]
    result, evidence = adapter.build_evidence(problem)
    assert result.status == "supported"
    assert evidence is not None
    assert evidence.encoder_version == ENCODER_VERSION
    assert evidence.encoder_hash == encoder_source_hash()
    assert evidence.theorem_fqn == THEOREM_FQN
    assert evidence.theorem_type_binding == THEOREM_TYPE_BINDING
    assert evidence.certificate_format in PRODUCTION_CERTIFICATE_FORMATS
    assert evidence.original_problem_digest == problem.digest()
    assert evidence.encoded_digest == result.encoded.digest()  # type: ignore[union-attr]
    assert evidence.checker_capability.available
    assert evidence.trust_domain


def test_valid_certificate_wrong_encoding_cannot_authorize() -> None:
    adapter = _adapter()
    problem = fixture_problems()["unsat_contradiction"]
    result, evidence = adapter.build_evidence(problem)
    assert evidence is not None and result.encoded is not None
    mutated = mutate_encoded_flip_first_literal(result.encoded)
    assert mutated.digest() != result.encoded.digest()

    # Certificate "checks" against the mutated encoding, but evidence still
    # claims the correct encoding — observed digest mismatch blocks authority.
    assert encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=result.encoded.digest(),
    )
    assert not encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=mutated.digest(),
    )

    # Swap evidence to the mutated formula while keeping the original problem:
    # bridge still fails because encoder identity is bound to the honest encode.
    forged = VerifiedEncodingEvidenceV2(
        original_problem_digest=problem.digest(),
        encoded_digest=mutated.digest(),
        encoder_version=evidence.encoder_version,
        encoder_hash=evidence.encoder_hash,
        theorem_fqn=evidence.theorem_fqn,
        theorem_type_binding=evidence.theorem_type_binding,
        certificate_format=evidence.certificate_format,
        checker_capability=evidence.checker_capability,
        trust_domain=evidence.trust_domain,
        support_status="supported",
    )
    # Honest re-encode digest differs from forged encoded_digest when observed
    # against the adapter's real encoding of the same problem.
    honest = adapter.encode(problem).encoded
    assert honest is not None
    assert not encoding_authorizes_semantic_result(
        forged,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=honest.digest(),
    )


def test_mint_checked_certificate_requires_encoding_bridge() -> None:
    adapter = _adapter()
    problem = fixture_problems()["unsat_contradiction"]
    _, evidence = adapter.build_evidence(problem)
    assert evidence is not None

    ok = mint_checked_certificate_via_encoding(
        _BINDING,
        encoding=evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=evidence.encoded_digest,
    )
    assert ok is not None
    assert evidence_authorizes_removal(ok, _BINDING)

    blocked = mint_checked_certificate_via_encoding(
        _BINDING,
        encoding=evidence,
        expected_problem_digest="0" * 64,
        certificate_checked=True,
        encoded_digest_observed=evidence.encoded_digest,
    )
    assert blocked is None

    no_cert = mint_checked_certificate_via_encoding(
        _BINDING,
        encoding=evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=False,
    )
    assert no_cert is None


def test_pilot_formats_named_but_not_production_authority() -> None:
    adapter = _adapter()
    problem = fixture_problems()["unsat_contradiction"]
    _, evidence = adapter.build_evidence(
        problem, certificate_format="lrat_pilot"
    )
    assert evidence is not None
    assert evidence.certificate_format == "lrat_pilot"
    assert not encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=evidence.encoded_digest,
        require_production_format=True,
    )
    # Explicit pilot opt-in still requires a checked capability — never a dep.
    assert encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=evidence.encoded_digest,
        require_production_format=False,
    )


def test_formal_authority_v2_carries_encoding_evidence() -> None:
    adapter = _adapter()
    problem = fixture_problems()["unsat_contradiction"]
    _, evidence = adapter.build_evidence(problem)
    assert evidence is not None
    cert = mint_checked_certificate_via_encoding(
        _BINDING,
        encoding=evidence,
        expected_problem_digest=problem.digest(),
        encoded_digest_observed=evidence.encoded_digest,
    )
    assert cert is not None
    digests, blob = attach_encoding_evidence_fields(encoding=evidence)
    judgment = SolverJudgmentV1(
        outcome=JudgmentOutcome.REFUTED,
        payload=RefutationPayloadV1(
            refutation_digest=cert.evidence_digest,
            source="checked_certificate",
        ),
        checked=True,
    )
    authority = build_formal_authority_v2(
        claim_id="evid10-encoding-bridge",
        surface="formal_object",
        judgment=judgment,
        v1_artifact={"refutation": cert.to_dict()},
        v1_content_digest=cert.evidence_digest,
        checker_results=(
            CheckerResultRefV1(
                backend="python_encoding_ref",
                status="pass",
                result_digest=evidence.encoded_digest,
            ),
        ),
        checker_capabilities=(
            CheckerCapabilityRefV1(backend="python_encoding_ref", available=True),
        ),
        source_digests=digests,
        certificate_digest=evidence.encoded_digest,
        encoding_evidence=blob,
    )
    assert authority.encoding_evidence is not None
    assert authority.encoding_evidence["original_problem_digest"] == problem.digest()
    rebuilt = type(authority).from_dict(authority.to_dict())
    assert rebuilt.content_digest == authority.content_digest
    assert rebuilt.encoding_evidence == blob


def test_lean_claim_catalog_includes_encoding_theorem() -> None:
    catalog = lean_claim_catalog()
    assert "VerifiedEncoding.sat_encode_iff_satisfies" in catalog
    meta = catalog["VerifiedEncoding.sat_encode_iff_satisfies"]
    assert meta["module"] == "LeverProofLean.VerifiedEncoding"


def test_empty_clause_and_non_bool_domain_unsupported() -> None:
    adapter = _adapter()
    empty = BoundedProblemV1(
        problem_id="empty-clause",
        domains={"x": (0, 1)},
        clauses=(tuple(),),
        features=frozenset({"bool_domain", "clause"}),
    )
    assert adapter.encode(empty).status == "unsupported"
    ternary = BoundedProblemV1(
        problem_id="ternary",
        domains={"x": (0, 1, 2)},
        clauses=((LiteralV1("x", True),),),
        features=frozenset({"bool_domain", "clause"}),
    )
    assert adapter.encode(ternary).status == "unsupported"


def test_unavailable_checker_capability_blocks_authority() -> None:
    adapter = _adapter()
    problem = fixture_problems()["unsat_contradiction"]
    _, evidence = adapter.build_evidence(
        problem,
        checker_capability=CheckerCapabilityRefV1(
            backend="python_encoding_ref", available=False
        ),
    )
    assert evidence is not None
    assert not encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=evidence.encoded_digest,
    )


@pytest.mark.parametrize(
    "fmt",
    sorted(PRODUCTION_CERTIFICATE_FORMATS),
)
def test_production_formats_are_fixture_checked(fmt: str) -> None:
    assert fmt == "fixture_checked"
