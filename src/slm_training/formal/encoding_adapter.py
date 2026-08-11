"""Verified encoding adapters (EVID-10 / SLM-553).

Trusted bridge from bounded problem semantics to external SAT/PB/SMT encodings.
A checked solver certificate has **no** semantic authority without this bridge —
reuse FormalAuthorityV2 + EVID-09 ``checked_certificate``; do not invent a third
evidence store.

Unsupported features return ``unsupported`` / ``unknown`` — never silently
dropped or approximated. Pilot certificate formats (LRAT / PBLean / Alethe) are
named for future RESEARCH pilots and are not production dependencies.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from slm_training.formal.authority import CheckerCapabilityRefV1
from slm_training.formal.refutation_authority import (
    BindingIdsV1,
    CheckedRefutationEvidenceV1,
    make_checked_certificate_evidence,
)

ENCODING_ADAPTER_SCHEMA = "verified_encoding_adapter/v1"
ENCODING_EVIDENCE_SCHEMA = "verified_encoding_evidence/v2"
ENCODING_FIXTURE_SCHEMA = "encoding_adapter_fixtures/v1"

ENCODER_FAMILY_CNF_REF = "cnf_ref"
ENCODER_VERSION = "cnf_ref_v1"
THEOREM_FQN = "LeverProofLean.VerifiedEncoding.sat_encode_iff_satisfies"
THEOREM_TYPE_BINDING = "Satisfiable(encode p) ↔ ∃ x, Satisfies p x"

SUPPORTED_FEATURES: frozenset[str] = frozenset({"bool_domain", "clause"})

CertificateFormatName = Literal[
    "fixture_checked",
    "lrat_pilot",
    "pblean_pilot",
    "alethe_pilot",
]
SupportStatus = Literal["supported", "unsupported", "unknown"]

CERTIFICATE_FORMATS: frozenset[str] = frozenset(
    {"fixture_checked", "lrat_pilot", "pblean_pilot", "alethe_pilot"}
)
# Production path may mint authority only with the fixture-checked format.
# Pilot formats are declared so RESEARCH-05/06/07 can plug in without deps here.
PRODUCTION_CERTIFICATE_FORMATS: frozenset[str] = frozenset({"fixture_checked"})

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "encoding_adapter_fixtures.v1.json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encoder_source_hash(*, family: str = ENCODER_FAMILY_CNF_REF, version: str = ENCODER_VERSION) -> str:
    """Stable encoder identity hash (versioned family contract, not file bytes)."""

    return _sha256(
        _canonical_json(
            {
                "family": family,
                "version": version,
                "theorem_fqn": THEOREM_FQN,
                "theorem_type_binding": THEOREM_TYPE_BINDING,
                "supported_features": sorted(SUPPORTED_FEATURES),
            }
        )
    )


@dataclass(frozen=True)
class LiteralV1:
    var: str
    positive: bool

    def to_dict(self) -> dict[str, Any]:
        return {"var": self.var, "positive": self.positive}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiteralV1:
        return cls(var=str(data["var"]), positive=bool(data["positive"]))


@dataclass(frozen=True)
class BoundedProblemV1:
    """Tiny bounded VSS-shaped problem (finite bool domains + clauses)."""

    problem_id: str
    domains: dict[str, tuple[int, ...]]
    clauses: tuple[tuple[LiteralV1, ...], ...]
    features: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "domains": {key: list(vals) for key, vals in sorted(self.domains.items())},
            "clauses": [[lit.to_dict() for lit in clause] for clause in self.clauses],
            "features": sorted(self.features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundedProblemV1:
        domains = {
            str(key): tuple(int(v) for v in vals)
            for key, vals in dict(data["domains"]).items()
        }
        clauses = tuple(
            tuple(LiteralV1.from_dict(lit) for lit in clause)
            for clause in data["clauses"]
        )
        features = frozenset(str(f) for f in data.get("features", ()))
        return cls(
            problem_id=str(data["problem_id"]),
            domains=domains,
            clauses=clauses,
            features=features,
        )

    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class EncodedFormulaV1:
    """Deterministic CNF encoding payload (DIMACS-shaped, JSON-stable)."""

    family: str
    var_order: tuple[str, ...]
    clauses: tuple[tuple[tuple[str, bool], ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "var_order": list(self.var_order),
            "clauses": [
                [{"var": var, "positive": pos} for var, pos in clause]
                for clause in self.clauses
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EncodedFormulaV1:
        clauses = tuple(
            tuple((str(lit["var"]), bool(lit["positive"])) for lit in clause)
            for clause in data["clauses"]
        )
        return cls(
            family=str(data["family"]),
            var_order=tuple(str(v) for v in data["var_order"]),
            clauses=clauses,
        )

    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class EncodingResultV1:
    status: SupportStatus
    encoded: EncodedFormulaV1 | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "reason": self.reason}
        if self.encoded is not None:
            out["encoded"] = self.encoded.to_dict()
            out["encoded_digest"] = self.encoded.digest()
        return out


@dataclass(frozen=True)
class TheoremCheckerRefsV1:
    theorem_fqn: str
    theorem_type_binding: str
    certificate_format: CertificateFormatName
    checker_backend: str
    trust_domain: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "theorem_fqn": self.theorem_fqn,
            "theorem_type_binding": self.theorem_type_binding,
            "certificate_format": self.certificate_format,
            "checker_backend": self.checker_backend,
            "trust_domain": self.trust_domain,
        }


@dataclass(frozen=True)
class VerifiedEncodingEvidenceV2:
    """v2 encoding-bridge evidence for FormalAuthorityV2 / checked certificates."""

    original_problem_digest: str
    encoded_digest: str
    encoder_version: str
    encoder_hash: str
    theorem_fqn: str
    theorem_type_binding: str
    certificate_format: CertificateFormatName
    checker_capability: CheckerCapabilityRefV1
    trust_domain: str
    support_status: SupportStatus
    schema: str = ENCODING_EVIDENCE_SCHEMA

    def well_formed(self) -> bool:
        if self.schema != ENCODING_EVIDENCE_SCHEMA:
            return False
        if self.support_status != "supported":
            return False
        if not (
            self.original_problem_digest
            and self.encoded_digest
            and self.encoder_version
            and self.encoder_hash
            and self.theorem_fqn
            and self.theorem_type_binding
            and self.trust_domain
        ):
            return False
        if self.certificate_format not in CERTIFICATE_FORMATS:
            return False
        return self.checker_capability.available

    def binds_problem(self, expected_problem_digest: str) -> bool:
        return (
            self.well_formed()
            and self.original_problem_digest == expected_problem_digest
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "original_problem_digest": self.original_problem_digest,
            "encoded_digest": self.encoded_digest,
            "encoder_version": self.encoder_version,
            "encoder_hash": self.encoder_hash,
            "theorem_fqn": self.theorem_fqn,
            "theorem_type_binding": self.theorem_type_binding,
            "certificate_format": self.certificate_format,
            "checker_capability": self.checker_capability.to_dict(),
            "trust_domain": self.trust_domain,
            "support_status": self.support_status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VerifiedEncodingEvidenceV2:
        schema = str(data.get("schema", ENCODING_EVIDENCE_SCHEMA))
        if schema != ENCODING_EVIDENCE_SCHEMA:
            raise ValueError(f"expected schema {ENCODING_EVIDENCE_SCHEMA!r}")
        fmt = str(data["certificate_format"])
        if fmt not in CERTIFICATE_FORMATS:
            raise ValueError(f"unknown certificate_format: {fmt!r}")
        status = str(data["support_status"])
        if status not in {"supported", "unsupported", "unknown"}:
            raise ValueError(f"invalid support_status: {status!r}")
        return cls(
            original_problem_digest=str(data["original_problem_digest"]),
            encoded_digest=str(data["encoded_digest"]),
            encoder_version=str(data["encoder_version"]),
            encoder_hash=str(data["encoder_hash"]),
            theorem_fqn=str(data["theorem_fqn"]),
            theorem_type_binding=str(data["theorem_type_binding"]),
            certificate_format=fmt,  # type: ignore[arg-type]
            checker_capability=CheckerCapabilityRefV1.from_dict(
                data["checker_capability"]
            ),
            trust_domain=str(data["trust_domain"]),
            support_status=status,  # type: ignore[arg-type]
            schema=schema,
        )

    def content_digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


class VerifiedEncodingAdapter(Protocol):
    """Generic encoding interface (supported-subset, encode, witness map, refs)."""

    family: str
    version: str
    supported_features: frozenset[str]

    def supports(self, problem: BoundedProblemV1) -> SupportStatus: ...

    def encode(self, problem: BoundedProblemV1) -> EncodingResultV1: ...

    def decode_witness(
        self, problem: BoundedProblemV1, assignment: Mapping[str, int]
    ) -> dict[str, int] | None: ...

    def theorem_checker_refs(
        self, *, certificate_format: CertificateFormatName, trust_domain: str
    ) -> TheoremCheckerRefsV1: ...

    def problem_encoding_identity(
        self, problem: BoundedProblemV1, encoded: EncodedFormulaV1
    ) -> dict[str, str]: ...


def _bool_domains(problem: BoundedProblemV1) -> bool:
    if not problem.domains:
        return False
    for values in problem.domains.values():
        if tuple(sorted(values)) != (0, 1):
            return False
    return True


def _clause_vars_declared(problem: BoundedProblemV1) -> bool:
    declared = set(problem.domains)
    for clause in problem.clauses:
        for lit in clause:
            if lit.var not in declared:
                return False
    return True


def satisfies(problem: BoundedProblemV1, assignment: Mapping[str, int]) -> bool:
    """Evaluate original problem semantics under a total 0/1 assignment."""

    if set(assignment) != set(problem.domains):
        return False
    for var, val in assignment.items():
        if val not in problem.domains[var]:
            return False
    for clause in problem.clauses:
        if not clause:
            return False
        ok = False
        for lit in clause:
            bit = int(assignment[lit.var])
            if (bit == 1) == lit.positive:
                ok = True
                break
        if not ok:
            return False
    return True


def encoded_satisfiable(encoded: EncodedFormulaV1) -> bool:
    """Brute-force SAT over the tiny reference CNF (fixture scale only)."""

    n = len(encoded.var_order)
    if n > 12:
        raise ValueError("reference SAT enumerator capped at 12 vars")
    for mask in range(1 << n):
        assign = {
            var: 1 if (mask >> idx) & 1 else 0
            for idx, var in enumerate(encoded.var_order)
        }
        good = True
        for clause in encoded.clauses:
            if not clause:
                good = False
                break
            if any((assign[var] == 1) == pos for var, pos in clause):
                continue
            good = False
            break
        if good:
            return True
    return False


def exists_satisfying_assignment(problem: BoundedProblemV1) -> bool:
    vars_ = tuple(sorted(problem.domains))
    if len(vars_) > 12:
        raise ValueError("reference enumerator capped at 12 vars")
    n = len(vars_)
    for mask in range(1 << n):
        assign = {var: 1 if (mask >> idx) & 1 else 0 for idx, var in enumerate(vars_)}
        if satisfies(problem, assign):
            return True
    return False


@dataclass(frozen=True)
class CnfRefEncodingAdapter:
    """Reference bool-CNF identity encoding (promoted family for EVID-10)."""

    family: str = ENCODER_FAMILY_CNF_REF
    version: str = ENCODER_VERSION
    supported_features: frozenset[str] = SUPPORTED_FEATURES

    def supports(self, problem: BoundedProblemV1) -> SupportStatus:
        unknown_feats = problem.features - self.supported_features
        if unknown_feats:
            return "unsupported"
        required = {"bool_domain", "clause"}
        if not required <= problem.features:
            return "unknown"
        if not _bool_domains(problem) or not _clause_vars_declared(problem):
            return "unsupported"
        if any(len(clause) == 0 for clause in problem.clauses):
            return "unsupported"
        return "supported"

    def encode(self, problem: BoundedProblemV1) -> EncodingResultV1:
        status = self.supports(problem)
        if status != "supported":
            return EncodingResultV1(
                status=status,
                encoded=None,
                reason=(
                    "unsupported features or non-bool domains"
                    if status == "unsupported"
                    else "missing required feature tags"
                ),
            )
        var_order = tuple(sorted(problem.domains))
        clauses = tuple(
            tuple((lit.var, lit.positive) for lit in clause) for clause in problem.clauses
        )
        encoded = EncodedFormulaV1(
            family=self.family, var_order=var_order, clauses=clauses
        )
        return EncodingResultV1(status="supported", encoded=encoded, reason="")

    def decode_witness(
        self, problem: BoundedProblemV1, assignment: Mapping[str, int]
    ) -> dict[str, int] | None:
        """Witness map: encoded assignment → original assignment (identity)."""

        if self.supports(problem) != "supported":
            return None
        mapped = {var: int(assignment[var]) for var in problem.domains if var in assignment}
        if set(mapped) != set(problem.domains):
            return None
        if not satisfies(problem, mapped):
            return None
        return mapped

    def theorem_checker_refs(
        self,
        *,
        certificate_format: CertificateFormatName = "fixture_checked",
        trust_domain: str = "encoding_bridge/cnf_ref",
        checker_backend: str = "python_encoding_ref",
    ) -> TheoremCheckerRefsV1:
        return TheoremCheckerRefsV1(
            theorem_fqn=THEOREM_FQN,
            theorem_type_binding=THEOREM_TYPE_BINDING,
            certificate_format=certificate_format,
            checker_backend=checker_backend,
            trust_domain=trust_domain,
        )

    def problem_encoding_identity(
        self, problem: BoundedProblemV1, encoded: EncodedFormulaV1
    ) -> dict[str, str]:
        return {
            "problem_digest": problem.digest(),
            "encoded_digest": encoded.digest(),
            "encoder_version": self.version,
            "encoder_hash": encoder_source_hash(family=self.family, version=self.version),
            "family": self.family,
        }

    def build_evidence(
        self,
        problem: BoundedProblemV1,
        *,
        certificate_format: CertificateFormatName = "fixture_checked",
        checker_capability: CheckerCapabilityRefV1 | None = None,
        trust_domain: str = "encoding_bridge/cnf_ref",
    ) -> tuple[EncodingResultV1, VerifiedEncodingEvidenceV2 | None]:
        result = self.encode(problem)
        if result.status != "supported" or result.encoded is None:
            return result, None
        refs = self.theorem_checker_refs(
            certificate_format=certificate_format, trust_domain=trust_domain
        )
        capability = checker_capability or CheckerCapabilityRefV1(
            backend=refs.checker_backend,
            available=True,
            notes="reference CNF encoding bridge",
        )
        evidence = VerifiedEncodingEvidenceV2(
            original_problem_digest=problem.digest(),
            encoded_digest=result.encoded.digest(),
            encoder_version=self.version,
            encoder_hash=encoder_source_hash(family=self.family, version=self.version),
            theorem_fqn=refs.theorem_fqn,
            theorem_type_binding=refs.theorem_type_binding,
            certificate_format=certificate_format,
            checker_capability=capability,
            trust_domain=trust_domain,
            support_status="supported",
        )
        return result, evidence


def encoding_authorizes_semantic_result(
    evidence: VerifiedEncodingEvidenceV2 | None,
    *,
    expected_problem_digest: str,
    certificate_checked: bool,
    encoded_digest_observed: str | None = None,
    require_production_format: bool = True,
) -> bool:
    """Authority gate: valid cert + wrong/mutated encoding cannot authorize."""

    if not certificate_checked or evidence is None:
        return False
    if not evidence.binds_problem(expected_problem_digest):
        return False
    if evidence.encoder_version != ENCODER_VERSION:
        return False
    if evidence.encoder_hash != encoder_source_hash():
        return False
    if evidence.theorem_fqn != THEOREM_FQN:
        return False
    if evidence.theorem_type_binding != THEOREM_TYPE_BINDING:
        return False
    if require_production_format:
        if evidence.certificate_format not in PRODUCTION_CERTIFICATE_FORMATS:
            return False
    elif evidence.certificate_format not in CERTIFICATE_FORMATS:
        return False
    if (
        encoded_digest_observed is not None
        and encoded_digest_observed != evidence.encoded_digest
    ):
        return False
    return evidence.well_formed()


def mint_checked_certificate_via_encoding(
    binding: BindingIdsV1,
    *,
    encoding: VerifiedEncodingEvidenceV2,
    expected_problem_digest: str,
    certificate_checked: bool = True,
    encoded_digest_observed: str | None = None,
    require_production_format: bool = True,
) -> CheckedRefutationEvidenceV1 | None:
    """Mint EVID-09 checked_certificate evidence only when the encoding bridge holds."""

    if not encoding_authorizes_semantic_result(
        encoding,
        expected_problem_digest=expected_problem_digest,
        certificate_checked=certificate_checked,
        encoded_digest_observed=encoded_digest_observed,
        require_production_format=require_production_format,
    ):
        return None
    digest = _sha256(
        _canonical_json(
            {
                "binding": binding.to_dict(),
                "encoding": encoding.to_dict(),
                "expected_problem_digest": expected_problem_digest,
                "encoded_digest_observed": encoded_digest_observed,
            }
        )
    )
    return make_checked_certificate_evidence(
        binding,
        evidence_digest=digest,
        capability_present=encoding.checker_capability.available,
        trust_checked=True,
    )


def mutate_encoded_flip_first_literal(encoded: EncodedFormulaV1) -> EncodedFormulaV1:
    """Deliberate wrong-encoding mutation (flip polarity of first literal)."""

    if not encoded.clauses or not encoded.clauses[0]:
        raise ValueError("encoded formula has no literals to mutate")
    first = encoded.clauses[0]
    flipped = ((first[0][0], not first[0][1]),) + first[1:]
    return EncodedFormulaV1(
        family=encoded.family,
        var_order=encoded.var_order,
        clauses=(flipped,) + encoded.clauses[1:],
    )


def load_fixture_payload() -> dict[str, Any]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != ENCODING_FIXTURE_SCHEMA:
        raise ValueError(f"unexpected fixture schema: {payload.get('schema')!r}")
    return payload


def fixture_problems() -> dict[str, BoundedProblemV1]:
    payload = load_fixture_payload()
    return {
        str(case["id"]): BoundedProblemV1.from_dict(case["problem"])
        for case in payload["cases"]
        if "problem" in case
    }


def authority_source_digests(evidence: VerifiedEncodingEvidenceV2) -> dict[str, str]:
    """Digests to merge into FormalAuthorityV2.source_digests."""

    return {
        "encoding.original_problem": evidence.original_problem_digest,
        "encoding.encoded": evidence.encoded_digest,
        "encoding.encoder": evidence.encoder_hash,
        "encoding.evidence": evidence.content_digest(),
    }


def attach_encoding_evidence_fields(
    *,
    encoding: VerifiedEncodingEvidenceV2,
    source_digests: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Helper for FormalAuthorityV2 builders — digests + optional evidence blob."""

    merged = dict(source_digests or {})
    merged.update(authority_source_digests(encoding))
    return merged, encoding.to_dict()


__all__ = [
    "CERTIFICATE_FORMATS",
    "ENCODING_ADAPTER_SCHEMA",
    "ENCODING_EVIDENCE_SCHEMA",
    "ENCODING_FIXTURE_SCHEMA",
    "ENCODER_FAMILY_CNF_REF",
    "ENCODER_VERSION",
    "PRODUCTION_CERTIFICATE_FORMATS",
    "SUPPORTED_FEATURES",
    "THEOREM_FQN",
    "THEOREM_TYPE_BINDING",
    "BoundedProblemV1",
    "CnfRefEncodingAdapter",
    "EncodedFormulaV1",
    "EncodingResultV1",
    "LiteralV1",
    "TheoremCheckerRefsV1",
    "VerifiedEncodingAdapter",
    "VerifiedEncodingEvidenceV2",
    "attach_encoding_evidence_fields",
    "authority_source_digests",
    "encoded_satisfiable",
    "encoder_source_hash",
    "encoding_authorizes_semantic_result",
    "exists_satisfying_assignment",
    "fixture_problems",
    "load_fixture_payload",
    "mint_checked_certificate_via_encoding",
    "mutate_encoded_flip_first_literal",
    "satisfies",
]
