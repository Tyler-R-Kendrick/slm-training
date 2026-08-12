"""RESEARCH-06 / SLM-564: default-off proof-producing VSS PB backend (PBLean pilot).

Isolated research adapter. Uses the EVID-10 verified-encoding bridge
(``CnfRefEncodingAdapter`` + ``pblean_pilot``) and never becomes a production
decode / ship-gate / serving dependency.

Control authority remains exhaustive VSS/CNF replay. Treatment emits a
deterministic CNF→pseudo-Boolean encoding plus checked VeriPB/PBLean-style
certificate for the declared supported unsat subset. When instances are also
in the RESEARCH-05 LRAT-supported subset, paired reports include an LRAT
comparison arm. Toolchain failure / timeout / unsupported features return
``unknown`` and preserve candidates (EVID-09).

External PBLean is optional — this pilot ships a hermetic Python checker stub
that still exercises the encoding adapter, mutation rejection, and paired
timing metrics. Missing external tools never fabricate success.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from slm_training.formal.encoding_adapter import (
    ENCODER_VERSION,
    BoundedProblemV1,
    CnfRefEncodingAdapter,
    EncodedFormulaV1,
    LiteralV1,
    encoded_satisfiable,
    encoding_authorizes_semantic_result,
    encoder_source_hash,
    exists_satisfying_assignment,
    mutate_encoded_flip_first_literal,
)
from slm_training.formal.vss_lrat_backend import (
    check_lrat_certificate as _check_lrat_certificate,
    generate_lrat_certificate as _generate_lrat_certificate,
    supports_lrat_subset as _supports_lrat_subset,
)

BACKEND_ID = "vss_pblean_pb_pilot"
CERTIFICATE_SCHEMA = "vss_pblean_certificate/v1"
TOOLCHAIN_ID = "hermetic_python_pblean_pilot_v1"
MAX_EXHAUSTIVE_VARS = 12

OutcomeKind = Literal["sat", "unsat", "unknown"]
SupportStatus = Literal["supported", "unsupported", "unknown"]


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


def pinned_toolchain() -> dict[str, str]:
    """Pinned identities for encoding + PBLean pilot checker (no external PB deps)."""

    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "encoder_family": "cnf_ref",
        "encoder_version": ENCODER_VERSION,
        "encoder_hash": encoder_source_hash(),
        "certificate_format": "pblean_pilot",
        "certificate_schema": CERTIFICATE_SCHEMA,
        "checker_backend": "python_pblean_veripb_pilot",
        "trust_domain": "encoding_bridge/cnf_ref+pblean_pilot",
        "external_pblean": "optional_unavailable_fixture_stub",
    }


@dataclass(frozen=True)
class PbConstraintV1:
    """Pseudo-Boolean inequality: sum(weight_i * lit_i) >= degree."""

    constraint_id: int
    terms: tuple[tuple[str, bool, int], ...]  # (var, positive, weight>0)
    degree: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "terms": [
                {"var": v, "positive": p, "weight": w} for v, p, w in self.terms
            ],
            "degree": self.degree,
        }


@dataclass(frozen=True)
class PbFormulaV1:
    """Deterministic CNF→PB encoding (each clause → sum lits >= 1)."""

    var_order: tuple[str, ...]
    constraints: tuple[PbConstraintV1, ...]
    source_encoded_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "var_order": list(self.var_order),
            "constraints": [c.to_dict() for c in self.constraints],
            "source_encoded_digest": self.source_encoded_digest,
        }

    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class PbLeanStepV1:
    """Pilot VeriPB-style step: derive contradiction from hint constraint ids."""

    step_id: int
    derived_degree: int  # empty / false constraint uses degree 1 with no terms
    hints: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "derived_degree": self.derived_degree,
            "hints": list(self.hints),
            "terms": [],  # pilot only derives ⊥ (0 >= 1)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PbLeanStepV1:
        return cls(
            step_id=int(data["step_id"]),
            derived_degree=int(data["derived_degree"]),
            hints=tuple(int(h) for h in data.get("hints", ())),
        )


@dataclass(frozen=True)
class PbLeanCertificateV1:
    schema: str
    problem_digest: str
    encoded_digest: str
    pb_digest: str
    encoder_hash: str
    toolchain: dict[str, str]
    steps: tuple[PbLeanStepV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "problem_digest": self.problem_digest,
            "encoded_digest": self.encoded_digest,
            "pb_digest": self.pb_digest,
            "encoder_hash": self.encoder_hash,
            "toolchain": dict(self.toolchain),
            "steps": [step.to_dict() for step in self.steps],
            "content_digest": self.content_digest(),
        }

    def content_digest(self) -> str:
        payload = {
            "schema": self.schema,
            "problem_digest": self.problem_digest,
            "encoded_digest": self.encoded_digest,
            "pb_digest": self.pb_digest,
            "encoder_hash": self.encoder_hash,
            "toolchain": self.toolchain,
            "steps": [step.to_dict() for step in self.steps],
        }
        return _sha256(_canonical_json(payload))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PbLeanCertificateV1:
        steps = tuple(PbLeanStepV1.from_dict(s) for s in data.get("steps", ()))
        return cls(
            schema=str(data["schema"]),
            problem_digest=str(data["problem_digest"]),
            encoded_digest=str(data["encoded_digest"]),
            pb_digest=str(data["pb_digest"]),
            encoder_hash=str(data["encoder_hash"]),
            toolchain={str(k): str(v) for k, v in dict(data.get("toolchain", {})).items()},
            steps=steps,
        )


@dataclass(frozen=True)
class BackendVerdictV1:
    outcome: OutcomeKind
    reason: str
    elapsed_s: float
    certificate: PbLeanCertificateV1 | None = None
    encoding_digest: str | None = None
    pb_digest: str | None = None
    identities: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "outcome": self.outcome,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "encoding_digest": self.encoding_digest,
            "pb_digest": self.pb_digest,
            "identities": self.identities or {},
        }
        if self.certificate is not None:
            out["certificate"] = self.certificate.to_dict()
        return out


def encode_cnf_to_pb(encoded: EncodedFormulaV1) -> PbFormulaV1:
    """Deterministic clause→PB map used by the RESEARCH-06 treatment arm."""

    constraints: list[PbConstraintV1] = []
    for idx, clause in enumerate(encoded.clauses, start=1):
        terms = tuple((var, pos, 1) for var, pos in clause)
        constraints.append(
            PbConstraintV1(constraint_id=idx, terms=terms, degree=1)
        )
    return PbFormulaV1(
        var_order=encoded.var_order,
        constraints=tuple(constraints),
        source_encoded_digest=encoded.digest(),
    )


def _unit_conflict_pair(
    encoded: EncodedFormulaV1,
) -> tuple[int, int, str] | None:
    pos: dict[str, int] = {}
    neg: dict[str, int] = {}
    for idx, clause in enumerate(encoded.clauses, start=1):
        if len(clause) != 1:
            continue
        var, positive = clause[0]
        if positive:
            pos[var] = idx
        else:
            neg[var] = idx
    for var, pos_id in pos.items():
        if var in neg:
            return pos_id, neg[var], var
    return None


def supports_pblean_subset(problem: BoundedProblemV1) -> SupportStatus:
    adapter = CnfRefEncodingAdapter()
    status = adapter.supports(problem)
    if status != "supported":
        return status
    result = adapter.encode(problem)
    if result.encoded is None:
        return "unsupported"
    if len(result.encoded.var_order) > MAX_EXHAUSTIVE_VARS:
        return "unsupported"
    if _unit_conflict_pair(result.encoded) is None:
        if exists_satisfying_assignment(problem):
            return "supported"
        return "unknown"
    return "supported"


def exhaustive_replay(problem: BoundedProblemV1) -> BackendVerdictV1:
    """Control: canonical exhaustive VSS/CNF replay (fixture-scale cap)."""

    started = time.perf_counter()
    adapter = CnfRefEncodingAdapter()
    status = adapter.supports(problem)
    if status != "supported":
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_replay_{status}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
        )
    try:
        sat = exists_satisfying_assignment(problem)
    except ValueError as exc:
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_replay_tool_failure:{exc}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
        )
    enc = adapter.encode(problem)
    if enc.encoded is None:
        return BackendVerdictV1(
            outcome="unknown",
            reason="exhaustive_replay_encode_failed",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
        )
    try:
        sat_enc = encoded_satisfiable(enc.encoded)
    except ValueError as exc:
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_replay_tool_failure:{exc}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
        )
    if sat != sat_enc:
        return BackendVerdictV1(
            outcome="unknown",
            reason="exhaustive_replay_encode_disagreement",
            elapsed_s=time.perf_counter() - started,
            encoding_digest=enc.encoded.digest(),
            identities=pinned_toolchain(),
        )
    return BackendVerdictV1(
        outcome="sat" if sat else "unsat",
        reason="exhaustive_replay_ok",
        elapsed_s=time.perf_counter() - started,
        encoding_digest=enc.encoded.digest(),
        identities={
            **pinned_toolchain(),
            "problem_digest": problem.digest(),
            "encoded_digest": enc.encoded.digest(),
        },
    )


def generate_pblean_certificate(
    problem: BoundedProblemV1,
) -> tuple[BackendVerdictV1, EncodedFormulaV1 | None, PbFormulaV1 | None]:
    """Cold treatment: EVID-10 encode + emit PBLean/VeriPB-style unit conflict cert."""

    started = time.perf_counter()
    adapter = CnfRefEncodingAdapter()
    status = supports_pblean_subset(problem)
    if status != "supported":
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason=f"pblean_generate_{status}",
                elapsed_s=time.perf_counter() - started,
                identities=pinned_toolchain(),
            ),
            None,
            None,
        )
    result, evidence = adapter.build_evidence(
        problem, certificate_format="pblean_pilot"
    )
    if result.encoded is None or evidence is None:
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason="pblean_generate_encode_failed",
                elapsed_s=time.perf_counter() - started,
                identities=pinned_toolchain(),
            ),
            None,
            None,
        )
    pb = encode_cnf_to_pb(result.encoded)
    if exists_satisfying_assignment(problem):
        return (
            BackendVerdictV1(
                outcome="sat",
                reason="pblean_generate_sat_no_refutation",
                elapsed_s=time.perf_counter() - started,
                encoding_digest=result.encoded.digest(),
                pb_digest=pb.digest(),
                identities={
                    **pinned_toolchain(),
                    "problem_digest": problem.digest(),
                    "encoded_digest": result.encoded.digest(),
                    "pb_digest": pb.digest(),
                    "encoder_hash": evidence.encoder_hash,
                },
            ),
            result.encoded,
            pb,
        )
    conflict = _unit_conflict_pair(result.encoded)
    if conflict is None:
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason="pblean_generate_no_unit_conflict",
                elapsed_s=time.perf_counter() - started,
                encoding_digest=result.encoded.digest(),
                pb_digest=pb.digest(),
                identities=pinned_toolchain(),
            ),
            result.encoded,
            pb,
        )
    pos_id, neg_id, _var = conflict
    empty_id = len(pb.constraints) + 1
    cert = PbLeanCertificateV1(
        schema=CERTIFICATE_SCHEMA,
        problem_digest=problem.digest(),
        encoded_digest=result.encoded.digest(),
        pb_digest=pb.digest(),
        encoder_hash=evidence.encoder_hash,
        toolchain=pinned_toolchain(),
        steps=(
            PbLeanStepV1(step_id=empty_id, derived_degree=1, hints=(pos_id, neg_id)),
        ),
    )
    check = check_pblean_certificate(problem, result.encoded, pb, cert)
    elapsed = time.perf_counter() - started
    if check.outcome != "unsat":
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason=f"pblean_generate_check_failed:{check.reason}",
                elapsed_s=elapsed,
                encoding_digest=result.encoded.digest(),
                pb_digest=pb.digest(),
                identities=pinned_toolchain(),
            ),
            result.encoded,
            pb,
        )
    return (
        BackendVerdictV1(
            outcome="unsat",
            reason="pblean_generate_ok",
            elapsed_s=elapsed,
            certificate=cert,
            encoding_digest=result.encoded.digest(),
            pb_digest=pb.digest(),
            identities={
                **pinned_toolchain(),
                "problem_digest": problem.digest(),
                "encoded_digest": result.encoded.digest(),
                "pb_digest": pb.digest(),
                "certificate_digest": cert.content_digest(),
                "encoder_hash": evidence.encoder_hash,
            },
        ),
        result.encoded,
        pb,
    )


def check_pblean_certificate(
    problem: BoundedProblemV1,
    encoded: EncodedFormulaV1,
    pb: PbFormulaV1,
    certificate: PbLeanCertificateV1,
) -> BackendVerdictV1:
    """Warm treatment: re-check an existing PBLean-style certificate (no enum)."""

    started = time.perf_counter()
    identities = pinned_toolchain()
    if certificate.schema != CERTIFICATE_SCHEMA:
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_schema_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.problem_digest != problem.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_stale_problem",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.encoded_digest != encoded.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_encoded_digest_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.pb_digest != pb.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_pb_digest_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if pb.source_encoded_digest != encoded.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_pb_source_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.encoder_hash != encoder_source_hash():
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_encoder_hash_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.toolchain.get("toolchain_id") != TOOLCHAIN_ID:
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_toolchain_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )

    adapter = CnfRefEncodingAdapter()
    result, evidence = adapter.build_evidence(
        problem, certificate_format="pblean_pilot"
    )
    if (
        result.encoded is None
        or evidence is None
        or result.encoded.digest() != encoded.digest()
    ):
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_encoding_bridge_failed",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if not encoding_authorizes_semantic_result(
        evidence,
        expected_problem_digest=problem.digest(),
        certificate_checked=True,
        encoded_digest_observed=encoded.digest(),
        require_production_format=False,
    ):
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_encoding_not_authorized",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )

    db: dict[int, PbConstraintV1] = {c.constraint_id: c for c in pb.constraints}
    derived_false = False
    for step in certificate.steps:
        if step.step_id in db:
            return BackendVerdictV1(
                outcome="unknown",
                reason="pblean_check_duplicate_step_id",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        if not step.hints or len(step.hints) != 2:
            return BackendVerdictV1(
                outcome="unknown",
                reason="pblean_check_hint_arity",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        if step.derived_degree != 1:
            return BackendVerdictV1(
                outcome="unknown",
                reason="pblean_check_non_false_derivation_unsupported",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        c1 = db.get(step.hints[0])
        c2 = db.get(step.hints[1])
        if c1 is None or c2 is None:
            return BackendVerdictV1(
                outcome="unknown",
                reason="pblean_check_hint_missing",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        # Pilot VeriPB RUP: two opposing unit PB constraints (1 x >= 1, 1 ~x >= 1).
        if len(c1.terms) != 1 or len(c2.terms) != 1 or c1.degree != 1 or c2.degree != 1:
            return BackendVerdictV1(
                outcome="unknown",
                reason="pblean_check_hint_not_unit",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        (v1, p1, w1), (v2, p2, w2) = c1.terms[0], c2.terms[0]
        if w1 != 1 or w2 != 1 or v1 != v2 or p1 == p2:
            return BackendVerdictV1(
                outcome="unknown",
                reason="pblean_check_not_opposing_units",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        db[step.step_id] = PbConstraintV1(
            constraint_id=step.step_id, terms=(), degree=1
        )
        derived_false = True

    if not derived_false:
        return BackendVerdictV1(
            outcome="unknown",
            reason="pblean_check_no_false_derived",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    return BackendVerdictV1(
        outcome="unsat",
        reason="pblean_check_ok",
        elapsed_s=time.perf_counter() - started,
        certificate=certificate,
        encoding_digest=encoded.digest(),
        pb_digest=pb.digest(),
        identities={
            **identities,
            "problem_digest": problem.digest(),
            "encoded_digest": encoded.digest(),
            "pb_digest": pb.digest(),
            "certificate_digest": certificate.content_digest(),
        },
    )


def mutate_certificate_flip_hint(
    certificate: PbLeanCertificateV1,
) -> PbLeanCertificateV1:
    if not certificate.steps:
        raise ValueError("certificate has no steps to mutate")
    first = certificate.steps[0]
    if not first.hints:
        raise ValueError("certificate step has no hints")
    bad_hints = (first.hints[0] + 99,) + first.hints[1:]
    mutated = PbLeanStepV1(
        step_id=first.step_id,
        derived_degree=first.derived_degree,
        hints=bad_hints,
    )
    return PbLeanCertificateV1(
        schema=certificate.schema,
        problem_digest=certificate.problem_digest,
        encoded_digest=certificate.encoded_digest,
        pb_digest=certificate.pb_digest,
        encoder_hash=certificate.encoder_hash,
        toolchain=certificate.toolchain,
        steps=(mutated,) + certificate.steps[1:],
    )


def mutation_rejection_suite(problem: BoundedProblemV1) -> dict[str, Any]:
    cold, encoded, pb = generate_pblean_certificate(problem)
    results: list[dict[str, Any]] = []
    if cold.outcome != "unsat" or cold.certificate is None or encoded is None or pb is None:
        return {
            "ok": False,
            "reason": "suite_requires_unsat_certificate",
            "cases": results,
            "rejection_rate": 0.0,
        }

    mutated_enc = mutate_encoded_flip_first_literal(encoded)
    mutated_pb = encode_cnf_to_pb(mutated_enc)
    enc_check = check_pblean_certificate(
        problem, mutated_enc, mutated_pb, cold.certificate
    )
    results.append(
        {
            "case_id": "encoding_flip_literal",
            "rejected": enc_check.outcome != "unsat",
            "outcome": enc_check.outcome,
            "reason": enc_check.reason,
        }
    )

    mutated_cert = mutate_certificate_flip_hint(cold.certificate)
    cert_check = check_pblean_certificate(problem, encoded, pb, mutated_cert)
    results.append(
        {
            "case_id": "certificate_hint_corrupt",
            "rejected": cert_check.outcome != "unsat",
            "outcome": cert_check.outcome,
            "reason": cert_check.reason,
        }
    )

    stale = PbLeanCertificateV1(
        schema=cold.certificate.schema,
        problem_digest="0" * 64,
        encoded_digest=cold.certificate.encoded_digest,
        pb_digest=cold.certificate.pb_digest,
        encoder_hash=cold.certificate.encoder_hash,
        toolchain=cold.certificate.toolchain,
        steps=cold.certificate.steps,
    )
    stale_check = check_pblean_certificate(problem, encoded, pb, stale)
    results.append(
        {
            "case_id": "stale_problem_digest",
            "rejected": stale_check.outcome != "unsat",
            "outcome": stale_check.outcome,
            "reason": stale_check.reason,
        }
    )

    rejected = sum(1 for row in results if row["rejected"])
    return {
        "ok": rejected == len(results),
        "reason": "all_mutations_rejected" if rejected == len(results) else "mutation_miss",
        "cases": results,
        "rejection_rate": rejected / len(results),
    }


def pad_unit_conflict_problem(
    *,
    problem_id: str,
    conflict_var: str = "x0",
    pad_vars: int = 10,
) -> BoundedProblemV1:
    if pad_vars < 0 or pad_vars > MAX_EXHAUSTIVE_VARS - 1:
        raise ValueError("pad_vars out of exhaustive range")
    domains = {conflict_var: (0, 1)}
    for idx in range(1, pad_vars + 1):
        domains[f"x{idx}"] = (0, 1)
    clauses = (
        (LiteralV1(conflict_var, True),),
        (LiteralV1(conflict_var, False),),
    )
    return BoundedProblemV1(
        problem_id=problem_id,
        domains=domains,
        clauses=clauses,
        features=frozenset({"bool_domain", "clause"}),
    )


def default_unsat_suite() -> tuple[BoundedProblemV1, ...]:
    """Frozen unsatisfiable suite for RESEARCH-06 paired timing (+ R05-comparable)."""

    return (
        pad_unit_conflict_problem(problem_id="research06/unsat_pad0", pad_vars=0),
        pad_unit_conflict_problem(problem_id="research06/unsat_pad4", pad_vars=4),
        pad_unit_conflict_problem(problem_id="research06/unsat_pad8", pad_vars=8),
        pad_unit_conflict_problem(problem_id="research06/unsat_pad10", pad_vars=10),
    )


def paired_warm_trials(
    problems: Sequence[BoundedProblemV1],
    *,
    warm_repeats: int = 5,
) -> dict[str, Any]:
    """Paired exhaustive vs warm-PBLean timings + optional LRAT comparison."""

    rows: list[dict[str, Any]] = []
    disagreements = 0
    ratios: list[float] = []
    cold_costs: list[float] = []
    cert_sizes: list[int] = []
    pb_sizes: list[int] = []
    covered = 0
    lrat_comparable = 0
    lrat_agree = 0

    for problem in problems:
        control = exhaustive_replay(problem)
        cold, encoded, pb = generate_pblean_certificate(problem)
        if cold.outcome == "unknown" or cold.certificate is None or encoded is None or pb is None:
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "status": "unknown",
                    "control": control.to_dict(),
                    "cold": cold.to_dict(),
                    "preserved_candidate": True,
                }
            )
            continue
        covered += 1
        if control.outcome != cold.outcome:
            disagreements += 1
        warm_times: list[float] = []
        warm_ok = True
        for _ in range(max(1, warm_repeats)):
            warm = check_pblean_certificate(problem, encoded, pb, cold.certificate)
            warm_times.append(warm.elapsed_s)
            if warm.outcome != "unsat":
                warm_ok = False
                disagreements += 1
        warm_median = sorted(warm_times)[len(warm_times) // 2]
        ratio = (
            warm_median / control.elapsed_s
            if control.elapsed_s > 0
            else float("inf")
        )
        ratios.append(ratio)
        cold_costs.append(cold.elapsed_s)
        cert_sizes.append(len(_canonical_json(cold.certificate.to_dict())))
        pb_sizes.append(len(_canonical_json(pb.to_dict())))

        lrat_cmp: dict[str, Any] | None = None
        if _supports_lrat_subset(problem) == "supported":
            lrat_comparable += 1
            lrat_cold, lrat_enc = _generate_lrat_certificate(problem)
            if (
                lrat_cold.outcome == "unsat"
                and lrat_cold.certificate is not None
                and lrat_enc is not None
            ):
                lrat_warm = _check_lrat_certificate(
                    problem, lrat_enc, lrat_cold.certificate
                )
                agree = lrat_warm.outcome == cold.outcome == "unsat"
                if agree:
                    lrat_agree += 1
                else:
                    disagreements += 1
                lrat_cmp = {
                    "comparable": True,
                    "lrat_outcome": lrat_warm.outcome,
                    "agree_with_pblean": agree,
                    "lrat_warm_elapsed_s": lrat_warm.elapsed_s,
                    "pblean_warm_median_s": warm_median,
                }
            else:
                lrat_cmp = {
                    "comparable": True,
                    "lrat_outcome": lrat_cold.outcome,
                    "agree_with_pblean": False,
                    "preserved_candidate": True,
                }
                disagreements += 1

        rows.append(
            {
                "problem_id": problem.problem_id,
                "status": "ok" if warm_ok and control.outcome == "unsat" else "disagree",
                "control_elapsed_s": control.elapsed_s,
                "cold_elapsed_s": cold.elapsed_s,
                "warm_median_s": warm_median,
                "warm_over_exhaustive_ratio": ratio,
                "certificate_bytes": cert_sizes[-1],
                "pb_encoding_bytes": pb_sizes[-1],
                "lrat_comparison": lrat_cmp,
                "identities": cold.identities,
            }
        )

    mutation = mutation_rejection_suite(problems[-1])
    ratios_sorted = sorted(ratios)
    median_ratio = (
        ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else None
    )
    correctness_ok = disagreements == 0 and mutation["ok"] and covered == len(problems)
    decision = (
        "accept"
        if correctness_ok and median_ratio is not None and median_ratio < 1.0
        else ("reject" if covered else "unknown")
    )
    exact_agreement = (
        1.0 if covered and disagreements == 0 else (0.0 if covered else None)
    )
    return {
        "schema": "research_06_paired_warm_report/v1",
        "covered": covered,
        "suite_n": len(problems),
        "supported_subset_coverage": covered / len(problems) if problems else 0.0,
        "witness_disagreement_count": disagreements,
        "exact_semantic_agreement_rate_on_supported_pb_subset": exact_agreement,
        "mutation_rejection_rate": mutation["rejection_rate"],
        "mutation": mutation,
        "median_paired_warm_pblean_check_over_exhaustive_replay_time_ratio": median_ratio,
        "lrat_comparable_n": lrat_comparable,
        "lrat_agreement_n": lrat_agree,
        "secondary": {
            "cold_elapsed_s_median": (
                sorted(cold_costs)[len(cold_costs) // 2] if cold_costs else None
            ),
            "certificate_bytes_median": (
                sorted(cert_sizes)[len(cert_sizes) // 2] if cert_sizes else None
            ),
            "pb_encoding_bytes_median": (
                sorted(pb_sizes)[len(pb_sizes) // 2] if pb_sizes else None
            ),
        },
        "correctness_ok": correctness_ok,
        "decision": decision,
        "rows": rows,
        "toolchain": pinned_toolchain(),
    }


__all__ = [
    "BACKEND_ID",
    "CERTIFICATE_SCHEMA",
    "PbLeanCertificateV1",
    "BackendVerdictV1",
    "check_pblean_certificate",
    "default_unsat_suite",
    "encode_cnf_to_pb",
    "exhaustive_replay",
    "generate_pblean_certificate",
    "mutation_rejection_suite",
    "pad_unit_conflict_problem",
    "paired_warm_trials",
    "pinned_toolchain",
    "supports_pblean_subset",
]
