"""RESEARCH-05 / SLM-563: default-off proof-producing VSS SAT backend (LRAT pilot).

Isolated research adapter. Uses the EVID-10 verified-encoding bridge
(``CnfRefEncodingAdapter`` + ``lrat_pilot``) and never becomes a production
decode / ship-gate / serving dependency.

Control authority remains exhaustive VSS/CNF replay. Treatment emits a
deterministic CNF + checked LRAT-style RUP certificate for the declared
supported unsat subset. Toolchain failure / timeout / unsupported features
return ``unknown`` and preserve candidates (EVID-09).
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

BACKEND_ID = "vss_lrat_sat_pilot"
CERTIFICATE_SCHEMA = "vss_lrat_certificate/v1"
TOOLCHAIN_ID = "hermetic_python_lrat_pilot_v1"
# Fixture-scale enumerator ceiling shared with encoding_adapter.
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
    """Pinned identities for encoding + LRAT pilot checker (no external SAT deps)."""

    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "encoder_family": "cnf_ref",
        "encoder_version": ENCODER_VERSION,
        "encoder_hash": encoder_source_hash(),
        "certificate_format": "lrat_pilot",
        "certificate_schema": CERTIFICATE_SCHEMA,
        "checker_backend": "python_lrat_rup_pilot",
        "trust_domain": "encoding_bridge/cnf_ref+lrat_pilot",
    }


@dataclass(frozen=True)
class LratStepV1:
    clause_id: int
    literals: tuple[tuple[str, bool], ...]  # empty ⇒ derived ⊥
    hints: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "literals": [{"var": v, "positive": p} for v, p in self.literals],
            "hints": list(self.hints),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LratStepV1:
        lits = tuple(
            (str(item["var"]), bool(item["positive"]))
            for item in data.get("literals", ())
        )
        hints = tuple(int(h) for h in data.get("hints", ()))
        return cls(clause_id=int(data["clause_id"]), literals=lits, hints=hints)


@dataclass(frozen=True)
class LratCertificateV1:
    schema: str
    problem_digest: str
    encoded_digest: str
    encoder_hash: str
    toolchain: dict[str, str]
    steps: tuple[LratStepV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "problem_digest": self.problem_digest,
            "encoded_digest": self.encoded_digest,
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
            "encoder_hash": self.encoder_hash,
            "toolchain": self.toolchain,
            "steps": [step.to_dict() for step in self.steps],
        }
        return _sha256(_canonical_json(payload))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LratCertificateV1:
        steps = tuple(LratStepV1.from_dict(s) for s in data.get("steps", ()))
        return cls(
            schema=str(data["schema"]),
            problem_digest=str(data["problem_digest"]),
            encoded_digest=str(data["encoded_digest"]),
            encoder_hash=str(data["encoder_hash"]),
            toolchain={str(k): str(v) for k, v in dict(data.get("toolchain", {})).items()},
            steps=steps,
        )


@dataclass(frozen=True)
class BackendVerdictV1:
    outcome: OutcomeKind
    reason: str
    elapsed_s: float
    certificate: LratCertificateV1 | None = None
    encoding_digest: str | None = None
    identities: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "outcome": self.outcome,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "encoding_digest": self.encoding_digest,
            "identities": self.identities or {},
        }
        if self.certificate is not None:
            out["certificate"] = self.certificate.to_dict()
        return out


def _unit_conflict_pair(
    encoded: EncodedFormulaV1,
) -> tuple[int, int, str] | None:
    """Return (pos_clause_id, neg_clause_id, var) for a unit x / ¬x conflict."""

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


def supports_lrat_subset(problem: BoundedProblemV1) -> SupportStatus:
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
        # Pilot subset: only unit-conflict unsat certificates are emitted.
        # Sat / other unsat shapes stay unknown so candidates are preserved.
        if exists_satisfying_assignment(problem):
            return "supported"  # sat path uses exhaustive agreement only
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


def generate_lrat_certificate(
    problem: BoundedProblemV1,
) -> tuple[BackendVerdictV1, EncodedFormulaV1 | None]:
    """Cold treatment path: encode via EVID-10 + emit LRAT RUP for unit conflict."""

    started = time.perf_counter()
    adapter = CnfRefEncodingAdapter()
    status = supports_lrat_subset(problem)
    if status != "supported":
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason=f"lrat_generate_{status}",
                elapsed_s=time.perf_counter() - started,
                identities=pinned_toolchain(),
            ),
            None,
        )
    result, evidence = adapter.build_evidence(
        problem, certificate_format="lrat_pilot"
    )
    if result.encoded is None or evidence is None:
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason="lrat_generate_encode_failed",
                elapsed_s=time.perf_counter() - started,
                identities=pinned_toolchain(),
            ),
            None,
        )
    if exists_satisfying_assignment(problem):
        # Sat: no LRAT refutation; agreement is via encoding bridge only.
        return (
            BackendVerdictV1(
                outcome="sat",
                reason="lrat_generate_sat_no_refutation",
                elapsed_s=time.perf_counter() - started,
                encoding_digest=result.encoded.digest(),
                identities={
                    **pinned_toolchain(),
                    "problem_digest": problem.digest(),
                    "encoded_digest": result.encoded.digest(),
                    "encoder_hash": evidence.encoder_hash,
                },
            ),
            result.encoded,
        )
    conflict = _unit_conflict_pair(result.encoded)
    if conflict is None:
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason="lrat_generate_no_unit_conflict",
                elapsed_s=time.perf_counter() - started,
                encoding_digest=result.encoded.digest(),
                identities=pinned_toolchain(),
            ),
            result.encoded,
        )
    pos_id, neg_id, _var = conflict
    empty_id = len(result.encoded.clauses) + 1
    cert = LratCertificateV1(
        schema=CERTIFICATE_SCHEMA,
        problem_digest=problem.digest(),
        encoded_digest=result.encoded.digest(),
        encoder_hash=evidence.encoder_hash,
        toolchain=pinned_toolchain(),
        steps=(
            LratStepV1(clause_id=empty_id, literals=(), hints=(pos_id, neg_id)),
        ),
    )
    check = check_lrat_certificate(problem, result.encoded, cert)
    elapsed = time.perf_counter() - started
    if check.outcome != "unsat":
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason=f"lrat_generate_check_failed:{check.reason}",
                elapsed_s=elapsed,
                encoding_digest=result.encoded.digest(),
                identities=pinned_toolchain(),
            ),
            result.encoded,
        )
    return (
        BackendVerdictV1(
            outcome="unsat",
            reason="lrat_generate_ok",
            elapsed_s=elapsed,
            certificate=cert,
            encoding_digest=result.encoded.digest(),
            identities={
                **pinned_toolchain(),
                "problem_digest": problem.digest(),
                "encoded_digest": result.encoded.digest(),
                "certificate_digest": cert.content_digest(),
                "encoder_hash": evidence.encoder_hash,
            },
        ),
        result.encoded,
    )


def check_lrat_certificate(
    problem: BoundedProblemV1,
    encoded: EncodedFormulaV1,
    certificate: LratCertificateV1,
) -> BackendVerdictV1:
    """Warm treatment path: re-check an existing LRAT certificate (no enumeration)."""

    started = time.perf_counter()
    identities = pinned_toolchain()
    if certificate.schema != CERTIFICATE_SCHEMA:
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_schema_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.problem_digest != problem.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_stale_problem",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.encoded_digest != encoded.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_encoded_digest_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.encoder_hash != encoder_source_hash():
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_encoder_hash_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    if certificate.toolchain.get("toolchain_id") != TOOLCHAIN_ID:
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_toolchain_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )

    adapter = CnfRefEncodingAdapter()
    result, evidence = adapter.build_evidence(
        problem, certificate_format="lrat_pilot"
    )
    if (
        result.encoded is None
        or evidence is None
        or result.encoded.digest() != encoded.digest()
    ):
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_encoding_bridge_failed",
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
            reason="lrat_check_encoding_not_authorized",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )

    db: dict[int, tuple[tuple[str, bool], ...]] = {
        idx: clause for idx, clause in enumerate(encoded.clauses, start=1)
    }
    derived_empty = False
    for step in certificate.steps:
        if step.clause_id in db:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_duplicate_clause_id",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        if not step.hints:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_missing_hints",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        # Pilot RUP: empty clause from two opposing unit hints.
        if step.literals:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_nonempty_addition_unsupported",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        if len(step.hints) != 2:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_hint_arity",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        c1 = db.get(step.hints[0])
        c2 = db.get(step.hints[1])
        if c1 is None or c2 is None:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_hint_missing",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        if len(c1) != 1 or len(c2) != 1:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_hint_not_unit",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        (v1, p1), (v2, p2) = c1[0], c2[0]
        if v1 != v2 or p1 == p2:
            return BackendVerdictV1(
                outcome="unknown",
                reason="lrat_check_not_opposing_units",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
            )
        db[step.clause_id] = ()
        derived_empty = True

    if not derived_empty:
        return BackendVerdictV1(
            outcome="unknown",
            reason="lrat_check_no_empty_derived",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
        )
    return BackendVerdictV1(
        outcome="unsat",
        reason="lrat_check_ok",
        elapsed_s=time.perf_counter() - started,
        certificate=certificate,
        encoding_digest=encoded.digest(),
        identities={
            **identities,
            "problem_digest": problem.digest(),
            "encoded_digest": encoded.digest(),
            "certificate_digest": certificate.content_digest(),
        },
    )


def mutate_certificate_flip_hint(
    certificate: LratCertificateV1,
) -> LratCertificateV1:
    """Deliberate certificate mutation (corrupt first hint id)."""

    if not certificate.steps:
        raise ValueError("certificate has no steps to mutate")
    first = certificate.steps[0]
    if not first.hints:
        raise ValueError("certificate step has no hints")
    bad_hints = (first.hints[0] + 99,) + first.hints[1:]
    mutated = LratStepV1(
        clause_id=first.clause_id, literals=first.literals, hints=bad_hints
    )
    return LratCertificateV1(
        schema=certificate.schema,
        problem_digest=certificate.problem_digest,
        encoded_digest=certificate.encoded_digest,
        encoder_hash=certificate.encoder_hash,
        toolchain=certificate.toolchain,
        steps=(mutated,) + certificate.steps[1:],
    )


def mutation_rejection_suite(problem: BoundedProblemV1) -> dict[str, Any]:
    """Encoding + certificate mutations must not authorize unsat."""

    cold, encoded = generate_lrat_certificate(problem)
    results: list[dict[str, Any]] = []
    if cold.outcome != "unsat" or cold.certificate is None or encoded is None:
        return {
            "ok": False,
            "reason": "suite_requires_unsat_certificate",
            "cases": results,
            "rejection_rate": 0.0,
        }

    # 1) Encoding polarity flip with original certificate.
    mutated_enc = mutate_encoded_flip_first_literal(encoded)
    enc_check = check_lrat_certificate(problem, mutated_enc, cold.certificate)
    results.append(
        {
            "case_id": "encoding_flip_literal",
            "rejected": enc_check.outcome != "unsat",
            "outcome": enc_check.outcome,
            "reason": enc_check.reason,
        }
    )

    # 2) Certificate hint corruption against honest encoding.
    mutated_cert = mutate_certificate_flip_hint(cold.certificate)
    cert_check = check_lrat_certificate(problem, encoded, mutated_cert)
    results.append(
        {
            "case_id": "certificate_hint_corrupt",
            "rejected": cert_check.outcome != "unsat",
            "outcome": cert_check.outcome,
            "reason": cert_check.reason,
        }
    )

    # 3) Stale problem digest.
    stale = LratCertificateV1(
        schema=cold.certificate.schema,
        problem_digest="0" * 64,
        encoded_digest=cold.certificate.encoded_digest,
        encoder_hash=cold.certificate.encoder_hash,
        toolchain=cold.certificate.toolchain,
        steps=cold.certificate.steps,
    )
    stale_check = check_lrat_certificate(problem, encoded, stale)
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
    """Bounded unsat VSS problem: unit conflict + free bool pads (cost foil)."""

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
    """Frozen unsatisfiable suite for RESEARCH-05 paired timing."""

    return (
        pad_unit_conflict_problem(problem_id="research05/unsat_pad0", pad_vars=0),
        pad_unit_conflict_problem(problem_id="research05/unsat_pad4", pad_vars=4),
        pad_unit_conflict_problem(problem_id="research05/unsat_pad8", pad_vars=8),
        pad_unit_conflict_problem(problem_id="research05/unsat_pad10", pad_vars=10),
    )


def paired_warm_trials(
    problems: Sequence[BoundedProblemV1],
    *,
    warm_repeats: int = 5,
) -> dict[str, Any]:
    """Paired exhaustive vs warm-LRAT timings + correctness on one suite."""

    rows: list[dict[str, Any]] = []
    disagreements = 0
    ratios: list[float] = []
    cold_costs: list[float] = []
    cert_sizes: list[int] = []
    covered = 0

    for problem in problems:
        control = exhaustive_replay(problem)
        cold, encoded = generate_lrat_certificate(problem)
        if cold.outcome == "unknown" or cold.certificate is None or encoded is None:
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
            warm = check_lrat_certificate(problem, encoded, cold.certificate)
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
        rows.append(
            {
                "problem_id": problem.problem_id,
                "status": "ok" if warm_ok and control.outcome == "unsat" else "disagree",
                "control_elapsed_s": control.elapsed_s,
                "cold_elapsed_s": cold.elapsed_s,
                "warm_median_s": warm_median,
                "warm_over_exhaustive_ratio": ratio,
                "certificate_bytes": cert_sizes[-1],
                "identities": cold.identities,
            }
        )

    mutation = mutation_rejection_suite(problems[-1])
    ratios_sorted = sorted(ratios)
    median_ratio = (
        ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else None
    )
    correctness_ok = disagreements == 0 and mutation["ok"] and covered == len(problems)
    decision = "accept" if correctness_ok and median_ratio is not None and median_ratio < 1.0 else (
        "reject" if covered else "unknown"
    )
    return {
        "schema": "research_05_paired_warm_report/v1",
        "covered": covered,
        "suite_n": len(problems),
        "supported_subset_coverage": covered / len(problems) if problems else 0.0,
        "witness_disagreement_count": disagreements,
        "mutation_rejection_rate": mutation["rejection_rate"],
        "mutation": mutation,
        "median_paired_warm_lrat_check_over_exhaustive_replay_time_ratio": median_ratio,
        "secondary": {
            "cold_elapsed_s_median": (
                sorted(cold_costs)[len(cold_costs) // 2] if cold_costs else None
            ),
            "certificate_bytes_median": (
                sorted(cert_sizes)[len(cert_sizes) // 2] if cert_sizes else None
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
    "LratCertificateV1",
    "BackendVerdictV1",
    "check_lrat_certificate",
    "default_unsat_suite",
    "exhaustive_replay",
    "generate_lrat_certificate",
    "mutation_rejection_suite",
    "pad_unit_conflict_problem",
    "paired_warm_trials",
    "pinned_toolchain",
    "supports_lrat_subset",
]
