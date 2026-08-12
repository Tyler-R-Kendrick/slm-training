"""RESEARCH-07 / SLM-565: default-off Alethe/SMT proof reconstruction pilot.

Isolated research adapter. Uses the EVID-10 verified-encoding bridge
(``CnfRefEncodingAdapter`` + ``alethe_pilot``) for the boolean overlap subset
and a hermetic fixture-backed Alethe reconstruction path for a declared
equality (QF_UF) subset. Never production decode / ship-gate / serving authority.

Control authority remains exhaustive finite-domain replay (plus SAT/CNF
agreement where the problem overlaps the boolean encoding). Treatment emits a
checked/reconstructed Alethe-style certificate. Unsupported theory steps,
incomplete reconstruction, missing cvc5 (fixture stub only), or tool failure
return ``unknown`` — never a fake ``refuted`` / ``unsat``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

from slm_training.formal.encoding_adapter import (
    ENCODER_VERSION,
    BoundedProblemV1,
    CnfRefEncodingAdapter,
    LiteralV1,
    encoded_satisfiable,
    encoding_authorizes_semantic_result,
    encoder_source_hash,
    exists_satisfying_assignment,
    mutate_encoded_flip_first_literal,
)

BACKEND_ID = "vss_alethe_smt_pilot"
CERTIFICATE_SCHEMA = "vss_alethe_certificate/v1"
PROOF_FORMAT_VERSION = "alethe_reconstructed_pilot/v1"
TOOLCHAIN_ID = "hermetic_python_alethe_stub_v1"
# Optional live solver pin (never required; missing → fixture stub, fail closed).
CVC5_PROOF_FORMAT = "alethe"
CVC5_VERSION_PIN = "cvc5>=1.0.0 (optional; fixture stub when unavailable)"

# Declared SMT theories for this pilot (no silent widening).
SUPPORTED_SMT_THEORIES: frozenset[str] = frozenset({"QF_BOOL", "QF_UF"})
UNSUPPORTED_SMT_THEORIES: frozenset[str] = frozenset(
    {"QF_LIA", "QF_LRA", "QF_NIA", "QF_AUFLIA", "ARRAYS", "QUANTIFIERS"}
)

MAX_EXHAUSTIVE_VARS = 12

OutcomeKind = Literal["sat", "unsat", "unknown"]
SupportStatus = Literal["supported", "unsupported", "unknown"]
TheoryName = Literal["QF_BOOL", "QF_UF"]
AtomKind = Literal["bool_lit", "eq", "neq"]


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


def cvc5_available() -> bool:
    """Live cvc5 is optional; pilot defaults to hermetic fixture stub."""

    return shutil.which("cvc5") is not None


def pinned_toolchain() -> dict[str, str]:
    """Pinned identities for SMT/Alethe pilot (fixture stub when cvc5 absent)."""

    stub = not cvc5_available()
    return {
        "toolchain_id": TOOLCHAIN_ID,
        "backend_id": BACKEND_ID,
        "encoder_family": "cnf_ref+smt_eq_stub",
        "encoder_version": ENCODER_VERSION,
        "encoder_hash": encoder_source_hash(),
        "certificate_format": "alethe_pilot",
        "certificate_schema": CERTIFICATE_SCHEMA,
        "proof_format_version": PROOF_FORMAT_VERSION,
        "checker_backend": "python_alethe_reconstruct_stub",
        "trust_domain": "encoding_bridge/cnf_ref+alethe_pilot",
        "supported_smt_theories": ",".join(sorted(SUPPORTED_SMT_THEORIES)),
        "cvc5_proof_format": CVC5_PROOF_FORMAT,
        "cvc5_version_pin": CVC5_VERSION_PIN,
        "solver_mode": "fixture_stub" if stub else "cvc5_optional_unused",
    }


@dataclass(frozen=True)
class TheoryAtomV1:
    kind: AtomKind
    left: str
    right: str | None = None
    positive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "left": self.left,
            "right": self.right,
            "positive": self.positive,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TheoryAtomV1:
        right = data.get("right")
        return cls(
            kind=str(data["kind"]),  # type: ignore[arg-type]
            left=str(data["left"]),
            right=None if right is None else str(right),
            positive=bool(data.get("positive", True)),
        )


@dataclass(frozen=True)
class TheoryRichProblemV1:
    """Bounded theory-rich VSS problem for the Alethe/SMT pilot."""

    problem_id: str
    theory: TheoryName
    domains: dict[str, tuple[int, ...]]
    atoms: tuple[TheoryAtomV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "theory": self.theory,
            "domains": {k: list(v) for k, v in sorted(self.domains.items())},
            "atoms": [a.to_dict() for a in self.atoms],
        }

    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class AletheStepV1:
    step_id: str
    rule: str
    clause: tuple[tuple[str, bool], ...]  # empty ⇒ ⊥
    premises: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "rule": self.rule,
            "clause": [{"atom": a, "positive": p} for a, p in self.clause],
            "premises": list(self.premises),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AletheStepV1:
        clause = tuple(
            (str(item["atom"]), bool(item["positive"]))
            for item in data.get("clause", ())
        )
        premises = tuple(str(p) for p in data.get("premises", ()))
        return cls(
            step_id=str(data["step_id"]),
            rule=str(data["rule"]),
            clause=clause,
            premises=premises,
        )


@dataclass(frozen=True)
class AletheCertificateV1:
    schema: str
    proof_format_version: str
    theory: str
    problem_digest: str
    encoded_digest: str
    encoder_hash: str
    toolchain: dict[str, str]
    steps: tuple[AletheStepV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "proof_format_version": self.proof_format_version,
            "theory": self.theory,
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
            "proof_format_version": self.proof_format_version,
            "theory": self.theory,
            "problem_digest": self.problem_digest,
            "encoded_digest": self.encoded_digest,
            "encoder_hash": self.encoder_hash,
            "toolchain": self.toolchain,
            "steps": [step.to_dict() for step in self.steps],
        }
        return _sha256(_canonical_json(payload))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AletheCertificateV1:
        steps = tuple(AletheStepV1.from_dict(s) for s in data.get("steps", ()))
        return cls(
            schema=str(data["schema"]),
            proof_format_version=str(data["proof_format_version"]),
            theory=str(data["theory"]),
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
    certificate: AletheCertificateV1 | None = None
    encoding_digest: str | None = None
    identities: dict[str, str] | None = None
    theory: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "outcome": self.outcome,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "encoding_digest": self.encoding_digest,
            "identities": self.identities or {},
            "theory": self.theory,
        }
        if self.certificate is not None:
            out["certificate"] = self.certificate.to_dict()
        return out


def bool_problem_to_theory(problem: BoundedProblemV1) -> TheoryRichProblemV1:
    atoms: list[TheoryAtomV1] = []
    for clause in problem.clauses:
        if len(clause) != 1:
            # Pilot reconstructs only unit-conflict bool refutations.
            continue
        lit = clause[0]
        atoms.append(
            TheoryAtomV1(kind="bool_lit", left=lit.var, positive=lit.positive)
        )
    return TheoryRichProblemV1(
        problem_id=problem.problem_id,
        theory="QF_BOOL",
        domains={k: tuple(v) for k, v in problem.domains.items()},
        atoms=tuple(atoms),
    )


def supports_theory(problem: TheoryRichProblemV1) -> SupportStatus:
    if problem.theory in UNSUPPORTED_SMT_THEORIES:
        return "unsupported"
    if problem.theory not in SUPPORTED_SMT_THEORIES:
        return "unsupported"
    if not problem.domains or len(problem.domains) > MAX_EXHAUSTIVE_VARS:
        return "unsupported"
    if any(len(dom) == 0 for dom in problem.domains.values()):
        return "unsupported"
    if problem.theory == "QF_BOOL":
        if not all(set(dom) <= {0, 1} for dom in problem.domains.values()):
            return "unsupported"
        if not problem.atoms:
            return "unknown"
        if not all(a.kind == "bool_lit" for a in problem.atoms):
            return "unsupported"
        return "supported"
    # QF_UF: equality / disequality over finite domains.
    if not problem.atoms:
        return "unknown"
    if not all(a.kind in {"eq", "neq"} for a in problem.atoms):
        return "unsupported"
    for atom in problem.atoms:
        if atom.right is None:
            return "unsupported"
        if atom.left not in problem.domains or atom.right not in problem.domains:
            return "unsupported"
    return "supported"


def _atom_holds(atom: TheoryAtomV1, assign: Mapping[str, int]) -> bool:
    if atom.kind == "bool_lit":
        val = int(assign[atom.left])
        return (val == 1) if atom.positive else (val == 0)
    assert atom.right is not None
    left, right = int(assign[atom.left]), int(assign[atom.right])
    if atom.kind == "eq":
        holds = left == right
    else:
        holds = left != right
    return holds if atom.positive else (not holds)


def theory_satisfiable(problem: TheoryRichProblemV1) -> bool:
    keys = tuple(sorted(problem.domains))
    domains = [problem.domains[k] for k in keys]
    for values in product(*domains):
        assign = dict(zip(keys, values, strict=True))
        if all(_atom_holds(atom, assign) for atom in problem.atoms):
            return True
    return False


def smt_encoding_digest(problem: TheoryRichProblemV1) -> str:
    """Deterministic SMT-shaped encoding identity (fixture stub, no cvc5)."""

    payload = {
        "family": "smt_eq_stub",
        "proof_format_version": PROOF_FORMAT_VERSION,
        "theory": problem.theory,
        "problem": problem.to_dict(),
        "logic": problem.theory,
    }
    return _sha256(_canonical_json(payload))


def exhaustive_replay_theory(problem: TheoryRichProblemV1) -> BackendVerdictV1:
    started = time.perf_counter()
    status = supports_theory(problem)
    if status != "supported":
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_replay_{status}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
            theory=problem.theory,
        )
    try:
        sat = theory_satisfiable(problem)
    except (ValueError, KeyError) as exc:
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_replay_tool_failure:{exc}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
            theory=problem.theory,
        )
    return BackendVerdictV1(
        outcome="sat" if sat else "unsat",
        reason="exhaustive_replay_ok",
        elapsed_s=time.perf_counter() - started,
        encoding_digest=smt_encoding_digest(problem),
        identities={
            **pinned_toolchain(),
            "problem_digest": problem.digest(),
            "encoded_digest": smt_encoding_digest(problem),
        },
        theory=problem.theory,
    )


def exhaustive_replay_bool(problem: BoundedProblemV1) -> BackendVerdictV1:
    """Control for boolean overlap (canonical VSS/CNF exhaustive replay)."""

    started = time.perf_counter()
    adapter = CnfRefEncodingAdapter()
    status = adapter.supports(problem)
    if status != "supported":
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_bool_{status}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
            theory="QF_BOOL",
        )
    try:
        sat = exists_satisfying_assignment(problem)
        enc = adapter.encode(problem)
        if enc.encoded is None:
            raise ValueError("encode_failed")
        sat_enc = encoded_satisfiable(enc.encoded)
    except ValueError as exc:
        return BackendVerdictV1(
            outcome="unknown",
            reason=f"exhaustive_bool_tool_failure:{exc}",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
            theory="QF_BOOL",
        )
    if sat != sat_enc:
        return BackendVerdictV1(
            outcome="unknown",
            reason="exhaustive_bool_encode_disagreement",
            elapsed_s=time.perf_counter() - started,
            identities=pinned_toolchain(),
            theory="QF_BOOL",
        )
    return BackendVerdictV1(
        outcome="sat" if sat else "unsat",
        reason="exhaustive_bool_ok",
        elapsed_s=time.perf_counter() - started,
        encoding_digest=enc.encoded.digest(),
        identities={
            **pinned_toolchain(),
            "problem_digest": problem.digest(),
            "encoded_digest": enc.encoded.digest(),
        },
        theory="QF_BOOL",
    )


def _unit_conflict_bool_pair(
    atoms: Sequence[TheoryAtomV1],
) -> tuple[TheoryAtomV1, TheoryAtomV1] | None:
    pos = {a.left: a for a in atoms if a.kind == "bool_lit" and a.positive}
    neg = {a.left: a for a in atoms if a.kind == "bool_lit" and not a.positive}
    for var, p_atom in pos.items():
        if var in neg:
            return p_atom, neg[var]
    return None


def _eq_neq_conflict_pair(
    atoms: Sequence[TheoryAtomV1],
) -> tuple[TheoryAtomV1, TheoryAtomV1] | None:
    eqs = [
        a
        for a in atoms
        if a.kind == "eq" and a.positive and a.right is not None
    ]
    neqs = [
        a
        for a in atoms
        if a.kind == "neq" and a.positive and a.right is not None
    ]
    for eq in eqs:
        for neq in neqs:
            pair_eq = frozenset({eq.left, eq.right})
            pair_neq = frozenset({neq.left, neq.right})
            if pair_eq == pair_neq:
                return eq, neq
    return None


def generate_alethe_certificate(
    problem: TheoryRichProblemV1,
) -> tuple[BackendVerdictV1, str | None]:
    """Cold treatment: SMT encode + reconstruct Alethe (fixture stub)."""

    started = time.perf_counter()
    status = supports_theory(problem)
    if status != "supported":
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason=f"alethe_generate_{status}",
                elapsed_s=time.perf_counter() - started,
                identities=pinned_toolchain(),
                theory=problem.theory,
            ),
            None,
        )
    encoded_digest = smt_encoding_digest(problem)
    if theory_satisfiable(problem):
        return (
            BackendVerdictV1(
                outcome="sat",
                reason="alethe_generate_sat_no_refutation",
                elapsed_s=time.perf_counter() - started,
                encoding_digest=encoded_digest,
                identities={
                    **pinned_toolchain(),
                    "problem_digest": problem.digest(),
                    "encoded_digest": encoded_digest,
                },
                theory=problem.theory,
            ),
            encoded_digest,
        )

    steps: list[AletheStepV1] = []
    if problem.theory == "QF_BOOL":
        conflict = _unit_conflict_bool_pair(problem.atoms)
        if conflict is None:
            # Incomplete reconstruction for other unsat shapes → unknown.
            return (
                BackendVerdictV1(
                    outcome="unknown",
                    reason="alethe_generate_incomplete_reconstruction",
                    elapsed_s=time.perf_counter() - started,
                    encoding_digest=encoded_digest,
                    identities=pinned_toolchain(),
                    theory=problem.theory,
                ),
                encoded_digest,
            )
        a_pos, a_neg = conflict
        steps = [
            AletheStepV1(
                step_id="h1",
                rule="assume",
                clause=((a_pos.left, True),),
                premises=(),
            ),
            AletheStepV1(
                step_id="h2",
                rule="assume",
                clause=((a_neg.left, False),),
                premises=(),
            ),
            AletheStepV1(
                step_id="t1",
                rule="resolution",
                clause=(),
                premises=("h1", "h2"),
            ),
        ]
    else:
        conflict = _eq_neq_conflict_pair(problem.atoms)
        if conflict is None:
            return (
                BackendVerdictV1(
                    outcome="unknown",
                    reason="alethe_generate_incomplete_reconstruction",
                    elapsed_s=time.perf_counter() - started,
                    encoding_digest=encoded_digest,
                    identities=pinned_toolchain(),
                    theory=problem.theory,
                ),
                encoded_digest,
            )
        eq, neq = conflict
        assert eq.right is not None and neq.right is not None
        atom_key = f"eq:{eq.left}:{eq.right}"
        steps = [
            AletheStepV1(
                step_id="h1",
                rule="assume",
                clause=((atom_key, True),),
                premises=(),
            ),
            AletheStepV1(
                step_id="h2",
                rule="assume",
                clause=((atom_key, False),),
                premises=(),
            ),
            AletheStepV1(
                step_id="t1",
                rule="eq_contradiction",
                clause=(),
                premises=("h1", "h2"),
            ),
        ]

    cert = AletheCertificateV1(
        schema=CERTIFICATE_SCHEMA,
        proof_format_version=PROOF_FORMAT_VERSION,
        theory=problem.theory,
        problem_digest=problem.digest(),
        encoded_digest=encoded_digest,
        encoder_hash=encoder_source_hash(),
        toolchain=pinned_toolchain(),
        steps=tuple(steps),
    )
    check = check_alethe_certificate(problem, encoded_digest, cert)
    elapsed = time.perf_counter() - started
    if check.outcome != "unsat":
        return (
            BackendVerdictV1(
                outcome="unknown",
                reason=f"alethe_generate_check_failed:{check.reason}",
                elapsed_s=elapsed,
                encoding_digest=encoded_digest,
                identities=pinned_toolchain(),
                theory=problem.theory,
            ),
            encoded_digest,
        )
    return (
        BackendVerdictV1(
            outcome="unsat",
            reason="alethe_generate_ok",
            elapsed_s=elapsed,
            certificate=cert,
            encoding_digest=encoded_digest,
            identities={
                **pinned_toolchain(),
                "problem_digest": problem.digest(),
                "encoded_digest": encoded_digest,
                "certificate_digest": cert.content_digest(),
            },
            theory=problem.theory,
        ),
        encoded_digest,
    )


def check_alethe_certificate(
    problem: TheoryRichProblemV1,
    encoded_digest: str,
    certificate: AletheCertificateV1,
) -> BackendVerdictV1:
    """Warm path: re-check reconstructed Alethe (never widen unknown→unsat)."""

    started = time.perf_counter()
    identities = pinned_toolchain()
    if certificate.schema != CERTIFICATE_SCHEMA:
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_schema_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.proof_format_version != PROOF_FORMAT_VERSION:
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_proof_format_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.theory != problem.theory:
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_theory_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.problem_digest != problem.digest():
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_stale_problem",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.encoded_digest != encoded_digest:
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_encoded_digest_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.encoded_digest != smt_encoding_digest(problem):
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_encoding_drift",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.encoder_hash != encoder_source_hash():
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_encoder_hash_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if certificate.toolchain.get("toolchain_id") != TOOLCHAIN_ID:
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_toolchain_mismatch",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    if problem.theory == "QF_BOOL":
        # EVID-10 bridge for boolean overlap (alethe_pilot is non-production).
        bool_problem = theory_to_bool_bounded(problem)
        if bool_problem is not None:
            adapter = CnfRefEncodingAdapter()
            result, evidence = adapter.build_evidence(
                bool_problem, certificate_format="alethe_pilot"
            )
            if (
                result.encoded is None
                or evidence is None
                or not encoding_authorizes_semantic_result(
                    evidence,
                    expected_problem_digest=bool_problem.digest(),
                    certificate_checked=True,
                    encoded_digest_observed=result.encoded.digest(),
                    require_production_format=False,
                )
            ):
                return BackendVerdictV1(
                    outcome="unknown",
                    reason="alethe_check_encoding_not_authorized",
                    elapsed_s=time.perf_counter() - started,
                    identities=identities,
                    theory=problem.theory,
                )

    db: dict[str, tuple[tuple[str, bool], ...]] = {}
    derived_empty = False
    for step in certificate.steps:
        if step.step_id in db:
            return BackendVerdictV1(
                outcome="unknown",
                reason="alethe_check_duplicate_step",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
                theory=problem.theory,
            )
        if step.rule == "assume":
            if step.premises or not step.clause:
                return BackendVerdictV1(
                    outcome="unknown",
                    reason="alethe_check_bad_assume",
                    elapsed_s=time.perf_counter() - started,
                    identities=identities,
                    theory=problem.theory,
                )
            db[step.step_id] = step.clause
            continue
        if step.rule not in {"resolution", "eq_contradiction"}:
            # Unknown theory rule → incomplete reconstruction, not refutation.
            return BackendVerdictV1(
                outcome="unknown",
                reason="alethe_check_unsupported_theory_rule",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
                theory=problem.theory,
            )
        if len(step.premises) != 2 or step.clause:
            return BackendVerdictV1(
                outcome="unknown",
                reason="alethe_check_bad_contradiction_step",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
                theory=problem.theory,
            )
        c1 = db.get(step.premises[0])
        c2 = db.get(step.premises[1])
        if c1 is None or c2 is None or len(c1) != 1 or len(c2) != 1:
            return BackendVerdictV1(
                outcome="unknown",
                reason="alethe_check_premise_missing",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
                theory=problem.theory,
            )
        (a1, p1), (a2, p2) = c1[0], c2[0]
        if a1 != a2 or p1 == p2:
            return BackendVerdictV1(
                outcome="unknown",
                reason="alethe_check_not_opposing_units",
                elapsed_s=time.perf_counter() - started,
                identities=identities,
                theory=problem.theory,
            )
        db[step.step_id] = ()
        derived_empty = True

    if not derived_empty:
        return BackendVerdictV1(
            outcome="unknown",
            reason="alethe_check_no_empty_derived",
            elapsed_s=time.perf_counter() - started,
            identities=identities,
            theory=problem.theory,
        )
    return BackendVerdictV1(
        outcome="unsat",
        reason="alethe_check_ok",
        elapsed_s=time.perf_counter() - started,
        certificate=certificate,
        encoding_digest=encoded_digest,
        identities={
            **identities,
            "problem_digest": problem.digest(),
            "encoded_digest": encoded_digest,
            "certificate_digest": certificate.content_digest(),
        },
        theory=problem.theory,
    )


def theory_to_bool_bounded(problem: TheoryRichProblemV1) -> BoundedProblemV1 | None:
    if problem.theory != "QF_BOOL":
        return None
    if not all(set(dom) <= {0, 1} for dom in problem.domains.values()):
        return None
    clauses = tuple(
        (LiteralV1(atom.left, atom.positive),)
        for atom in problem.atoms
        if atom.kind == "bool_lit"
    )
    if not clauses:
        return None
    return BoundedProblemV1(
        problem_id=problem.problem_id,
        domains={k: (0, 1) for k in problem.domains},
        clauses=clauses,
        features=frozenset({"bool_domain", "clause"}),
    )


def mutate_certificate_flip_premise(
    certificate: AletheCertificateV1,
) -> AletheCertificateV1:
    if not certificate.steps:
        raise ValueError("certificate has no steps")
    last = certificate.steps[-1]
    if not last.premises:
        raise ValueError("final step has no premises")
    bad = AletheStepV1(
        step_id=last.step_id,
        rule=last.rule,
        clause=last.clause,
        premises=(last.premises[0] + "_x",) + last.premises[1:],
    )
    return AletheCertificateV1(
        schema=certificate.schema,
        proof_format_version=certificate.proof_format_version,
        theory=certificate.theory,
        problem_digest=certificate.problem_digest,
        encoded_digest=certificate.encoded_digest,
        encoder_hash=certificate.encoder_hash,
        toolchain=certificate.toolchain,
        steps=certificate.steps[:-1] + (bad,),
    )


def mutation_rejection_suite(problem: TheoryRichProblemV1) -> dict[str, Any]:
    cold, encoded_digest = generate_alethe_certificate(problem)
    results: list[dict[str, Any]] = []
    if cold.outcome != "unsat" or cold.certificate is None or encoded_digest is None:
        return {
            "ok": False,
            "reason": "suite_requires_unsat_certificate",
            "cases": results,
            "rejection_rate": 0.0,
        }

    mutated_cert = mutate_certificate_flip_premise(cold.certificate)
    cert_check = check_alethe_certificate(problem, encoded_digest, mutated_cert)
    results.append(
        {
            "case_id": "certificate_premise_corrupt",
            "rejected": cert_check.outcome != "unsat",
            "outcome": cert_check.outcome,
            "reason": cert_check.reason,
        }
    )

    stale = AletheCertificateV1(
        schema=cold.certificate.schema,
        proof_format_version=cold.certificate.proof_format_version,
        theory=cold.certificate.theory,
        problem_digest="0" * 64,
        encoded_digest=cold.certificate.encoded_digest,
        encoder_hash=cold.certificate.encoder_hash,
        toolchain=cold.certificate.toolchain,
        steps=cold.certificate.steps,
    )
    stale_check = check_alethe_certificate(problem, encoded_digest, stale)
    results.append(
        {
            "case_id": "stale_problem_digest",
            "rejected": stale_check.outcome != "unsat",
            "outcome": stale_check.outcome,
            "reason": stale_check.reason,
        }
    )

    wrong_theory = AletheCertificateV1(
        schema=cold.certificate.schema,
        proof_format_version=cold.certificate.proof_format_version,
        theory="QF_LIA",
        problem_digest=cold.certificate.problem_digest,
        encoded_digest=cold.certificate.encoded_digest,
        encoder_hash=cold.certificate.encoder_hash,
        toolchain=cold.certificate.toolchain,
        steps=cold.certificate.steps,
    )
    theory_check = check_alethe_certificate(problem, encoded_digest, wrong_theory)
    results.append(
        {
            "case_id": "theory_metadata_corrupt",
            "rejected": theory_check.outcome != "unsat",
            "outcome": theory_check.outcome,
            "reason": theory_check.reason,
        }
    )

    if problem.theory == "QF_BOOL":
        bounded = theory_to_bool_bounded(problem)
        if bounded is not None:
            adapter = CnfRefEncodingAdapter()
            enc = adapter.encode(bounded)
            if enc.encoded is not None:
                flipped = mutate_encoded_flip_first_literal(enc.encoded)
                # Encoding mutation must not authorize via digest match.
                enc_check = check_alethe_certificate(
                    problem, flipped.digest(), cold.certificate
                )
                results.append(
                    {
                        "case_id": "encoding_digest_mismatch",
                        "rejected": enc_check.outcome != "unsat",
                        "outcome": enc_check.outcome,
                        "reason": enc_check.reason,
                    }
                )

    rejected = sum(1 for row in results if row["rejected"])
    return {
        "ok": rejected == len(results),
        "reason": "all_mutations_rejected" if rejected == len(results) else "mutation_miss",
        "cases": results,
        "rejection_rate": rejected / len(results) if results else 0.0,
    }


def pad_bool_unit_conflict(*, problem_id: str, pad_vars: int = 2) -> TheoryRichProblemV1:
    domains = {"x0": (0, 1)}
    for idx in range(1, pad_vars + 1):
        domains[f"x{idx}"] = (0, 1)
    return TheoryRichProblemV1(
        problem_id=problem_id,
        theory="QF_BOOL",
        domains=domains,
        atoms=(
            TheoryAtomV1(kind="bool_lit", left="x0", positive=True),
            TheoryAtomV1(kind="bool_lit", left="x0", positive=False),
        ),
    )


def eq_neq_conflict(*, problem_id: str, domain: tuple[int, ...] = (0, 1, 2)) -> TheoryRichProblemV1:
    return TheoryRichProblemV1(
        problem_id=problem_id,
        theory="QF_UF",
        domains={"a": domain, "b": domain},
        atoms=(
            TheoryAtomV1(kind="eq", left="a", right="b", positive=True),
            TheoryAtomV1(kind="neq", left="a", right="b", positive=True),
        ),
    )


def default_theory_suite() -> tuple[TheoryRichProblemV1, ...]:
    """Frozen theory-rich + bool-overlap suite for RESEARCH-07."""

    return (
        pad_bool_unit_conflict(problem_id="research07/bool_unit", pad_vars=2),
        pad_bool_unit_conflict(problem_id="research07/bool_unit_pad6", pad_vars=6),
        eq_neq_conflict(problem_id="research07/eq_neq"),
        eq_neq_conflict(problem_id="research07/eq_neq_dom4", domain=(0, 1, 2, 3)),
    )


def incomplete_reconstruction_fixture() -> TheoryRichProblemV1:
    """Unsat without reconstructible eq/neq pair → must stay ``unknown``."""

    # Singleton domain + disequality is unsat, but the pilot only reconstructs
    # explicit eq∧neq conflicts — incomplete reconstruction must not refute.
    return TheoryRichProblemV1(
        problem_id="research07/singleton_neq_unsat",
        theory="QF_UF",
        domains={"a": (0,), "b": (0,)},
        atoms=(TheoryAtomV1(kind="neq", left="a", right="b", positive=True),),
    )


def unsupported_theory_fixture() -> TheoryRichProblemV1:
    """Bool atoms under QF_UF are outside the declared theory fragment."""

    return TheoryRichProblemV1(
        problem_id="research07/uf_bool_atoms_unsupported",
        theory="QF_UF",
        domains={"x": (0, 1, 2)},
        atoms=(TheoryAtomV1(kind="bool_lit", left="x", positive=True),),
    )


def paired_agreement_report(
    problems: Sequence[TheoryRichProblemV1],
) -> dict[str, Any]:
    """Control vs Alethe treatment agreement on the declared supported subset."""

    rows: list[dict[str, Any]] = []
    disagreements = 0
    supported = 0
    agreed = 0
    unknown_preserved = 0
    fake_refutations = 0
    sat_overlap_agreements = 0
    sat_overlap_n = 0

    for problem in problems:
        control = exhaustive_replay_theory(problem)
        cold, encoded_digest = generate_alethe_certificate(problem)
        status = supports_theory(problem)
        if status != "supported":
            preserved = cold.outcome == "unknown"
            if preserved:
                unknown_preserved += 1
            if cold.outcome == "unsat":
                fake_refutations += 1
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "theory": problem.theory,
                    "status": status,
                    "control": control.to_dict(),
                    "treatment": cold.to_dict(),
                    "preserved_candidate": preserved,
                }
            )
            continue

        supported += 1
        warm_ok = False
        if cold.certificate is not None and encoded_digest is not None:
            warm = check_alethe_certificate(problem, encoded_digest, cold.certificate)
            warm_ok = warm.outcome == cold.outcome
        else:
            warm = None

        if control.outcome == cold.outcome and (
            cold.outcome != "unsat" or warm_ok
        ):
            agreed += 1
        else:
            disagreements += 1

        if cold.outcome == "unknown" and control.outcome == "unsat":
            # Incomplete reconstruction on unsat → unknown is correct preservation.
            unknown_preserved += 1

        # Boolean/SAT overlap: when QF_BOOL maps to BoundedProblemV1, compare CNF replay.
        if problem.theory == "QF_BOOL":
            bounded = theory_to_bool_bounded(problem)
            if bounded is not None:
                sat_overlap_n += 1
                bool_control = exhaustive_replay_bool(bounded)
                if bool_control.outcome == control.outcome == cold.outcome:
                    sat_overlap_agreements += 1

        rows.append(
            {
                "problem_id": problem.problem_id,
                "theory": problem.theory,
                "status": "ok" if control.outcome == cold.outcome else "disagree",
                "control_outcome": control.outcome,
                "treatment_outcome": cold.outcome,
                "warm_ok": warm_ok,
                "identities": cold.identities,
            }
        )

    # Incomplete reconstruction must not become refutation.
    incomplete = incomplete_reconstruction_fixture()
    inc_cold, _ = generate_alethe_certificate(incomplete)
    if inc_cold.outcome == "unsat":
        fake_refutations += 1
    elif inc_cold.outcome == "unknown":
        unknown_preserved += 1

    unsupported = unsupported_theory_fixture()
    uns_status = supports_theory(unsupported)
    uns_cold, _ = generate_alethe_certificate(unsupported)
    if uns_status == "unsupported" and uns_cold.outcome == "unknown":
        unknown_preserved += 1
    if uns_cold.outcome == "unsat":
        fake_refutations += 1

    mutation = mutation_rejection_suite(problems[-1])
    agreement_rate = agreed / supported if supported else 0.0
    correctness_ok = (
        disagreements == 0
        and mutation["ok"]
        and fake_refutations == 0
        and supported == len(problems)
        and agreement_rate == 1.0
    )
    decision = "accept" if correctness_ok else ("reject" if supported else "unknown")
    return {
        "schema": "research_07_theory_agreement_report/v1",
        "supported": supported,
        "suite_n": len(problems),
        "exact_agreement_rate_on_supported_theory_subset": agreement_rate,
        "witness_disagreement_count": disagreements,
        "mutation_rejection_rate": mutation["rejection_rate"],
        "mutation": mutation,
        "unknown_preservation_rate": (
            unknown_preserved / max(1, unknown_preserved + fake_refutations)
        ),
        "reconstruction_failure_as_refutation_count": fake_refutations,
        "sat_overlap_agreement_rate": (
            sat_overlap_agreements / sat_overlap_n if sat_overlap_n else None
        ),
        "correctness_ok": correctness_ok,
        "decision": decision,
        "rows": rows,
        "toolchain": pinned_toolchain(),
        "supported_smt_theories": sorted(SUPPORTED_SMT_THEORIES),
        "proof_format_version": PROOF_FORMAT_VERSION,
    }


__all__ = [
    "BACKEND_ID",
    "CERTIFICATE_SCHEMA",
    "PROOF_FORMAT_VERSION",
    "SUPPORTED_SMT_THEORIES",
    "AletheCertificateV1",
    "BackendVerdictV1",
    "TheoryAtomV1",
    "TheoryRichProblemV1",
    "check_alethe_certificate",
    "cvc5_available",
    "default_theory_suite",
    "exhaustive_replay_bool",
    "exhaustive_replay_theory",
    "generate_alethe_certificate",
    "incomplete_reconstruction_fixture",
    "mutation_rejection_suite",
    "paired_agreement_report",
    "pinned_toolchain",
    "supports_theory",
    "unsupported_theory_fixture",
]
