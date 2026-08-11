"""KERN-07 / SLM-530: singleton zero-neural-forward policy + runtime refinement.

Mirrors ``LeverProofLean.ConstrainedDiffusion`` singleton policy theorems under
KERN-06 ``NeuralForwardOnlyModel``. Lean proves a valid complete singleton
*admits* a policy with neural-forward cost 0. This module:

* checks the strengthened singleton-validity contract (forced token = sole
  complete-domain candidate);
* projects existing ``DecodeStats.forwards_count`` telemetry through the named
  forward-count model;
* records fixture evidence that certified singletons observe zero neural
  forwards — without promoting that evidence to implementation optimality
  (grammar / certificate / memory / wall-clock stay outside the theorem).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.formal.event_trace import (
    NEURAL_FORWARD_ONLY_MODEL,
    Event,
    EventKind,
    MachineModel,
    project_decode_stats,
    resource_cost_model_id,
    trace_cost,
)
from slm_training.models.decode_stats import (
    DecodeStats,
    DecodeStatsRecordV1,
    proves_zero_neural_work,
)

SINGLETON_NEURAL_POLICY_SCHEMA = "singleton_neural_policy/v1"
FIXTURE_SCHEMA = "singleton_neural_policy_fixtures/v1"
THEOREM_REF = "ConstrainedDiffusion.singleton_policy_neural_cost_zero"
COST_MODEL = NEURAL_FORWARD_ONLY_MODEL

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "singleton_neural_policy_fixtures.v1.json"
)


@dataclass(frozen=True)
class SingletonProofV1:
    """Python mirror of Lean ``ConstrainedDiffusion.SingletonProof``."""

    status: str
    candidates: tuple[tuple[int, ...], ...]
    forced_token: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "candidates": [list(c) for c in self.candidates],
            "forced_token": self.forced_token,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SingletonProofV1:
        candidates_raw = payload.get("candidates")
        if not isinstance(candidates_raw, Sequence):
            raise ValueError("candidates must be a sequence")
        candidates: list[tuple[int, ...]] = []
        for row in candidates_raw:
            if not isinstance(row, Sequence):
                raise ValueError("each candidate must be a sequence of ints")
            candidates.append(tuple(int(x) for x in row))
        forced = payload.get("forced_token")
        return cls(
            status=str(payload.get("status", "")),
            candidates=tuple(candidates),
            forced_token=None if forced is None else int(forced),
        )


def singleton_proof_valid(proof: SingletonProofV1) -> bool:
    """Lean ``SingletonProof.valid`` / ``validBool``: complete + forced = sole."""

    if proof.status != "complete":
        return False
    if proof.forced_token is None:
        return False
    if len(proof.candidates) != 1:
        return False
    return proof.candidates[0] == (proof.forced_token,)


def singleton_bypass_neural_trace(_proof: SingletonProofV1) -> tuple[Event, ...]:
    """Empty neural-forward trace for the bypass policy (Lean mirror)."""

    return ()


def policy_neural_cost(
    trace: Sequence[Event],
    *,
    model: MachineModel = COST_MODEL,
) -> int:
    """Named forward-count cost (``NeuralForwardOnlyModel``)."""

    return trace_cost(model, trace)


def singleton_policy_neural_cost_zero(proof: SingletonProofV1) -> bool:
    """True when a valid proof's bypass policy has neural cost 0."""

    if not singleton_proof_valid(proof):
        return False
    return policy_neural_cost(singleton_bypass_neural_trace(proof)) == 0


@dataclass(frozen=True)
class SingletonRuntimeRefinementV1:
    """Empirical refinement: observed decode telemetry under the named model.

    Status is always ``empirical_remainder`` — never a stronger theorem than
    Lean refinement supports.
    """

    schema: str
    theorem_ref: str
    cost_model_id: str
    proof: SingletonProofV1
    proof_valid: bool
    observed_forwards_count: int
    policy_neural_cost: int
    projected_neural_cost: int
    proves_zero_neural_work: bool
    grammar_certificate_memory_wall_clock_outside_theorem: bool
    claim_class: str
    wall_clock_claim: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "theorem_ref": self.theorem_ref,
            "cost_model_id": self.cost_model_id,
            "proof": self.proof.to_dict(),
            "proof_valid": self.proof_valid,
            "observed_forwards_count": self.observed_forwards_count,
            "policy_neural_cost": self.policy_neural_cost,
            "projected_neural_cost": self.projected_neural_cost,
            "proves_zero_neural_work": self.proves_zero_neural_work,
            "grammar_certificate_memory_wall_clock_outside_theorem": (
                self.grammar_certificate_memory_wall_clock_outside_theorem
            ),
            "claim_class": self.claim_class,
            "wall_clock_claim": self.wall_clock_claim,
            "notes": list(self.notes),
            "implementation_refinement_status": "empirical_remainder",
        }


def observe_singleton_runtime(
    proof: SingletonProofV1,
    stats: DecodeStats | DecodeStatsRecordV1 | Mapping[str, Any],
    *,
    claim_class: str = "fixture",
) -> SingletonRuntimeRefinementV1:
    """Project decode telemetry; agree with Lean only as empirical remainder."""

    if isinstance(stats, DecodeStatsRecordV1):
        decode_stats = stats.stats
    elif isinstance(stats, DecodeStats):
        decode_stats = stats
    else:
        decode_stats = DecodeStats(
            forwards_count=int(stats.get("forwards_count", 0) or 0),
            tokens_emitted=int(stats.get("tokens_emitted", 0) or 0),
            dfa_sync_count=int(stats.get("dfa_sync_count", 0) or 0),
            solver_expanded_nodes=int(stats.get("solver_expanded_nodes", 0) or 0),
            solver_verifier_calls=int(stats.get("solver_verifier_calls", 0) or 0),
            denoiser_rows_evaluated=int(stats.get("denoiser_rows_evaluated", 0) or 0),
            ambiguous_rows_forwarded=int(stats.get("ambiguous_rows_forwarded", 0) or 0),
            total_ms=float(stats.get("total_ms", 0.0) or 0.0),
        )

    projection = project_decode_stats(decode_stats, model=COST_MODEL)
    policy_cost = policy_neural_cost(singleton_bypass_neural_trace(proof))
    valid = singleton_proof_valid(proof)
    return SingletonRuntimeRefinementV1(
        schema=SINGLETON_NEURAL_POLICY_SCHEMA,
        theorem_ref=THEOREM_REF,
        cost_model_id=resource_cost_model_id(COST_MODEL),
        proof=proof,
        proof_valid=valid,
        observed_forwards_count=int(decode_stats.forwards_count),
        policy_neural_cost=policy_cost,
        projected_neural_cost=int(projection.abstract_cost),
        proves_zero_neural_work=proves_zero_neural_work(decode_stats),
        grammar_certificate_memory_wall_clock_outside_theorem=True,
        claim_class=claim_class,
        wall_clock_claim=False,
        notes=(
            "Lean theorem scopes NeuralForwardOnlyModel policy cost only",
            "DecodeStats ms / grammar / certificate / memory remain outside",
            "Nat.zero_le is not an implementation-optimality certificate",
            "fixture evidence is empirical_remainder, not a stronger theorem",
        ),
    )


def certified_singleton_fixture_rows() -> list[dict[str, Any]]:
    """Live rows derived from the frozen fixture file + DecodeStats projection."""

    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"unexpected fixture schema: {payload.get('schema')!r}")
    rows: list[dict[str, Any]] = []
    for case in payload["cases"]:
        proof = SingletonProofV1.from_dict(case["proof"])
        stats = DecodeStats(**case.get("decode_stats", {}))
        observed = observe_singleton_runtime(
            proof, stats, claim_class=str(case.get("claim_class", "fixture"))
        )
        rows.append(
            {
                "id": case["id"],
                "expected_proof_valid": bool(case["expected_proof_valid"]),
                "expected_observed_forwards": int(case["expected_observed_forwards"]),
                "expected_projected_neural_cost": int(
                    case["expected_projected_neural_cost"]
                ),
                "live": observed.to_dict(),
            }
        )
    return rows


def mutation_rejects_mismatched_forced_token() -> bool:
    """Mutation: forced token ≠ sole candidate ⇒ invalid."""

    proof = SingletonProofV1(
        status="complete",
        candidates=((0,),),
        forced_token=1,
    )
    return not singleton_proof_valid(proof)


def mutation_rejects_incomplete_singleton_domain() -> bool:
    """Mutation: incomplete status with one candidate ⇒ invalid."""

    proof = SingletonProofV1(
        status="incomplete",
        candidates=((0,),),
        forced_token=0,
    )
    return not singleton_proof_valid(proof)


__all__ = [
    "COST_MODEL",
    "FIXTURE_SCHEMA",
    "SINGLETON_NEURAL_POLICY_SCHEMA",
    "THEOREM_REF",
    "SingletonProofV1",
    "SingletonRuntimeRefinementV1",
    "certified_singleton_fixture_rows",
    "mutation_rejects_incomplete_singleton_domain",
    "mutation_rejects_mismatched_forced_token",
    "observe_singleton_runtime",
    "policy_neural_cost",
    "singleton_bypass_neural_trace",
    "singleton_policy_neural_cost_zero",
    "singleton_proof_valid",
    "Event",
    "EventKind",
]
