"""Exact-closure strict progress and finite stabilization (KERN-05 / SLM-523).

Mirrors ``LeverProofLean.ExactClosure`` KERN-05 theorems:

* non-fixed pass removes ≥1 previously live candidate (strict progress);
* cumulative strict removals ≤ ``Σ d_i`` (product-lattice height);
* under nonempty-domain invariance, ≤ ``Σ (d_i - 1)``;
* closure-pass complexity (``Σ d_i``) is separated from assignment search (``∏ d_i``).

Batch fairness/coverage is an explicit precondition. Fixed points are **not**
claimed maximal or oracle-complete without additional proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from slm_training.formal.refutation_authority import (
    BindingIdsV1,
    CheckedRefutationEvidenceV1,
    make_exact_replay_evidence,
)
from slm_training.formal.structural import close_pass, removable
from slm_training.harness_core.lineage.records import content_sha

CLOSURE_STABILIZATION_SCHEMA = "closure_stabilization/v1"
BOUND_AST_STRICT_REMOVALS = "bound.closure.strict_removals.v1"
BOUND_AST_LIVE_UPPER = "bound.closure.live_upper.v1"

# Shared fixture binding for frozen closure traces (identity-bound replay).
_FIXTURE_BINDING = BindingIdsV1(
    state_id="closure-fixture-state",
    problem_id="closure-fixture-problem",
    source_id="exact_closure",
    tool_id="python_replay",
)


def _fixture_evidence(candidate: int) -> CheckedRefutationEvidenceV1:
    return make_exact_replay_evidence(
        _FIXTURE_BINDING,
        evidence_digest=f"closure-replay-{candidate}",
    )


def list_sum(sizes: Sequence[int]) -> int:
    total = 0
    for d in sizes:
        if d < 0:
            raise ValueError("domain sizes must be non-negative")
        total += int(d)
    return total


def strict_removal_upper_bound(domain_sizes: Sequence[int]) -> int:
    """``Σ d_i`` — product-lattice height / closure-pass removal bound."""

    return list_sum(domain_sizes)


def nonempty_invariant_upper_bound(domain_sizes: Sequence[int]) -> int:
    """``Σ (d_i - 1)`` with saturating Nat subtraction (``0 - 1 = 0``)."""

    caps: list[int] = []
    for d in domain_sizes:
        di = int(d)
        if di < 0:
            raise ValueError("domain sizes must be non-negative")
        caps.append(di - 1 if di >= 1 else 0)
    return list_sum(caps)


def list_prod(sizes: Sequence[int]) -> int:
    """Empty product is 1 (matches Lean ``FiniteSearchBounds.listProd``)."""

    out = 1
    for d in sizes:
        if d < 0:
            raise ValueError("domain sizes must be non-negative")
        out *= int(d)
    return out


def assignment_search_bound(domain_sizes: Sequence[int]) -> int:
    """``∏ d_i`` — KERN-03 assignment-search complexity (not closure passes)."""

    return list_prod(domain_sizes)


def closure_complexity_separated(domain_sizes: Sequence[int]) -> bool:
    """True when ``Σ d_i ≠ ∏ d_i`` (witness that the measures differ)."""

    return strict_removal_upper_bound(domain_sizes) != assignment_search_bound(
        domain_sizes
    )


@dataclass(frozen=True)
class QueryResultV1:
    hole: int
    candidate: int
    verdict: Literal["supported", "unsupported", "unknown"]
    replay_ok: bool = False  # telemetry only (EVID-09)
    refutation_evidence: CheckedRefutationEvidenceV1 | None = None
    expected_binding: BindingIdsV1 | None = None

    def is_removable(self) -> bool:
        return removable(
            self.verdict,
            self.replay_ok,
            evidence=self.refutation_evidence,
            expected=self.expected_binding,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hole": self.hole,
            "candidate": self.candidate,
            "verdict": self.verdict,
            "replay_ok": self.replay_ok,
        }
        if self.refutation_evidence is not None:
            out["refutation_evidence"] = self.refutation_evidence.to_dict()
        if self.expected_binding is not None:
            out["expected_binding"] = self.expected_binding.to_dict()
        return out


def is_removable_in_batch(results: Sequence[QueryResultV1], candidate: int) -> bool:
    return any(r.candidate == candidate and r.is_removable() for r in results)


def batch_mentions(results: Sequence[QueryResultV1], candidate: int) -> bool:
    return any(r.candidate == candidate for r in results)


def batch_covers_removable(
    live: Sequence[int], results: Sequence[QueryResultV1]
) -> bool:
    """Fairness/coverage: every live removable candidate is named in the batch."""

    for c in live:
        if is_removable_in_batch(results, c) and not batch_mentions(results, c):
            return False
    return True


def close_pass_results(live: Sequence[int], results: Sequence[QueryResultV1]) -> list[int]:
    rem = [r.candidate for r in results if r.is_removable()]
    return close_pass(live, rem)


def removed_candidates(
    live: Sequence[int], results: Sequence[QueryResultV1]
) -> list[int]:
    return [c for c in live if is_removable_in_batch(results, c)]


def is_fixed_point(live: Sequence[int], results: Sequence[QueryResultV1]) -> bool:
    survivors = set(close_pass_results(live, results))
    return survivors == set(live)


def strict_progress_or_fixed_point(
    live: Sequence[int], results: Sequence[QueryResultV1]
) -> bool:
    """Dichotomy mirror of ``ExactClosure.strict_progress_or_fixed_point``."""

    if is_fixed_point(live, results):
        return True
    survivors = set(close_pass_results(live, results))
    return any(c not in survivors for c in live)


def removed_count(live: Sequence[int], results: Sequence[QueryResultV1]) -> int:
    return len(live) - len(close_pass_results(live, results))


def total_removed_across(
    live: Sequence[int], batches: Sequence[Sequence[QueryResultV1]]
) -> int:
    total = 0
    cur = list(live)
    for batch in batches:
        nxt = close_pass_results(cur, batch)
        total += len(cur) - len(nxt)
        cur = nxt
    return total


def hole_sizes(holes: Sequence[Sequence[int]]) -> tuple[int, ...]:
    return tuple(len(h) for h in holes)


def total_live_count(holes: Sequence[Sequence[int]]) -> int:
    return list_sum(hole_sizes(holes))


def close_pass_holes(
    holes: Sequence[Sequence[int]], results: Sequence[QueryResultV1]
) -> list[list[int]]:
    return [close_pass_results(h, results) for h in holes]


def all_nonempty(holes: Sequence[Sequence[int]]) -> bool:
    return all(len(h) > 0 for h in holes)


def total_removed_across_holes(
    holes: Sequence[Sequence[int]], batches: Sequence[Sequence[QueryResultV1]]
) -> int:
    total = 0
    cur = [list(h) for h in holes]
    for batch in batches:
        nxt = close_pass_holes(cur, batch)
        total += total_live_count(cur) - total_live_count(nxt)
        cur = nxt
    return total


def unique_live(live: Sequence[int]) -> bool:
    return len(live) == len(set(live))


@dataclass(frozen=True)
class ClosureTraceV1:
    """Frozen multi-pass closure trace for Lean↔Python parity."""

    id: str
    live: tuple[int, ...]
    batches: tuple[tuple[QueryResultV1, ...], ...]
    holes: tuple[tuple[int, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "live": list(self.live),
            "batches": [[r.to_dict() for r in batch] for batch in self.batches],
            "holes": [list(h) for h in self.holes] if self.holes is not None else None,
        }


@dataclass(frozen=True)
class ClosureStabilizationEvidenceV1:
    domain_sizes: tuple[int, ...]
    strict_removal_bound: int
    nonempty_invariant_bound: int
    assignment_search_bound: int
    total_removed: int
    within_strict_bound: bool
    within_nonempty_bound: bool | None
    unique_live: bool
    batch_coverage: bool
    preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLOSURE_STABILIZATION_SCHEMA,
            "domain_sizes": list(self.domain_sizes),
            "strict_removal_upper_bound": self.strict_removal_bound,
            "nonempty_invariant_upper_bound": self.nonempty_invariant_bound,
            "assignment_search_bound": self.assignment_search_bound,
            "total_removed": self.total_removed,
            "within_strict_bound": self.within_strict_bound,
            "within_nonempty_bound": self.within_nonempty_bound,
            "unique_live": self.unique_live,
            "batch_coverage": self.batch_coverage,
            "preconditions": list(self.preconditions),
            "wall_clock_claim": False,
            "maximality_claimed": False,
            "bound_ast_id": BOUND_AST_STRICT_REMOVALS,
        }

    def certificate_sha256(self) -> str:
        return content_sha(self.to_dict())


def build_stabilization_evidence(
    *,
    domain_sizes: Sequence[int],
    total_removed: int,
    unique: bool,
    coverage: bool,
    nonempty_invariant: bool | None = None,
) -> ClosureStabilizationEvidenceV1:
    sizes = tuple(int(d) for d in domain_sizes)
    strict_b = strict_removal_upper_bound(sizes)
    nonempty_b = nonempty_invariant_upper_bound(sizes)
    assign_b = assignment_search_bound(sizes)
    preconditions = [
        "exact_closure_certificate_checked",
        "finite_live_domain",
        "batch_fairness_coverage",
        "not_wall_clock",
        "no_maximality_without_proof",
    ]
    if unique:
        preconditions.append("unique_live_candidates")
    if nonempty_invariant:
        preconditions.append("nonempty_domains_invariant")
    return ClosureStabilizationEvidenceV1(
        domain_sizes=sizes,
        strict_removal_bound=strict_b,
        nonempty_invariant_bound=nonempty_b,
        assignment_search_bound=assign_b,
        total_removed=int(total_removed),
        within_strict_bound=int(total_removed) <= strict_b,
        within_nonempty_bound=(
            int(total_removed) <= nonempty_b if nonempty_invariant else None
        ),
        unique_live=unique,
        batch_coverage=coverage,
        preconditions=tuple(preconditions),
    )


def cross_check_trace(trace: ClosureTraceV1) -> dict[str, Any]:
    """Cross-check a frozen Python closure trace against KERN-05 bounds."""

    cur = list(trace.live)
    coverage = True
    progress_ok = True
    for batch in trace.batches:
        if not batch_covers_removable(cur, batch):
            coverage = False
        if not strict_progress_or_fixed_point(cur, batch):
            progress_ok = False
        cur = close_pass_results(cur, batch)

    if trace.holes is not None:
        sizes = hole_sizes(trace.holes)
        removed_holes = total_removed_across_holes(trace.holes, trace.batches)
        nonempty = all_nonempty(trace.holes)
    else:
        # Flat live bag: treat cardinality as a single domain size for Σ bound.
        sizes = (len(trace.live),)
        removed_holes = total_removed_across(trace.live, trace.batches)
        nonempty = len(trace.live) > 0

    evidence = build_stabilization_evidence(
        domain_sizes=sizes,
        total_removed=removed_holes,
        unique=unique_live(trace.live),
        coverage=coverage,
        nonempty_invariant=nonempty,
    )
    return {
        "id": trace.id,
        "total_removed": removed_holes,
        "strict_removal_upper_bound": evidence.strict_removal_bound,
        "nonempty_invariant_upper_bound": evidence.nonempty_invariant_bound,
        "assignment_search_bound": evidence.assignment_search_bound,
        "within_strict_bound": evidence.within_strict_bound,
        "within_nonempty_bound": evidence.within_nonempty_bound,
        "unique_live": evidence.unique_live,
        "batch_coverage": evidence.batch_coverage,
        "strict_progress_or_fixed_point": progress_ok,
        "closure_vs_assignment_separated": closure_complexity_separated(sizes),
        "certificate_sha256": evidence.certificate_sha256(),
    }


def _qr(
    hole: int, candidate: int, verdict: str, replay_ok: bool
) -> QueryResultV1:
    evidence = None
    binding = None
    if verdict == "unsupported" and replay_ok:
        evidence = _fixture_evidence(candidate)
        binding = _FIXTURE_BINDING
    return QueryResultV1(
        hole=hole,
        candidate=candidate,
        verdict=verdict,  # type: ignore[arg-type]
        replay_ok=replay_ok,
        refutation_evidence=evidence,
        expected_binding=binding,
    )


def frozen_closure_traces() -> tuple[ClosureTraceV1, ...]:
    """Frozen traces mirroring Lean ``fixture_*`` and Python parity cases."""

    return (
        ClosureTraceV1(
            id="flat_two_removals",
            live=(0, 1, 2, 3),
            batches=(
                (_qr(0, 1, "unsupported", True),),
                (_qr(0, 3, "unsupported", True),),
            ),
        ),
        ClosureTraceV1(
            id="fixed_point_empty_batch",
            live=(0, 1, 2),
            batches=((),),
        ),
        ClosureTraceV1(
            id="holes_two_three_one_removal",
            live=(10, 11, 20, 21, 22),
            batches=((_qr(0, 11, "unsupported", True),),),
            holes=((10, 11), (20, 21, 22)),
        ),
        ClosureTraceV1(
            id="nonempty_invariant_shrink",
            live=(10, 11, 20, 21, 22),
            batches=(
                (_qr(0, 11, "unsupported", True),),
                (_qr(1, 22, "unsupported", True),),
            ),
            holes=((10, 11), (20, 21, 22)),
        ),
    )


def frozen_fixture_rows() -> list[dict[str, Any]]:
    rows = []
    for trace in frozen_closure_traces():
        check = cross_check_trace(trace)
        rows.append(
            {
                "id": trace.id,
                "live": list(trace.live),
                "holes": [list(h) for h in trace.holes] if trace.holes else None,
                "predicted_total_removed": check["total_removed"],
                "strict_removal_upper_bound": check["strict_removal_upper_bound"],
                "nonempty_invariant_upper_bound": check[
                    "nonempty_invariant_upper_bound"
                ],
                "assignment_search_bound": check["assignment_search_bound"],
                "within_strict_bound": check["within_strict_bound"],
                "batch_coverage": check["batch_coverage"],
            }
        )
    # Static lattice fixtures (no trace).
    for sizes, sid in (([2, 3], "lattice_two_three"), ([1, 1, 1], "lattice_three_ones")):
        rows.append(
            {
                "id": sid,
                "domain_sizes": sizes,
                "strict_removal_upper_bound": strict_removal_upper_bound(sizes),
                "nonempty_invariant_upper_bound": nonempty_invariant_upper_bound(sizes),
                "assignment_search_bound": assignment_search_bound(sizes),
                "closure_vs_assignment_separated": closure_complexity_separated(sizes),
            }
        )
    return rows


def mutate_duplicate_live(live: Sequence[int]) -> list[int]:
    """Mutation helper: duplicate the first live candidate (bookkeeping error)."""

    xs = list(live)
    if not xs:
        return xs
    return [xs[0], *xs]


def mutate_drop_removal_bookkeeping(
    live: Sequence[int], results: Sequence[QueryResultV1]
) -> list[int]:
    """Mutation: pretend a non-fixed pass made no progress (ignore removals)."""

    del results
    return list(live)


__all__ = [
    "BOUND_AST_LIVE_UPPER",
    "BOUND_AST_STRICT_REMOVALS",
    "CLOSURE_STABILIZATION_SCHEMA",
    "ClosureStabilizationEvidenceV1",
    "ClosureTraceV1",
    "QueryResultV1",
    "all_nonempty",
    "assignment_search_bound",
    "batch_covers_removable",
    "batch_mentions",
    "build_stabilization_evidence",
    "close_pass_holes",
    "close_pass_results",
    "closure_complexity_separated",
    "cross_check_trace",
    "frozen_closure_traces",
    "frozen_fixture_rows",
    "hole_sizes",
    "is_fixed_point",
    "is_removable_in_batch",
    "list_sum",
    "mutate_drop_removal_bookkeeping",
    "mutate_duplicate_live",
    "nonempty_invariant_upper_bound",
    "removed_candidates",
    "removed_count",
    "strict_progress_or_fixed_point",
    "strict_removal_upper_bound",
    "total_live_count",
    "total_removed_across",
    "total_removed_across_holes",
    "unique_live",
]
