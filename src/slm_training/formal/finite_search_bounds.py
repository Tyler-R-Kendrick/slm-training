"""Finite completion-search upper bounds (KERN-03 / SLM-521).

Mirrors ``LeverProofLean.FiniteSearchBounds``: abstract finite-search model
parameterized by hole/domain sizes (from proof-bearing complete domains) and
per-expansion / per-verification query costs under the named
``ExpansionVerificationQueryModel``.

Bounds are query/machine-model counts — **not** wall-clock latency claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Sequence

from slm_training.formal.complete_domain import (
    CompleteDomainEvidenceV1,
    build_complete_domain,
)
from slm_training.harness_core.lineage.records import content_sha

FINITE_SEARCH_BOUNDS_SCHEMA = "finite_search_bounds/v1"
BOUND_AST_ID = "bound.finite_search.prefix_tree.v1"
COST_MODEL_NAME = "ExpansionVerificationQueryModel"
U64_MAX = (1 << 64) - 1

OverflowStatus = Literal["fits_u64", "overflow_u64"]


@dataclass(frozen=True)
class ExpansionVerificationQueryModel:
    """Named query-cost model (not wall-clock)."""

    cost_per_expansion: int
    cost_per_verification: int

    def __post_init__(self) -> None:
        if self.cost_per_expansion < 0 or self.cost_per_verification < 0:
            raise ValueError("query costs must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": COST_MODEL_NAME,
            "cost_per_expansion": self.cost_per_expansion,
            "cost_per_verification": self.cost_per_verification,
            "wall_clock": False,
        }


UNIT_COST_MODEL = ExpansionVerificationQueryModel(
    cost_per_expansion=1,
    cost_per_verification=1,
)


def list_prod(sizes: Sequence[int]) -> int:
    """Empty product is 1 (matches Lean ``listProd``)."""

    out = 1
    for d in sizes:
        if d < 0:
            raise ValueError("domain sizes must be non-negative")
        out *= int(d)
    return out


def complete_assignment_count(sizes: Sequence[int]) -> int:
    """``P = ∏ d_i``."""

    return list_prod(sizes)


def prefix_products(sizes: Sequence[int]) -> tuple[int, ...]:
    """Cumulative products ``∏_{i ≤ k} d_i`` for each hole index ``k``."""

    out: list[int] = []
    pref = 1
    for d in sizes:
        if d < 0:
            raise ValueError("domain sizes must be non-negative")
        pref *= int(d)
        out.append(pref)
    return tuple(out)


def prefix_tree_node_bound(sizes: Sequence[int]) -> int:
    """``1 + Σ_k ∏_{i ≤ k} d_i``."""

    return 1 + sum(prefix_products(sizes))


def coarse_node_bound(sizes: Sequence[int]) -> int:
    """``1 + h·P`` (valid upper bound when every ``d_i ≥ 1``)."""

    return 1 + len(sizes) * complete_assignment_count(sizes)


def all_domains_positive(sizes: Sequence[int]) -> bool:
    return all(int(d) >= 1 for d in sizes)


def prefix_tree_le_coarse(sizes: Sequence[int]) -> bool:
    """Python mirror of ``prefix_tree_le_one_plus_holes_mul_assignments``."""

    if not all_domains_positive(sizes):
        return True  # antecedent false — implication holds vacuously for callers
    return prefix_tree_node_bound(sizes) <= coarse_node_bound(sizes)


def sizes_from_proved_domains(
    domains: Sequence[CompleteDomainEvidenceV1],
) -> tuple[int, ...]:
    """Consume proved domain sizes — never bare Boolean coverage flags."""

    sizes: list[int] = []
    for domain in domains:
        if not domain.is_proved_complete():
            raise ValueError(
                "finite-search bounds require proved-complete domains "
                f"(authority={domain.authority_name()!r})"
            )
        sizes.append(len(domain.candidates))
    return tuple(sizes)


def u64_status(*values: int) -> OverflowStatus:
    return "fits_u64" if all(0 <= v <= U64_MAX for v in values) else "overflow_u64"


@dataclass(frozen=True)
class SearchTraceV1:
    expansions: int
    verifications: int

    def __post_init__(self) -> None:
        if self.expansions < 0 or self.verifications < 0:
            raise ValueError("trace counters must be non-negative")


def respects_upper_bounds(sizes: Sequence[int], trace: SearchTraceV1) -> bool:
    return (
        trace.expansions <= prefix_tree_node_bound(sizes)
        and trace.verifications <= complete_assignment_count(sizes)
    )


def pruning_memo_cannot_increase_bound(
    sizes: Sequence[int],
    before: SearchTraceV1,
    after: SearchTraceV1,
) -> bool:
    """Pruning/memo that only reduces visits stays inside the certified bound."""

    if not respects_upper_bounds(sizes, before):
        return False
    if after.expansions > before.expansions or after.verifications > before.verifications:
        return False
    return respects_upper_bounds(sizes, after)


def enumerate_complete_assignments(domain_values: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """Exhaustive assignment enumeration (algorithm correspondence fixture)."""

    if not domain_values:
        return [()]
    return [tuple(row) for row in product(*domain_values)]


def enumerate_prefix_tree_nodes(domain_values: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """Prefix-tree nodes including the empty root prefix."""

    nodes: list[tuple[int, ...]] = [()]
    if not domain_values:
        return nodes
    # Depth-k nodes are assignments to the first k holes.
    for depth in range(1, len(domain_values) + 1):
        nodes.extend(tuple(row) for row in product(*domain_values[:depth]))
    return nodes


@dataclass(frozen=True)
class FiniteSearchBoundEvidenceV1:
    """Checked finite-search bound evidence for one instance."""

    domain_sizes: tuple[int, ...]
    cost_model: ExpansionVerificationQueryModel
    assignment_count: int
    prefix_tree_nodes: int
    coarse_nodes: int
    expansion_query_bound: int
    verification_query_bound: int
    total_query_bound: int
    u64_overflow: OverflowStatus
    preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FINITE_SEARCH_BOUNDS_SCHEMA,
            "domain_sizes": list(self.domain_sizes),
            "cost_model": self.cost_model.to_dict(),
            "complete_assignment_count": self.assignment_count,
            "prefix_tree_node_bound": self.prefix_tree_nodes,
            "coarse_node_bound": self.coarse_nodes,
            "expansion_query_bound": self.expansion_query_bound,
            "verification_query_bound": self.verification_query_bound,
            "total_query_bound": self.total_query_bound,
            "u64_overflow": self.u64_overflow,
            "preconditions": list(self.preconditions),
            "wall_clock_claim": False,
            "bound_ast_id": BOUND_AST_ID,
        }

    def certificate_sha256(self) -> str:
        return content_sha(self.to_dict())


def build_finite_search_bound(
    domain_sizes: Sequence[int],
    *,
    cost_model: ExpansionVerificationQueryModel = UNIT_COST_MODEL,
) -> FiniteSearchBoundEvidenceV1:
    sizes = tuple(int(d) for d in domain_sizes)
    if any(d < 0 for d in sizes):
        raise ValueError("domain sizes must be non-negative")
    p = complete_assignment_count(sizes)
    nodes = prefix_tree_node_bound(sizes)
    coarse = coarse_node_bound(sizes)
    exp = nodes * cost_model.cost_per_expansion
    ver = p * cost_model.cost_per_verification
    total = exp + ver
    preconditions = (
        "finite_domain_sizes",
        "algorithm_correspondence_prefix_tree_enumeration",
        f"cost_model:{COST_MODEL_NAME}",
        "not_wall_clock",
    )
    if all_domains_positive(sizes):
        preconditions = (*preconditions, "all_domain_sizes_ge_1_for_coarse_bound")
    return FiniteSearchBoundEvidenceV1(
        domain_sizes=sizes,
        cost_model=cost_model,
        assignment_count=p,
        prefix_tree_nodes=nodes,
        coarse_nodes=coarse,
        expansion_query_bound=exp,
        verification_query_bound=ver,
        total_query_bound=total,
        u64_overflow=u64_status(p, nodes, coarse, exp, ver, total),
        preconditions=preconditions,
    )


def build_finite_search_bound_from_domains(
    domains: Sequence[CompleteDomainEvidenceV1],
    *,
    cost_model: ExpansionVerificationQueryModel = UNIT_COST_MODEL,
) -> FiniteSearchBoundEvidenceV1:
    return build_finite_search_bound(
        sizes_from_proved_domains(domains),
        cost_model=cost_model,
    )


def export_four_axis_bound_evidence(
    evidence: FiniteSearchBoundEvidenceV1,
    *,
    formal_preflight_sha256: str | None = None,
):
    """Export bound evidence through the four-axis ledger interface."""

    # Lazy import: revmath.__init__ eagerly loads quantitative_bound which
    # imports this module (HARN-07). Avoid the cycle at module import time.
    from slm_training.harnesses.reasoning.revmath.schemas import (
        AxisCertificateRefV1,
        RevmathFourAxisAnalysisV1,
    )

    cert = evidence.certificate_sha256()
    return RevmathFourAxisAnalysisV1(
        assumption_strength=AxisCertificateRefV1(
            axis="assumption_strength",
            status="not_claimed",
            notes="KERN-03 states resource bounds, not RM subsystem strength",
        ),
        computability=AxisCertificateRefV1(
            axis="computability",
            status="proved_axis",
            certificate_sha256=cert,
            theorem_ref="FiniteSearchBounds.completeAssignmentCount_eq",
            notes="Finite complete domains imply finite exhaustive assignment search",
        ),
        resource_bounds=AxisCertificateRefV1(
            axis="resource_bounds",
            status="proved_axis",
            certificate_sha256=cert,
            theorem_ref=(
                "FiniteSearchBounds.prefix_tree_le_one_plus_holes_mul_assignments"
            ),
            bound_ast_id=BOUND_AST_ID,
            notes=(
                f"{COST_MODEL_NAME} query bound; "
                "not a wall-clock claim"
            ),
        ),
        implementation_refinement=AxisCertificateRefV1(
            axis="implementation_refinement",
            status="empirical_remainder",
            notes=(
                "Frozen Python fixtures cross-check VSS/enumerator counts; "
                "runtime wall-clock remains empirical"
            ),
        ),
        formal_preflight_sha256=formal_preflight_sha256,
    )


def cross_check_enumeration(
    domain_values: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Cross-check exact counts against exhaustive Python enumeration."""

    sizes = tuple(len(vals) for vals in domain_values)
    assignments = enumerate_complete_assignments(domain_values)
    nodes = enumerate_prefix_tree_nodes(domain_values)
    evidence = build_finite_search_bound(sizes)
    return {
        "domain_sizes": list(sizes),
        "enumerated_assignments": len(assignments),
        "enumerated_prefix_nodes": len(nodes),
        "predicted_assignments": evidence.assignment_count,
        "predicted_prefix_nodes": evidence.prefix_tree_nodes,
        "assignments_match": len(assignments) == evidence.assignment_count,
        "prefix_nodes_match": len(nodes) == evidence.prefix_tree_nodes,
        "coarse_ok": prefix_tree_le_coarse(sizes),
        "u64_overflow": evidence.u64_overflow,
    }


# Frozen fixture table (mirrors resources/formal/finite_search_bounds_fixtures.v1.json).
_FIXTURE_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "empty_holes", "domains": []},
    {"id": "single_hole_three", "domains": [[0, 1, 2]]},
    {"id": "two_by_three", "domains": [[0, 1], [0, 1, 2]]},
    {"id": "zero_domain_middle", "domains": [[0, 1], [], [0, 1, 2, 3]]},
    {"id": "all_singletons", "domains": [[7], [8], [9]]},
    {
        "id": "overflow_u64_product",
        "domain_sizes": [1 << 32, 1 << 32, 4],
        "size_only": True,
    },
)


def frozen_fixture_rows() -> list[dict[str, Any]]:
    """Materialize fixture cross-check rows for tests and export."""

    rows: list[dict[str, Any]] = []
    for spec in _FIXTURE_SPECS:
        if spec.get("size_only"):
            sizes = tuple(int(d) for d in spec["domain_sizes"])
            evidence = build_finite_search_bound(sizes)
            rows.append(
                {
                    "id": spec["id"],
                    "domain_sizes": list(sizes),
                    "enumerated_assignments": None,
                    "predicted_assignments": evidence.assignment_count,
                    "predicted_prefix_nodes": evidence.prefix_tree_nodes,
                    "u64_overflow": evidence.u64_overflow,
                    "assignments_match": None,
                    "prefix_nodes_match": None,
                }
            )
            continue
        check = cross_check_enumeration(spec["domains"])
        check["id"] = spec["id"]
        rows.append(check)
    return rows


def proved_singleton_domains_fixture() -> FiniteSearchBoundEvidenceV1:
    """Connect proved CompleteDomainEvidence sizes into the bound model."""

    domains = (
        build_complete_domain("s1", [7], [7]),
        build_complete_domain("s1", [8], [8]),
    )
    return build_finite_search_bound_from_domains(domains)


__all__ = [
    "BOUND_AST_ID",
    "COST_MODEL_NAME",
    "FINITE_SEARCH_BOUNDS_SCHEMA",
    "UNIT_COST_MODEL",
    "U64_MAX",
    "ExpansionVerificationQueryModel",
    "FiniteSearchBoundEvidenceV1",
    "SearchTraceV1",
    "all_domains_positive",
    "build_finite_search_bound",
    "build_finite_search_bound_from_domains",
    "coarse_node_bound",
    "complete_assignment_count",
    "cross_check_enumeration",
    "enumerate_complete_assignments",
    "enumerate_prefix_tree_nodes",
    "export_four_axis_bound_evidence",
    "frozen_fixture_rows",
    "list_prod",
    "prefix_products",
    "prefix_tree_le_coarse",
    "prefix_tree_node_bound",
    "proved_singleton_domains_fixture",
    "pruning_memo_cannot_increase_bound",
    "respects_upper_bounds",
    "sizes_from_proved_domains",
    "u64_status",
]
