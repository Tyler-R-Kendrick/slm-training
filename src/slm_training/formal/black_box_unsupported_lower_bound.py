"""Black-box UNSUPPORTED query lower bound (KERN-04 / SLM-522).

Mirrors ``LeverProofLean.BlackBoxUnsupportedLowerBound``: over a finite complete
domain of ``P`` assignments, a solver that learns only one Boolean validity
answer per complete-assignment query must query all ``P`` assignments in the
worst case to soundly certify UNSUPPORTED.

Model-relative — **not** a universal solver lower bound. Symbolic constraints,
certificates, shared residual states, and structure escape the assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Sequence

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathFourAxisAnalysisV1,
)


def _complete_assignment_count(sizes: Sequence[int]) -> int:
    """``P = ∏ d_i`` (empty product 1). Local copy avoids formal↔revmath import cycles."""

    out = 1
    for d in sizes:
        if int(d) < 0:
            raise ValueError("domain sizes must be non-negative")
        out *= int(d)
    return out

BLACK_BOX_UNSUPPORTED_SCHEMA = "black_box_unsupported_lower_bound/v1"
BOUND_AST_ID = "bound.black_box.unsupported_query_lower.v1"
COST_MODEL_NAME = "BlackBoxVerifierQueryModel"
U64_MAX = (1 << 64) - 1

Oracle = Callable[[int], bool]


class EscapeClass(str, Enum):
    """Optimization classes that leave the black-box Boolean-query model."""

    SYMBOLIC_CONSTRAINT = "symbolic_constraint"
    CERTIFICATE = "certificate"
    SHARED_RESIDUAL_STATE = "shared_residual_state"
    STRUCTURE = "structure"


ESCAPE_NOTES: dict[EscapeClass, str] = {
    EscapeClass.SYMBOLIC_CONSTRAINT: (
        "symbolic constraints query regions, not black-box assignment Booleans"
    ),
    EscapeClass.CERTIFICATE: (
        "certificates carry proof structure beyond one Bool per assignment"
    ),
    EscapeClass.SHARED_RESIDUAL_STATE: (
        "shared residual states correlate answers across assignments"
    ),
    EscapeClass.STRUCTURE: (
        "structure/prefix pruning rejects many leaves without leaf Boolean queries"
    ),
}


@dataclass(frozen=True)
class BlackBoxVerifierQueryModel:
    """Named query model: one Bool validity answer per complete assignment."""

    assignment_count: int

    def __post_init__(self) -> None:
        if self.assignment_count < 0:
            raise ValueError("assignment_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": COST_MODEL_NAME,
            "assignment_count": self.assignment_count,
            "wall_clock": False,
            "universal_solver_claim": False,
        }


def model_from_domain_sizes(sizes: Sequence[int]) -> BlackBoxVerifierQueryModel:
    return BlackBoxVerifierQueryModel(
        assignment_count=_complete_assignment_count(sizes)
    )


def worst_case_query_lower_bound(model: BlackBoxVerifierQueryModel) -> int:
    return model.assignment_count


def all_invalid(_index: int) -> bool:
    return False


def one_valid(i: int) -> Oracle:
    return lambda j: j == i


def oracles_agree_on(omega1: Oracle, omega2: Oracle, queried: Sequence[int]) -> bool:
    return all(omega1(j) == omega2(j) for j in queried)


def sound_unsupported_claim(p: int, omega: Oracle) -> bool:
    return all(omega(k) is False for k in range(p))


def early_stop_distinguishes(
    p: int, i: int, queried: Sequence[int]
) -> tuple[bool, bool]:
    """Return (oracles_agree, one_valid_not_sound) for a missing index ``i``."""

    if not (0 <= i < p):
        raise ValueError("i must satisfy 0 <= i < P")
    if i in queried:
        raise ValueError("i must be missing from queried")
    agree = oracles_agree_on(all_invalid, one_valid(i), queried)
    not_sound = not sound_unsupported_claim(p, one_valid(i))
    return agree, not_sound


def sound_unsupported_requires_full_coverage(
    p: int,
    queried: Sequence[int],
    *,
    assume_sound_for_all_consistent: bool = True,
) -> bool:
    """Python mirror of the coverage corollary.

    When every all-false-consistent oracle must admit a sound UNSUPPORTED claim,
    every index ``0 .. P-1`` must appear in ``queried``. If coverage is incomplete,
    the distinguisher shows the assumption fails.
    """

    if not assume_sound_for_all_consistent:
        return False
    missing = [i for i in range(p) if i not in queried]
    if not missing:
        return True
    for i in missing:
        agree, not_sound = early_stop_distinguishes(p, i, queried)
        if not (agree and not_sound):
            return False
    return False  # incomplete coverage ⇒ soundness-for-all-consistent fails


def black_box_unsupported_query_lower_bound(
    p: int, queried: Sequence[int]
) -> bool:
    """True iff ``queried`` covers every index (Lean main theorem conclusion)."""

    return all(i in queried for i in range(p))


def escape_leaves_black_box(escape: EscapeClass) -> bool:
    return True


@dataclass(frozen=True)
class BlackBoxLowerBoundEvidenceV1:
    domain_sizes: tuple[int, ...]
    model: BlackBoxVerifierQueryModel
    lower_bound_queries: int
    preconditions: tuple[str, ...]
    escape_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BLACK_BOX_UNSUPPORTED_SCHEMA,
            "domain_sizes": list(self.domain_sizes),
            "cost_model": self.model.to_dict(),
            "worst_case_query_lower_bound": self.lower_bound_queries,
            "preconditions": list(self.preconditions),
            "escape_classes": list(self.escape_classes),
            "wall_clock_claim": False,
            "universal_solver_claim": False,
            "bound_ast_id": BOUND_AST_ID,
            "theorem_ref": (
                "BlackBoxUnsupportedLowerBound.black_box_unsupported_query_lower_bound"
            ),
        }

    def certificate_sha256(self) -> str:
        return content_sha(self.to_dict())


def build_black_box_lower_bound(
    domain_sizes: Sequence[int] | None = None,
    *,
    assignment_count: int | None = None,
) -> BlackBoxLowerBoundEvidenceV1:
    if domain_sizes is not None:
        sizes = tuple(int(d) for d in domain_sizes)
        if any(d < 0 for d in sizes):
            raise ValueError("domain sizes must be non-negative")
        model = model_from_domain_sizes(sizes)
    elif assignment_count is not None:
        sizes = ()
        model = BlackBoxVerifierQueryModel(assignment_count=int(assignment_count))
    else:
        raise ValueError("provide domain_sizes or assignment_count")
    return BlackBoxLowerBoundEvidenceV1(
        domain_sizes=sizes,
        model=model,
        lower_bound_queries=worst_case_query_lower_bound(model),
        preconditions=(
            "finite_complete_domain",
            f"cost_model:{COST_MODEL_NAME}",
            "one_boolean_validity_answer_per_complete_assignment",
            "deterministic_sound_refuter",
            "model_relative_not_universal_solver",
            "not_wall_clock",
        ),
        escape_classes=tuple(e.value for e in EscapeClass),
    )


def export_four_axis_lower_bound_evidence(
    evidence: BlackBoxLowerBoundEvidenceV1,
    *,
    formal_preflight_sha256: str | None = None,
) -> RevmathFourAxisAnalysisV1:
    """Export the lower-bound claim through the four-axis ledger."""

    cert = evidence.certificate_sha256()
    return RevmathFourAxisAnalysisV1(
        assumption_strength=AxisCertificateRefV1(
            axis="assumption_strength",
            status="not_claimed",
            notes=(
                "KERN-04 is model-relative to BlackBoxVerifierQueryModel; "
                "not an RM subsystem-strength claim"
            ),
        ),
        computability=AxisCertificateRefV1(
            axis="computability",
            status="proved_axis",
            certificate_sha256=cert,
            theorem_ref=(
                "BlackBoxUnsupportedLowerBound.early_stop_distinguishes"
            ),
            notes=(
                "Early stop leaves an unqueried assignment distinguishing "
                "all-invalid from one-valid worlds"
            ),
        ),
        resource_bounds=AxisCertificateRefV1(
            axis="resource_bounds",
            status="proved_axis",
            certificate_sha256=cert,
            theorem_ref=(
                "BlackBoxUnsupportedLowerBound.black_box_unsupported_query_lower_bound"
            ),
            bound_ast_id=BOUND_AST_ID,
            notes=(
                f"{COST_MODEL_NAME}: worst-case queries ≥ P; "
                "not a wall-clock or universal-solver claim"
            ),
        ),
        implementation_refinement=AxisCertificateRefV1(
            axis="implementation_refinement",
            status="empirical_remainder",
            notes=(
                "Escapes (symbolic/certificate/residual/structure) and runtime "
                "schedulers remain outside the black-box model"
            ),
        ),
        formal_preflight_sha256=formal_preflight_sha256,
    )


def exhaustive_partial_query_unsound(p: int) -> dict[str, Any]:
    """Exhaustively check: every proper subset of queries is unsound for all-false."""

    if p < 0:
        raise ValueError("P must be non-negative")
    if p > 12:
        raise ValueError("exhaustive check capped at P<=12")
    universe = list(range(p))
    failures: list[dict[str, Any]] = []
    checked = 0
    for k in range(p):  # proper subsets only (length < P)
        for queried in combinations(universe, k):
            checked += 1
            missing = [i for i in universe if i not in queried]
            # For soundness-for-all-consistent to hold, every missing i must fail
            # the distinguisher — it never does.
            for i in missing:
                agree, not_sound = early_stop_distinguishes(p, i, queried)
                if not (agree and not_sound):
                    failures.append(
                        {
                            "queried": list(queried),
                            "missing": i,
                            "agree": agree,
                            "not_sound": not_sound,
                        }
                    )
            # Coverage conclusion of the main theorem fails for proper subsets.
            if black_box_unsupported_query_lower_bound(p, queried):
                failures.append(
                    {"queried": list(queried), "unexpected_full_coverage": True}
                )
    # Full cover succeeds.
    full_ok = black_box_unsupported_query_lower_bound(p, universe)
    return {
        "P": p,
        "partial_subsets_checked": checked,
        "failures": failures,
        "full_cover_ok": full_ok,
        "lower_bound": p,
        "ok": full_ok and not failures,
    }


# Frozen fixture table (mirrors resources/formal/black_box_unsupported_fixtures.v1.json).
_FIXTURE_SPECS: tuple[dict[str, Any], ...] = (
    {"id": "P0_empty", "P": 0},
    {"id": "P1_singleton", "P": 1},
    {"id": "P2_pair", "P": 2},
    {"id": "P3_triple", "P": 3},
    {"id": "P4_exhaustive", "P": 4},
    {"id": "sizes_2x3", "domain_sizes": [2, 3]},
    {"id": "sizes_empty_holes", "domain_sizes": []},
)


def frozen_fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _FIXTURE_SPECS:
        if "domain_sizes" in spec:
            evidence = build_black_box_lower_bound(spec["domain_sizes"])
            p = evidence.lower_bound_queries
            check = exhaustive_partial_query_unsound(p) if p <= 6 else None
            rows.append(
                {
                    "id": spec["id"],
                    "domain_sizes": list(spec["domain_sizes"]),
                    "P": p,
                    "lower_bound": p,
                    "exhaustive_ok": None if check is None else check["ok"],
                    "u64_overflow": (
                        "fits_u64" if p <= U64_MAX else "overflow_u64"
                    ),
                }
            )
            continue
        p = int(spec["P"])
        check = exhaustive_partial_query_unsound(p)
        rows.append(
            {
                "id": spec["id"],
                "domain_sizes": None,
                "P": p,
                "lower_bound": p,
                "exhaustive_ok": check["ok"],
                "partial_subsets_checked": check["partial_subsets_checked"],
                "u64_overflow": "fits_u64",
            }
        )
    return rows


__all__ = [
    "BLACK_BOX_UNSUPPORTED_SCHEMA",
    "BOUND_AST_ID",
    "COST_MODEL_NAME",
    "ESCAPE_NOTES",
    "U64_MAX",
    "BlackBoxLowerBoundEvidenceV1",
    "BlackBoxVerifierQueryModel",
    "EscapeClass",
    "all_invalid",
    "black_box_unsupported_query_lower_bound",
    "build_black_box_lower_bound",
    "early_stop_distinguishes",
    "escape_leaves_black_box",
    "exhaustive_partial_query_unsound",
    "export_four_axis_lower_bound_evidence",
    "frozen_fixture_rows",
    "model_from_domain_sizes",
    "one_valid",
    "oracles_agree_on",
    "sound_unsupported_claim",
    "sound_unsupported_requires_full_coverage",
    "worst_case_query_lower_bound",
]
