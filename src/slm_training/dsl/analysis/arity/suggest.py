"""Suggest feasible robust coding arms and filter disproven ones (CAP0-03)."""

from __future__ import annotations

from dataclasses import dataclass

from slm_training.dsl.analysis.arity.coding import (
    build_mds_7_4_2_3,
    build_shortened_ternary_hamming_7_4_3,
    singleton_upper_bound,
    smallest_injective_arity,
    verify_code,
)


@dataclass(frozen=True)
class RobustArm:
    """One candidate robust coding arm for a state-count target."""

    q: int
    n: int
    d: int
    feasible: bool
    reason: str
    construction: str | None


#: Arms that CAP0-03 explicitly removes from the toy robust matrix.
# The notation (K, d) in the issue maps to alphabet size q=K and block length n.
_REMOVED_ARMS: frozenset[tuple[int, int, int]] = frozenset({(6, 4, 4), (3, 6, 6)})


def suggest_robust_arms(
    state_count: int,
    *,
    dimensions: int = 4,
    max_alphabet: int = 8,
    max_distance: int = 8,
) -> tuple[RobustArm, ...]:
    """Return candidate robust arms, excluding arms disproven by CAP0-03.

    For the toy M=41, n=4 target this keeps:
      * (q=7, n=4, d=3) MDS construction;
      * (q=3, n=7, d=3) shortened Hamming construction;
      * optionally a ternary d=7 slack control when requested by the caller.

    It removes the infeasible (K=6, d=4) -> (q=6, n=4, d=4) and
    (K=3, d=6) -> (q=3, n=6, d=6) arms.
    """
    if state_count <= 0:
        raise ValueError("state_count must be positive")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    arms: list[RobustArm] = []
    for q in range(2, max_alphabet + 1):
        for d in range(1, min(max_distance, dimensions + 1) + 1):
            if (q, dimensions, d) in _REMOVED_ARMS:
                continue
            bound = singleton_upper_bound(q, dimensions, d)
            if bound < state_count:
                continue
            construction: str | None = None
            if q == 7 and dimensions == 4 and d == 3:
                code = build_mds_7_4_2_3()
                result = verify_code(code, q=7, n=4, required_size=state_count, required_distance=d)
                if result.ok:
                    construction = "mds_7_4_2_3"
            elif q == 3 and dimensions == 7 and d == 3:
                code = build_shortened_ternary_hamming_7_4_3()
                result = verify_code(code, q=3, n=7, required_size=state_count, required_distance=d)
                if result.ok:
                    construction = "shortened_ternary_hamming_7_4_3"
            arms.append(
                RobustArm(
                    q=q,
                    n=dimensions,
                    d=d,
                    feasible=True,
                    reason=f"Singleton bound {bound} >= {state_count}",
                    construction=construction,
                )
            )

    # Ternary shortened-Hamming arm with n=7, d=3 (verified construction).
    # CAP0-03 keeps the ternary n=7 construction; d=6 and d=7 unverified arms are
    # not emitted because no local construction is available.
    code = build_shortened_ternary_hamming_7_4_3()
    result = verify_code(code, q=3, n=7, required_size=state_count, required_distance=3)
    if result.ok and 3 <= max_distance:
        arms.append(
            RobustArm(
                q=3,
                n=7,
                d=3,
                feasible=True,
                reason="verified shortened ternary Hamming construction",
                construction="shortened_ternary_hamming_7_4_3",
            )
        )

    return tuple(arms)


def smallest_feasible_alphabet(state_count: int, dimensions: int) -> int:
    """Smallest q such that q^dimensions >= state_count and Singleton does not reject it.

    For the toy case this returns 7 because q=6 is infeasible for d=3, n=4.
    """
    q = max(2, smallest_injective_arity(state_count, dimensions))
    # Conservatively require that Singleton with d=3 admits it; this is the toy robust bound.
    while singleton_upper_bound(q, dimensions, 3) < state_count:
        q += 1
    return q
