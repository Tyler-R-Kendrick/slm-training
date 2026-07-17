"""Exact coding-theory reference functions and verified constructions (CAP0-03)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable


@dataclass(frozen=True)
class CodeVerification:
    """Result of verifying a code against declared parameters."""

    ok: bool
    q: int
    n: int
    size: int
    minimum_distance: int
    required_size: int
    required_distance: int
    messages: tuple[str, ...]


def _validate_q(q: int) -> None:
    if not isinstance(q, int) or q < 2:
        raise ValueError(f"alphabet size q must be an integer >= 2, got {q}")


def _validate_finite_field_symbol(symbol: int, q: int) -> None:
    if not isinstance(symbol, int) or not 0 <= symbol < q:
        raise ValueError(f"symbol {symbol} is not in F_{q}")


def smallest_injective_arity(states: int, dimensions: int) -> int:
    """Smallest alphabet size K such that K**dimensions >= states."""
    if states < 0 or dimensions <= 0:
        raise ValueError("states must be non-negative and dimensions positive")
    if states <= 1:
        return 1
    return math.ceil(states ** (1.0 / dimensions))


def hamming_ball_volume(q: int, n: int, radius: int) -> int:
    """Volume of a Hamming ball of given radius over an alphabet of size q."""
    _validate_q(q)
    if n < 0 or radius < 0:
        raise ValueError("n and radius must be non-negative")
    total = 0
    for i in range(min(radius, n) + 1):
        total += math.comb(n, i) * ((q - 1) ** i)
    return total


def hamming_sphere_packing_holds(M: int, q: int, n: int, t: int) -> bool:
    """Check the sphere-packing (Hamming) bound for a t-error-correcting code."""
    if M <= 0 or n <= 0 or t < 0:
        raise ValueError("M and n must be positive and t non-negative")
    return M * hamming_ball_volume(q, n, t) <= (q ** n)


def gilbert_greedy_guarantees(M: int, q: int, n: int, distance: int) -> bool:
    """Gilbert-Varshamov existence guarantee for a code of size M and distance."""
    if M <= 0 or n <= 0 or distance <= 0:
        raise ValueError("M, n, distance must be positive")
    return M * hamming_ball_volume(q, n, distance - 1) <= (q ** n)


def singleton_upper_bound(q: int, n: int, distance: int) -> int:
    """Singleton upper bound on code size."""
    _validate_q(q)
    if n <= 0 or distance <= 0 or distance > n + 1:
        raise ValueError("invalid n or distance for Singleton bound")
    return q ** (n - distance + 1)


def minimum_distance(codewords: Iterable[tuple[int, ...]], q: int) -> int:
    """Minimum Hamming distance between distinct codewords over F_q."""
    _validate_q(q)
    words = tuple(codewords)
    if not words:
        raise ValueError("empty code")
    n = len(words[0])
    for word in words:
        if len(word) != n:
            raise ValueError("all codewords must have the same length")
        for sym in word:
            _validate_finite_field_symbol(sym, q)
    min_dist = n
    for a, b in combinations(words, 2):
        dist = sum(1 for x, y in zip(a, b) if x != y)
        if dist < min_dist:
            min_dist = dist
            if min_dist == 0:
                return 0
    return min_dist


def verify_code(
    codewords: Iterable[tuple[int, ...]],
    *,
    q: int,
    n: int,
    required_size: int,
    required_distance: int,
) -> CodeVerification:
    """Exhaustively verify a code and return a structured result."""
    words = tuple(codewords)
    actual_size = len(words)
    actual_distance = minimum_distance(words, q)
    ok = (
        actual_size >= required_size
        and actual_distance >= required_distance
        and all(len(word) == n for word in words)
    )
    messages: list[str] = []
    if actual_size < required_size:
        messages.append(f"size {actual_size} < required {required_size}")
    if actual_distance < required_distance:
        messages.append(f"distance {actual_distance} < required {required_distance}")
    return CodeVerification(
        ok=ok,
        q=q,
        n=n,
        size=actual_size,
        minimum_distance=actual_distance,
        required_size=required_size,
        required_distance=required_distance,
        messages=tuple(messages),
    )


def build_mds_7_4_2_3(*, a: int = 2) -> tuple[tuple[int, ...], ...]:
    """Build the [4,2,3]_7 MDS code used by CAP0-01 robust comparisons.

    Messages (x, y) in F_7^2 are encoded as (x, y, x+y, x+a*y) mod 7.
    """
    q = 7
    if not 0 <= a < q:
        raise ValueError(f"multiplier a={a} is not in F_7")
    codewords: list[tuple[int, ...]] = []
    for x in range(q):
        for y in range(q):
            codewords.append((x, y, (x + y) % q, (x + a * y) % q))
    return tuple(codewords)


def build_shortened_ternary_hamming_7_4_3() -> tuple[tuple[int, ...], ...]:
    """Build the shortened ternary [7,4,3]_3 Hamming code.

    Uses a 3x7 parity-check matrix over F_3 with seven distinct non-proportional
    projective columns. The code is the nullspace of H.
    """
    q = 3
    # Columns of the 3x7 parity-check matrix over F_3.
    columns = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 2, 0),
        (1, 0, 1),
        (1, 0, 2),
    ]
    # H is a 3x7 matrix; rows are indexed by coordinate position.
    rows = tuple(tuple(col[i] for col in columns) for i in range(3))

    def syndrome(word: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            sum((hij * wj) % q for hij, wj in zip(row, word)) % q for row in rows
        )

    codewords: list[tuple[int, ...]] = []
    for word in product(range(q), repeat=7):
        if syndrome(word) == (0, 0, 0):
            codewords.append(word)
    return tuple(codewords)
