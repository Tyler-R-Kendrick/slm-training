"""Semantic confusion cost and cost-aware ternary ECOC assignment (CAP2-03).

Pairwise costs between actions are constructed from semantic fingerprints.  For
small action sets an exact search assigns codewords maximizing cost-weighted
Hamming distance.  For larger sets a deterministic greedy heuristic is used and
labelled as heuristic.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

from slm_training.models.action_code_registry import (
    ActionCodeEntry,
    ActionSchema,
    CodeAssignment,
)


def _validate_trit_word(word: tuple[int, ...]) -> None:
    if any(t not in (0, 1, 2) for t in word):
        raise ValueError(f"invalid trit in {word}")


def hamming_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Hamming distance between two equal-length words."""
    if len(a) != len(b):
        raise ValueError("words must have equal length")
    return sum(1 for x, y in zip(a, b) if x != y)


def trit_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Ternary Hamming distance (symbols in {0,1,2})."""
    _validate_trit_word(a)
    _validate_trit_word(b)
    return hamming_distance(a, b)


def ordinal_base3_codeword(index: int, m: int) -> tuple[int, ...]:
    """Lexicographic base-3 digits for ``index`` using ``m`` trits."""
    if index < 0 or m < 0:
        raise ValueError("index and m must be non-negative")
    digits: list[int] = []
    n = index
    for _ in range(m):
        digits.append(n % 3)
        n //= 3
    if n != 0:
        raise ValueError(f"index {index} does not fit in {m} trits")
    return tuple(reversed(digits))


def add_parity_trit(word: tuple[int, ...]) -> tuple[int, ...]:
    """Append a single parity trit equal to ``sum(word) mod 3``.

    This gives a distance-2 ternary code: any single-trit corruption changes
    the parity, so the syndrome is non-zero.
    """
    _validate_trit_word(word)
    return word + (sum(word) % 3,)


def base3_codewords(b: int, m: int) -> tuple[tuple[int, ...], ...]:
    """Return the first ``b`` ordinal base-3 codewords of length ``m``."""
    if b > 3 ** m:
        raise ValueError(f"cannot encode {b} actions in {m} trits")
    return tuple(ordinal_base3_codeword(i, m) for i in range(b))


def ternary_ecoc_codewords(b: int, *, detect_single_trit_error: bool) -> tuple[tuple[int, ...], ...]:
    """Return ternary codewords for ``b`` actions.

    Without detection: ``m = ceil(log_3 b)`` trits.
    With detection: add one parity trit for distance 2.
    """
    if b <= 0:
        raise ValueError("b must be positive")
    if b == 1:
        base = ()
    else:
        m = math.ceil(math.log(b, 3))
        base = base3_codewords(b, m)
    if detect_single_trit_error:
        return tuple(add_parity_trit(w) for w in base)
    return base


CostMatrix = dict[tuple[str, str], float]


def uniform_cost(actions: tuple[str, ...]) -> CostMatrix:
    """Uniform unit cost for every action pair."""
    costs: CostMatrix = {}
    for i, a in enumerate(actions):
        for b in actions[i + 1 :]:
            costs[(a, b)] = 1.0
            costs[(b, a)] = 1.0
    return costs


def fingerprint_cost(
    actions: tuple[str, ...],
    fingerprints: dict[str, tuple[Any, ...]],
) -> CostMatrix:
    """Cost proportional to the number of differing fingerprint fields."""
    costs: CostMatrix = {}
    for a in actions:
        for b in actions:
            if a == b:
                continue
            fa = fingerprints.get(a, ())
            fb = fingerprints.get(b, ())
            max_len = max(len(fa), len(fb))
            diff = sum(
                1
                for i in range(max_len)
                if (fa[i] if i < len(fa) else None)
                != (fb[i] if i < len(fb) else None)
            )
            costs[(a, b)] = float(diff)
    return costs


def _weighted_distance_sum(
    assignment: dict[str, tuple[int, ...]],
    costs: CostMatrix,
) -> float:
    total = 0.0
    actions = list(assignment.keys())
    for i, a in enumerate(actions):
        for b in actions[i + 1 :]:
            cost = costs.get((a, b), 0.0)
            total += cost * trit_distance(assignment[a], assignment[b])
    return total


def exact_cost_aware_assignment(
    actions: tuple[str, ...],
    costs: CostMatrix,
    m: int,
    *,
    detect_single_trit_error: bool,
    seed: int = 0,
) -> tuple[dict[str, tuple[int, ...]], tuple[tuple[int, ...], ...]]:
    """Exact search for small action sets maximizing cost-weighted distance.

    Returns:
        mapping from action identity to codeword,
        tuple of unused codewords.
    """
    b = len(actions)
    if b == 0:
        return {}, ()
    if b == 1:
        word = add_parity_trit(()) if detect_single_trit_error else ()
        return {actions[0]: word}, ()
    base_m = m - 1 if detect_single_trit_error else m
    base_words = [ordinal_base3_codeword(i, base_m) for i in range(3 ** base_m)]
    if detect_single_trit_error:
        candidate_words = [add_parity_trit(w) for w in base_words]
    else:
        candidate_words = base_words

    best_score = -1.0
    best_assignment: dict[str, tuple[int, ...]] = {}
    # Enumerate all injections of b actions into candidate_words.
    for chosen in itertools.permutations(candidate_words, b):
        assignment = {action: word for action, word in zip(actions, chosen)}
        score = _weighted_distance_sum(assignment, costs)
        if score > best_score:
            best_score = score
            best_assignment = assignment
    unused = tuple(w for w in candidate_words if w not in best_assignment.values())
    return best_assignment, unused


def greedy_cost_aware_assignment(
    actions: tuple[str, ...],
    costs: CostMatrix,
    m: int,
    *,
    detect_single_trit_error: bool,
    seed: int = 0,
) -> tuple[dict[str, tuple[int, ...]], tuple[tuple[int, ...], ...]]:
    """Deterministic greedy heuristic for larger action sets.

    Places actions in descending order of total pairwise cost, then greedily
    picks the codeword maximizing marginal weighted distance to already placed
    actions.
    """
    rng = random.Random(seed)
    b = len(actions)
    if b == 0:
        return {}, ()
    if b == 1:
        word = add_parity_trit(()) if detect_single_trit_error else ()
        return {actions[0]: word}, ()
    base_m = m - 1 if detect_single_trit_error else m
    base_words = [ordinal_base3_codeword(i, base_m) for i in range(3 ** base_m)]
    if detect_single_trit_error:
        candidate_words = [add_parity_trit(w) for w in base_words]
    else:
        candidate_words = base_words

    # Order actions by descending total cost.
    total_cost = {
        a: sum(costs.get((a, b), 0.0) for b in actions if b != a) for a in actions
    }
    ordered = sorted(actions, key=lambda a: (-total_cost[a], a))

    assignment: dict[str, tuple[int, ...]] = {}
    for action in ordered:
        best_word = None
        best_score = -1.0
        # Shuffle candidate words for tie-breaking determinism controlled by seed.
        rng.shuffle(candidate_words)
        for word in candidate_words:
            if word in assignment.values():
                continue
            score = sum(
                costs.get((action, other), 0.0) * trit_distance(word, assignment[other])
                for other in assignment
            )
            if score > best_score:
                best_score = score
                best_word = word
        if best_word is None:
            raise RuntimeError("ran out of codewords during greedy assignment")
        assignment[action] = best_word
    unused = tuple(w for w in candidate_words if w not in assignment.values())
    return assignment, unused


def build_ternary_ecoc_entry(
    schema: ActionSchema,
    costs: CostMatrix | None = None,
    *,
    detect_single_trit_error: bool = False,
    use_exact_search: bool = True,
    invalid_code_policy: str = "abstain",
    cost_matrix_source: str = "uniform",
    seed: int = 0,
) -> ActionCodeEntry:
    """Build an ActionCodeEntry with ternary ECOC codewords.

    For small action sets (<=6) the default exact search finds the cost-weighted
    maximum-distance assignment.  Larger sets use a deterministic greedy
    heuristic and the entry notes that optimality is unknown.
    """
    from slm_training.dsl.analysis.arity.precision import ternary_ecoc_width

    actions = schema.action_identities
    b = len(actions)
    if b == 0:
        raise ValueError("schema must have at least one action")
    if costs is None:
        costs = uniform_cost(actions)
        cost_matrix_source = "uniform"
    m = ternary_ecoc_width(b, detect_single_trit_error=detect_single_trit_error)
    if use_exact_search and b <= 6:
        assignment, unused = exact_cost_aware_assignment(
            actions, costs, m, detect_single_trit_error=detect_single_trit_error, seed=seed
        )
        code_family = (
            "ternary_ecoc_exact" if detect_single_trit_error else "base3_exact"
        )
    else:
        assignment, unused = greedy_cost_aware_assignment(
            actions, costs, m, detect_single_trit_error=detect_single_trit_error, seed=seed
        )
        code_family = (
            "ternary_ecoc_heuristic" if detect_single_trit_error else "base3_heuristic"
        )

    min_dist = None
    if assignment:
        words = list(assignment.values())
        if len(words) >= 2:
            min_dist = min(
                trit_distance(words[i], words[j])
                for i in range(len(words))
                for j in range(i + 1, len(words))
            )

    assignments = tuple(
        CodeAssignment(action, assignment[action], 3) for action in actions
    )
    return ActionCodeEntry(
        schema=schema,
        code_family=code_family,
        alphabet_radices=(3,) * m,
        assignments=assignments,
        minimum_hamming_distance=min_dist,
        unused_codewords=unused,
        invalid_code_policy=invalid_code_policy,  # type: ignore[arg-type]
        cost_matrix_source=cost_matrix_source,
    )
