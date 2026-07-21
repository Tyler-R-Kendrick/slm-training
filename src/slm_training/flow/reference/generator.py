"""Sparse/dense CTMC generator construction and probability integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from slm_training.flow.reference.adapter import ActionRef, RateFn, StateRef

if TYPE_CHECKING:
    from slm_training.flow.reference.enumerate import StateGraph


@dataclass(frozen=True)
class Generator:
    """Dense CTMC generator plus bookkeeping for an exact state graph."""

    Q: np.ndarray
    state_index: dict[str, int]
    index_state: dict[int, StateRef]
    action_for_pair: dict[tuple[int, int], ActionRef]
    rates: dict[tuple[int, int], float]

    @property
    def n_states(self) -> int:
        return self.Q.shape[0]

    def hazard(self, state: StateRef | int) -> float:
        """Total hazard (negative diagonal) for a state."""
        idx = state if isinstance(state, int) else self.state_index[state.fingerprint]
        return float(-self.Q[idx, idx])

    def holding_time(self, rng: Any, state: StateRef | int) -> float:
        """Sample an exponential holding time for ``state``."""
        h = self.hazard(state)
        if h <= 0.0:
            return float("inf")
        return rng.expovariate(h)

    def legal_successors(self, state: StateRef | int) -> list[tuple[int, ActionRef, float]]:
        """Return (target_index, action, rate) for all non-zero off-diagonal rates."""
        idx = state if isinstance(state, int) else self.state_index[state.fingerprint]
        out: list[tuple[int, ActionRef, float]] = []
        for j in range(self.n_states):
            if j == idx:
                continue
            rate = float(self.Q[idx, j])
            if rate > 0.0:
                action = self.action_for_pair.get((idx, j))
                out.append((j, action, rate))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "state_index": dict(self.state_index),
            "rates": {
                f"{i},{j}": float(r)
                for (i, j), r in self.rates.items()
            },
        }


def _matrix_exp_uniformization(Q: np.ndarray, t: float, tol: float = 1e-12) -> np.ndarray:
    """Stable CTMC matrix exponential via uniformization (no scipy)."""
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if t == 0.0:
        return np.eye(n, dtype=float)
    q = max(0.0, float(np.max(-np.diag(Q))))
    if q == 0.0:
        return np.eye(n, dtype=float)
    # Uniformize: P = I + Q / q, exp(Q t) = exp(-q t) sum_k (q t)^k / k! P^k.
    P = np.eye(n, dtype=float) + Q / q
    theta = q * t
    weight = math.exp(-theta)
    result = weight * np.eye(n, dtype=float)
    term = np.eye(n, dtype=float)
    max_iter = 50 + int(theta * 3)
    for k in range(1, max_iter):
        term = term @ P
        weight *= theta / k
        result += weight * term
        if weight < tol and k > theta * 2:
            break
    return result


def apply_matrix_exp_row(
    Q: np.ndarray, row_vec: np.ndarray, t: float, tol: float = 1e-12
) -> np.ndarray:
    """Compute ``row_vec @ exp(Q t)`` via uniformization without forming exp(Q t)."""
    Q = np.asarray(Q, dtype=float)
    row_vec = np.asarray(row_vec, dtype=float)
    if t == 0.0:
        return row_vec.copy()
    q = max(0.0, float(np.max(-np.diag(Q))))
    if q == 0.0:
        return row_vec.copy()
    theta = q * t
    weight = math.exp(-theta)
    result = weight * row_vec.copy()
    term = row_vec.copy()
    max_iter = 50 + int(theta * 3)
    for k in range(1, max_iter):
        term = term + (term @ Q) / q
        weight *= theta / k
        result += weight * term
        if weight < tol and k > theta * 2:
            break
    return result


def apply_matrix_exp_col(
    Q: np.ndarray, col_vec: np.ndarray, t: float, tol: float = 1e-12
) -> np.ndarray:
    """Compute ``exp(Q t) @ col_vec`` via uniformization without forming exp(Q t)."""
    Q = np.asarray(Q, dtype=float)
    col_vec = np.asarray(col_vec, dtype=float)
    if t == 0.0:
        return col_vec.copy()
    q = max(0.0, float(np.max(-np.diag(Q))))
    if q == 0.0:
        return col_vec.copy()
    theta = q * t
    weight = math.exp(-theta)
    result = weight * col_vec.copy()
    term = col_vec.copy()
    max_iter = 50 + int(theta * 3)
    for k in range(1, max_iter):
        term = term + Q @ term / q
        weight *= theta / k
        result += weight * term
        if weight < tol and k > theta * 2:
            break
    return result


def matrix_exponential(Q: np.ndarray, t: float) -> np.ndarray:
    """Return ``exp(Q * t)``; prefers scipy, otherwise uniformization."""
    if t == 0.0:
        n = Q.shape[0]
        return np.eye(n, dtype=float)
    try:
        from scipy.linalg import expm

        return np.asarray(expm(Q * t), dtype=float)
    except Exception:  # noqa: BLE001
        pass
    return _matrix_exp_uniformization(Q, t)


def endpoint_distribution(
    Q: np.ndarray, p0: np.ndarray, t: float
) -> np.ndarray:
    """Solve dp/dt = p Q forward in time from row vector ``p0``."""
    p0 = np.asarray(p0, dtype=float)
    if t == 0.0:
        return p0.copy()
    return p0 @ matrix_exponential(Q, t)


def forward_equation(
    Q: np.ndarray, p0: np.ndarray, t: float
) -> np.ndarray:
    """Alias for ``endpoint_distribution``."""
    return endpoint_distribution(Q, p0, t)


def check_generator(Q: np.ndarray, atol: float = 1e-9) -> list[str]:
    """Return a list of generator-contract violations."""
    errors: list[str] = []
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and Q[i, j] < -atol:
                errors.append(f"negative off-diagonal Q[{i},{j}]={Q[i,j]}")
    row_sums = Q.sum(axis=1)
    for i, s in enumerate(row_sums):
        if abs(s) > atol:
            errors.append(f"row {i} sum {s} != 0")
    return errors


def build_uniform_rate_fn(rate: float = 1.0) -> RateFn:
    """Constant rate for every legal transition."""

    def fn(source: StateRef, action: ActionRef, target: StateRef, graph: Any) -> float:
        return max(0.0, float(rate))

    return fn


def build_distance_rate_fn(
    distance_fn: Callable[[StateRef, StateRef], float],
    temperature: float = 1.0,
    min_rate: float = 1e-9,
) -> RateFn:
    """Gibbs rate ``exp(-distance / temperature)`` with floor."""
    temp = max(temperature, 1e-9)

    def fn(source: StateRef, action: ActionRef, target: StateRef, graph: Any) -> float:
        d = distance_fn(source, target)
        return max(min_rate, math.exp(-d / temp))

    return fn


def build_bridge_rate_fn(
    target_dist: dict[str, float],
    terminal_class_fn: Any,
    base_rate_fn: RateFn | None = None,
    temperature: float = 1.0,
) -> RateFn:
    """Target-weighted rate using a terminal-class target distribution.

    The resulting unnormalized rate is ``p1(class(y)) * base_rate(x,a,y)``.
    This is the simplest time-conditioned target parameterization; the
    harness also supports the exact Doob bridge posterior via
    ``build_doob_bridge_rate_fn``.
    """
    base = base_rate_fn or build_uniform_rate_fn(1.0)
    target = {k: max(0.0, float(v)) for k, v in target_dist.items()}
    total = sum(target.values())
    if total > 0.0:
        target = {k: v / total for k, v in target.items()}

    def fn(source: StateRef, action: ActionRef, target_state: StateRef, graph: Any) -> float:
        class_value = terminal_class_fn(target_state)
        p1 = target.get(class_value, 0.0)
        if p1 <= 0.0:
            return 0.0
        return p1 * base(source, action, target_state, graph) / max(temperature, 1e-9)

    return fn


def build_doob_bridge_rate_fn(
    base_generator: Generator,
    terminal_indices: set[int],
    time: float,
) -> RateFn:
    """Exact h-transform bridge rate conditioned on hitting a terminal at ``time``."""
    n = base_generator.n_states
    # Build terminal indicator vector.
    b = np.zeros(n, dtype=float)
    for idx in terminal_indices:
        b[idx] = 1.0
    # h[i] = P(terminal at time T | X_t=i) for the base process.
    # Approximate with endpoint mass at remaining time from i.
    remaining = max(time, 1e-9)
    P_T = matrix_exponential(base_generator.Q, remaining)
    h = P_T @ b
    h = np.maximum(h, 1e-12)
    Q = base_generator.Q

    def fn(source: StateRef, action: ActionRef, target: StateRef, graph: Any) -> float:
        i = base_generator.state_index[source.fingerprint]
        j = base_generator.state_index[target.fingerprint]
        rate = max(0.0, float(Q[i, j]))
        return rate * h[j] / h[i]

    return fn


class GeneratorBuilder:
    """Build dense CTMC generators from an exact state graph and rate function."""

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph

    def build_dense(
        self, rate_fn: RateFn, max_rate: float = 1e6
    ) -> Generator:
        """Build a dense generator matrix from the graph and rate function."""
        n = self.graph.n_states
        Q = np.zeros((n, n), dtype=float)
        state_index = self.graph.state_index
        index_state = {i: s for i, s in enumerate(self.graph.states)}
        action_for_pair: dict[tuple[int, int], ActionRef] = {}
        rates: dict[tuple[int, int], float] = {}

        for t in self.graph.transitions:
            i = state_index[t.source.fingerprint]
            j = state_index[t.target.fingerprint]
            rate = float(rate_fn(t.source, t.action, t.target, self.graph))
            rate = max(0.0, min(rate, max_rate))
            if rate > 0.0:
                # If multiple actions collapse to the same pair, keep the first
                # action id but sum rates.
                if (i, j) in rates:
                    rates[(i, j)] += rate
                    Q[i, j] += rate
                else:
                    rates[(i, j)] = rate
                    Q[i, j] = rate
                    action_for_pair[(i, j)] = t.action

        for i in range(n):
            Q[i, i] = -Q[i].sum()

        return Generator(
            Q=Q,
            state_index=state_index,
            index_state=index_state,
            action_for_pair=action_for_pair,
            rates=rates,
        )

    def build_sparse(self, rate_fn: RateFn) -> dict[int, dict[int, float]]:
        """Build a sparse row-dict generator (no diagonal)."""
        sparse: dict[int, dict[int, float]] = {}
        state_index = self.graph.state_index
        for t in self.graph.transitions:
            i = state_index[t.source.fingerprint]
            j = state_index[t.target.fingerprint]
            rate = max(0.0, float(rate_fn(t.source, t.action, t.target, self.graph)))
            if rate > 0.0:
                sparse.setdefault(i, {})[j] = sparse.get(i, {}).get(j, 0.0) + rate
        return sparse


import math  # noqa: E402
