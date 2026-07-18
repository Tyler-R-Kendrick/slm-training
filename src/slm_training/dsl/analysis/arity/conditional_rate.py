"""Conditional task rate, Fano bounds, and posterior effective support.

CAP1-04 (SLM-84): estimate how many bits a neural latent must preserve, given
exact compiler state Q, to select semantically acceptable legal actions or
completions at a declared distortion. This module is Torch-free and works on
empirical distributions produced by CAP1-03 state profiles.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from slm_training.dsl.analysis.arity.task_quotient import (
    AlignedActionRecord,
    ConfusabilityGraph,
    QuotientReport,
    StateProfile,
    TaskDistortionSpec,
    build_state_profiles,
    color_graph,
    policy_distance,
)


@dataclass(frozen=True)
class RateDistortionPoint:
    """One point on a finite rate-distortion curve."""

    distortion: float
    rate_bits: float
    beta: float
    exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "distortion": self.distortion,
            "rate_bits": self.rate_bits,
            "beta": self.beta,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class FanoBound:
    """Fano-style lower bound on Bayes error rate from posterior entropy."""

    conditional_entropy_bits: float
    alphabet_size: int
    lower_bound_error: float
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditional_entropy_bits": self.conditional_entropy_bits,
            "alphabet_size": self.alphabet_size,
            "lower_bound_error": self.lower_bound_error,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class PosteriorEffectiveSupport:
    """Dynamic-support diagnostics for a conditional distribution."""

    per_state: dict[str, float]
    mean: float
    max: float
    min: float
    median: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_state": self.per_state,
            "mean": self.mean,
            "max": self.max,
            "min": self.min,
            "median": self.median,
        }


@dataclass
class ConditionalRateReport:
    """Aggregate report for a conditional task-rate analysis."""

    spec: TaskDistortionSpec
    state_count: int
    action_alphabet_size: int
    conditional_entropy_bits: float
    mutual_information_bits: float | None
    fano_bounds: tuple[FanoBound, ...]
    posterior_support: PosteriorEffectiveSupport
    rate_distortion_curve: tuple[RateDistortionPoint, ...]
    estimated: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "state_count": self.state_count,
            "action_alphabet_size": self.action_alphabet_size,
            "conditional_entropy_bits": self.conditional_entropy_bits,
            "mutual_information_bits": self.mutual_information_bits,
            "fano_bounds": [b.to_dict() for b in self.fano_bounds],
            "posterior_support": self.posterior_support.to_dict(),
            "rate_distortion_curve": [p.to_dict() for p in self.rate_distortion_curve],
            "estimated": self.estimated,
        }


def entropy(dist: Mapping[str, float]) -> float:
    """Shannon entropy in bits for a discrete distribution."""
    total = 0.0
    for p in dist.values():
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def conditional_entropy(conditional: Mapping[str, Mapping[str, float]]) -> float:
    """H(Y|X) for p(y|x) weighted by the empirical frequency of x."""
    if not conditional:
        return 0.0
    # Empirical frequency of x is proportional to the number of observations.
    # When distributions come from build_state_profiles, every state is weighted
    # equally here; caller can pre-weight if production weighting is required.
    weights: dict[str, float] = defaultdict(float)
    for x, dist in conditional.items():
        weights[x] = sum(dist.values())
    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return 0.0
    result = 0.0
    for x, dist in conditional.items():
        wx = weights[x] / total_weight
        result += wx * entropy(dist)
    return result


def mutual_information(joint: Mapping[tuple[str, str], float]) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y) in bits."""
    if not joint:
        return 0.0
    total = sum(joint.values())
    if total == 0.0:
        return 0.0
    p_xy = {k: v / total for k, v in joint.items()}
    p_x: dict[str, float] = defaultdict(float)
    p_y: dict[str, float] = defaultdict(float)
    for (x, y), p in p_xy.items():
        p_x[x] += p
        p_y[y] += p
    h_xy = entropy(p_xy)
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    return max(0.0, h_x + h_y - h_xy)


def fano_lower_bound(
    conditional_entropy_bits: float,
    alphabet_size: int,
    *,
    exact_forced: bool = False,
) -> FanoBound:
    """Finite-class Fano lower bound on error probability.

    Returns the standard bound P_e >= (H(Y|X) - log2(2)) / log2(|Y| - 1)
    when |Y| > 1, with safe handling of the degenerate cases.
    """
    if alphabet_size <= 1:
        return FanoBound(
            conditional_entropy_bits=conditional_entropy_bits,
            alphabet_size=alphabet_size,
            lower_bound_error=0.0,
            exact=True,
        )
    numerator = max(0.0, conditional_entropy_bits - 1.0)
    denominator = math.log2(alphabet_size - 1)
    if denominator <= 0.0:
        bound = 0.0
    else:
        bound = min(1.0, numerator / denominator)
    return FanoBound(
        conditional_entropy_bits=conditional_entropy_bits,
        alphabet_size=alphabet_size,
        lower_bound_error=bound,
        exact=exact_forced or bound == 0.0,
    )


def posterior_effective_support(
    conditional: Mapping[str, Mapping[str, float]]
) -> PosteriorEffectiveSupport:
    """exp(H(Y|X=x)) for each x and aggregate statistics."""
    per_state = {x: math.exp(entropy(dist) * math.log(2)) for x, dist in conditional.items()}
    values = sorted(per_state.values())
    n = len(values)
    if n == 0:
        return PosteriorEffectiveSupport(
            per_state={},
            mean=0.0,
            max=0.0,
            min=0.0,
            median=0.0,
        )
    mean = sum(values) / n
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2.0
    return PosteriorEffectiveSupport(
        per_state=per_state,
        mean=mean,
        max=max(values),
        min=min(values),
        median=median,
    )


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total == 0.0:
        return dist
    return {k: v / total for k, v in dist.items()}


def _kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    total = 0.0
    for x, px in p.items():
        if px > 0.0:
            qx = q.get(x, 0.0)
            if qx <= 0.0:
                return float("inf")
            total += px * math.log2(px / qx)
    return total


def _ba_step(
    source_dist: Sequence[float],
    distortion: Sequence[Sequence[float]],
    beta: float,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> tuple[list[list[float]], list[float], float, float, bool]:
    """Run Blahut-Arimoto for a fixed beta. Return p(z|x), q(z), D, R, converged."""
    n = len(source_dist)
    m = len(distortion[0])
    # Initialize p(z|x) uniformly.
    p_z_given_x = [[1.0 / m for _ in range(m)] for _ in range(n)]
    q_z = [1.0 / m for _ in range(m)]
    converged = False

    for _ in range(max_iterations):
        # q(z) = sum_x p(x) p(z|x)
        new_q = [0.0] * m
        for i in range(n):
            for j in range(m):
                new_q[j] += source_dist[i] * p_z_given_x[i][j]
        # Avoid zeros.
        new_q = [max(q, 1e-12) for q in new_q]
        q_sum = sum(new_q)
        new_q = [q / q_sum for q in new_q]

        # p(z|x) ∝ q(z) * exp(-beta * d(x,z))
        new_p = [[0.0] * m for _ in range(n)]
        for i in range(n):
            row_sum = 0.0
            for j in range(m):
                new_p[i][j] = new_q[j] * math.exp(-beta * distortion[i][j])
                row_sum += new_p[i][j]
            if row_sum > 0.0:
                new_p[i] = [v / row_sum for v in new_p[i]]
            else:
                new_p[i] = [1.0 / m for _ in range(m)]

        # Check convergence.
        max_diff = max(
            abs(new_p[i][j] - p_z_given_x[i][j])
            for i in range(n)
            for j in range(m)
        )
        p_z_given_x = new_p
        q_z = new_q
        if max_diff < tolerance:
            converged = True
            break

    # Compute D and R.
    d = 0.0
    mi = 0.0
    for i in range(n):
        for j in range(m):
            p = p_z_given_x[i][j]
            if p > 0.0:
                d += source_dist[i] * p * distortion[i][j]
                mi += source_dist[i] * p * math.log2(p / q_z[j])
    return p_z_given_x, q_z, d, max(0.0, mi), converged


def blahut_arimoto_rate_distortion(
    source_dist: Mapping[str, float],
    distortion_matrix: Mapping[tuple[str, str], float],
    reproduction_symbols: Sequence[str],
    *,
    betas: Sequence[float] | None = None,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> tuple[RateDistortionPoint, ...]:
    """Compute a finite rate-distortion curve via Blahut-Arimoto.

    `source_dist` maps source symbol names to probabilities.
    `distortion_matrix` maps (source_name, repro_name) to non-negative distortion.
    `reproduction_symbols` is the finite reproduction alphabet Z.

    Returns points ordered by decreasing distortion (increasing rate).
    """
    source_names = list(source_dist)
    n = len(source_names)
    m = len(reproduction_symbols)
    if n == 0 or m == 0:
        return ()

    src = [float(source_dist[s]) for s in source_names]
    src_sum = sum(src)
    if src_sum == 0.0:
        return ()
    src = [s / src_sum for s in src]

    # Build numeric distortion matrix.
    dist = [
        [
            float(distortion_matrix.get((source_names[i], reproduction_symbols[j]), 1.0))
            for j in range(m)
        ]
        for i in range(n)
    ]

    if betas is None:
        # Sweep from near-zero rate (large beta) to near-zero distortion (small beta).
        betas = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    points: list[RateDistortionPoint] = []
    for beta in betas:
        _, _, d, r, converged = _ba_step(
            src, dist, beta, max_iterations=max_iterations, tolerance=tolerance
        )
        points.append(
            RateDistortionPoint(
                distortion=d,
                rate_bits=r,
                beta=beta,
                exact=converged,
            )
        )

    # Sort by distortion ascending; rate should decrease as distortion increases.
    points.sort(key=lambda p: (p.distortion, p.rate_bits))
    # Keep the Pareto frontier: each kept point must have a strictly lower rate
    # than every point with lower or equal distortion.
    pruned: list[RateDistortionPoint] = []
    min_rate_so_far = float("inf")
    for p in points:
        if p.rate_bits < min_rate_so_far - 1e-12:
            pruned.append(p)
            min_rate_so_far = p.rate_bits
    return tuple(pruned)


def _quotient_rate_distortion_points(
    profiles: Mapping[str, StateProfile],
    spec: TaskDistortionSpec,
    tolerances: Sequence[float] | None = None,
) -> tuple[RateDistortionPoint, ...]:
    """Estimate R(D) by recomputing CAP1-03 quotients at several tolerances.

    Each quotient color is a reproduction symbol; the distortion at a source
    state is its distance to the closest other state in the same color class,
    and the rate is the empirical entropy of the color assignment.
    """
    if not profiles:
        return ()

    source_names = sorted(profiles)

    if tolerances is None:
        # Log-spaced tolerances from very strict to very loose.
        tolerances = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    points: list[RateDistortionPoint] = []
    for tol in tolerances:
        tol_spec = TaskDistortionSpec(
            spec_id=f"{spec.spec_id}-tol{tol}",
            action_alignment=spec.action_alignment,
            policy_metric=spec.policy_metric,
            policy_tolerance=tol,
            value_weight=spec.value_weight,
            execution_weight=spec.execution_weight,
            semantic_fingerprint_weight=spec.semantic_fingerprint_weight,
            average_tolerance=spec.average_tolerance,
            cvar_alpha=spec.cvar_alpha,
            cvar_tolerance=spec.cvar_tolerance,
            hard_forbidden_confusions=spec.hard_forbidden_confusions,
            horizon=spec.horizon,
        )
        graph = ConfusabilityGraph(
            vertices=set(source_names),
            edges=set(),  # placeholder; build below
        )
        # Build edges for this tolerance.
        for i, a in enumerate(source_names):
            for b in source_names[i + 1 :]:
                d = policy_distance(
                    profiles[a].action_distribution,
                    profiles[b].action_distribution,
                    tol_spec.policy_metric,
                )
                if d > tol:
                    graph.edges.add(frozenset({a, b}))
        coloring = color_graph(graph)
        colors = coloring.colors

        # Distortion = average distance from each state to the nearest other state
        # in the same color class (a simple intra-color worst-case proxy).
        total_distortion = 0.0
        count = 0
        color_groups: dict[int, set[str]] = defaultdict(set)
        for v in source_names:
            color_groups[colors.get(v, 0)].add(v)
        for group in color_groups.values():
            members = sorted(group)
            for a in members:
                min_dist = float("inf")
                for b in members:
                    if a == b:
                        continue
                    d = policy_distance(
                        profiles[a].action_distribution,
                        profiles[b].action_distribution,
                        spec.policy_metric,
                    )
                    if d < min_dist:
                        min_dist = d
                if min_dist == float("inf"):
                    min_dist = 0.0
                total_distortion += min_dist
                count += 1
        avg_distortion = total_distortion / count if count else 0.0

        # Rate = entropy of color assignment.
        color_counts: dict[int, float] = defaultdict(float)
        for name in source_names:
            color_counts[colors.get(name, 0)] += 1.0 / len(source_names)
        rate = entropy(color_counts)

        points.append(
            RateDistortionPoint(
                distortion=avg_distortion,
                rate_bits=rate,
                beta=1.0 / (tol + 1e-12),
                exact=coloring.exact,
            )
        )

    points.sort(key=lambda p: (p.distortion, p.rate_bits))
    pruned: list[RateDistortionPoint] = []
    min_rate_so_far = float("inf")
    for p in points:
        if p.rate_bits < min_rate_so_far - 1e-12:
            pruned.append(p)
            min_rate_so_far = p.rate_bits
    return tuple(pruned)


def analyze_conditional_rate(
    records: Sequence[AlignedActionRecord],
    spec: TaskDistortionSpec,
    *,
    quotient: QuotientReport | None = None,
    rd_tolerances: Sequence[float] | None = None,
) -> ConditionalRateReport:
    """Estimate conditional task rate and related diagnostics.

    If `quotient` is provided, its coloring is used as a reproduction alphabet
    for one point on the rate-distortion curve. The full curve is estimated by
    recomputing CAP1-03 quotients across a tolerance sweep.
    """
    profiles = build_state_profiles(records)
    state_names = sorted(profiles)
    state_count = len(state_names)

    # Build empirical conditional distribution p(action | state).
    conditional: dict[str, dict[str, float]] = {
        name: dict(profiles[name].action_distribution) for name in state_names
    }
    conditional_entropy_bits = conditional_entropy(conditional)

    # Action alphabet size over all observed aligned families.
    action_alphabet = set()
    for dist in conditional.values():
        action_alphabet.update(dist)
    action_alphabet_size = len(action_alphabet)

    # Mutual information I(input proxy; action) is not directly available from
    # action records alone; leave it None unless caller provides a quotient.
    mutual_info_bits: float | None = None
    if quotient is not None and quotient.coloring.num_colors > 0:
        colors = quotient.coloring.colors
        joint: dict[tuple[str, str], float] = defaultdict(float)
        for name in state_names:
            color = str(colors.get(name, 0))
            dist = conditional[name]
            for action, p in dist.items():
                joint[(color, action)] += p / state_count
        mutual_info_bits = mutual_information(joint)

    # Fano bounds for the aggregate posterior entropy and per-state entropies.
    fano_bounds: list[FanoBound] = [
        fano_lower_bound(conditional_entropy_bits, action_alphabet_size)
    ]
    for name in state_names:
        h = entropy(conditional[name])
        fano_bounds.append(
            fano_lower_bound(h, len(conditional[name]), exact_forced=True)
        )

    posterior_support = posterior_effective_support(conditional)

    # Rate-distortion curve from quotient tolerance sweep.
    rd_curve = _quotient_rate_distortion_points(profiles, spec, rd_tolerances)

    return ConditionalRateReport(
        spec=spec,
        state_count=state_count,
        action_alphabet_size=action_alphabet_size,
        conditional_entropy_bits=conditional_entropy_bits,
        mutual_information_bits=mutual_info_bits,
        fano_bounds=tuple(fano_bounds),
        posterior_support=posterior_support,
        rate_distortion_curve=rd_curve,
        estimated=True,
    )
