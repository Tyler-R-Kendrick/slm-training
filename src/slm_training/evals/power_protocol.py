"""SLM-183: pure statistical utilities for powered, cluster-aware confirmation.

Default-off / fixture-only helpers.  No model is trained and no GPU is required.
All functions degrade gracefully on zero-denominator inputs.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "wilson_interval",
    "binomial_rate_evidence",
    "plan_binomial_rate_test",
    "exact_binomial_interval",
    "bootstrap_paired_ci",
    "cluster_bootstrap_ci",
    "intraclass_correlation",
    "mde_simulation",
    "benjamini_hochberg",
    "holm_bonferroni",
    "classify_power",
]


def wilson_interval(
    successes: int,
    n: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int | None]:
    """Return a two-sided Wilson score interval for a binomial proportion.

    Invalid counts fail closed instead of being clamped. A zero denominator is
    explicitly unmeasured, so its estimate and bounds are ``None``.
    """
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if n == 0:
        return {
            "n": 0,
            "estimate": None,
            "low": None,
            "high": None,
            "confidence_level": confidence_level,
        }
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    rate = successes / n
    denominator = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return {
        "n": n,
        "estimate": rate,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "confidence_level": confidence_level,
    }


def binomial_rate_evidence(
    successes: int,
    n: int,
    *,
    seed_count: int,
    evidence_class: str,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Build the count provenance attached to one binomial scoreboard rate."""
    if isinstance(seed_count, bool) or not isinstance(seed_count, int):
        raise TypeError("seed_count must be an integer")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if not evidence_class:
        raise ValueError("evidence_class must be non-empty")
    return {
        "schema": "binomial_rate_evidence/v1",
        "numerator": successes,
        "denominator": n,
        "seed_count": seed_count,
        "interval": {
            "method": "wilson_score",
            **wilson_interval(
                successes,
                n,
                confidence_level=confidence_level,
            ),
        },
        "evidence_class": evidence_class,
    }


def plan_binomial_rate_test(
    *,
    null_rate: float,
    target_delta: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
    sides: int = 2,
    seeds: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Prospectively plan a one-proportion rate test.

    This score-normal approximation accepts design inputs only. Observed
    outcomes are deliberately absent: post-hoc power is not success evidence.
    Seeds are reported separately and never multiplied into the sample size.
    """
    if not 0.0 < null_rate < 1.0:
        raise ValueError("null_rate must be in (0, 1)")
    if target_delta == 0.0:
        raise ValueError("target_delta must be non-zero")
    target_rate = null_rate + target_delta
    if not 0.0 < target_rate < 1.0:
        raise ValueError("null_rate + target_delta must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if isinstance(sides, bool) or not isinstance(sides, int) or sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise ValueError("seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in normalized_seeds):
        raise TypeError("seeds must contain only integer identifiers")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")

    z_alpha = NormalDist().inv_cdf(1.0 - alpha / sides)
    z_power = NormalDist().inv_cdf(target_power)
    null_sd = math.sqrt(null_rate * (1.0 - null_rate))
    target_sd = math.sqrt(target_rate * (1.0 - target_rate))
    required_n = math.ceil(
        ((z_alpha * null_sd + z_power * target_sd) / abs(target_delta)) ** 2
    )
    return {
        "schema": "binomial_power_preregistration/v1",
        "method": "one_proportion_score_normal_approximation",
        "null_rate": null_rate,
        "target_delta": target_delta,
        "target_rate": target_rate,
        "alpha": alpha,
        "target_power": target_power,
        "sides": sides,
        "confidence_level": 1.0 - alpha,
        "planned_sample_size_per_seed": required_n,
        "required_n": required_n,
        "seeds": list(normalized_seeds),
        "seed_count": len(normalized_seeds),
        "seed_aggregation": "report_separately_no_pooling",
        "use": "preregistration_only_not_post_hoc_success_evidence",
    }


def _log_sum_exp(log_values: Sequence[float]) -> float:
    """Stable log-sum-exp."""
    if not log_values:
        return -float("inf")
    max_log = max(log_values)
    if math.isinf(max_log):
        return max_log
    return max_log + math.log(sum(math.exp(v - max_log) for v in log_values))


def _binom_tail_prob(n: int, successes: int, p: float, upper: bool) -> float:
    """Compute P(X <= successes) or P(X >= successes) for X ~ Binom(n, p)."""
    if p <= 0.0:
        return 1.0 if upper and successes <= 0 else 0.0
    if p >= 1.0:
        return 1.0 if upper or successes >= n else 0.0
    if upper:
        # P(X >= successes) = sum_{k=successes}^{n} C(n,k) p^k (1-p)^(n-k)
        k_values = range(successes, n + 1)
    else:
        # P(X <= successes)
        k_values = range(0, successes + 1)
    log_p = math.log(p)
    log_1p = math.log(1.0 - p)
    log_probs = [
        math.log(math.comb(n, k)) + k * log_p + (n - k) * log_1p
        for k in k_values
    ]
    return math.exp(_log_sum_exp(log_probs))


def exact_binomial_interval(
    successes: int, n: int, *, alpha: float = 0.05
) -> dict[str, float | int]:
    """Clopper-Pearson exact binomial confidence interval.

    Implemented without scipy via binary search on the binomial tail, so it
    only depends on the stdlib + numpy already used elsewhere in the repo.
    """
    if n < 1:
        return {"n": n, "estimate": 0.0, "low": 0.0, "high": 0.0}
    if successes < 0:
        successes = 0
    if successes > n:
        successes = n
    rate = successes / n
    half_alpha = alpha / 2.0

    if successes == 0:
        low = 0.0
    else:
        # Find p in (0, rate) with P(X >= successes) = alpha/2.
        lo, hi = 1e-12, max(1e-12, min(rate, 1.0 - 1e-12))
        for _ in range(64):
            mid = (lo + hi) / 2.0
            tail = _binom_tail_prob(n, successes, mid, upper=True)
            if tail > half_alpha:
                hi = mid
            else:
                lo = mid
        low = (lo + hi) / 2.0

    if successes == n:
        high = 1.0
    else:
        # Find p in (rate, 1) with P(X <= successes) = alpha/2.
        lo, hi = min(1.0 - 1e-12, max(rate, 1e-12)), 1.0 - 1e-12
        for _ in range(64):
            mid = (lo + hi) / 2.0
            tail = _binom_tail_prob(n, successes, mid, upper=False)
            if tail > half_alpha:
                lo = mid
            else:
                hi = mid
        high = (lo + hi) / 2.0

    return {
        "n": n,
        "estimate": rate,
        "low": max(0.0, min(low, high)),
        "high": min(1.0, max(low, high)),
    }


def bootstrap_paired_ci(
    left: Sequence[float],
    right: Sequence[float],
    metric: Callable[[Sequence[float], Sequence[float]], float],
    *,
    seed: int = 0,
    resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap confidence interval for a two-sample metric.

    ``metric(left, right)`` must return a scalar.  Pairs are resampled by index.
    """
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    n = len(left_arr)
    if n == 0 or len(right_arr) != n:
        return {
            "estimate": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "resamples": resamples,
        }
    observed = float(metric(left_arr, right_arr))
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=float)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        replicates[i] = float(metric(left_arr[idx], right_arr[idx]))
    low = float(np.percentile(replicates, 100.0 * alpha / 2.0))
    high = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "estimate": observed,
        "low": low,
        "high": high,
        "resamples": resamples,
        "alpha": alpha,
    }


def cluster_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence[Hashable],
    metric: Callable[[Sequence[float]], float],
    *,
    seed: int = 0,
    resamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster-aware bootstrap CI respecting ``target_cluster_id`` structure.

    Resamples clusters with replacement, keeps all observations within each
    selected cluster, and recomputes the metric.
    """
    values_arr = np.asarray(values, dtype=float)
    if len(values_arr) == 0 or len(cluster_ids) != len(values_arr):
        return {
            "estimate": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "resamples": resamples,
        }
    by_cluster: dict[Any, list[int]] = defaultdict(list)
    for i, cid in enumerate(cluster_ids):
        by_cluster[cid].append(i)
    clusters = list(by_cluster.keys())
    observed = float(metric(values_arr))
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=float)
    for r in range(resamples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        idx: list[int] = []
        for cid in selected:
            idx.extend(by_cluster[cid])
        replicates[r] = float(metric(values_arr[idx]))
    low = float(np.percentile(replicates, 100.0 * alpha / 2.0))
    high = float(np.percentile(replicates, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "estimate": observed,
        "low": low,
        "high": high,
        "resamples": resamples,
        "alpha": alpha,
    }


def intraclass_correlation(
    values: Sequence[float], cluster_ids: Sequence[Hashable]
) -> dict[str, float]:
    """Simple one-way ICC estimate via variance decomposition.

    Returns ICC(1,1) along with between-cluster and within-cluster variance
    components.  Degrades to 0.0 ICC when clusters are degenerate.
    """
    values_arr = np.asarray(values, dtype=float)
    if len(values_arr) == 0 or len(cluster_ids) != len(values_arr):
        return {"icc": float("nan"), "between": 0.0, "within": 0.0, "n_clusters": 0}
    by_cluster: dict[Any, list[float]] = defaultdict(list)
    for cid, val in zip(cluster_ids, values_arr):
        by_cluster[cid].append(float(val))
    clusters = list(by_cluster.values())
    k = len(clusters)
    if k < 2:
        return {"icc": 0.0, "between": 0.0, "within": 0.0, "n_clusters": k}
    grand_mean = float(np.mean(values_arr))
    cluster_means = [np.mean(c) for c in clusters]
    cluster_sizes = [len(c) for c in clusters]
    n_total = sum(cluster_sizes)
    # MS_between = sum n_i (mean_i - grand_mean)^2 / (k - 1)
    ms_between = sum(
        n_i * (m - grand_mean) ** 2 for n_i, m in zip(cluster_sizes, cluster_means)
    ) / (k - 1)
    # MS_within = sum sum (x_ij - mean_i)^2 / (N - k)
    ms_within = sum(
        sum((x - m) ** 2 for x in cluster)
        for cluster, m in zip(clusters, cluster_means)
    ) / max(1, n_total - k)
    # n0 correction for unequal cluster sizes
    sum_sq = sum(n_i * n_i for n_i in cluster_sizes)
    n0 = (n_total - sum_sq / n_total) / max(1, k - 1)
    denom = ms_between + (n0 - 1.0) * ms_within
    if denom <= 0.0 or not math.isfinite(denom):
        return {
            "icc": 0.0,
            "between": float(ms_between),
            "within": float(ms_within),
            "n_clusters": k,
        }
    icc = (ms_between - ms_within) / denom
    return {
        "icc": max(-1.0, min(1.0, float(icc))),
        "between": float(ms_between),
        "within": float(ms_within),
        "n_clusters": k,
    }


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mde_simulation(
    base_rate: float,
    sigma_seed: float,
    sigma_target: float,
    n_targets: int,
    paths_per_target: int,
    n_seeds: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    n_simulations: int = 200,
    effect_sizes: Sequence[float] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate statistical power across effect sizes under seed + target variance.

    Binary outcomes are generated from a mixed-effects logit model with target
    and seed random effects.  Power is estimated as the rejection rate of a
    one-sided paired z-test on target-level mean differences between treatment
    and control.  The z-test uses only stdlib + numpy (no scipy).
    """
    base_rate = float(np.clip(base_rate, 1e-6, 1.0 - 1e-6))
    sigma_seed = max(0.0, float(sigma_seed))
    sigma_target = max(0.0, float(sigma_target))
    n_targets = max(1, int(n_targets))
    paths_per_target = max(1, int(paths_per_target))
    n_seeds = max(1, int(n_seeds))
    n_simulations = max(10, int(n_simulations))
    if effect_sizes is None:
        effect_sizes = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    effect_sizes = [float(e) for e in effect_sizes]

    rng = np.random.default_rng(seed)
    base_logit = math.log(base_rate / (1.0 - base_rate))
    z_crit = _normal_cdf_inv(1.0 - alpha)

    def _simulate_one(effect: float) -> bool:
        target_effects = rng.normal(0.0, sigma_target, size=n_targets)
        control = np.empty((n_targets, n_seeds, paths_per_target), dtype=float)
        treatment = np.empty((n_targets, n_seeds, paths_per_target), dtype=float)
        for t in range(n_targets):
            seed_effects = rng.normal(0.0, sigma_seed, size=n_seeds)
            for s in range(n_seeds):
                logit_c = base_logit + target_effects[t] + seed_effects[s]
                logit_t = logit_c + effect
                p_c = 1.0 / (1.0 + math.exp(-logit_c))
                p_t = 1.0 / (1.0 + math.exp(-logit_t))
                control[t, s, :] = rng.random(paths_per_target) < p_c
                treatment[t, s, :] = rng.random(paths_per_target) < p_t
        control_mean = control.mean(axis=(1, 2))
        treatment_mean = treatment.mean(axis=(1, 2))
        diff = treatment_mean - control_mean
        n = len(diff)
        if n < 2:
            return False
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / math.sqrt(n))
        if se <= 0.0 or not math.isfinite(se):
            return mean_diff > 0.0
        z_stat = mean_diff / se
        return z_stat > z_crit

    curve: list[dict[str, float]] = []
    for effect in effect_sizes:
        rejections = sum(_simulate_one(effect) for _ in range(n_simulations))
        est_power = rejections / n_simulations
        curve.append({"effect_size": effect, "power": est_power})

    # Find smallest effect size reaching target power by linear interpolation.
    mde: float | None = None
    above = [pt for pt in curve if pt["power"] >= power]
    if above:
        mde = float(min(pt["effect_size"] for pt in above))
    else:
        # Extrapolate from last two points if all are below target power.
        if len(curve) >= 2:
            last = curve[-1]
            prev = curve[-2]
            denom = last["power"] - prev["power"]
            if denom > 0.0:
                mde = prev["effect_size"] + (power - prev["power"]) * (
                    last["effect_size"] - prev["effect_size"]
                ) / denom
    return {
        "base_rate": base_rate,
        "sigma_seed": sigma_seed,
        "sigma_target": sigma_target,
        "n_targets": n_targets,
        "paths_per_target": paths_per_target,
        "n_seeds": n_seeds,
        "alpha": alpha,
        "target_power": power,
        "n_simulations": n_simulations,
        "curve": curve,
        "mde": mde,
    }


def _normal_cdf_inv(p: float) -> float:
    """Inverse standard normal CDF (quantile) using a rational approximation."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    if p > 0.5:
        r = math.sqrt(-2.0 * math.log(1.0 - p))
    else:
        r = math.sqrt(-2.0 * math.log(p))
    # Abramowitz & Stegun formula 26.2.23
    c0, c1, c2, d1, d2, d3 = (
        2.515517,
        0.802853,
        0.010328,
        1.432788,
        0.189269,
        0.001308,
    )
    num = c0 + c1 * r + c2 * r * r
    den = 1.0 + d1 * r + d2 * r * r + d3 * r * r * r
    x = r - num / den
    return x if p > 0.5 else -x


def benjamini_hochberg(
    p_values: Sequence[float], *, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Benjamini-Hochberg false-discovery-rate correction.

    Returns one entry per original p-value with its rank, BH threshold, and
    rejection flag.  The list preserves the original input order.
    """
    entries = [
        {"index": i, "p_value": float(p), "rank": 0, "threshold": 0.0, "rejected": False}
        for i, p in enumerate(p_values)
    ]
    n = len(entries)
    if n == 0:
        return []
    sorted_entries = sorted(entries, key=lambda e: e["p_value"])
    for rank, entry in enumerate(sorted_entries, start=1):
        entry["rank"] = rank
        entry["threshold"] = rank * alpha / n
    # Find largest rank where p <= threshold
    max_rank = 0
    for entry in sorted_entries:
        if entry["p_value"] <= entry["threshold"]:
            max_rank = max(max_rank, entry["rank"])
    for entry in sorted_entries:
        entry["rejected"] = entry["rank"] <= max_rank
    # Restore original order
    entries.sort(key=lambda e: e["index"])
    return entries


def holm_bonferroni(
    hypotheses: Sequence[tuple[str, float]], *, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Apply Holm's step-down family-wise error correction.

    The family must be declared prospectively as ``(hypothesis_id, p_value)``
    pairs. Results retain caller order while rank ties are resolved by stable
    hypothesis identifier.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    ids = [item[0] for item in hypotheses]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("hypothesis identifiers must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis identifiers must be unique")
    entries: list[dict[str, Any]] = []
    for index, (hypothesis_id, raw_p) in enumerate(hypotheses):
        if isinstance(raw_p, bool) or not isinstance(raw_p, (int, float)):
            raise TypeError("p-values must be numeric")
        p_value = float(raw_p)
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("p-values must be finite and in [0, 1]")
        entries.append(
            {
                "index": index,
                "hypothesis_id": hypothesis_id,
                "p_value": p_value,
                "rank": 0,
                "threshold": 0.0,
                "adjusted_p_value": 0.0,
                "rejected": False,
            }
        )
    family_size = len(entries)
    ordered = sorted(entries, key=lambda item: (item["p_value"], item["hypothesis_id"]))
    prior_adjusted = 0.0
    still_rejecting = True
    for rank, entry in enumerate(ordered, start=1):
        remaining = family_size - rank + 1
        threshold = alpha / remaining
        adjusted = min(1.0, max(prior_adjusted, remaining * entry["p_value"]))
        rejected = still_rejecting and entry["p_value"] <= threshold
        if not rejected:
            still_rejecting = False
        entry.update(
            {
                "rank": rank,
                "threshold": threshold,
                "adjusted_p_value": adjusted,
                "rejected": rejected,
            }
        )
        prior_adjusted = adjusted
    return sorted(entries, key=lambda item: item["index"])


def classify_power(
    conclusion: bool | str | float, mde: float, effect_size: float
) -> str:
    """Classify a power conclusion as decidable / large_effect_only / underpowered.

    - ``conclusion`` can be a boolean (power met?), a string such as
      ``"power_met"``, or a numeric power estimate.
    """
    if isinstance(conclusion, bool):
        power_met = conclusion
    elif isinstance(conclusion, str):
        power_met = conclusion.lower() in {"power_met", "significant", "yes", "true"}
    elif isinstance(conclusion, (int, float)):
        power_met = float(conclusion) >= 0.8
    else:
        power_met = bool(conclusion)

    effect_size = float(effect_size)
    mde = float(mde)
    if mde <= 0.0 or not math.isfinite(mde):
        return "decidable" if power_met else "underpowered"
    if power_met or effect_size >= mde:
        return "decidable"
    if effect_size >= 0.5 * mde:
        return "large_effect_only"
    return "underpowered"
