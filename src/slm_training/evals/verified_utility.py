"""SLM-186 (FFE0-04): verified-utility ladder and Goodhart canary wiring.

Pure CPU helpers for combining multiple eval signals into a single
verified-utility estimate while exposing the channels along which a model
could game the metric.  No neural judge is called here; real judge scores
are expected to be supplied by callers.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "SCHEMA_VERSION",
    "FACTOR_NAMES",
    "VerifiedUtilityV1",
    "UtilityWeightManifestV1",
    "lexicographic_score",
    "scalarized_score",
    "pareto_dominance",
    "pareto_front",
    "cvar_tail",
    "abstention_economics",
    "sensitivity_rank_reversals",
    "safe_ratio",
    "canonical_equivalent_utility",
]

SCHEMA_VERSION = "verified_utility/v1"

# Canonical factor names in the order they appear on VerifiedUtilityV1.
# Keep this in sync with the dataclass fields below.
FACTOR_NAMES = (
    "hard_valid",
    "support_status",
    "contract_coverage",
    "binding_aware_meaningful_v2",
    "component_role_recall",
    "topology_node_f1",
    "topology_edge_f1",
    "reference_graph_exactness",
    "behavior_evidence",
    "render_evidence",
    "independent_judge_score",
    "human_pair_preference",
    "complexity_cost",
    "inference_cost",
)

# Numeric factors used for scalarization / pareto / sensitivity.
_NUMERIC_FACTORS = frozenset(
    {
        "contract_coverage",
        "binding_aware_meaningful_v2",
        "component_role_recall",
        "topology_node_f1",
        "topology_edge_f1",
        "reference_graph_exactness",
        "behavior_evidence",
        "render_evidence",
        "independent_judge_score",
        "human_pair_preference",
        "complexity_cost",
        "inference_cost",
    }
)

# Factors where higher is better (benefit).  Costs are lower-is-better.
_BENEFIT_FACTORS = frozenset(
    {
        "contract_coverage",
        "binding_aware_meaningful_v2",
        "component_role_recall",
        "topology_node_f1",
        "topology_edge_f1",
        "reference_graph_exactness",
        "behavior_evidence",
        "render_evidence",
        "independent_judge_score",
        "human_pair_preference",
    }
)

_COST_FACTORS = frozenset({"complexity_cost", "inference_cost"})

# Default [low, high] range for each numeric factor when normalizing.
_DEFAULT_FACTOR_RANGES: dict[str, tuple[float, float]] = {
    "contract_coverage": (0.0, 1.0),
    "binding_aware_meaningful_v2": (0.0, 1.0),
    "component_role_recall": (0.0, 1.0),
    "topology_node_f1": (0.0, 1.0),
    "topology_edge_f1": (0.0, 1.0),
    "reference_graph_exactness": (0.0, 1.0),
    "behavior_evidence": (0.0, 1.0),
    "render_evidence": (0.0, 1.0),
    "independent_judge_score": (0.0, 1.0),
    "human_pair_preference": (0.0, 1.0),
    "complexity_cost": (0.0, 1.0),
    "inference_cost": (0.0, 1.0),
}


@dataclass(frozen=True)
class VerifiedUtilityV1:
    """Frozen multi-factor utility record for one candidate program or model.

    ``availability`` tells consumers which factors are real measurements
    (``available``), missing/unimplemented (``unavailable``), or explicitly
    turned off (``disabled``).  A real eval run should set every used factor
    to ``available``; fixture runs may leave judges as ``unavailable``.
    """

    hard_valid: bool = False
    support_status: str = "unsupported"
    contract_coverage: float = 0.0
    binding_aware_meaningful_v2: float = 0.0
    component_role_recall: float = 0.0
    topology_node_f1: float = 0.0
    topology_edge_f1: float = 0.0
    reference_graph_exactness: float = 0.0
    behavior_evidence: float = 0.0
    render_evidence: float = 0.0
    independent_judge_score: float | None = None
    human_pair_preference: float | None = None
    complexity_cost: float = 0.0
    inference_cost: float = 0.0
    abstained: bool = False
    failure_reason_codes: tuple[str, ...] = ()
    availability: dict[str, str] = field(
        default_factory=lambda: {name: "unavailable" for name in FACTOR_NAMES}
    )

    def __post_init__(self) -> None:
        # Coerce availability keys to the canonical set on construction.
        object.__setattr__(
            self,
            "availability",
            {name: self.availability.get(name, "unavailable") for name in FACTOR_NAMES},
        )

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["failure_reason_codes"] = list(self.failure_reason_codes)
        data["availability"] = dict(self.availability)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedUtilityV1":
        return cls(
            hard_valid=bool(data.get("hard_valid", False)),
            support_status=str(data.get("support_status", "unsupported")),
            contract_coverage=float(data.get("contract_coverage", 0.0)),
            binding_aware_meaningful_v2=float(data.get("binding_aware_meaningful_v2", 0.0)),
            component_role_recall=float(data.get("component_role_recall", 0.0)),
            topology_node_f1=float(data.get("topology_node_f1", 0.0)),
            topology_edge_f1=float(data.get("topology_edge_f1", 0.0)),
            reference_graph_exactness=float(data.get("reference_graph_exactness", 0.0)),
            behavior_evidence=float(data.get("behavior_evidence", 0.0)),
            render_evidence=float(data.get("render_evidence", 0.0)),
            independent_judge_score=(
                None
                if data.get("independent_judge_score") is None
                else float(data["independent_judge_score"])
            ),
            human_pair_preference=(
                None
                if data.get("human_pair_preference") is None
                else float(data["human_pair_preference"])
            ),
            complexity_cost=float(data.get("complexity_cost", 0.0)),
            inference_cost=float(data.get("inference_cost", 0.0)),
            abstained=bool(data.get("abstained", False)),
            failure_reason_codes=tuple(str(r) for r in data.get("failure_reason_codes", [])),
            availability={
                name: str(data.get("availability", {}).get(name, "unavailable"))
                for name in FACTOR_NAMES
            },
        )

    def numeric_factor(self, name: str) -> float | None:
        """Return the numeric value of a factor, or None if not available."""
        if name not in _NUMERIC_FACTORS:
            return None
        if self.availability.get(name) != "available":
            return None
        value = getattr(self, name)
        if value is None:
            return None
        return float(value)


@dataclass(frozen=True)
class UtilityWeightManifestV1:
    """Preregistered weight policy for scalarizing VerifiedUtilityV1.

    ``permitted_ranges`` maps factor names to (min_weight, max_weight).  It is
    used by :func:`sensitivity_rank_reversals` to bound weight perturbations.
    ``dev_fit_hash`` and ``confirmation_hash`` are placeholders for the future
    fit/confirmation split; in wiring mode they carry deterministic fixture
    values.
    """

    weights: dict[str, float] = field(default_factory=dict)
    normalization: str = "unit"
    primary_policy: str = "scalarized"
    dev_fit_hash: str = ""
    confirmation_hash: str = ""
    permitted_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["weights"] = dict(self.weights)
        data["permitted_ranges"] = {
            k: list(v) for k, v in self.permitted_ranges.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UtilityWeightManifestV1":
        ranges = data.get("permitted_ranges") or {}
        return cls(
            weights={k: float(v) for k, v in (data.get("weights") or {}).items()},
            normalization=str(data.get("normalization", "unit")),
            primary_policy=str(data.get("primary_policy", "scalarized")),
            dev_fit_hash=str(data.get("dev_fit_hash", "")),
            confirmation_hash=str(data.get("confirmation_hash", "")),
            permitted_ranges={
                k: (float(v[0]), float(v[1])) for k, v in ranges.items() if isinstance(v, (list, tuple)) and len(v) == 2
            },
            version=str(data.get("version", SCHEMA_VERSION)),
        )

    def validate(self) -> list[str]:
        """Return validation errors; empty list means valid."""
        errors: list[str] = []
        for name in self.weights:
            if name not in _NUMERIC_FACTORS:
                errors.append(f"unknown weight factor: {name}")
        for name, (lo, hi) in self.permitted_ranges.items():
            if name not in _NUMERIC_FACTORS:
                errors.append(f"unknown permitted_range factor: {name}")
            if lo > hi:
                errors.append(f"permitted_range for {name} has lo > hi")
            if name in self.weights and not (lo <= self.weights[name] <= hi):
                errors.append(f"weight for {name} outside its permitted_range")
        if self.primary_policy not in {"scalarized", "lexicographic", "pareto"}:
            errors.append(f"unknown primary_policy: {self.primary_policy}")
        if self.normalization not in {"unit", "minmax"}:
            errors.append(f"unknown normalization: {self.normalization}")
        return errors


def _factor_value(util: VerifiedUtilityV1, name: str) -> float:
    """Numeric value for ordering, treating missing/unavailable as worst case."""
    value = util.numeric_factor(name)
    if value is None:
        return -float("inf")
    if name in _COST_FACTORS:
        # For lexicographic ordering, lower cost is better -> negate.
        return -value
    return value


def lexicographic_score(
    util: VerifiedUtilityV1,
    policy: Sequence[str],
) -> dict[str, Any]:
    """Return a rank vector under the given priority order.

    ``policy`` is a list of factor names from most important to least
    important.  The function returns the value of each factor in that order,
    with unavailable factors mapped to negative infinity so they never win a
    tie on a higher-priority factor.
    """
    policy = [name for name in policy if name in _NUMERIC_FACTORS]
    rank_vector = [_factor_value(util, name) for name in policy]
    # Hard validity is a gate: if not hard_valid, the whole vector is penalized.
    if not util.hard_valid:
        rank_vector = [-float("inf"), *rank_vector]
    else:
        rank_vector = [1.0, *rank_vector]
    tier = "|".join(
        "-inf" if not math.isfinite(v) else f"{v:.4f}" for v in rank_vector
    )
    return {
        "policy": list(policy),
        "rank_vector": rank_vector,
        "tier": tier,
        "hard_valid": util.hard_valid,
    }


def _normalize_factor(value: float, name: str, normalization: str) -> float:
    if normalization == "unit":
        return value
    if normalization == "minmax":
        lo, hi = _DEFAULT_FACTOR_RANGES.get(name, (0.0, 1.0))
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)
    return value


def scalarized_score(
    util: VerifiedUtilityV1,
    manifest: UtilityWeightManifestV1,
) -> dict[str, Any]:
    """Weighted scalar score over available numeric factors.

    Missing/unavailable factors contribute 0.0 to the numerator so that the
    score degrades gracefully rather than raising.  Costs are inverted before
    weighting (lower cost -> higher contribution).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    contributions: dict[str, float] = {}
    for name, weight in manifest.weights.items():
        value = util.numeric_factor(name)
        if value is None:
            continue
        normalized = _normalize_factor(value, name, manifest.normalization)
        if name in _COST_FACTORS:
            contribution = weight * (1.0 - normalized)
        else:
            contribution = weight * normalized
        weighted_sum += contribution
        total_weight += abs(weight)
        contributions[name] = contribution

    score = weighted_sum / total_weight if total_weight > 0.0 else 0.0
    return {
        "score": score,
        "total_weight": total_weight,
        "contributions": contributions,
        "normalization": manifest.normalization,
        "hard_valid": util.hard_valid,
    }


def pareto_dominance(
    left: VerifiedUtilityV1,
    right: VerifiedUtilityV1,
    factors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare two utilities along the Pareto frontier.

    For benefit factors, higher is better.  For cost factors, lower is better.
    Only factors that are available on both sides are compared.
    """
    factors = [name for name in (factors or _NUMERIC_FACTORS) if name in _NUMERIC_FACTORS]
    comparisons: dict[str, dict[str, Any]] = {}
    left_better = 0
    right_better = 0
    comparable = 0
    for name in factors:
        lv = left.numeric_factor(name)
        rv = right.numeric_factor(name)
        if lv is None or rv is None:
            comparisons[name] = {"left": None, "right": None, "compared": False}
            continue
        comparable += 1
        if name in _COST_FACTORS:
            lv, rv = -lv, -rv
        if lv > rv:
            left_better += 1
        elif rv > lv:
            right_better += 1
        comparisons[name] = {
            "left": float(lv),
            "right": float(rv),
            "compared": True,
            "direction": "left" if lv > rv else ("right" if rv > lv else "tie"),
        }

    left_dominates = left_better > 0 and right_better == 0
    right_dominates = right_better > 0 and left_better == 0
    return {
        "left_dominates": left_dominates,
        "right_dominates": right_dominates,
        "incomparable": comparable > 0 and not left_dominates and not right_dominates,
        "comparable_factors": comparable,
        "left_better_count": left_better,
        "right_better_count": right_better,
        "factor_comparisons": comparisons,
    }


def pareto_front(
    points: Iterable[tuple[str, VerifiedUtilityV1]],
    factors: Sequence[str] | None = None,
) -> list[tuple[str, VerifiedUtilityV1]]:
    """Return the Pareto-optimal subset of (label, utility) points."""
    point_list = list(points)
    front: list[tuple[str, VerifiedUtilityV1]] = []
    for label, util in point_list:
        dominated = False
        kept: list[tuple[str, VerifiedUtilityV1]] = []
        for other_label, other_util in front:
            dom = pareto_dominance(other_util, util, factors=factors)
            if dom["left_dominates"]:
                dominated = True
                break
            rev = pareto_dominance(util, other_util, factors=factors)
            if not rev["left_dominates"]:
                kept.append((other_label, other_util))
        if not dominated:
            kept.append((label, util))
            front = kept
    return front


def cvar_tail(values: Sequence[float], alpha: float = 0.05) -> dict[str, Any]:
    """Conditional Value at Risk: mean of the worst ``alpha`` tail."""
    clean = [float(v) for v in values if math.isfinite(v)]
    n = len(clean)
    if n == 0:
        return {"alpha": alpha, "n": 0, "cvar": float("nan"), "tail": []}
    sorted_vals = sorted(clean)
    k = max(1, math.ceil(alpha * n))
    tail = sorted_vals[:k]
    return {
        "alpha": alpha,
        "n": n,
        "cvar": sum(tail) / len(tail),
        "tail": tail,
    }


def abstention_economics(
    util_list: Sequence[VerifiedUtilityV1],
    risk_threshold: float = 0.3,
) -> dict[str, Any]:
    """Measure the value of abstaining on risky candidates.

    A candidate abstains when ``abstained`` is True or when its scalar score
    (under a uniform unit-weight policy) is below ``risk_threshold``.
    """
    if not util_list:
        return {
            "n": 0,
            "accepted": 0,
            "abstained": 0,
            "mean_utility_accepted": float("nan"),
            "mean_utility_if_forced": float("nan"),
            "value_of_abstention": 0.0,
            "risk_threshold": risk_threshold,
        }

    uniform = UtilityWeightManifestV1(
        weights={name: 1.0 for name in _NUMERIC_FACTORS},
        normalization="unit",
        primary_policy="scalarized",
    )

    accepted_utils: list[VerifiedUtilityV1] = []
    abstained_utils: list[VerifiedUtilityV1] = []
    forced_scores: list[float] = []
    for util in util_list:
        score = scalarized_score(util, uniform)["score"]
        forced_scores.append(score)
        if util.abstained or score < risk_threshold:
            abstained_utils.append(util)
        else:
            accepted_utils.append(util)

    accepted_scores = [scalarized_score(u, uniform)["score"] for u in accepted_utils]
    mean_accepted = sum(accepted_scores) / len(accepted_scores) if accepted_scores else 0.0
    mean_forced = sum(forced_scores) / len(forced_scores)
    value_of_abstention = mean_accepted - mean_forced

    return {
        "n": len(util_list),
        "accepted": len(accepted_utils),
        "abstained": len(abstained_utils),
        "mean_utility_accepted": mean_accepted,
        "mean_utility_if_forced": mean_forced,
        "value_of_abstention": value_of_abstention,
        "risk_threshold": risk_threshold,
    }


def sensitivity_rank_reversals(
    candidates: Sequence[tuple[str, VerifiedUtilityV1]],
    manifests: Sequence[UtilityWeightManifestV1],
    *,
    perturbations_per_manifest: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Perturb weights inside permitted_ranges and report rank reversals.

    For each manifest, ``perturbations_per_manifest`` random weight vectors are
    drawn uniformly from the permitted ranges.  A reversal is recorded whenever
    the scalarized ordering of two candidates flips relative to the base
    manifest ordering.
    """
    rng = random.Random(seed)
    candidate_list = list(candidates)
    utilities = [util for _, util in candidate_list]

    base_orders: list[list[int]] = []
    for manifest in manifests:
        scored = [
            (i, scalarized_score(util, manifest)["score"])
            for i, util in enumerate(utilities)
        ]
        base_orders.append([i for i, _ in sorted(scored, key=lambda x: (-x[1], x[0]))])

    reversals: list[dict[str, Any]] = []
    for m_idx, manifest in enumerate(manifests):
        factors = list(manifest.weights.keys())
        ranges = manifest.permitted_ranges
        for _ in range(perturbations_per_manifest):
            perturbed = UtilityWeightManifestV1(
                weights={
                    name: rng.uniform(
                        ranges.get(name, (max(0.0, manifest.weights[name] - 0.1), manifest.weights[name] + 0.1))[0],
                        ranges.get(name, (max(0.0, manifest.weights[name] - 0.1), manifest.weights[name] + 0.1))[1],
                    )
                    for name in factors
                },
                normalization=manifest.normalization,
                primary_policy=manifest.primary_policy,
                version=manifest.version,
            )
            scored = [
                (i, scalarized_score(util, perturbed)["score"])
                for i, util in enumerate(utilities)
            ]
            order = [i for i, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
            if order != base_orders[m_idx]:
                reversals.append(
                    {
                        "manifest_index": m_idx,
                        "base_order": list(base_orders[m_idx]),
                        "perturbed_order": list(order),
                        "perturbed_weights": dict(perturbed.weights),
                    }
                )

    return {
        "n_candidates": len(candidate_list),
        "n_manifests": len(manifests),
        "perturbations_per_manifest": perturbations_per_manifest,
        "total_perturbations": len(manifests) * perturbations_per_manifest,
        "reversal_count": len(reversals),
        "reversal_rate": len(reversals)
        / max(1, len(manifests) * perturbations_per_manifest),
        "reversals": reversals[:20],  # Cap detail to keep reports small.
    }


def safe_ratio(
    numerator: float,
    denominator: float,
    name: str,
) -> dict[str, Any]:
    """Compute a ratio that degrades to None on zero denominator."""
    return {
        "name": name,
        "numerator": float(numerator),
        "denominator": float(denominator),
        "ratio": float(numerator) / float(denominator) if denominator != 0.0 else None,
    }


def canonical_equivalent_utility(
    a: VerifiedUtilityV1,
    b: VerifiedUtilityV1,
    margin: float = 0.01,
) -> bool:
    """True if hard facts and all available numeric factors match within margin.

    Failure reason codes are ignored intentionally; two canonically equivalent
    programs may fail for different surface reasons.
    """
    if a.hard_valid != b.hard_valid:
        return False
    for name in _NUMERIC_FACTORS:
        av = a.numeric_factor(name)
        bv = b.numeric_factor(name)
        if av is None and bv is None:
            continue
        if av is None or bv is None:
            return False
        if abs(av - bv) > margin:
            return False
    return True
