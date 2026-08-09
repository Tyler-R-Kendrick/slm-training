"""Deterministic cross-version evidence ledger for the autotrain hill climb.

The continuous loop's live memory (``outputs/autoresearch``) is ephemeral and
capped, while the durable record — 1,500+ result JSONs under ``docs/design``
spanning many harness versions — was never machine-consumed. This module closes
that gap with four deterministic pieces (no LLM, no RNG: identical inputs
produce identical outputs):

1. **Mining** — extract per-arm observations from the committed cycle-record
   and measured-result schema families, joining "the same arm" across harness
   versions by slug (the stable identity `_SCREENING_ARM_BANK` already uses)
   and partitioning by the version-stamp eval components so results produced
   under different metric constraints are pooled knowingly, never silently.
2. **Posterior** — an exact Normal-Inverse-Gamma conjugate update per arm over
   the primary-metric effect. Unseen arms get prior-width optimism; repeatedly
   null arms sink; one noisy win shifts a posterior instead of erasing a null
   ledger.
3. **Ranking** — a deterministic upper-confidence score (posterior mean plus
   ``exploration_c`` posterior standard deviations of the mean). Residual
   boosts stay lexicographically dominant, mirroring ``soft_rank_slugs``.
4. **Power feasibility** — exact sign-test arithmetic (``fractions.Fraction``)
   answering, before any run is spent, whether a measurement of ``n`` paired
   documents can reject at ``alpha`` at all, and how many independent null
   cycles are required before an arm may be closed.

The committed ledger artifact lives beside the climb policy so a fresh
container inherits the full cross-session memory:
``resources/experiments/autotrain_climb/evidence_ledger.v1.json``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LEDGER_SCHEMA = "autotrain_evidence_ledger/v1"
VERDICT_SCHEMA = "regime_exhausted_verdict/v1"

#: version_stamp components whose versions define eval comparability. Results
#: stamped with equal tuples were produced under the same metric constraints.
EVAL_KEY_COMPONENTS = (
    "evals.scoring",
    "evals.meaningful_program",
    "harness.model_build.eval",
    "gates.ship",
)

#: Observations with no version stamp (the cycle-record family is unstamped)
#: or from a different eval partition still inform the posterior, but with
#: reduced weight — they are evidence about the arm, not about the exact
#: current measurement.
CROSS_PARTITION_WEIGHT = 0.5

_CYCLE_SCHEMAS = frozenset(
    {
        "continuous_cycle_results/v1",
        "continuous_autotrain_cycle_results/v1",
        "autotrain_continuous_cycle_results/v1",
    }
)
_MEASURED_SCHEMAS = frozenset(
    {"autotrain_measured_result/v1", "autotrain_measured_results/v1"}
)
#: Candidate-id suffixes that are roles, not screening arms.
_NON_ARM_SLUGS = frozenset({"control", "confirm", "promote", "promotion"})
_CANDIDATE_ID_RE = re.compile(r"-c\d+-([a-z0-9][a-z0-9-]*?)(?:-confirm|-promote)?$")

DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "experiments"
    / "autotrain_climb"
    / "evidence_ledger.v1.json"
)


# ---------------------------------------------------------------------------
# Power feasibility (exact arithmetic)
# ---------------------------------------------------------------------------


def sign_test_min_two_sided_p(n: int) -> Fraction:
    """Smallest attainable two-sided sign-test p-value with ``n`` paired items.

    With every pair agreeing in direction the two-sided p is ``2 * (1/2)^n``.
    ``n <= 0`` carries no evidence and returns 1.
    """
    if n <= 0:
        return Fraction(1)
    return min(Fraction(1), Fraction(2, 2**n))


def sign_test_feasible(n: int, alpha: Fraction) -> bool:
    """True when a sign test over ``n`` pairs can reject at ``alpha`` at all."""
    return sign_test_min_two_sided_p(n) <= alpha


def required_sign_test_n(alpha: Fraction) -> int:
    """Smallest ``n`` for which a sign-test rejection at ``alpha`` is possible."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    n = 1
    while not sign_test_feasible(n, alpha):
        n += 1
    return n


def parse_alpha(value: Any, default: Fraction = Fraction(1, 20)) -> Fraction:
    """Parse a policy alpha ("1/20", "0.05", or number) into an exact Fraction."""
    if value is None:
        return default
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def power_scaled_null_seeds(base_min: int, per_cycle_n: int, alpha: Fraction) -> int:
    """Null cycles required before arm closure carries a decidable signal.

    A single cycle of ``per_cycle_n`` documents may be individually
    undecidable; closure then needs enough independent null cycles that the
    pooled sign test could have rejected at ``alpha`` had the arm worked.
    Never returns less than ``base_min``.
    """
    base = max(1, int(base_min))
    if per_cycle_n <= 0:
        return base
    required = required_sign_test_n(alpha)
    cycles = math.ceil(required / per_cycle_n)
    return max(base, cycles)


def power_feasibility_report(n: int, alpha: Fraction) -> dict[str, Any]:
    """Typed pre-run feasibility statement for a proposed measurement."""
    min_p = sign_test_min_two_sided_p(n)
    return {
        "schema": "power_feasibility/v1",
        "n": int(n),
        "alpha": str(alpha),
        "min_two_sided_p": str(min_p),
        "decisive": bool(min_p <= alpha),
        "required_n": required_sign_test_n(alpha),
    }


# ---------------------------------------------------------------------------
# Observation mining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmObservation:
    """One historical measurement of a screening arm against its control."""

    slug: str
    delta: float | None
    positive: bool | None
    complete: bool
    seed: int | None
    eval_key: str | None
    campaign_id: str
    cycle_index: int | None
    source: str


def slug_from_candidate_token(token: str) -> str | None:
    """Extract the arm slug from a candidate-id-shaped token, if present."""
    match = _CANDIDATE_ID_RE.search(token.strip())
    if not match:
        return None
    slug = match.group(1)
    if slug in _NON_ARM_SLUGS or slug.isdigit():
        return None
    return slug


def _slug_from_reasons(reasons: Iterable[Any]) -> str | None:
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        for token in re.split(r"[\s:,;]+", reason):
            slug = slug_from_candidate_token(token)
            if slug:
                return slug
    return None


def _metric_value(metrics: Mapping[str, Any], primary: str) -> float | None:
    """Read ``primary`` from a flat metric dict, tolerating suite prefixes."""
    for key in (primary, primary.split(".", 1)[-1]):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def eval_key_from_stamp(payload: Mapping[str, Any]) -> str | None:
    stamp = payload.get("version_stamp")
    if not isinstance(stamp, Mapping):
        return None
    components = stamp.get("components")
    if not isinstance(components, Mapping):
        return None
    parts = [
        f"{name}={components[name]}"
        for name in EVAL_KEY_COMPONENTS
        if isinstance(components.get(name), str)
    ]
    return "|".join(parts) if parts else None


def _cycle_record_observations(
    payload: Mapping[str, Any], source: str
) -> list[ArmObservation]:
    reasons = payload.get("reasons")
    slug = _slug_from_reasons(reasons if isinstance(reasons, list) else [])
    if not slug:
        return []
    primary = str(payload.get("primary_metric") or "smoke.structural_similarity")
    candidate = payload.get("candidate_metrics")
    control = payload.get("control_metrics")
    delta: float | None = None
    if isinstance(candidate, Mapping) and isinstance(control, Mapping):
        cand_value = _metric_value(candidate, primary)
        ctrl_value = _metric_value(control, primary)
        if cand_value is not None and ctrl_value is not None:
            delta = cand_value - ctrl_value
    cycle_index = payload.get("cycle_index")
    return [
        ArmObservation(
            slug=slug,
            delta=delta,
            positive=bool(payload["positive"])
            if isinstance(payload.get("positive"), bool)
            else None,
            complete=bool(payload.get("measurement_complete")),
            seed=None,
            eval_key=eval_key_from_stamp(payload),
            campaign_id=str(payload.get("campaign_id") or ""),
            cycle_index=int(cycle_index) if isinstance(cycle_index, int) else None,
            source=source,
        )
    ]


def _measured_record_observations(
    payload: Mapping[str, Any], source: str
) -> list[ArmObservation]:
    arms = payload.get("arms")
    if not isinstance(arms, list):
        return []
    primary = str(payload.get("primary_metric") or "smoke.structural_similarity")
    bare = primary.split(".", 1)[-1]
    control_value: float | None = None
    candidates: list[tuple[str, float | None]] = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            continue
        experiment_id = str(arm.get("experiment_id") or arm.get("label") or "")
        smoke = arm.get("smoke")
        value = (
            _metric_value(smoke, bare) if isinstance(smoke, Mapping) else None
        )
        slug = slug_from_candidate_token(experiment_id)
        if slug is None and experiment_id.endswith("-control"):
            control_value = value
        elif slug is None and "control" in experiment_id:
            control_value = value
        elif slug is not None:
            candidates.append((slug, value))
    seed = payload.get("seed")
    cycle_index = payload.get("cycle_index")
    eval_key = eval_key_from_stamp(payload)
    observations = []
    for slug, value in candidates:
        delta = (
            value - control_value
            if value is not None and control_value is not None
            else None
        )
        observations.append(
            ArmObservation(
                slug=slug,
                delta=delta,
                positive=None,
                complete=delta is not None,
                seed=int(seed) if isinstance(seed, int) else None,
                eval_key=eval_key,
                campaign_id=str(payload.get("campaign_id") or ""),
                cycle_index=int(cycle_index) if isinstance(cycle_index, int) else None,
                source=source,
            )
        )
    return observations


def extract_observations(
    payload: Mapping[str, Any], source: str
) -> list[ArmObservation]:
    """Extract arm observations from one result payload (empty if not one)."""
    schema = payload.get("schema") or payload.get("schema_version")
    if schema in _CYCLE_SCHEMAS:
        return _cycle_record_observations(payload, source)
    if schema in _MEASURED_SCHEMAS:
        return _measured_record_observations(payload, source)
    return []


# ---------------------------------------------------------------------------
# Ledger build / IO
# ---------------------------------------------------------------------------


def build_ledger(design_dir: Path) -> dict[str, Any]:
    """Mine ``design_dir`` (recursively, ``*.json``) into a ledger payload.

    Deterministic: files are processed in sorted order and observations are
    deduplicated on ``(campaign_id, cycle_index, slug, seed)`` keeping the
    first occurrence.
    """
    scanned = 0
    unreadable = 0
    seen: set[tuple[str, int | None, str, int | None]] = set()
    per_arm: dict[str, dict[str, Any]] = {}
    for path in sorted(design_dir.rglob("*.json")):
        scanned += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        if not isinstance(payload, Mapping):
            continue
        for obs in extract_observations(payload, source=path.name):
            key = (obs.campaign_id, obs.cycle_index, obs.slug, obs.seed)
            if obs.campaign_id and key in seen:
                continue
            seen.add(key)
            _fold_observation(per_arm, obs)
    arms = {
        slug: _finalize_arm(stats)
        for slug, stats in sorted(per_arm.items())
    }
    return {
        "schema": LEDGER_SCHEMA,
        "generated_from": str(design_dir),
        "eval_key_components": list(EVAL_KEY_COMPONENTS),
        "file_counts": {
            "scanned": scanned,
            "unreadable": unreadable,
            "observations": sum(a["n_obs"] for a in arms.values()),
        },
        "arms": arms,
    }


def _fold_observation(per_arm: dict[str, dict[str, Any]], obs: ArmObservation) -> None:
    stats = per_arm.setdefault(
        obs.slug,
        {
            "n_obs": 0,
            "n_complete": 0,
            "n_positive": 0,
            "n_null": 0,
            "deltas": [],
            "by_eval_key": {},
        },
    )
    stats["n_obs"] += 1
    if obs.complete:
        stats["n_complete"] += 1
        if obs.positive is True:
            stats["n_positive"] += 1
        elif obs.positive is False:
            stats["n_null"] += 1
    if obs.delta is not None and obs.complete:
        stats["deltas"].append(obs.delta)
        bucket = stats["by_eval_key"].setdefault(
            obs.eval_key or "unstamped", {"n": 0, "sum": 0.0, "sumsq": 0.0}
        )
        bucket["n"] += 1
        bucket["sum"] += obs.delta
        bucket["sumsq"] += obs.delta * obs.delta


def _finalize_arm(stats: dict[str, Any]) -> dict[str, Any]:
    deltas = stats.pop("deltas")
    n = len(deltas)
    mean = sum(deltas) / n if n else 0.0
    m2 = sum((d - mean) ** 2 for d in deltas) if n else 0.0
    finalized = dict(stats)
    finalized["n_delta"] = n
    finalized["mean_delta"] = mean
    finalized["m2_delta"] = m2
    finalized["by_eval_key"] = {
        key: dict(bucket) for key, bucket in sorted(stats["by_eval_key"].items())
    }
    return finalized


def write_ledger(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_ledger(path: Path | None = None) -> dict[str, Any]:
    """Load the committed ledger; empty arms map when missing or invalid."""
    target = path or DEFAULT_LEDGER_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, Mapping) or payload.get("schema") != LEDGER_SCHEMA:
        return {}
    arms = payload.get("arms")
    return dict(arms) if isinstance(arms, Mapping) else {}


# ---------------------------------------------------------------------------
# Posterior + deterministic UCB ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmPosterior:
    """Normal-Inverse-Gamma posterior summary for one arm's effect."""

    mean: float
    sd_of_mean: float
    n_eff: float

    def ucb(self, exploration_c: float) -> float:
        return self.mean + exploration_c * self.sd_of_mean


def arm_posterior(
    *,
    n: float,
    mean: float,
    m2: float,
    prior_mean: float = 0.0,
    prior_scale: float = 0.05,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
) -> ArmPosterior:
    """Exact conjugate NIG update from sufficient statistics.

    ``prior_scale`` is the prior standard deviation of the arm's mean effect
    (with the defaults, an unseen arm has ``sd_of_mean == prior_scale``).
    Fractional ``n`` supports down-weighted cross-partition observations.
    """
    beta0 = alpha0 * kappa0 * prior_scale * prior_scale
    kappa_n = kappa0 + n
    mu_n = (kappa0 * prior_mean + n * mean) / kappa_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * m2 + (kappa0 * n * (mean - prior_mean) ** 2) / (2.0 * kappa_n)
    variance_of_mean = beta_n / (alpha_n * kappa_n)
    return ArmPosterior(
        mean=mu_n, sd_of_mean=math.sqrt(max(0.0, variance_of_mean)), n_eff=n
    )


def _arm_sufficient_stats(
    ledger_arm: Mapping[str, Any] | None,
    live: tuple[float, float] | None,
) -> tuple[float, float, float]:
    """Combine ledger stats and live loop stats into ``(n, mean, m2)``.

    Live stats arrive as ``(n_complete, mean_delta)`` with unknown within-run
    variance; they are folded via the parallel-axis rule with zero internal
    spread (the prior's ``alpha0`` keeps the posterior width positive).
    Ledger observations count at ``CROSS_PARTITION_WEIGHT`` because the
    committed record spans eval partitions.
    """
    n = mean = m2 = 0.0
    if isinstance(ledger_arm, Mapping):
        raw_n = float(ledger_arm.get("n_delta") or 0.0)
        if raw_n > 0:
            n = raw_n * CROSS_PARTITION_WEIGHT
            mean = float(ledger_arm.get("mean_delta") or 0.0)
            m2 = float(ledger_arm.get("m2_delta") or 0.0) * CROSS_PARTITION_WEIGHT
    if live is not None:
        live_n, live_mean = live
        if live_n > 0:
            total = n + live_n
            delta = live_mean - mean
            m2 = m2 + (n * live_n / total) * delta * delta
            mean = mean + delta * live_n / total
            n = total
    return n, mean, m2


def rank_arms_by_evidence(
    candidates: Sequence[str],
    ledger_arms: Mapping[str, Any],
    *,
    exploration_c: float = 1.0,
    prior_scale: float = 0.05,
    residual_boosts: Mapping[str, float] | None = None,
    live_stats: Mapping[str, tuple[float, float]] | None = None,
    rotation_order: Sequence[str] | None = None,
) -> list[str]:
    """Deterministic evidence ranking of candidate slugs.

    Sort key mirrors ``soft_rank_slugs``: residual boost lexicographically
    dominates, then the posterior UCB, then rotation order, then slug.
    """
    boosts = residual_boosts or {}
    live = live_stats or {}
    rotation = {slug: idx for idx, slug in enumerate(rotation_order or candidates)}

    def key(slug: str) -> tuple[float, float, int, str]:
        n, mean, m2 = _arm_sufficient_stats(ledger_arms.get(slug), live.get(slug))
        posterior = arm_posterior(n=n, mean=mean, m2=m2, prior_scale=prior_scale)
        return (
            -float(boosts.get(slug, 0.0)),
            -posterior.ucb(exploration_c),
            rotation.get(slug, 10_000),
            slug,
        )

    return sorted(dict.fromkeys(candidates), key=key)


def pick_evidence_ranked_slug(
    candidates: Sequence[str],
    ledger_arms: Mapping[str, Any],
    **kwargs: Any,
) -> str | None:
    ordered = rank_arms_by_evidence(candidates, ledger_arms, **kwargs)
    return ordered[0] if ordered else None


# ---------------------------------------------------------------------------
# Terminal verdict
# ---------------------------------------------------------------------------


def build_regime_exhausted_verdict(
    *,
    campaign_id: str,
    loop_id: str,
    cycle_index: int,
    binding_constraint: str,
    closed_slugs: Sequence[str],
    policy_sha256: str | None,
    resume_predicate: str,
) -> dict[str, Any]:
    """Typed terminal verdict: the legal experiment domain is empty.

    This is a conclusion, not a failure — it names the constraint whose change
    re-opens the region. Consumers must not resume the parked loop until the
    ``resume_predicate`` holds.
    """
    return {
        "schema": VERDICT_SCHEMA,
        "campaign_id": campaign_id,
        "loop_id": loop_id,
        "cycle_index": int(cycle_index),
        "binding_constraint": binding_constraint,
        "closed_slugs": sorted(set(closed_slugs)),
        "policy_sha256": policy_sha256,
        "resume_predicate": resume_predicate,
    }
