"""Deterministic cross-version evidence ledger for the autotrain hill climb.

The continuous loop's live memory (``outputs/autoresearch``) is ephemeral and
capped, while the durable record — 1,500+ result JSONs under ``docs/design``
spanning many harness versions — was never machine-consumed. This module closes
that gap with four deterministic pieces (no LLM, no RNG: identical inputs
produce identical outputs):

1. **Mining** — extract per-arm observations from the committed cycle-record,
   measured-result, and sdlc-delivery schema families (standalone or embedded
   as a cycle record's ``"delivery"`` object), joining "the same arm" across
   harness versions by slug (the stable identity `_SCREENING_ARM_BANK` already uses)
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

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LEDGER_SCHEMA = "autotrain_evidence_ledger/v1"
VERDICT_SCHEMA = "regime_exhausted_verdict/v1"
DELIVERY_SCHEMA = "autotrain_sdlc_delivery/v1"

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

#: Stamped cross-partition buckets decay per version step behind the current
#: eval key; unstamped buckets keep the flat ``CROSS_PARTITION_WEIGHT``. The
#: climb policy ``selection`` block may override both knobs.
STALENESS_DECAY = 0.9
STALENESS_FLOOR = 0.1

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
_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _delivery_observations(
    payload: Mapping[str, Any], source: str, *, eval_key: str | None = None
) -> list[ArmObservation]:
    """Observations from an ``autotrain_sdlc_delivery/v1`` payload.

    The rich per-cycle delivery record names its arm directly
    (``candidate_id``/``arm_seed``) — no reasons-string recovery needed.
    """
    slug = slug_from_candidate_token(str(payload.get("candidate_id") or ""))
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
    seed = payload.get("arm_seed")
    cycle_index = payload.get("cycle_index")
    return [
        ArmObservation(
            slug=slug,
            delta=delta,
            positive=bool(payload["positive"])
            if isinstance(payload.get("positive"), bool)
            else None,
            complete=bool(payload.get("measurement_complete")),
            seed=int(seed) if isinstance(seed, int) else None,
            eval_key=eval_key_from_stamp(payload) or eval_key,
            campaign_id=str(payload.get("campaign_id") or ""),
            cycle_index=int(cycle_index) if isinstance(cycle_index, int) else None,
            source=source,
        )
    ]


def _cycle_record_observations(
    payload: Mapping[str, Any], source: str
) -> list[ArmObservation]:
    # An embedded delivery record supersedes reasons-string recovery: same
    # measurement, richer identity (candidate_id/arm_seed) — never both.
    delivery = payload.get("delivery")
    if isinstance(delivery, Mapping) and delivery.get("schema") == DELIVERY_SCHEMA:
        embedded = _delivery_observations(
            delivery, source, eval_key=eval_key_from_stamp(payload)
        )
        if embedded:
            return embedded
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
        elif slug is None and experiment_id.rsplit("-", 1)[-1] in _NON_ARM_SLUGS:
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
    if schema == DELIVERY_SCHEMA:
        return _delivery_observations(payload, source)
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
        # Campaign-less records fall back to a content digest so a payload
        # duplicated across files (e.g. an iteration doc and its closeout
        # copy) still folds exactly once.
        payload_digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        for obs in extract_observations(payload, source=path.name):
            identity = obs.campaign_id or f"sha:{payload_digest}"
            key = (identity, obs.cycle_index, obs.slug, obs.seed)
            if key in seen:
                continue
            seen.add(key)
            _fold_observation(per_arm, obs)
    arms = {
        slug: _finalize_arm(stats)
        for slug, stats in sorted(per_arm.items())
    }
    try:
        generated_from = str(design_dir.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        generated_from = design_dir.name
    return {
        "schema": LEDGER_SCHEMA,
        "generated_from": generated_from,
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
    stats = dict(stats)
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


def current_eval_key() -> str | None:
    """Current eval-comparability key from the live version registry.

    Mirrors ``eval_key_from_stamp``'s format so ledger buckets produced under
    the same metric constraints match exactly. ``None`` when the registry is
    unavailable.
    """
    try:
        from slm_training.versioning import component_version

        parts = []
        for name in EVAL_KEY_COMPONENTS:
            try:
                parts.append(f"{name}={component_version(name)}")
            except KeyError:
                continue
        return "|".join(parts) if parts else None
    except Exception:  # noqa: BLE001 — selection must not require the registry
        return None


_VERSIONS_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "resources" / "versions.json"
)
_VERSION_HISTORY_CACHE: dict[str, tuple[str, ...]] | None = None


def parse_eval_key(key: str | None) -> dict[str, str]:
    """Parse a ``comp=ver|comp=ver`` bucket key; missing components tolerated."""
    parts: dict[str, str] = {}
    for token in (key or "").split("|"):
        name, sep, version = token.partition("=")
        if sep and name and version:
            parts[name] = version
    return parts


def load_version_histories(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Newest-first distinct version history per eval-key component.

    Reads ``components.<id>.history`` from the version registry directly —
    the same newest-first walk as ``scripts/verify_version_stamps``'s
    ``_history_versions``, reimplemented here so selection never imports the
    checker script. No-bump rows repeat a version; each version keeps its
    first (newest) index. Empty on a missing or malformed registry.
    """
    target = path or _VERSIONS_REGISTRY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    components = payload.get("components")
    if not isinstance(components, Mapping):
        return {}
    histories: dict[str, tuple[str, ...]] = {}
    for name in EVAL_KEY_COMPONENTS:
        entry = components.get(name)
        rows = entry.get("history") if isinstance(entry, Mapping) else None
        versions = [
            str(row.get("version"))
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, Mapping) and row.get("version")
        ]
        histories[name] = tuple(dict.fromkeys(versions))
    return histories


def _cached_version_histories() -> dict[str, tuple[str, ...]]:
    global _VERSION_HISTORY_CACHE
    if _VERSION_HISTORY_CACHE is None:
        _VERSION_HISTORY_CACHE = load_version_histories()
    return _VERSION_HISTORY_CACHE


def staleness_behind_by(
    bucket_key: str | None,
    eval_key: str | None,
    histories: Mapping[str, Sequence[str]],
) -> int:
    """Summed history-index distance between a bucket's stamp and the current.

    Per ``EVAL_KEY_COMPONENTS`` component: 0 when equal, missing, or either
    version is unknown to the history; otherwise the index distance in the
    newest-first distinct history.
    """
    bucket = parse_eval_key(bucket_key)
    current = parse_eval_key(eval_key)
    behind = 0
    for name in EVAL_KEY_COMPONENTS:
        bucket_version = bucket.get(name)
        current_version = current.get(name)
        if not bucket_version or not current_version:
            continue
        if bucket_version == current_version:
            continue
        history = histories.get(name) or ()
        if bucket_version in history and current_version in history:
            behind += abs(
                history.index(bucket_version) - history.index(current_version)
            )
    return behind


def stale_bucket_weight(
    bucket_key: str,
    eval_key: str | None,
    *,
    decay: float = STALENESS_DECAY,
    floor: float = STALENESS_FLOOR,
    histories: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Staleness-decayed pooling weight for a stamped cross-partition bucket.

    ``max(floor, CROSS_PARTITION_WEIGHT * decay ** behind_by)`` — unknown
    versions contribute no distance, so a bucket with no resolvable history
    keeps the flat ``CROSS_PARTITION_WEIGHT``.
    """
    resolved = histories if histories is not None else _cached_version_histories()
    behind = staleness_behind_by(bucket_key, eval_key, resolved)
    return max(floor, CROSS_PARTITION_WEIGHT * decay**behind)


def _weighted_bucket_stats(
    buckets: Mapping[str, Any],
    eval_key: str | None,
    *,
    staleness_decay: float = STALENESS_DECAY,
    staleness_floor: float = STALENESS_FLOOR,
    histories: Mapping[str, Sequence[str]] | None = None,
) -> tuple[float, float, float]:
    """Fold per-eval-partition buckets into weighted ``(n, mean, m2)``.

    Same-partition observations count at full weight; unstamped history at
    the flat ``CROSS_PARTITION_WEIGHT``; stamped cross-partition buckets at
    the per-version staleness decay (``stale_bucket_weight``).
    """
    rows: list[tuple[float, float, float, float]] = []
    total_w = weighted_sum = 0.0
    for key, bucket in sorted(buckets.items()):
        if not isinstance(bucket, Mapping):
            continue
        bucket_n = float(bucket.get("n") or 0.0)
        if bucket_n <= 0:
            continue
        if eval_key is not None and key == eval_key:
            weight = 1.0
        elif eval_key is None or key == "unstamped":
            weight = CROSS_PARTITION_WEIGHT
        else:
            weight = stale_bucket_weight(
                key,
                eval_key,
                decay=staleness_decay,
                floor=staleness_floor,
                histories=histories,
            )
        bucket_sum = float(bucket.get("sum") or 0.0)
        bucket_sumsq = float(bucket.get("sumsq") or 0.0)
        rows.append((weight, bucket_n, bucket_sum, bucket_sumsq))
        total_w += weight * bucket_n
        weighted_sum += weight * bucket_sum
    if total_w <= 0:
        return 0.0, 0.0, 0.0
    mean = weighted_sum / total_w
    m2 = sum(
        w * (sumsq - 2.0 * mean * s + bn * mean * mean)
        for w, bn, s, sumsq in rows
    )
    return total_w, mean, max(0.0, m2)


def _arm_sufficient_stats(
    ledger_arm: Mapping[str, Any] | None,
    live: tuple[float, float] | None,
    eval_key: str | None = None,
    *,
    staleness_decay: float = STALENESS_DECAY,
    staleness_floor: float = STALENESS_FLOOR,
    histories: Mapping[str, Sequence[str]] | None = None,
) -> tuple[float, float, float]:
    """Combine ledger stats and live loop stats into ``(n, mean, m2)``.

    Ledger evidence is folded per eval partition: same-partition buckets at
    full weight, unstamped at ``CROSS_PARTITION_WEIGHT``, stamped
    cross-partition buckets at the staleness decay (uniform flat discount
    when no bucket detail exists). Live stats arrive as
    ``(n_complete, mean_delta)`` with unknown within-run variance; they are
    folded via the parallel-axis rule with zero internal spread (the prior's
    ``alpha0`` keeps the posterior width positive).
    """
    n = mean = m2 = 0.0
    if isinstance(ledger_arm, Mapping):
        buckets = ledger_arm.get("by_eval_key")
        if isinstance(buckets, Mapping) and buckets:
            n, mean, m2 = _weighted_bucket_stats(
                buckets,
                eval_key,
                staleness_decay=staleness_decay,
                staleness_floor=staleness_floor,
                histories=histories,
            )
        else:
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
    eval_key: str | None = None,
    staleness_decay: float = STALENESS_DECAY,
    staleness_floor: float = STALENESS_FLOOR,
    histories: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Deterministic evidence ranking of candidate slugs.

    Sort key mirrors ``soft_rank_slugs``: residual boost lexicographically
    dominates, then the posterior UCB, then rotation order, then slug.
    """
    boosts = residual_boosts or {}
    live = live_stats or {}
    rotation = {slug: idx for idx, slug in enumerate(rotation_order or candidates)}

    def key(slug: str) -> tuple[float, float, int, str]:
        n, mean, m2 = _arm_sufficient_stats(
            ledger_arms.get(slug),
            live.get(slug),
            eval_key,
            staleness_decay=staleness_decay,
            staleness_floor=staleness_floor,
            histories=histories,
        )
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
        "schema_version": VERDICT_SCHEMA,
        "campaign_id": campaign_id,
        "loop_id": loop_id,
        "cycle_index": int(cycle_index),
        "binding_constraint": binding_constraint,
        "closed_slugs": sorted(set(closed_slugs)),
        "policy_sha256": policy_sha256,
        "resume_predicate": resume_predicate,
    }
