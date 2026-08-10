"""Cheap thrash steering from completed deliveries (no retrain).

Pure helpers: classify interesting residuals and soft-rank screening slugs
from historical ``sdlc_delivery.json`` evidence. Interesting residual ≠
promotable; multi-seed close and hard skips still win.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "RESIDUAL_HIGH_BAND_ABSOLUTE",
    "RESIDUAL_PRIMARY_UP_BINDER_DOWN",
    "RESIDUAL_EFFICIENCY_WIN_QUALITY_HELD",
    "RESIDUAL_CONTROL_SPIKE_SHARED",
    "DEFAULT_HIGH_SS_BAND",
    "BINDER_FAMILY_SLUGS",
    "ResidualObservation",
    "SlugStats",
    "metric_from_delivery",
    "slug_from_candidate_id",
    "classify_delivery_residual",
    "aggregate_slug_stats",
    "soft_rank_slugs",
    "pick_soft_ranked_slug",
    "residual_boosts_from_observations",
    "expand_family_boosts",
    "build_slug_stats_payload",
    "screening_tie_saturation",
    "rank_absolute_regimes",
]

# Soft-related screening arms for binder non-regression residuals (not densify
# only the exact failed slug — also try binder-focused thrash bank members).
BINDER_FAMILY_SLUGS: frozenset[str] = frozenset(
    {
        "binder-arity",
        "binder-component-plan",
        "binder-topology",
        "slot-contract-context",
        "constraint-graph",
        "literal-close",
        "literal-margin",
        "compose-ltr-tail-symbol-boundary",
        "symbol-boundary",
        "symbol-boundary-structure",
    }
)

RESIDUAL_HIGH_BAND_ABSOLUTE = "high_band_absolute"
RESIDUAL_PRIMARY_UP_BINDER_DOWN = "primary_up_binder_down"
RESIDUAL_EFFICIENCY_WIN_QUALITY_HELD = "efficiency_win_quality_held"
RESIDUAL_CONTROL_SPIKE_SHARED = "control_spike_shared"

DEFAULT_HIGH_SS_BAND = 0.35
_DELTA_EPS = 1e-9


@dataclass(frozen=True)
class ResidualObservation:
    """One complete dual-arm delivery classified as interesting (or empty)."""

    campaign_id: str
    cycle_index: int | None
    slug: str | None
    residual_class: str | None
    score: float
    ss_control: float | None
    ss_cand: float | None
    mpr_control: float | None
    mpr_cand: float | None
    binder_control: float | None
    binder_cand: float | None
    positive: bool | None
    measurement_complete: bool | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "thrash_residual_observation/v1",
            "campaign_id": self.campaign_id,
            "cycle_index": self.cycle_index,
            "slug": self.slug,
            "residual_class": self.residual_class,
            "score": self.score,
            "ss_control": self.ss_control,
            "ss_cand": self.ss_cand,
            "mpr_control": self.mpr_control,
            "mpr_cand": self.mpr_cand,
            "binder_control": self.binder_control,
            "binder_cand": self.binder_cand,
            "positive": self.positive,
            "measurement_complete": self.measurement_complete,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SlugStats:
    """Aggregated complete-delivery stats for one thrash slug."""

    slug: str
    n_complete: int
    n_positive: int
    mean_delta_ss: float
    binder_fail_rate: float
    incomplete_n: int
    residual_hits: int

    @property
    def win_rate(self) -> float:
        if self.n_complete <= 0:
            return 0.0
        return self.n_positive / float(self.n_complete)

    def prior_score(self) -> float:
        """Soft ranking score (higher = prefer among open arms)."""
        return (
            1.5 * self.mean_delta_ss
            + 0.5 * self.win_rate
            + 0.25 * (self.residual_hits / max(1, self.n_complete))
            - 0.75 * self.binder_fail_rate
            - 0.1 * (self.incomplete_n / max(1, self.n_complete + self.incomplete_n))
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "n_complete": self.n_complete,
            "n_positive": self.n_positive,
            "win_rate": self.win_rate,
            "mean_delta_ss": self.mean_delta_ss,
            "binder_fail_rate": self.binder_fail_rate,
            "incomplete_n": self.incomplete_n,
            "residual_hits": self.residual_hits,
            "prior_score": self.prior_score(),
        }


def metric_from_delivery(
    metrics: Mapping[str, Any] | None,
    *keys: str,
) -> float | None:
    """Pull a finite metric from flat or smoke-prefixed delivery metrics."""
    if not isinstance(metrics, dict):
        return None
    candidates: list[str] = []
    for key in keys:
        candidates.append(key)
        if not key.startswith("smoke."):
            candidates.append(f"smoke.{key}")
    for key in candidates:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        return value
    smoke = metrics.get("smoke")
    if isinstance(smoke, dict):
        for key in keys:
            bare = key.removeprefix("smoke.")
            if bare not in smoke:
                continue
            try:
                value = float(smoke[bare])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if value == value:
                return value
    return None


def screening_tie_saturation(
    deliveries: Sequence[Mapping[str, Any]],
    *,
    threshold: int,
    epsilon: float = _DELTA_EPS,
) -> tuple[int, int | None]:
    """Return the current complete-tie streak and its threshold-crossing cycle.

    Deliveries may be in either order. Incomplete rows are ignored rather than
    converted to quality zeros; the newest complete non-tie resets the streak.
    """

    complete: list[tuple[int, float]] = []
    for row in deliveries:
        if row.get("measurement_complete") is not True:
            continue
        ctrl = row.get("control_metrics")
        cand = row.get("candidate_metrics")
        ss_c = metric_from_delivery(ctrl, "structural_similarity")
        ss_k = metric_from_delivery(cand, "structural_similarity")
        if ss_c is None or ss_k is None:
            continue
        try:
            cycle = int(row.get("cycle_index") or 0)
        except (TypeError, ValueError):
            continue
        complete.append((cycle, ss_k - ss_c))
    complete.sort(reverse=True)
    ties: list[int] = []
    for cycle, delta in complete:
        if abs(delta) > epsilon:
            break
        ties.append(cycle)
    required = max(1, int(threshold))
    trigger = sorted(ties)[required - 1] if len(ties) >= required else None
    return len(ties), trigger


def rank_absolute_regimes(
    deliveries: Sequence[Mapping[str, Any]],
    *,
    through_cycle: int,
    max_regimes: int,
    decode_cost_slugs: Iterable[str] = (),
    excluded_slugs: Iterable[str] = (),
    minimum_latency_gain_fraction: float = 0.05,
) -> list[str]:
    """Rank size-matched complete ties by absolute guarded quality.

    Decode-cost treatments are useful residuals only when they also clear the
    existing latency-effect floor. This is screening recovery, not promotion.
    """

    decode_cost = set(decode_cost_slugs)
    excluded = set(excluded_slugs)
    best: dict[str, tuple[float, float, float, float, int]] = {}
    for row in deliveries:
        if row.get("measurement_complete") is not True:
            continue
        try:
            cycle = int(row.get("cycle_index") or 0)
        except (TypeError, ValueError):
            continue
        if cycle > int(through_cycle):
            continue
        slug = slug_from_candidate_id(str(row.get("candidate_id") or "") or None)
        if not slug:
            continue
        if slug in excluded:
            continue
        ctrl = row.get("control_metrics")
        cand = row.get("candidate_metrics")
        ss_c = metric_from_delivery(ctrl, "structural_similarity")
        ss_k = metric_from_delivery(cand, "structural_similarity")
        parse_c = metric_from_delivery(ctrl, "parse_rate")
        parse_k = metric_from_delivery(cand, "parse_rate")
        mpr_c = metric_from_delivery(ctrl, "meaningful_program_rate")
        mpr_k = metric_from_delivery(cand, "meaningful_program_rate")
        binder_c = metric_from_delivery(ctrl, "binder_reference_f1")
        binder_k = metric_from_delivery(cand, "binder_reference_f1")
        latency_c = metric_from_delivery(ctrl, "latency_ms_p50")
        latency_k = metric_from_delivery(cand, "latency_ms_p50")
        metrics = (
            ss_c,
            ss_k,
            parse_c,
            parse_k,
            mpr_c,
            mpr_k,
            binder_c,
            binder_k,
            latency_c,
            latency_k,
        )
        if any(value is None for value in metrics):
            continue
        assert ss_c is not None and ss_k is not None
        assert parse_c is not None and parse_k is not None
        assert mpr_c is not None and mpr_k is not None
        assert binder_c is not None and binder_k is not None
        assert latency_c is not None and latency_k is not None
        if abs(ss_k - ss_c) > _DELTA_EPS:
            continue
        if parse_k + _DELTA_EPS < parse_c or mpr_k + _DELTA_EPS < mpr_c:
            continue
        if binder_k + _DELTA_EPS < binder_c:
            continue
        try:
            baseline_params = int(row["baseline_trainable_params"])
            candidate_params = int(row["candidate_trainable_params"])
        except (KeyError, TypeError, ValueError):
            continue
        if candidate_params > baseline_params:
            continue
        if slug in decode_cost:
            gain = (latency_c - latency_k) / max(latency_c, _DELTA_EPS)
            if gain + _DELTA_EPS < float(minimum_latency_gain_fraction):
                continue
        score = (ss_k, mpr_k, binder_k, -latency_k, -cycle)
        if score > best.get(slug, (-1.0, -1.0, -1.0, float("-inf"), 0)):
            best[slug] = score
    ranked = sorted(best, key=lambda slug: (*(-v for v in best[slug][:-1]), slug))
    return ranked[: max(0, int(max_regimes))]


def slug_from_candidate_id(candidate_id: str | None) -> str | None:
    """Extract thrash arm slug from continuous experiment id suffix."""
    if not candidate_id:
        return None
    text = str(candidate_id)
    # ...-cN-<slug>  (slug may contain hyphens)
    match = re.search(r"-c\d+-(.+)$", text)
    if match:
        slug = match.group(1)
        if slug in {"control", "confirm", "promote"}:
            return None
        return slug
    return None


def _reasons_blob(reasons: Sequence[str]) -> str:
    return " ".join(str(r) for r in reasons)


def classify_delivery_residual(
    delivery: Mapping[str, Any],
    *,
    campaign_id: str = "",
    cycle_index: int | None = None,
    high_ss_band: float = DEFAULT_HIGH_SS_BAND,
) -> ResidualObservation | None:
    """Classify a complete delivery; return None when not interesting."""
    complete = delivery.get("measurement_complete")
    if complete is False:
        return None
    reasons = tuple(str(r) for r in (delivery.get("reasons") or ()))
    ctrl = delivery.get("control_metrics") or {}
    cand = delivery.get("candidate_metrics") or {}
    if not isinstance(ctrl, dict) or not isinstance(cand, dict):
        return None
    ss_c = metric_from_delivery(ctrl, "structural_similarity")
    ss_k = metric_from_delivery(cand, "structural_similarity")
    mpr_c = metric_from_delivery(ctrl, "meaningful_program_rate")
    mpr_k = metric_from_delivery(cand, "meaningful_program_rate")
    b_c = metric_from_delivery(ctrl, "binder_reference_f1")
    b_k = metric_from_delivery(cand, "binder_reference_f1")
    if ss_c is None and ss_k is None:
        return None

    blob = _reasons_blob(reasons)
    delta = None
    if ss_c is not None and ss_k is not None:
        delta = ss_k - ss_c
    max_ss = max(x for x in (ss_c, ss_k) if x is not None)

    residual_class: str | None = None
    score = 0.0

    # Clear primary lift (win label or delta past noise). null_or_worse may
    # still carry a positive improvement string on fixture noise — treat Δ>0.02
    # as lift for residual steering only (not Phase-A positive).
    primary_up = "primary_metric_win:" in blob or (delta is not None and delta > 0.02)
    binder_down = "non_regression_fail:binder_reference_f1" in blob or (
        b_c is not None
        and b_k is not None
        and b_k + 1e-6 < b_c
        and "non_regression_fail" in blob
    )
    efficiency = "efficiency_win:" in blob and "efficiency_win_rejected" not in blob
    quality_held = "quality_held:" in blob
    both_high = (
        ss_c is not None
        and ss_k is not None
        and ss_c >= high_ss_band
        and ss_k >= high_ss_band
    )
    near_tie = delta is not None and abs(delta) < 0.01

    if both_high and near_tie and not primary_up:
        residual_class = RESIDUAL_CONTROL_SPIKE_SHARED
        score = 0.1  # informative but do not densify
    elif binder_down and (primary_up or efficiency):
        # Binder non-regression fails are the load-bearing residual for c251-like
        # spikes; outrank pure efficiency densify.
        residual_class = RESIDUAL_PRIMARY_UP_BINDER_DOWN
        score = 1.0 + max(0.0, delta or 0.0)
    elif efficiency and quality_held:
        residual_class = RESIDUAL_EFFICIENCY_WIN_QUALITY_HELD
        score = 0.8 + max(0.0, delta or 0.0)
    elif max_ss >= high_ss_band and (
        primary_up or (delta is not None and delta > 0.02)
    ):
        residual_class = RESIDUAL_HIGH_BAND_ABSOLUTE
        score = 0.6 + max_ss
    elif max_ss >= high_ss_band and both_high:
        residual_class = RESIDUAL_CONTROL_SPIKE_SHARED
        score = 0.15
    else:
        return None

    slug = slug_from_candidate_id(str(delivery.get("candidate_id") or "") or None)
    return ResidualObservation(
        campaign_id=str(campaign_id or delivery.get("campaign_id") or ""),
        cycle_index=cycle_index,
        slug=slug,
        residual_class=residual_class,
        score=float(score),
        ss_control=ss_c,
        ss_cand=ss_k,
        mpr_control=mpr_c,
        mpr_cand=mpr_k,
        binder_control=b_c,
        binder_cand=b_k,
        positive=delivery.get("positive")
        if isinstance(delivery.get("positive"), bool)
        else None,
        measurement_complete=bool(complete) if complete is not None else None,
        reasons=reasons,
    )


def aggregate_slug_stats(
    deliveries: Sequence[Mapping[str, Any]],
    *,
    residuals: Sequence[ResidualObservation] | None = None,
) -> dict[str, SlugStats]:
    """Fold deliveries into per-slug stats for soft ranking."""
    complete: dict[str, list[float]] = defaultdict(list)
    positive: dict[str, int] = defaultdict(int)
    binder_fail: dict[str, int] = defaultdict(int)
    incomplete: dict[str, int] = defaultdict(int)
    residual_hits: dict[str, int] = defaultdict(int)

    for row in deliveries:
        slug = slug_from_candidate_id(str(row.get("candidate_id") or "") or None)
        if not slug:
            continue
        if row.get("measurement_complete") is False:
            incomplete[slug] += 1
            continue
        ctrl = row.get("control_metrics") or {}
        cand = row.get("candidate_metrics") or {}
        if not isinstance(ctrl, dict) or not isinstance(cand, dict):
            continue
        ss_c = metric_from_delivery(ctrl, "structural_similarity")
        ss_k = metric_from_delivery(cand, "structural_similarity")
        if ss_c is None or ss_k is None:
            incomplete[slug] += 1
            continue
        complete[slug].append(ss_k - ss_c)
        if row.get("positive") is True:
            positive[slug] += 1
        blob = _reasons_blob(list(row.get("reasons") or ()))
        if "non_regression_fail:binder_reference_f1" in blob:
            binder_fail[slug] += 1

    for obs in residuals or ():
        if obs.slug and obs.residual_class not in {
            None,
            RESIDUAL_CONTROL_SPIKE_SHARED,
        }:
            residual_hits[obs.slug] += 1

    out: dict[str, SlugStats] = {}
    slugs = set(complete) | set(incomplete) | set(residual_hits)
    for slug in slugs:
        deltas = complete.get(slug) or []
        n = len(deltas)
        mean_delta = sum(deltas) / n if n else 0.0
        out[slug] = SlugStats(
            slug=slug,
            n_complete=n,
            n_positive=positive.get(slug, 0),
            mean_delta_ss=mean_delta,
            binder_fail_rate=(binder_fail.get(slug, 0) / n) if n else 0.0,
            incomplete_n=incomplete.get(slug, 0),
            residual_hits=residual_hits.get(slug, 0),
        )
    return out


def soft_rank_slugs(
    candidates: Sequence[str],
    *,
    stats: Mapping[str, SlugStats] | None = None,
    residual_boosts: Mapping[str, float] | None = None,
    rotation_order: Sequence[str] | None = None,
) -> list[str]:
    """Order open slugs: residual boost + prior score, then rotation order.

    Multi-seed-closed arms must already be removed from ``candidates``.
    """
    stats = stats or {}
    residual_boosts = residual_boosts or {}
    rot_index = {slug: i for i, slug in enumerate(rotation_order or candidates)}

    def key(slug: str) -> tuple[float, float, int, str]:
        st = stats.get(slug)
        prior = st.prior_score() if st is not None else 0.0
        boost = float(residual_boosts.get(slug, 0.0))
        # Higher score first; lower rotation index next.
        return (-boost, -prior, rot_index.get(slug, 10_000), slug)

    return sorted(candidates, key=key)


def pick_soft_ranked_slug(
    candidates: Sequence[str],
    *,
    stats: Mapping[str, SlugStats] | None = None,
    residual_boosts: Mapping[str, float] | None = None,
    rotation_order: Sequence[str] | None = None,
) -> str | None:
    """Return best open slug under soft ranking, or None if empty."""
    ordered = soft_rank_slugs(
        candidates,
        stats=stats,
        residual_boosts=residual_boosts,
        rotation_order=rotation_order,
    )
    return ordered[0] if ordered else None


def residual_boosts_from_observations(
    observations: Iterable[ResidualObservation],
    *,
    max_age: int | None = None,
    latest_cycle: int | None = None,
) -> dict[str, float]:
    """Map slug → soft boost from recent interesting residuals.

    ``control_spike_shared`` does not boost (explicit non-densify).
    """
    boosts: dict[str, float] = defaultdict(float)
    for obs in observations:
        if not obs.slug or not obs.residual_class:
            continue
        if obs.residual_class == RESIDUAL_CONTROL_SPIKE_SHARED:
            continue
        if (
            max_age is not None
            and latest_cycle is not None
            and obs.cycle_index is not None
            and (latest_cycle - int(obs.cycle_index)) > max_age
        ):
            continue
        weight = {
            RESIDUAL_PRIMARY_UP_BINDER_DOWN: 2.0,
            RESIDUAL_EFFICIENCY_WIN_QUALITY_HELD: 1.5,
            RESIDUAL_HIGH_BAND_ABSOLUTE: 1.0,
        }.get(obs.residual_class, 0.5)
        amount = weight * max(0.1, float(obs.score))
        boosts[obs.slug] += amount
        if obs.residual_class == RESIDUAL_PRIMARY_UP_BINDER_DOWN:
            # Also steer binder-focused bank members (cheap family prior).
            for fam in BINDER_FAMILY_SLUGS:
                if fam != obs.slug:
                    boosts[fam] += 0.25 * amount
    return expand_family_boosts(dict(boosts), observations=observations)


def expand_family_boosts(
    boosts: Mapping[str, float],
    *,
    observations: Iterable[ResidualObservation] | None = None,
) -> dict[str, float]:
    """Expand exact-slug boosts with light compose-family siblings.

    When a residual hits ``compose-ltr-tail-X``, sibling compose slugs already
    in ``boosts`` get a small shared-family bump so rotation does not abandon
    the productive compose band after one non-positive.
    """
    out: dict[str, float] = dict(boosts)
    compose_hits = [
        o.slug
        for o in (observations or ())
        if o.slug
        and o.slug.startswith("compose-")
        and o.residual_class not in {None, RESIDUAL_CONTROL_SPIKE_SHARED}
    ]
    if not compose_hits:
        return out
    # Light shared bump among observed compose residual slugs only (not whole bank).
    unique = sorted(set(compose_hits))
    if len(unique) < 2:
        return out
    for slug in unique:
        out[slug] = float(out.get(slug, 0.0)) + 0.15
    return out


def build_slug_stats_payload(
    *,
    loop_id: str,
    deliveries: Sequence[Mapping[str, Any]],
    residuals: Sequence[ResidualObservation],
    residual_boosts: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """JSON-serializable slug_stats ledger body (regenerable)."""
    stats = aggregate_slug_stats(deliveries, residuals=residuals)
    ranked = sorted(
        stats.values(),
        key=lambda s: (-s.prior_score(), -s.n_complete, s.slug),
    )
    return {
        "schema": "thrash_slug_stats/v1",
        "loop_id": loop_id,
        "n_deliveries": len(deliveries),
        "n_residuals": len(residuals),
        "slugs": {s.slug: s.as_dict() for s in ranked},
        "residual_boosts": dict(residual_boosts or {}),
    }
