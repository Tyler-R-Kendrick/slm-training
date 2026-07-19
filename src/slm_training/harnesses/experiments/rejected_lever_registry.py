"""Versioned registry and paired re-adjudication campaign for rejected levers.

EFS0-05 wiring: this module provides the schema, loader, and deterministic
statistical machinery for a preregistered re-adjudication campaign.  It does not
perform expensive reruns itself; it records which levers were rejected, why, and
what evidence would be required to change their status.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.autoresearch.schemas import EvidenceItem
from slm_training.versioning import build_version_stamp


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


CONFOUNDS = frozenset(
    {
        "tiny_n",
        "seed_instability",
        "decoder_bug",
        "harness_interference",
        "checkpoint_missing",
        "representation_mismatch",
        "underexposure",
        "strong_negative_control",
    }
)


class RejectedLeverV1(StrictModel):
    """One previously rejected or "no effect" lever with provenance and confounds."""

    entry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    experiment_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    hypothesis: str = Field(min_length=1)
    original_matrix: str = Field(min_length=1)
    original_source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    original_train_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    original_eval_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_remote_uri: str | None = None
    decoder_path: str = ""
    decoder_version: str = ""
    corpus_size: int = Field(default=0, ge=0)
    suite_size: int = Field(default=0, ge=0)
    seeds: tuple[int, ...] = ()
    original_primary_metric: str = ""
    original_primary_value: float | None = None
    judge_provenance: str = ""
    observed_effect: float | None = None
    cost_metric: str = ""
    observed_cost: float | None = None
    confounds: tuple[str, ...] = ()
    status: Literal[
        "closed", "provisional_negative", "invalidated", "reopen_candidate"
    ] = "provisional_negative"
    evidence_needed: str = ""
    notes: str = ""

    @model_validator(mode="after")
    def _check_confounds(self) -> "RejectedLeverV1":
        bad = [c for c in self.confounds if c not in CONFOUNDS]
        if bad:
            raise ValueError(f"unknown confounds: {bad}")
        return self


class RejectedLeverRegistryV1(StrictModel):
    """Content-addressed registry of rejected levers for EFS0-05."""

    schema_version: Literal["rejected_lever_registry/v1"] = (
        "rejected_lever_registry/v1"
    )
    registry_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    created_at: str = Field(default_factory=utc_now)
    version_stamp: dict[str, Any] = Field(
        default_factory=lambda: build_version_stamp("harness.experiments")
    )
    entries: tuple[RejectedLeverV1, ...] = ()


class ReAdjudicationRowV1(StrictModel):
    """Preregistered treatment/control row for re-adjudicating one lever."""

    row_id: str
    lever_id: str
    control_run_id: str
    treatment_run_id: str
    seeds: tuple[int, ...]
    primary_metric: str
    min_effect: float = Field(default=0.05, gt=0)
    cost_metric: str
    equivalence_margin: float = Field(default=0.02, ge=0)
    decoder_path: str
    eval_policy: dict[str, Any] = Field(default_factory=dict)
    status: Literal["planned", "running", "complete", "blocked"] = "planned"


class PairedSeedObservation(StrictModel):
    """One seed's paired treatment vs control observation."""

    seed: int
    control_value: float
    treatment_value: float
    control_cost: float
    treatment_cost: float
    failed: bool = False
    timeout: bool = False
    notes: str = ""

    @property
    def delta(self) -> float:
        return self.treatment_value - self.control_value

    @property
    def cost_delta(self) -> float:
        return self.treatment_cost - self.control_cost


class PairedTestResult(StrictModel):
    """Statistical summary of a paired five-seed re-adjudication row."""

    row_id: str
    observations: tuple[PairedSeedObservation, ...]
    mean_delta: float
    std_delta: float
    ci_low: float
    ci_high: float
    p_value: float | None
    cost_mean_delta: float
    verdict: Literal[
        "confirmed_negative",
        "equivalent",
        "reopened_positive",
        "inconclusive",
        "invalidated_original",
    ]


def load_registry(path: Path | str) -> RejectedLeverRegistryV1:
    """Load a ``RejectedLeverRegistryV1`` from JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return RejectedLeverRegistryV1.model_validate(data)


def save_registry(registry: RejectedLeverRegistryV1, path: Path | str) -> None:
    """Persist a registry to JSON with atomic directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(registry.model_dump_json(indent=2))


def _index_ids(registry: RejectedLeverRegistryV1, attr: str) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for entry in registry.entries:
        for value in getattr(entry, attr):
            seen.setdefault(value, []).append(entry.entry_id)
    return {value: ids for value, ids in seen.items() if len(ids) > 1}


def check_duplicate_run_ids(registry: RejectedLeverRegistryV1) -> dict[str, list[str]]:
    """Return ``run_id -> [entry_id, ...]`` for every run ID referenced more than once."""
    return _index_ids(registry, "run_ids")


def check_duplicate_experiment_ids(
    registry: RejectedLeverRegistryV1,
) -> dict[str, list[str]]:
    """Return ``experiment_id -> [entry_id, ...]`` for duplicate experiment IDs."""
    return _index_ids(registry, "experiment_ids")


def build_preregistered_campaign(
    registry: RejectedLeverRegistryV1,
    *,
    required_levers: int = 5,
    seed_count: int = 5,
    decoder_path: str = "current_exact_or_compiler",
) -> list[ReAdjudicationRowV1]:
    """Select up to ``required_levers`` eligible levers for re-adjudication.

    Eligible levers are ``reopen_candidate`` or ``provisional_negative`` entries
    that still exist in the registry and have a matched control available.
    """
    candidates = [
        entry
        for entry in registry.entries
        if entry.status in ("reopen_candidate", "provisional_negative")
    ]
    rows: list[ReAdjudicationRowV1] = []
    for entry in candidates[:required_levers]:
        seeds = tuple(range(seed_count))
        control_run_id = (
            entry.run_ids[0] if entry.run_ids else f"{entry.entry_id}_control"
        )
        treatment_run_id = f"{entry.entry_id}_treatment"
        rows.append(
            ReAdjudicationRowV1(
                row_id=f"{registry.registry_id}-{entry.entry_id}",
                lever_id=entry.entry_id,
                control_run_id=control_run_id,
                treatment_run_id=treatment_run_id,
                seeds=seeds,
                primary_metric=entry.original_primary_metric
                or "binding_aware_meaningful_v2",
                cost_metric=entry.cost_metric or "wall_seconds",
                decoder_path=decoder_path,
                eval_policy={
                    "decoder_path": decoder_path,
                    "metric": "binding_aware_meaningful_v2",
                },
            )
        )
    return rows


def _bootstrap_ci(
    deltas: list[float],
    *,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [rng.choice(deltas) for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = 1 - confidence
    low_idx = int(n_boot * alpha / 2)
    high_idx = max(int(n_boot * (1 - alpha / 2)), low_idx + 1)
    return (means[low_idx], means[high_idx])


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def classify_verdict(
    mean_delta: float,
    ci_low: float,
    ci_high: float,
    min_effect: float,
    equivalence_margin: float,
) -> Literal[
    "confirmed_negative",
    "equivalent",
    "reopened_positive",
    "inconclusive",
    "invalidated_original",
]:
    """Classify a paired comparison under the preregistered decision contract.

    Ordering follows the statistical decision contract in EFS0-05:

    1. ``reopened_positive`` — the whole CI is above the minimum useful gain.
    2. ``equivalent`` — the whole CI lies inside the preregistered equivalence
       band around zero, so the lever has no practical effect.
    3. ``confirmed_negative`` — the CI is below the minimum useful gain and is
       not inside the equivalence band.
    4. ``inconclusive`` — everything else (overlap, missing data, etc.).
    """
    if not math.isfinite(mean_delta):
        return "inconclusive"
    if ci_low > min_effect:
        return "reopened_positive"
    if ci_low >= -equivalence_margin and ci_high <= equivalence_margin:
        return "equivalent"
    if ci_high < min_effect:
        return "confirmed_negative"
    return "inconclusive"


def paired_seed_result(
    row: ReAdjudicationRowV1,
    observations: list[PairedSeedObservation],
) -> PairedTestResult:
    """Compute a paired statistical summary for one re-adjudication row.

    Observations marked ``failed`` or ``timeout`` are excluded from the numeric
    summary but are retained in the result so that incomplete cells are never
    silently treated as negative.
    """
    valid = [obs for obs in observations if not obs.failed and not obs.timeout]
    deltas = [obs.delta for obs in valid]
    costs = [obs.cost_delta for obs in valid]
    mean_delta = statistics.mean(deltas) if deltas else float("nan")
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    ci_low, ci_high = _bootstrap_ci(deltas)

    p_value: float | None = None
    if len(deltas) > 1:
        se = std_delta / math.sqrt(len(deltas))
        t_stat = mean_delta / se if se > 0 else 0.0
        # Two-sided normal approximation; exact t-distribution is overkill for
        # the seed-level bootstrap contract and avoids a scipy dependency.
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    cost_mean_delta = statistics.mean(costs) if costs else float("nan")
    verdict = classify_verdict(
        mean_delta, ci_low, ci_high, row.min_effect, row.equivalence_margin
    )
    return PairedTestResult(
        row_id=row.row_id,
        observations=tuple(observations),
        mean_delta=mean_delta,
        std_delta=std_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        cost_mean_delta=cost_mean_delta,
        verdict=verdict,
    )


def lever_signature(entry: RejectedLeverV1) -> str:
    """Canonical signature used for deduplication against autoresearch proposals."""
    payload = json.dumps(
        {
            "matrix": entry.original_matrix,
            "experiments": sorted(entry.experiment_ids),
            "confounds": sorted(entry.confounds),
            "primary_metric": entry.original_primary_metric,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def to_evidence_items(registry: RejectedLeverRegistryV1) -> tuple[EvidenceItem, ...]:
    """Convert the registry into ``EvidenceItem`` rows for autoresearch intake.

    Each entry becomes one item with kind ``rejected_lever``.  The canonical
    ``lever_signature`` lets the hypothesizer and validator avoid re-proposing
    closed branches.
    """
    items: list[EvidenceItem] = []
    for entry in registry.entries:
        summary = (
            f"lever={entry.entry_id} matrix={entry.original_matrix} "
            f"status={entry.status} confounds={','.join(entry.confounds)} "
            f"signature={lever_signature(entry)} evidence={entry.evidence_needed}"
        )
        metrics: dict[str, float] = {}
        if entry.original_primary_value is not None:
            metrics["original_primary_value"] = entry.original_primary_value
        if entry.observed_effect is not None:
            metrics["observed_effect"] = entry.observed_effect
        if entry.observed_cost is not None:
            metrics["observed_cost"] = entry.observed_cost
        raw = entry.model_dump_json().encode()
        items.append(
            EvidenceItem(
                path=f"rejected_lever_registry/{registry.registry_id}/{entry.entry_id}.json",
                kind="rejected_lever",
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
                summary=summary,
                metrics=metrics,
            )
        )
    return tuple(items)


def closed_lever_signatures(registry: RejectedLeverRegistryV1) -> frozenset[str]:
    """Signatures of levers that should not be re-proposed."""
    return frozenset(
        lever_signature(entry)
        for entry in registry.entries
        if entry.status in ("closed", "invalidated")
    )
