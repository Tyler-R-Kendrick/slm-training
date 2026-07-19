"""EFS3-01 wiring: compare solver-state supervision sources.

Provides a typed, serializable schema for training examples derived from
solver states, plus a deterministic mixer that can produce pure-gold,
pure-on-policy, and mixed (DAgger-style) corpora from the same source rows.

This module is eval/data wiring only: it loads no checkpoint, runs no model,
and makes no quality or ship claim.  It reuses the lineage and replay-verified
row kinds produced by ``slm_training.harnesses.distill.solver_supervision``,
but does not require torch or a solver backend.

Honesty invariants:

1. ``UNKNOWN`` support verdicts are preserved; they are never relabeled as
   positive or negative.
2. Cross-split leakage is rejected by ``split_group_id``: any group that
   appears in both a train split and any held-out split is excluded from
   train mixes.
3. Mixed supervision is sampled per-row (coin-flip or weighted draw), not by
   blending labels for the same state.
4. Cost-to-go is carried with an ``observed`` flag; censored costs stay
   censored.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator

from slm_training.versioning import UNKNOWN, build_version_stamp

SUPERVISION_SCHEMA_VERSION = 1

_HELD_OUT_SPLITS = frozenset({"val", "validation", "dev", "test", "heldout", "held_out"})


class SupervisionSource(str, Enum):
    """Where a solver-state training example's hard label came from."""

    GOLD = "gold"
    ON_POLICY = "on_policy"
    MIXED = "mixed"


@dataclass(frozen=True)
class SolverStateTrainingExampleV1:
    """One solver-state × decision point, ready for supervision.

    A single example captures the legal actions at a state, the subset that
    are acceptable according to the chosen supervision source, and enough
    lineage to enforce train/held-out isolation.  It is intentionally
    source-agnostic: the same state may appear once as ``GOLD`` and once as
    ``ON_POLICY`` with different acceptability labels, and a ``MIXED`` corpus
    selects between them.
    """

    problem_id: str
    state_fingerprint: str
    supervision_source: SupervisionSource
    legal_actions: tuple[dict[str, Any], ...]
    acceptable_actions: tuple[dict[str, Any], ...]
    support_verdict: str
    cost_to_go: float | None
    cost_observed: bool
    split_group_id: str
    split: str
    lineage_id: str
    program_family_id: str
    replay_certified: bool
    schema_version: int = SUPERVISION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supervision_source"] = self.supervision_source.value
        data["legal_actions"] = list(self.legal_actions)
        data["acceptable_actions"] = list(self.acceptable_actions)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolverStateTrainingExampleV1":
        source = data.get("supervision_source")
        if isinstance(source, str):
            source = SupervisionSource(source)
        return cls(
            problem_id=str(data["problem_id"]),
            state_fingerprint=str(data["state_fingerprint"]),
            supervision_source=source,  # type: ignore[arg-type]
            legal_actions=tuple(data.get("legal_actions") or ()),
            acceptable_actions=tuple(data.get("acceptable_actions") or ()),
            support_verdict=str(data.get("support_verdict") or "UNKNOWN"),
            cost_to_go=float(data["cost_to_go"]) if data.get("cost_to_go") is not None else None,
            cost_observed=bool(data.get("cost_observed", False)),
            split_group_id=str(data["split_group_id"]),
            split=str(data["split"]),
            lineage_id=str(data["lineage_id"]),
            program_family_id=str(data["program_family_id"]),
            replay_certified=bool(data.get("replay_certified", False)),
            schema_version=int(data.get("schema_version", SUPERVISION_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class SolverStateMixSpec:
    """Specification for one mixed supervision corpus."""

    mix_id: str
    source_weights: dict[SupervisionSource | str, float]
    seed: int = 42
    max_rows_per_source: int | None = None
    notes: str = ""

    def normalized_weights(self) -> dict[SupervisionSource, float]:
        clean: dict[SupervisionSource, float] = {}
        for key, value in self.source_weights.items():
            source = SupervisionSource(key) if isinstance(key, str) else key
            v = float(value)
            if v > 0:
                clean[source] = clean.get(source, 0.0) + v
        total = sum(clean.values())
        if total <= 0:
            raise ValueError("source_weights must sum to a positive value")
        return {source: weight / total for source, weight in sorted(clean.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mix_id": self.mix_id,
            "source_weights": {
                (k.value if isinstance(k, SupervisionSource) else str(k)): float(v)
                for k, v in self.source_weights.items()
            },
            "seed": self.seed,
            "max_rows_per_source": self.max_rows_per_source,
            "notes": self.notes,
        }


@dataclass
class MixResult:
    """Outcome of building one supervision mix."""

    spec: SolverStateMixSpec
    rows: list[SolverStateTrainingExampleV1] = field(default_factory=list)
    rejected_rows: list[dict[str, Any]] = field(default_factory=list)
    source_counts: Counter = field(default_factory=Counter)

    def counts(self) -> dict[str, int]:
        return {
            "rows": len(self.rows),
            "rejected_rows": len(self.rejected_rows),
            "source_counts": dict(self.source_counts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "counts": self.counts(),
            "rows": [row.to_dict() for row in self.rows],
            "rejected_rows": self.rejected_rows,
        }


@dataclass
class CompareResult:
    """Container for the EFS3-01 comparison of the three canonical mixes."""

    gold: MixResult
    on_policy: MixResult
    mixed: MixResult
    held_out_group_ids: frozenset[str]
    version_stamp: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gold": self.gold.to_dict(),
            "on_policy": self.on_policy.to_dict(),
            "mixed": self.mixed.to_dict(),
            "held_out_group_ids": sorted(self.held_out_group_ids),
            "version_stamp": self.version_stamp,
        }


def _group_ids_by_split(
    rows: Iterable[SolverStateTrainingExampleV1],
) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row.split, set()).add(row.split_group_id)
    return by_split


def _held_out_group_ids(rows: Iterable[SolverStateTrainingExampleV1]) -> set[str]:
    by_split = _group_ids_by_split(rows)
    held: set[str] = set()
    for split, groups in by_split.items():
        if split.lower() in _HELD_OUT_SPLITS:
            held.update(groups)
    return held


def _state_key(row: SolverStateTrainingExampleV1) -> str:
    payload = {
        "problem_id": row.problem_id,
        "state_fingerprint": row.state_fingerprint,
        "lineage_id": row.lineage_id,
        "split_group_id": row.split_group_id,
        "split": row.split,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _unique_rows(
    rows: Iterable[SolverStateTrainingExampleV1],
) -> list[SolverStateTrainingExampleV1]:
    """Drop duplicate state keys, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[SolverStateTrainingExampleV1] = []
    for row in rows:
        key = _state_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _sample_by_source(
    rows: list[SolverStateTrainingExampleV1],
    weights: dict[SupervisionSource, float],
    rng: random.Random,
    max_per_source: int | None,
) -> Iterator[SolverStateTrainingExampleV1]:
    """Yield rows selected from each source according to ``weights``.

    For each source, up to ``max_per_source`` rows are sampled without
    replacement (or all of them if the cap is larger).  The caller is expected
    to have already removed cross-split leakage duplicates.
    """
    by_source: dict[SupervisionSource, list[SolverStateTrainingExampleV1]] = {
        source: [] for source in weights
    }
    for row in rows:
        if row.supervision_source in by_source:
            by_source[row.supervision_source].append(row)

    for source in sorted(weights):
        pool = by_source[source]
        cap = max_per_source if max_per_source is not None else len(pool)
        n = min(cap, len(pool))
        if n <= 0:
            continue
        selected = rng.sample(pool, n) if n < len(pool) else pool
        yield from selected


def build_solver_state_mix(
    rows: Iterable[SolverStateTrainingExampleV1],
    spec: SolverStateMixSpec,
    *,
    held_out_group_ids: Iterable[str] | None = None,
    include_held_out_rows: bool = False,
) -> MixResult:
    """Build one supervision corpus from source rows.

    Steps:

    1. Discover held-out ``split_group_id`` values from the union of provided
       rows and any explicitly supplied held-out set.
    2. Reject rows whose group id appears in both a train split and a held-out
       split.
    3. Unless ``include_held_out_rows`` is true, drop rows from held-out splits
       (training corpora should not include them).
    4. For pure specs, keep rows matching the single source.  For mixed specs,
       sample from each source proportionally.
    5. Return selected rows with source counts and a rejection log.
    """
    result = MixResult(spec=spec)
    all_rows = list(rows)
    held_out = set(held_out_group_ids or ())
    held_out.update(_held_out_group_ids(all_rows))

    weights = spec.normalized_weights()
    unique_rows = _unique_rows(all_rows)

    train_groups: set[str] = set()
    for row in unique_rows:
        if row.split.lower() not in _HELD_OUT_SPLITS:
            train_groups.add(row.split_group_id)

    leaked_groups = train_groups & held_out
    allowed_rows: list[SolverStateTrainingExampleV1] = []
    for row in unique_rows:
        if row.split_group_id in leaked_groups and row.split.lower() not in _HELD_OUT_SPLITS:
            result.rejected_rows.append(
                {
                    "state_fingerprint": row.state_fingerprint,
                    "split_group_id": row.split_group_id,
                    "split": row.split,
                    "supervision_source": row.supervision_source.value,
                    "reason": "cross_split_leak",
                }
            )
            continue
        if not include_held_out_rows and row.split.lower() in _HELD_OUT_SPLITS:
            continue
        allowed_rows.append(row)

    rng = random.Random(spec.seed)
    selected = list(
        _sample_by_source(
            allowed_rows,
            weights,
            rng,
            spec.max_rows_per_source,
        )
    )

    # Re-label selected rows to the spec's effective source so a 50/50 mix
    # example is tagged ``MIXED`` while still carrying its origin lineage.
    relabeled: list[SolverStateTrainingExampleV1] = []
    effective_source: SupervisionSource
    if len(weights) == 1:
        effective_source = next(iter(weights))
    else:
        effective_source = SupervisionSource.MIXED
    for row in selected:
        if row.supervision_source != effective_source:
            row = SolverStateTrainingExampleV1(
                problem_id=row.problem_id,
                state_fingerprint=row.state_fingerprint,
                supervision_source=effective_source,
                legal_actions=row.legal_actions,
                acceptable_actions=row.acceptable_actions,
                support_verdict=row.support_verdict,
                cost_to_go=row.cost_to_go,
                cost_observed=row.cost_observed,
                split_group_id=row.split_group_id,
                split=row.split,
                lineage_id=row.lineage_id,
                program_family_id=row.program_family_id,
                replay_certified=row.replay_certified,
                schema_version=row.schema_version,
            )
        relabeled.append(row)
        result.source_counts[row.supervision_source.value] += 1

    result.rows = relabeled
    return result


def compare_solver_state_supervision(
    rows: Iterable[SolverStateTrainingExampleV1],
    *,
    seed: int = 42,
    max_rows_per_source: int | None = None,
    stamp_components: tuple[str, ...] = ("evals.scoring",),
) -> CompareResult:
    """Build the three canonical EFS3-01 mixes: gold, on-policy, and 50/50."""
    all_rows = list(rows)
    held_out = _held_out_group_ids(all_rows)

    gold_spec = SolverStateMixSpec(
        mix_id="efs3-01-gold",
        source_weights={SupervisionSource.GOLD: 1.0},
        seed=seed,
        max_rows_per_source=max_rows_per_source,
        notes="Replay-verified exact-closure states only.",
    )
    on_policy_spec = SolverStateMixSpec(
        mix_id="efs3-01-on-policy",
        source_weights={SupervisionSource.ON_POLICY: 1.0},
        seed=seed,
        max_rows_per_source=max_rows_per_source,
        notes="Solver rollout states only.",
    )
    mixed_spec = SolverStateMixSpec(
        mix_id="efs3-01-mixed-50-50",
        source_weights={SupervisionSource.GOLD: 0.5, SupervisionSource.ON_POLICY: 0.5},
        seed=seed,
        max_rows_per_source=max_rows_per_source,
        notes="DAgger-style 50/50 mix of gold and on-policy solver states.",
    )

    result = CompareResult(
        gold=build_solver_state_mix(all_rows, gold_spec, held_out_group_ids=held_out),
        on_policy=build_solver_state_mix(all_rows, on_policy_spec, held_out_group_ids=held_out),
        mixed=build_solver_state_mix(all_rows, mixed_spec, held_out_group_ids=held_out),
        held_out_group_ids=frozenset(held_out),
    )
    try:
        result.version_stamp = build_version_stamp(*stamp_components)
    except KeyError:
        result.version_stamp = {
            "stamp_schema": UNKNOWN,
            "components": {cid: UNKNOWN for cid in stamp_components},
            "note": "version stamp unavailable",
        }
    return result
