"""SPV2-04: dense teacher distributions on the winning gold/on-policy mixture.

This module extends the EFS3-01 (SLM-118) state-source comparison with the
SPV2-03 (SLM-151) dense legal-set teacher objective.  It builds immutable round
snapshots that mix replay-verified gold rows with teacher-labeled on-policy
rows, under a fixed decision/teacher-label budget and a gold replay floor.

Torch-free: teacher probabilities are stored as normalized float tuples and all
acquisition scoring is plain math.  No checkpoint is loaded and no quality or
ship claim is made here.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from slm_training.evals.solver_state_supervision import (
    SupervisionSource,
    SolverStateTrainingExampleV1,
    _HELD_OUT_SPLITS,
    _held_out_group_ids,
)

DENSE_SCHEMA_VERSION = 2


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #


class AcquisitionPolicy(str, Enum):
    """How to choose which student-visited states receive expensive teacher scores."""

    UNIFORM = "uniform"
    HIGH_DIVERGENCE = "high_divergence"
    LOW_ACCEPTED_RANK = "low_accepted_rank"
    VERIFIER_FAILURE_CONE = "verifier_failure_cone"
    HIGH_REGRET = "high_regret"
    STRATIFIED_MIXTURE = "stratified_mixture"


@dataclass(frozen=True)
class DenseTeacherExampleV1:
    """One solver-state example, optionally carrying a dense teacher distribution.

    The teacher distribution is aligned with ``legal_actions`` and sums to 1.0
    when present.  Gold rows carry no teacher distribution; on-policy rows may
    or may not have received a teacher label depending on the acquisition policy.
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
    teacher_probs: tuple[float, ...] | None = None
    teacher_source: str | None = None
    acquisition_score: float | None = None
    round_id: str | None = None
    schema_version: int = DENSE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["supervision_source"] = self.supervision_source.value
        data["legal_actions"] = list(self.legal_actions)
        data["acceptable_actions"] = list(self.acceptable_actions)
        data["teacher_probs"] = (
            list(self.teacher_probs) if self.teacher_probs is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DenseTeacherExampleV1":
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
            teacher_probs=tuple(float(x) for x in data["teacher_probs"])
            if data.get("teacher_probs") is not None
            else None,
            teacher_source=data.get("teacher_source"),
            acquisition_score=float(data["acquisition_score"]) if data.get("acquisition_score") is not None else None,
            round_id=data.get("round_id"),
            schema_version=int(data.get("schema_version", DENSE_SCHEMA_VERSION)),
        )


# --------------------------------------------------------------------------- #
# Teacher-trace alignment
# --------------------------------------------------------------------------- #


def _align_teacher_probs(
    trace: Any,
    legal_actions: tuple[dict[str, Any], ...],
    align_key: str | Callable[[dict[str, Any]], Any] = "value",
) -> tuple[float, ...] | None:
    """Return teacher probs aligned to ``legal_actions`` order.

    ``trace`` must expose ``legal_action_ids`` and either ``teacher_probs`` or
    ``teacher_logits``.  The alignment key maps a row's action dict to the same
    space as the trace's action ids.
    """
    trace_legal = tuple(trace.legal_action_ids)
    if not trace_legal:
        return None

    if callable(align_key):
        row_keys = [align_key(a) for a in legal_actions]
    else:
        row_keys = [a.get(align_key) for a in legal_actions]

    probs: tuple[float, ...] | None = getattr(trace, "teacher_probs", None)
    if probs is None:
        logits = getattr(trace, "teacher_logits", None)
        if logits is None:
            return None
        # Softmax over the trace's legal set.
        mx = max(float(v) for v in logits)
        exps = [math.exp(float(v) - mx) for v in logits]
        total = sum(exps)
        probs = tuple(v / total for v in exps)

    if len(probs) != len(trace_legal):
        return None

    mapping = {k: float(p) for k, p in zip(trace_legal, probs)}
    aligned: list[float] = []
    for key in row_keys:
        if key not in mapping:
            return None
        aligned.append(mapping[key])

    total = sum(aligned) or 1.0
    aligned = [p / total for p in aligned]
    return tuple(aligned)


def attach_teacher_distribution(
    rows: Iterable[SolverStateTrainingExampleV1],
    teacher_traces: Iterable[Any],
    *,
    align_key: str | Callable[[dict[str, Any]], Any] = "value",
    teacher_source: str | None = None,
    round_id: str | None = None,
) -> list[DenseTeacherExampleV1]:
    """Attach aligned teacher distributions to solver-state rows by fingerprint."""
    trace_by_state: dict[str, Any] = {}
    for trace in teacher_traces:
        sid = getattr(trace, "state_id", None)
        if sid:
            trace_by_state[str(sid)] = trace

    out: list[DenseTeacherExampleV1] = []
    for row in rows:
        trace = trace_by_state.get(row.state_fingerprint)
        teacher_probs = None
        # SPV2-04 labels student-visited states; gold states stay as validated
        # hard-label controls and do not receive a dense teacher distribution here.
        if trace is not None and row.supervision_source is not SupervisionSource.GOLD:
            teacher_probs = _align_teacher_probs(trace, row.legal_actions, align_key)
        out.append(
            DenseTeacherExampleV1(
                problem_id=row.problem_id,
                state_fingerprint=row.state_fingerprint,
                supervision_source=row.supervision_source,
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
                teacher_probs=teacher_probs,
                teacher_source=teacher_source if teacher_probs is not None else None,
                round_id=round_id,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #


def _action_family(action: dict[str, Any]) -> str:
    return str(action.get("family") or action.get("kind") or "unknown")


def _teacher_entropy(probs: tuple[float, ...]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0.0) if probs else 0.0


def _accepted_rank(
    row: DenseTeacherExampleV1,
) -> int:
    """Rank of the (first) accepted action under the teacher distribution."""
    if not row.teacher_probs or not row.acceptable_actions:
        return len(row.legal_actions)
    # Map accepted action value to index in legal_actions.
    accepted_values = {a.get("value") for a in row.acceptable_actions}
    order = sorted(
        range(len(row.teacher_probs)),
        key=lambda i: row.teacher_probs[i],
        reverse=True,
    )
    for rank, idx in enumerate(order):
        if row.legal_actions[idx].get("value") in accepted_values:
            return rank
    return len(row.legal_actions)


def compute_acquisition_score(
    row: DenseTeacherExampleV1,
    policy: AcquisitionPolicy,
) -> float:
    """Deterministic score used to rank rows for teacher labeling."""
    if policy is AcquisitionPolicy.UNIFORM:
        return 0.0

    if policy is AcquisitionPolicy.HIGH_DIVERGENCE:
        # Proxy: peaked teacher distributions are more informative for the
        # student to match (high KL against a uniform baseline).
        if not row.teacher_probs:
            return -1.0
        entropy = _teacher_entropy(row.teacher_probs)
        max_entropy = math.log2(max(len(row.teacher_probs), 2))
        return max_entropy - entropy

    if policy is AcquisitionPolicy.LOW_ACCEPTED_RANK:
        # Lower rank is worse; score is inverse rank so high score = low rank.
        rank = _accepted_rank(row)
        return float(len(row.legal_actions) - rank)

    if policy is AcquisitionPolicy.VERIFIER_FAILURE_CONE:
        return 1.0 if row.support_verdict != "SUPPORTED" else 0.0

    if policy is AcquisitionPolicy.HIGH_REGRET:
        # Wiring placeholder: random but deterministic per row.
        return float(
            int(hashlib.sha256((row.state_fingerprint + "regret").encode()).hexdigest()[:8], 16)
            % 1000
        ) / 1000.0

    if policy is AcquisitionPolicy.STRATIFIED_MIXTURE:
        # Score by rarity of the top teacher action family.
        if not row.teacher_probs:
            return 0.0
        top_idx = max(range(len(row.teacher_probs)), key=lambda i: row.teacher_probs[i])
        family = _action_family(row.legal_actions[top_idx])
        return float(
            int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 1000
        ) / 1000.0

    return 0.0


def acquire_teacher_labeled_states(
    rows: Iterable[DenseTeacherExampleV1],
    budget: int,
    policy: AcquisitionPolicy,
    *,
    require_teacher: bool = True,
    seed: int = 42,
) -> list[DenseTeacherExampleV1]:
    """Select up to ``budget`` rows to receive (or keep) teacher labels.

    The result preserves the row order of the input as much as possible after
    score sorting; ties are broken by a deterministic random shuffle.
    """
    pool = list(rows)
    if require_teacher:
        pool = [r for r in pool if r.teacher_probs is not None]
    if not pool or budget <= 0:
        return []

    scored = [
        (compute_acquisition_score(r, policy), i, r) for i, r in enumerate(pool)
    ]
    rng = random.Random(seed)
    # Sort by score descending, with a deterministic tie-breaker.
    scored.sort(key=lambda x: (x[0], rng.random()), reverse=True)
    selected = [r for _, _, r in scored[:budget]]
    return selected


# --------------------------------------------------------------------------- #
# Snapshot / arms
# --------------------------------------------------------------------------- #


@dataclass
class DenseTeacherSnapshot:
    """Immutable-style round snapshot with per-arm corpora."""

    round_id: str
    decision_budget: int
    teacher_label_budget: int
    gold_floor_count: int
    arms: dict[str, list[DenseTeacherExampleV1]] = field(default_factory=dict)
    source_counts: Counter = field(default_factory=Counter)
    lineage: dict[str, Any] = field(default_factory=dict)
    version_stamp: dict[str, Any] | None = None

    def counts(self) -> dict[str, int]:
        return {arm: len(rows) for arm, rows in self.arms.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "decision_budget": self.decision_budget,
            "teacher_label_budget": self.teacher_label_budget,
            "gold_floor_count": self.gold_floor_count,
            "counts": self.counts(),
            "source_counts": dict(self.source_counts),
            "lineage": dict(self.lineage),
            "version_stamp": self.version_stamp,
            "arms": {
                arm: [r.to_dict() for r in rows] for arm, rows in self.arms.items()
            },
        }


def _drop_held_out(rows: Iterable[DenseTeacherExampleV1]) -> list[DenseTeacherExampleV1]:
    return [r for r in rows if r.split.lower() not in _HELD_OUT_SPLITS]


def _sample(
    rows: list[DenseTeacherExampleV1], n: int, rng: random.Random
) -> list[DenseTeacherExampleV1]:
    if n <= 0:
        return []
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def _with_argmax_acceptable(row: DenseTeacherExampleV1) -> DenseTeacherExampleV1:
    """Return a copy where acceptable_actions contains only the teacher argmax."""
    if not row.teacher_probs:
        return row
    top_idx = max(range(len(row.teacher_probs)), key=lambda i: row.teacher_probs[i])
    top_action = row.legal_actions[top_idx]
    return DenseTeacherExampleV1(
        problem_id=row.problem_id,
        state_fingerprint=row.state_fingerprint,
        supervision_source=row.supervision_source,
        legal_actions=row.legal_actions,
        acceptable_actions=(top_action,),
        support_verdict=row.support_verdict,
        cost_to_go=row.cost_to_go,
        cost_observed=row.cost_observed,
        split_group_id=row.split_group_id,
        split=row.split,
        lineage_id=row.lineage_id,
        program_family_id=row.program_family_id,
        replay_certified=row.replay_certified,
        teacher_probs=row.teacher_probs,
        teacher_source=row.teacher_source,
        acquisition_score=row.acquisition_score,
        round_id=row.round_id,
    )


def build_dense_teacher_snapshot(
    rows: Iterable[SolverStateTrainingExampleV1],
    teacher_traces: Iterable[Any],
    *,
    round_id: str,
    decision_budget: int,
    teacher_label_budget: int,
    acquisition_policy: AcquisitionPolicy,
    gold_floor_fraction: float = 0.5,
    seed: int = 42,
    teacher_source: str | None = None,
    align_key: str | Callable[[dict[str, Any]], Any] = "value",
    manifest: dict[str, Any] | None = None,
) -> DenseTeacherSnapshot:
    """Build a deterministic round snapshot with matched arms.

    The arms follow the SPV2-04 acceptance criteria:

    * ``gold_only`` — replay-verified gold states, no teacher labels.
    * ``mixed_no_teacher`` — the winning gold/on-policy mixture without teacher.
    * ``mixed_teacher_argmax`` — winning mixture + argmax teacher label.
    * ``mixed_teacher_kl`` — winning mixture + full-set teacher KL.
    * ``targeted_teacher_kl`` — targeted acquisition + full-set KL.
    * ``on_policy_teacher_kl`` — pure on-policy rows with teacher labels.

    Every train arm respects the gold replay floor and the same decision budget.
    """
    all_dense = attach_teacher_distribution(
        rows,
        teacher_traces,
        align_key=align_key,
        teacher_source=teacher_source,
        round_id=round_id,
    )

    # Discover held-out groups *before* dropping held-out rows so the cross-split
    # leak guard can reject any train-row group that also appears in held-out.
    held_out_groups = _held_out_group_ids(all_dense)
    train_groups = {
        r.split_group_id
        for r in all_dense
        if r.split.lower() not in _HELD_OUT_SPLITS
    }
    leaked_groups = train_groups & held_out_groups
    all_dense = [
        r
        for r in all_dense
        if r.split_group_id not in leaked_groups or r.split.lower() in _HELD_OUT_SPLITS
    ]

    all_dense = _drop_held_out(all_dense)

    gold_pool = [r for r in all_dense if r.supervision_source is SupervisionSource.GOLD]
    on_policy_pool = [
        r for r in all_dense if r.supervision_source is SupervisionSource.ON_POLICY
    ]

    rng = random.Random(seed)

    # Label budget drives how many on-policy states get teacher scores.
    labeled = acquire_teacher_labeled_states(
        on_policy_pool,
        budget=min(teacher_label_budget, decision_budget),
        policy=acquisition_policy,
        seed=seed,
    )

    # Gold floor cannot be silently lowered (recorded in lineage; mixed arms
    # refill from the gold pool, so the floor is preserved by construction).
    gold_floor_count = int(decision_budget * max(0.0, min(1.0, gold_floor_fraction)))

    def _fill_from(pool: list[DenseTeacherExampleV1], already: list[DenseTeacherExampleV1]) -> list[DenseTeacherExampleV1]:
        need = decision_budget - len(already)
        if need <= 0:
            return already
        extra = _sample(pool, need, rng)
        # Avoid exact duplicate state fingerprints within an arm.
        seen = {r.state_fingerprint for r in already}
        out = list(already)
        for r in extra:
            if r.state_fingerprint not in seen:
                out.append(r)
                seen.add(r.state_fingerprint)
        return out

    arms: dict[str, list[DenseTeacherExampleV1]] = {}

    # Gold-only control.
    arms["gold_only"] = _fill_from(gold_pool, [])

    # Winning mixture without teacher labels.
    mixed_no_teacher = _sample(gold_pool, decision_budget // 2, rng) + _sample(
        on_policy_pool, decision_budget - decision_budget // 2, rng
    )
    arms["mixed_no_teacher"] = _dedupe_by_fingerprint(mixed_no_teacher)[:decision_budget]

    # Mixed + argmax teacher.
    argmax_labeled = [_with_argmax_acceptable(r) for r in labeled]
    arms["mixed_teacher_argmax"] = _fill_from(gold_pool, argmax_labeled)[:decision_budget]

    # Mixed + full-set teacher KL.
    arms["mixed_teacher_kl"] = _fill_from(gold_pool, list(labeled))[:decision_budget]

    # Targeted acquisition (different policy) + KL.
    targeted = acquire_teacher_labeled_states(
        on_policy_pool,
        budget=min(teacher_label_budget, decision_budget),
        policy=AcquisitionPolicy.HIGH_DIVERGENCE,
        seed=seed + 1,
    )
    arms["targeted_teacher_kl"] = _fill_from(gold_pool, targeted)[:decision_budget]

    # Pure on-policy + KL, filled back with unlabeled on-policy to meet budget.
    arms["on_policy_teacher_kl"] = _fill_from(on_policy_pool, list(labeled))[:decision_budget]

    source_counts: Counter = Counter()
    for arm_rows in arms.values():
        for r in arm_rows:
            source_counts[r.supervision_source.value] += 1
            if r.teacher_probs is not None:
                source_counts["teacher_labeled"] += 1

    lineage = {
        "parent_round_id": round_id,
        "decision_budget": decision_budget,
        "teacher_label_budget": teacher_label_budget,
        "acquisition_policy": acquisition_policy.value,
        "gold_floor_fraction": gold_floor_fraction,
        "gold_pool_size": len(gold_pool),
        "on_policy_pool_size": len(on_policy_pool),
        "teacher_trace_manifest": manifest or {},
        "held_out_group_ids": sorted(_held_out_group_ids(all_dense)),
    }

    return DenseTeacherSnapshot(
        round_id=round_id,
        decision_budget=decision_budget,
        teacher_label_budget=teacher_label_budget,
        gold_floor_count=gold_floor_count,
        arms=arms,
        source_counts=source_counts,
        lineage=lineage,
    )


def _dedupe_by_fingerprint(
    rows: list[DenseTeacherExampleV1],
) -> list[DenseTeacherExampleV1]:
    seen: set[str] = set()
    out: list[DenseTeacherExampleV1] = []
    for r in rows:
        if r.state_fingerprint in seen:
            continue
        seen.add(r.state_fingerprint)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Multi-seed comparison
# --------------------------------------------------------------------------- #


def compare_dense_teacher_mixtures(
    rows: Iterable[SolverStateTrainingExampleV1],
    teacher_traces: Iterable[Any],
    *,
    seeds: Iterable[int] = (0, 1, 2),
    decision_budget: int = 64,
    teacher_label_budget: int = 32,
    acquisition_policy: AcquisitionPolicy = AcquisitionPolicy.UNIFORM,
    round_id: str = "spv2-04-round-0",
    teacher_source: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build SPV2-04 snapshots for several seeds and return a summary."""
    snapshots: list[DenseTeacherSnapshot] = []
    for seed in seeds:
        snapshot = build_dense_teacher_snapshot(
            rows,
            teacher_traces,
            round_id=f"{round_id}-seed{seed}",
            decision_budget=decision_budget,
            teacher_label_budget=teacher_label_budget,
            acquisition_policy=acquisition_policy,
            seed=seed,
            teacher_source=teacher_source,
            manifest=manifest,
        )
        snapshots.append(snapshot)

    arm_names = sorted({arm for s in snapshots for arm in s.arms})
    per_seed_counts = [s.counts() for s in snapshots]
    aggregate: dict[str, dict[str, float]] = {}
    for arm in arm_names:
        sizes = [c.get(arm, 0) for c in per_seed_counts]
        aggregate[arm] = {
            "mean": sum(sizes) / len(sizes),
            "min": min(sizes),
            "max": max(sizes),
        }

    return {
        "round_id": round_id,
        "seeds": list(seeds),
        "decision_budget": decision_budget,
        "teacher_label_budget": teacher_label_budget,
        "acquisition_policy": acquisition_policy.value,
        "snapshots": [s.to_dict() for s in snapshots],
        "aggregate_arm_sizes": aggregate,
    }


__all__ = [
    "AcquisitionPolicy",
    "DenseTeacherExampleV1",
    "DenseTeacherSnapshot",
    "acquire_teacher_labeled_states",
    "attach_teacher_distribution",
    "build_dense_teacher_snapshot",
    "compare_dense_teacher_mixtures",
    "compute_acquisition_score",
]
