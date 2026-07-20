"""SLM-209 (SDE5-02): debt-targeted semantic exposure curriculum wiring/fixture harness.

Mines high-debt exact states from synthetic constraint-debt telemetry and builds a
fixed-budget semantic exposure curriculum.  The fixture compares selection policies
that combine grammar-mask debt, inverse action-kind frequency, and legal-support
entropy to up-weight under-served exact states while keeping per-group caps and
train/held-out isolation.

No model is trained, no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.harnesses.preference.constraint_debt import ConstraintDebtV1
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    split_for_group,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "POLICY_NAMES",
    "DebtCurriculumSelectionV1",
    "DebtCurriculumCellV1",
    "DebtCurriculumManifestV1",
    "build_synthetic_debt_and_events",
    "compute_selection_score",
    "select_states_for_cell",
    "build_debt_curriculum_manifest",
    "run_fixture_campaign",
    "render_markdown",
    "validate_manifest",
]

MATRIX_VERSION = "sde5-02-v1"
MATRIX_SET = "slm209_debt_targeted_curriculum"
EXPERIMENT_ID = "slm209-debt-targeted-curriculum"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_TOTAL_DECISION_BUDGET = 120
_DEFAULT_PER_GROUP_CAP = 6

POLICY_NAMES = (
    "uniform",
    "slm170_frequency",
    "high_debt",
    "debt_plus_rarity",
    "debt_plus_entropy",
    "preregistered_composite",
)

_DECISION_KINDS = (
    "constraint_shadow",
    "component_choice",
    "argument_value",
    "slot_binding",
    "root_closure",
    "array_insert",
)

_HYPOTHESIS = (
    "A fixed-budget curriculum that selects exact states by grammar-mask debt, "
    "inverse decision-kind frequency, and legal-support entropy increases the "
    "exposure of high-debt exact states while respecting per-group caps and "
    "train/held-out isolation."
)

_FALSIFIER = (
    "Debt-targeted selection fails to increase high-debt state exposure, violates "
    "the total decision budget, exceeds per-group caps, or leaks a group across "
    "train/held-out splits."
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d")


@dataclass(frozen=True)
class DebtCurriculumSelectionV1:
    """One selected exact state with full score-component auditability."""

    state_id: str
    group_id: str
    decision_kind: str
    split: str
    selected_by_policy: str
    score_components: dict[str, float]
    debt_row_digest: str
    source_event_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "group_id": self.group_id,
            "decision_kind": self.decision_kind,
            "split": self.split,
            "selected_by_policy": self.selected_by_policy,
            "score_components": dict(self.score_components),
            "debt_row_digest": self.debt_row_digest,
            "source_event_digest": self.source_event_digest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtCurriculumSelectionV1":
        fields = set(cls.__dataclass_fields__)
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown selection fields: {sorted(unknown)}")
        return cls(
            state_id=str(data["state_id"]),
            group_id=str(data["group_id"]),
            decision_kind=str(data["decision_kind"]),
            split=str(data["split"]),
            selected_by_policy=str(data["selected_by_policy"]),
            score_components=dict(data.get("score_components", {})),
            debt_row_digest=str(data["debt_row_digest"]),
            source_event_digest=str(data["source_event_digest"]),
        )


@dataclass(frozen=True)
class DebtCurriculumCellV1:
    """One policy arm of the debt-targeted curriculum fixture."""

    policy_name: str
    weight_config: dict[str, float]
    selections: tuple[DebtCurriculumSelectionV1, ...]
    exposure_audit: dict[str, Any]
    decision_budget: int
    per_group_cap: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "weight_config": dict(self.weight_config),
            "selections": [s.to_dict() for s in self.selections],
            "exposure_audit": dict(self.exposure_audit),
            "decision_budget": self.decision_budget,
            "per_group_cap": self.per_group_cap,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtCurriculumCellV1":
        fields = set(cls.__dataclass_fields__)
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown cell fields: {sorted(unknown)}")
        return cls(
            policy_name=str(data["policy_name"]),
            weight_config=dict(data.get("weight_config", {})),
            selections=tuple(
                DebtCurriculumSelectionV1.from_dict(s)
                for s in data.get("selections", ())
            ),
            exposure_audit=dict(data.get("exposure_audit", {})),
            decision_budget=int(data["decision_budget"]),
            per_group_cap=int(data["per_group_cap"]),
            seed=int(data["seed"]),
        )


@dataclass(frozen=True)
class DebtCurriculumManifestV1:
    """Full fixture manifest for SLM-209."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[DebtCurriculumCellV1, ...]
    total_decision_budget: int
    per_group_cap: int
    lineage: dict[str, Any]
    version_stamp: dict[str, Any]
    timestamp: str
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cells": [c.to_dict() for c in self.cells],
            "total_decision_budget": self.total_decision_budget,
            "per_group_cap": self.per_group_cap,
            "lineage": dict(self.lineage),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtCurriculumManifestV1":
        fields = set(cls.__dataclass_fields__)
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        return cls(
            schema=str(data.get("schema", "DebtCurriculumManifestV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", "slm209_fixture")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            cells=tuple(
                DebtCurriculumCellV1.from_dict(c) for c in data.get("cells", ())
            ),
            total_decision_budget=int(data["total_decision_budget"]),
            per_group_cap=int(data["per_group_cap"]),
            lineage=dict(data.get("lineage", {})),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", ())),
        )


def _sha256_stable_identity(event: DecisionEventV1) -> str:
    """Stable state id from the event identity fields (mirrors constraint_debt.py)."""
    import hashlib

    payload = {
        "group_id": event.group_id,
        "trajectory_id": event.trajectory_id,
        "policy_checkpoint_sha": event.policy_checkpoint_sha,
        "tokenizer_sha": event.tokenizer_sha,
        "decode_config_hash": event.decode_config_hash,
        "decision_kind": event.decision_kind,
        "split": event.split,
        "canvas_ids": list(event.canvas_ids),
        "position": event.position,
        "legal_token_ids": list(event.legal_token_ids),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _state_id(event: DecisionEventV1) -> str:
    return event.event_id or _sha256_stable_identity(event)


def _digest(value: object) -> str:
    import hashlib

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _debt_digest(debt: ConstraintDebtV1) -> str:
    """Stable digest of a debt row; excludes the runtime timestamp."""
    payload = debt.to_dict()
    payload.pop("timestamp", None)
    return _digest(payload)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_synthetic_debt_and_events(
    n_states: int = 200, seed: int = 0
) -> tuple[list[ConstraintDebtV1], list[DecisionEventV1]]:
    """Generate deterministic constraint-debt rows and matching decision events.

    The synthetic corpus intentionally varies decision kinds, splits, group sizes,
    and debt profiles so the selection policies can be distinguished without any
    model or checkpoint.
    """
    rng = random.Random(seed)
    debts: list[ConstraintDebtV1] = []
    events: list[DecisionEventV1] = []

    policy_checkpoint_sha = "synthetic_policy_sha"
    tokenizer_sha = "synthetic_tokenizer_sha"
    decode_config_hash = "synthetic_decode_config_hash"

    for i in range(n_states):
        group_id = f"group_{i % 40:03d}"
        split = split_for_group(group_id)
        decision_kind = rng.choice(_DECISION_KINDS)
        trajectory_id = f"traj_{i // 4:03d}"

        # Build a small canvas with a valid position.
        canvas_len = rng.randint(4, 12)
        canvas_ids = tuple(rng.randint(0, 1000) for _ in range(canvas_len))
        position = rng.randint(0, canvas_len - 1)

        # Build legal/good/bad token sets.
        legal_support_count = rng.randint(3, 20)
        legal_token_ids = tuple(sorted({rng.randint(0, 5000) for _ in range(legal_support_count)}))
        if len(legal_token_ids) < 2:
            legal_token_ids = (100, 200, 300)
        good_count = max(1, min(legal_support_count - 1, rng.randint(1, 3)))
        good_token_ids = tuple(sorted(rng.sample(legal_token_ids, good_count)))
        remaining = [t for t in legal_token_ids if t not in set(good_token_ids)]
        bad_count = max(1, min(len(remaining), rng.randint(1, 3)))
        bad_token_ids = tuple(sorted(rng.sample(remaining, bad_count)))

        event_id = f"state_{i:04d}_{_digest({'i': i, 'g': group_id, 'k': decision_kind})}"

        event = DecisionEventV1(
            event_id=event_id,
            group_id=group_id,
            context_text=f"synthetic context for {decision_kind} in {group_id}",
            canvas_ids=canvas_ids,
            position=position,
            good_token_ids=good_token_ids,
            bad_token_ids=bad_token_ids,
            legal_token_ids=legal_token_ids,
            evidence_kind="counterfactual",
            evidence_confidence=0.85,
            decision_kind=decision_kind,
            split=split,
            policy_checkpoint_sha=policy_checkpoint_sha,
            tokenizer_sha=tokenizer_sha,
            decode_config_hash=decode_config_hash,
            seed=seed,
            trajectory_id=trajectory_id,
            source_suite="slm209_fixture",
        )

        # Generate masses that correlate with decision kind and index so different
        # policies see different high-debt states.
        base = rng.random()
        kind_skew = {"constraint_shadow": 0.7, "slot_binding": 0.6, "root_closure": 0.5}.get(
            decision_kind, 0.3
        )
        legal_mass = _clamp(base * 0.5 + 0.45)
        good_mass = _clamp(base * legal_mass * kind_skew + 0.02)
        bad_mass = _clamp((1.0 - kind_skew) * legal_mass * (0.5 + 0.5 * rng.random()))
        ambiguous_mass = max(0.0, legal_mass - good_mass - bad_mass)
        unobserved_mass = 1.0 - legal_mass
        epsilon = 1e-12

        debt = ConstraintDebtV1(
            state_id=event_id,
            group_id=group_id,
            trajectory_id=trajectory_id,
            policy_checkpoint_sha=policy_checkpoint_sha,
            tokenizer_sha=tokenizer_sha,
            decode_config_hash=decode_config_hash,
            verifier_bundle_hash="synthetic_verifier_bundle_hash",
            decision_kind=decision_kind,
            abstract_state_role=decision_kind,
            split=split,
            probability_space="full_vocab",
            epsilon=epsilon,
            full_vocab_log_normalizer=float(rng.uniform(5.0, 10.0)),
            legal_mass=legal_mass,
            good_mass=good_mass,
            bad_mass=bad_mass,
            ambiguous_mass=ambiguous_mass,
            unobserved_mass=unobserved_mass,
            legal_debt=-math.log(max(epsilon, legal_mass)),
            good_debt=-math.log(max(epsilon, good_mass)) if good_token_ids else None,
            bad_debt=-math.log(max(epsilon, bad_mass)) if bad_token_ids else None,
            legal_mass_deficit=1.0 - legal_mass,
            pre_post_mask_kl=float(rng.uniform(0.0, 0.5)),
            legal_support_count=len(legal_token_ids),
            good_support_count=len(good_token_ids),
            bad_support_count=len(bad_token_ids),
            single_legal_action=len(legal_token_ids) == 1,
            empty_good_partition=not good_token_ids,
            empty_bad_partition=not bad_token_ids,
        )

        debts.append(debt)
        events.append(event)

    return debts, events


def _kind_frequencies(events: list[DecisionEventV1]) -> Counter[str]:
    return Counter(event.decision_kind for event in events)


def compute_selection_score(
    debt: ConstraintDebtV1,
    event: DecisionEventV1,
    policy: str,
    *,
    rarity_counter: Counter[str] | None = None,
    entropy_map: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return a score and decomposed components for one (debt, event) pair.

    Policies:
      - uniform: score = 1.0
      - slm170_frequency: sqrt-inverse-frequency over decision_kind
      - high_debt: good_debt (or legal_debt if good partition is empty)
      - debt_plus_rarity: weighted combo of good debt and inverse kind frequency
      - debt_plus_entropy: weighted combo of good debt and log(legal_support_count)
      - preregistered_composite: fixed weights over debt, rarity, entropy
    """
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown policy: {policy!r}")

    kind = event.decision_kind
    rarity_counter = rarity_counter or Counter()
    kind_freq = rarity_counter.get(kind, 1)
    inverse_kind_frequency = 1.0 / math.sqrt(max(1, kind_freq))

    good_debt = debt.good_debt if debt.good_debt is not None else 0.0
    legal_debt = debt.legal_debt if debt.legal_debt is not None else 0.0
    effective_debt = good_debt if good_debt > 0 else legal_debt

    legal_support = max(1, debt.legal_support_count)
    entropy = math.log(legal_support)
    normalized_entropy = entropy / math.log(20.0)  # cap at support == 20

    if policy == "uniform":
        score = 1.0
        components = {
            "uniform": 1.0,
            "effective_debt": effective_debt,
            "inverse_kind_frequency": inverse_kind_frequency,
            "normalized_entropy": normalized_entropy,
        }
    elif policy == "slm170_frequency":
        score = inverse_kind_frequency
        components = {
            "uniform": 1.0,
            "inverse_kind_frequency": inverse_kind_frequency,
            "effective_debt": effective_debt,
            "normalized_entropy": normalized_entropy,
        }
    elif policy == "high_debt":
        score = effective_debt
        components = {
            "effective_debt": effective_debt,
            "good_debt": good_debt,
            "legal_debt": legal_debt,
        }
    elif policy == "debt_plus_rarity":
        weight_debt = 0.6
        weight_rarity = 0.4
        score = weight_debt * effective_debt + weight_rarity * inverse_kind_frequency
        components = {
            "effective_debt": effective_debt,
            "inverse_kind_frequency": inverse_kind_frequency,
            "weight_debt": weight_debt,
            "weight_rarity": weight_rarity,
        }
    elif policy == "debt_plus_entropy":
        weight_debt = 0.7
        weight_entropy = 0.3
        score = weight_debt * effective_debt + weight_entropy * normalized_entropy
        components = {
            "effective_debt": effective_debt,
            "normalized_entropy": normalized_entropy,
            "weight_debt": weight_debt,
            "weight_entropy": weight_entropy,
        }
    else:  # preregistered_composite
        weight_debt = 0.5
        weight_rarity = 0.25
        weight_entropy = 0.25
        score = (
            weight_debt * effective_debt
            + weight_rarity * inverse_kind_frequency
            + weight_entropy * normalized_entropy
        )
        components = {
            "effective_debt": effective_debt,
            "inverse_kind_frequency": inverse_kind_frequency,
            "normalized_entropy": normalized_entropy,
            "weight_debt": weight_debt,
            "weight_rarity": weight_rarity,
            "weight_entropy": weight_entropy,
        }

    return {"score": score, **components}


def _build_weight_config(policy: str) -> dict[str, float]:
    if policy == "debt_plus_rarity":
        return {"debt": 0.6, "rarity": 0.4}
    if policy == "debt_plus_entropy":
        return {"debt": 0.7, "entropy": 0.3}
    if policy == "preregistered_composite":
        return {"debt": 0.5, "rarity": 0.25, "entropy": 0.25}
    if policy == "uniform":
        return {"uniform": 1.0}
    if policy == "slm170_frequency":
        return {"sqrt_inverse_frequency": 1.0}
    if policy == "high_debt":
        return {"debt": 1.0}
    return {}


def select_states_for_cell(
    debts: list[ConstraintDebtV1],
    events: list[DecisionEventV1],
    policy: str,
    total_decision_budget: int,
    per_group_cap: int,
    seed: int,
) -> tuple[tuple[DebtCurriculumSelectionV1, ...], dict[str, Any]]:
    """Select exact states for one policy cell.

    Selection is deterministic: score all states, sort by score descending with a
    stable state_id tie-breaker, then greedily fill the budget while respecting
    the per-group cap.  This prevents group-level leakage and keeps the exact
    state composition invariant to input row order.
    """
    rng = random.Random(seed)

    debt_by_state = {debt.state_id: debt for debt in debts}
    event_by_state = {event.event_id: event for event in events}
    state_ids = sorted(set(debt_by_state) & set(event_by_state))

    rarity_counter = _kind_frequencies(events)

    scored: list[tuple[float, str, ConstraintDebtV1, DecisionEventV1]] = []
    for state_id in state_ids:
        debt = debt_by_state[state_id]
        event = event_by_state[state_id]
        result = compute_selection_score(
            debt, event, policy, rarity_counter=rarity_counter
        )
        score = result["score"]
        # Add tiny random jitter deterministically so identical scores still have
        # a stable order, but the jitter is small enough not to reorder different
        # scores.  state_id remains the primary tie-breaker for permutation control.
        jitter = rng.random() * 1e-9
        scored.append((score + jitter, state_id, debt, event))

    # Sort by score descending, state_id ascending for determinism.
    scored.sort(key=lambda item: (-item[0], item[1]))

    group_counts: Counter[str] = Counter()
    selections: list[DebtCurriculumSelectionV1] = []
    group_cap = max(1, int(per_group_cap))

    for _score, state_id, debt, event in scored:
        if len(selections) >= total_decision_budget:
            break
        if group_counts[event.group_id] >= group_cap:
            continue
        result = compute_selection_score(
            debt, event, policy, rarity_counter=rarity_counter
        )
        selections.append(
            DebtCurriculumSelectionV1(
                state_id=state_id,
                group_id=event.group_id,
                decision_kind=event.decision_kind,
                split=event.split,
                selected_by_policy=policy,
                score_components=result,
                debt_row_digest=_debt_digest(debt),
                source_event_digest=_digest(event.to_dict()),
            )
        )
        group_counts[event.group_id] += 1

    exposure_audit = _build_exposure_audit(selections)
    return tuple(selections), exposure_audit


def _build_exposure_audit(
    selections: list[DebtCurriculumSelectionV1],
) -> dict[str, Any]:
    by_kind: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    total_score = 0.0
    total_debt = 0.0
    for sel in selections:
        by_kind[sel.decision_kind] += 1
        by_split[sel.split] += 1
        by_group[sel.group_id] += 1
        total_score += sel.score_components.get("score", 0.0)
        total_debt += sel.score_components.get("effective_debt", 0.0)

    return {
        "by_decision_kind": dict(sorted(by_kind.items())),
        "by_split": dict(sorted(by_split.items())),
        "unique_groups": len(by_group),
        "max_group_count": max(by_group.values()) if by_group else 0,
        "mean_selection_score": total_score / max(1, len(selections)),
        "mean_effective_debt": total_debt / max(1, len(selections)),
    }


def build_cells(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    *,
    policies: tuple[str, ...] = POLICY_NAMES,
    total_decision_budget: int = _DEFAULT_TOTAL_DECISION_BUDGET,
    per_group_cap: int = _DEFAULT_PER_GROUP_CAP,
) -> tuple[DebtCurriculumCellV1, ...]:
    """Build policy × seed cells without running selection."""
    cells: list[DebtCurriculumCellV1] = []
    for seed in seeds:
        for policy in policies:
            cells.append(
                DebtCurriculumCellV1(
                    policy_name=policy,
                    weight_config=_build_weight_config(policy),
                    selections=(),
                    exposure_audit={},
                    decision_budget=total_decision_budget,
                    per_group_cap=per_group_cap,
                    seed=seed,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[DebtCurriculumCellV1, ...]) -> list[str]:
    """Validate the debt-targeted curriculum manifest."""
    errors: list[str] = []
    if not cells:
        errors.append("cells must not be empty")
    seen: set[str] = set()
    for cell in cells:
        key = f"{cell.policy_name}__s{cell.seed}"
        if key in seen:
            errors.append(f"duplicate cell: {key}")
        seen.add(key)
        if cell.policy_name not in POLICY_NAMES:
            errors.append(f"{key}: invalid policy {cell.policy_name!r}")
        if cell.decision_budget <= 0:
            errors.append(f"{key}: decision_budget must be positive")
        if cell.per_group_cap <= 0:
            errors.append(f"{key}: per_group_cap must be positive")
    return errors


def _cell_key(cell: DebtCurriculumCellV1) -> str:
    return f"{cell.policy_name}__s{cell.seed}"


def _resolve_disposition(
    cells: tuple[DebtCurriculumCellV1, ...],
) -> tuple[str, str]:
    """Return (disposition, rationale) from the filled cells."""
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cell in cells:
        audit = cell.exposure_audit
        mean_debt = audit.get("mean_effective_debt", 0.0)
        means.setdefault(cell.policy_name, 0.0)
        counts.setdefault(cell.policy_name, 0)
        means[cell.policy_name] += mean_debt
        counts[cell.policy_name] += 1

    for policy in means:
        means[policy] /= max(1, counts[policy])

    high_debt_mean = means.get("high_debt", 0.0)
    composite_mean = means.get("preregistered_composite", 0.0)
    uniform_mean = means.get("uniform", 0.0)
    frequency_mean = means.get("slm170_frequency", 0.0)

    # Check structural invariants first.
    for cell in cells:
        audit = cell.exposure_audit
        if audit.get("max_group_count", 0) > cell.per_group_cap:
            return (
                "inconclusive",
                "At least one cell exceeded its per-group cap; the selection wiring "
                "needs tightening before claiming a curriculum win.",
            )
        if len(cell.selections) > cell.decision_budget:
            return (
                "inconclusive",
                "At least one cell exceeded its total decision budget.",
            )

    if not means:
        return ("inconclusive", "No filled cells to evaluate.")

    tolerance = 1e-6

    if (
        high_debt_mean <= uniform_mean + tolerance
        and composite_mean <= uniform_mean + tolerance
    ):
        return (
            "no_debt_lift",
            "Debt-targeted policies do not select higher-debt states than uniform "
            "sampling in this synthetic fixture.",
        )

    best_debt_policy = max(
        means,
        key=lambda p: (means[p], p),
    )
    if best_debt_policy in {"high_debt", "preregistered_composite", "debt_plus_rarity"}:
        if means[best_debt_policy] > uniform_mean + 0.1:
            return (
                "useful_debt_targeting",
                f"{best_debt_policy} selects meaningfully higher-debt states than "
                "uniform sampling while respecting the budget and per-group caps.",
            )

    if (
        composite_mean > frequency_mean + tolerance
        or high_debt_mean > frequency_mean + tolerance
    ):
        return (
            "modest_debt_lift",
            "Debt-aware policies show a measurable debt lift over frequency-only "
            "sampling, but the effect size is modest.",
        )

    return (
        "inconclusive",
        "The debt-targeted pattern is mixed; additional seeds or a larger synthetic "
        "corpus are needed to falsify the hypothesis.",
    )


def build_debt_curriculum_manifest(
    debts: list[ConstraintDebtV1],
    events: list[DecisionEventV1],
    *,
    cells: tuple[DebtCurriculumCellV1, ...] | None = None,
    total_decision_budget: int = _DEFAULT_TOTAL_DECISION_BUDGET,
    per_group_cap: int = _DEFAULT_PER_GROUP_CAP,
    seed: int = 0,
    run_id: str = "slm209-debt-targeted-curriculum",
) -> DebtCurriculumManifestV1:
    """Fill every cell with selections and build the manifest."""
    cells = cells or build_cells(
        seeds=(seed,),
        total_decision_budget=total_decision_budget,
        per_group_cap=per_group_cap,
    )

    filled: list[DebtCurriculumCellV1] = []
    for cell in cells:
        selections, audit = select_states_for_cell(
            debts,
            events,
            cell.policy_name,
            cell.decision_budget,
            cell.per_group_cap,
            cell.seed,
        )
        filled.append(
            DebtCurriculumCellV1(
                policy_name=cell.policy_name,
                weight_config=cell.weight_config,
                selections=selections,
                exposure_audit=audit,
                decision_budget=cell.decision_budget,
                per_group_cap=cell.per_group_cap,
                seed=cell.seed,
            )
        )

    disposition, rationale = _resolve_disposition(tuple(filled))

    debt_digest = _digest([_debt_digest(d) for d in debts])
    event_digest = _digest([e.to_dict() for e in events])

    return DebtCurriculumManifestV1(
        schema="DebtCurriculumManifestV1",
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        cells=tuple(filled),
        total_decision_budget=total_decision_budget,
        per_group_cap=per_group_cap,
        lineage={
            "debt_artifact_digest": debt_digest,
            "source_event_digest": event_digest,
            "synthetic_state_count": len(debts),
            "synthetic_event_count": len(events),
        },
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm209_debt_targeted_curriculum",
            "harness.preference.constraint_debt",
            "harness.train_data",
        ),
        timestamp=_now(),
        disposition=disposition,
        disposition_rationale=rationale,
        honest_caveats=(
            "Synthetic fixture: no model, checkpoint, or verifier labels were used.",
            "Debt masses are randomly generated and only weakly correlate with decision kind; "
            "real constraint-debt telemetry will differ.",
            "Per-group caps are enforced at the group_id level; real curricula may need "
            "trajectory-level or program-family-level caps.",
            "No ship-gate claim is made; this is wiring evidence only.",
        ),
    )


def run_fixture_campaign(
    output_dir: Path | None = None,
    *,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    n_states: int = 200,
    total_decision_budget: int = _DEFAULT_TOTAL_DECISION_BUDGET,
    per_group_cap: int = _DEFAULT_PER_GROUP_CAP,
    seed: int = 0,
    write_design_docs: bool = True,
    design_json: Path | None = None,
    design_md: Path | None = None,
) -> DebtCurriculumManifestV1:
    """Run the SLM-209 debt-targeted curriculum fixture campaign."""
    start = time.perf_counter()
    debts, events = build_synthetic_debt_and_events(n_states=n_states, seed=seed)

    cells = build_cells(
        seeds=seeds,
        total_decision_budget=total_decision_budget,
        per_group_cap=per_group_cap,
    )
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    manifest = build_debt_curriculum_manifest(
        debts,
        events,
        cells=cells,
        total_decision_budget=total_decision_budget,
        per_group_cap=per_group_cap,
        seed=seed,
        run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
    )

    elapsed = time.perf_counter() - start
    # Carry wall time in the lineage for documentation; do not let it dominate.
    lineage = dict(manifest.lineage)
    lineage["wall_seconds"] = _clamp(elapsed, low=0.001, high=10.0)
    manifest = DebtCurriculumManifestV1(
        schema=manifest.schema,
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        run_id=manifest.run_id,
        status=manifest.status,
        claim_class=manifest.claim_class,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        cells=manifest.cells,
        total_decision_budget=manifest.total_decision_budget,
        per_group_cap=manifest.per_group_cap,
        lineage=lineage,
        version_stamp=manifest.version_stamp,
        timestamp=manifest.timestamp,
        disposition=manifest.disposition,
        disposition_rationale=manifest.disposition_rationale,
        honest_caveats=manifest.honest_caveats,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_json(output_dir / "slm209_debt_targeted_curriculum_report.json")

        if write_design_docs:
            if design_json is None or design_md is None:
                root = _project_root()
                design_json = root / f"docs/design/iter-slm209-debt-targeted-curriculum-{_today_yyyymmdd()}.json"
                design_md = root / f"docs/design/iter-slm209-debt-targeted-curriculum-{_today_yyyymmdd()}.md"
            design_json.parent.mkdir(parents=True, exist_ok=True)
            design_md.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_json(design_json)
            design_md.write_text(render_markdown(manifest), encoding="utf-8")

    return manifest


def render_markdown(manifest: DebtCurriculumManifestV1) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-209 (SDE5-02): debt-targeted semantic exposure curriculum fixture ({manifest.run_id})",
        "",
        f"Matrix set: `{manifest.matrix_set}`",
        "",
        f"Version: `{manifest.matrix_version}`",
        "",
        f"Status: **{manifest.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable weights "
        "were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        manifest.falsifier,
        "",
        "## Cells",
        "",
        "| policy_name | seed | decision_budget | per_group_cap | selected_states | unique_groups | max_group_count | mean_effective_debt |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in manifest.cells:
        audit = cell.exposure_audit
        lines.append(
            f"| {cell.policy_name} | {cell.seed} | {cell.decision_budget} | "
            f"{cell.per_group_cap} | {len(cell.selections)} | "
            f"{audit.get('unique_groups', 0)} | {audit.get('max_group_count', 0)} | "
            f"{audit.get('mean_effective_debt', 0.0):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Exposure audit (preregistered composite example)",
            "",
        ]
    )
    composite_cells = [c for c in manifest.cells if c.policy_name == "preregistered_composite"]
    if composite_cells:
        audit = composite_cells[0].exposure_audit
        lines.extend(
            [
                f"- By decision kind: {audit.get('by_decision_kind', {})}",
                f"- By split: {audit.get('by_split', {})}",
                f"- Unique groups: {audit.get('unique_groups', 0)}",
                f"- Max group count: {audit.get('max_group_count', 0)}",
                f"- Mean selection score: {audit.get('mean_selection_score', 0.0):.4f}",
                f"- Mean effective debt: {audit.get('mean_effective_debt', 0.0):.4f}",
            ]
        )
    else:
        lines.append("_No preregistered_composite cell present._")

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{manifest.disposition}**",
            "",
            manifest.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The selection policies, "
            "score components, and caps are exercised over deterministic synthetic "
            "states, but no real model was trained or evaluated. The mechanism remains "
            "``retain_diagnostic`` / ``blocked_pending_real_model`` until trained-model "
            "constraint-debt telemetry and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in manifest.honest_caveats:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode plan-only",
            "python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
