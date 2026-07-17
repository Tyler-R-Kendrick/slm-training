"""DecisionEventV2: a state / action-evidence / objective-view model.

Extends (does not replace) :mod:`slm_training.harnesses.preference.local_decisions`.
See ``docs/design/local-decision-interventions.md``. The V2 contract separates
three concerns that ``DecisionEventV1`` conflated, which is the fix for the E284
blocker (*stable grammar-state support does not imply objective/action-partition
support*):

* :class:`DecisionStateV2` — the exact, replayable model decision state plus
  immutable runtime identities. Its ``state_id`` is a canonical hash of state and
  identity **only** — never of sampled good/bad labels, rollout outcomes, or
  candidate order — so two independent action-label samples of one exact state
  produce the *same* state id and merge into one state row.
* :class:`ActionOutcomeV2` — append-only, per-candidate verifier evidence. The
  complete ordered G0-G12 gate vector is preserved verbatim (never set-collapsed).
* :class:`ObjectiveView` — a pure, versioned materialization of one state's action
  table into a trainable objective. A ``constraint_shadow`` view is explicitly
  ``trainable=False`` so semantic trainers cannot consume legality-only evidence.

V1 corpora remain readable; :func:`migrate_v1_event` performs a one-way, lossless,
evidence-honest migration (it marks partial evidence incomplete and never
fabricates rollout/verifier outcomes). This module writes no model, checkpoint, or
training result.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.data.verify import Gate
from slm_training.harnesses.preference.counterfactuals import (
    SEMANTIC_VERIFIER_V1,
    _METRICS,
)
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    Split,
    split_for_group,
)
from slm_training.lineage.records import content_sha

SCHEMA_VERSION = 2
Architecture = Literal["twotower", "causal"]
GateStatusStr = Literal["pass", "fail", "skip"]

# The complete ordered G0-G12 gate identity a verifier vector must carry.
GATE_ORDER: tuple[str, ...] = tuple(gate.value for gate in Gate)
_GATE_STATUSES = frozenset({"pass", "fail", "skip"})
_CONSTRAINT_SHADOW_VERIFIER = "constraint_shadow_legality_only"

__all__ = [
    "SCHEMA_VERSION",
    "GATE_ORDER",
    "ActionOutcomeV2",
    "DecisionStateV2",
    "ObjectiveView",
    "append_action_outcomes",
    "materialize_constraint_shadow",
    "materialize_pareto",
    "migrate_v1_event",
    "verifier_bundle_hash",
]


def _ids(values: Iterable[int]) -> tuple[int, ...]:
    """Sorted, deduplicated int tuple (matches the V1 legal/candidate convention)."""
    return tuple(sorted({int(value) for value in values}))


def _int_tuple(values: Iterable[int]) -> tuple[int, ...]:
    """Order-preserving int tuple (for a causal prefix, where order is identity)."""
    return tuple(int(value) for value in values)


def verifier_bundle_hash(name: str, metrics: Sequence[str]) -> str:
    """Stable identity for a verifier bundle (name + ordered metric list)."""
    return content_sha({"name": str(name), "metrics": [str(metric) for metric in metrics]})


# --------------------------------------------------------------------------- #
# Decision state.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecisionStateV2:
    """One exact, replayable model decision state.

    ``state_id`` is derived on construction from the state and identity fields
    only; it is deliberately independent of any action evidence.
    """

    group_id: str
    architecture: Architecture
    context_text: str
    decision_position: int
    legal_action_ids: tuple[int, ...]
    decision_kind: str
    abstract_state_role: str
    grammar_state_hash: str
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    verifier_bundle_hash: str
    split: Split
    context_ids: tuple[int, ...] | None = None
    canvas_ids: tuple[int, ...] | None = None
    generation_step: int | None = None
    version: int = SCHEMA_VERSION
    state_id: str = ""

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError("unsupported decision state version")
        if self.architecture not in ("twotower", "causal"):
            raise ValueError(f"unknown architecture {self.architecture!r}")
        object.__setattr__(self, "legal_action_ids", _ids(self.legal_action_ids))
        if not self.legal_action_ids:
            raise ValueError("legal_action_ids must be non-empty")
        if self.canvas_ids is not None:
            object.__setattr__(self, "canvas_ids", _int_tuple(self.canvas_ids))
        if self.context_ids is not None:
            object.__setattr__(self, "context_ids", _int_tuple(self.context_ids))
        if not self.group_id or not self.context_text:
            raise ValueError("group_id and context_text are required")
        if self.architecture == "twotower":
            if not self.canvas_ids:
                raise ValueError("twotower state requires canvas_ids")
            if not 0 <= self.decision_position < len(self.canvas_ids):
                raise ValueError("decision_position must address canvas_ids")
        else:  # causal
            if self.context_ids is None:
                raise ValueError("causal state requires context_ids (the prefix)")
            if self.decision_position < 0:
                raise ValueError("decision_position must be non-negative")
        if self.generation_step is not None and self.generation_step < 0:
            raise ValueError("generation_step must be non-negative")
        for name in (
            "grammar_state_hash",
            "policy_checkpoint_sha",
            "tokenizer_sha",
            "decode_config_hash",
            "verifier_bundle_hash",
            "abstract_state_role",
            "decision_kind",
        ):
            if not getattr(self, name):
                raise ValueError(f"state identity field {name!r} must be non-empty")
        if self.split != split_for_group(self.group_id):
            raise ValueError("split must be derived from group_id")
        object.__setattr__(self, "state_id", self._compute_state_id())

    def _compute_state_id(self) -> str:
        # Exact model state + immutable runtime identity only. No labels, no
        # rollout outcomes, no candidate order -> reordering/augmenting action
        # evidence never changes the state id.
        return content_sha(
            {
                "group_id": self.group_id,
                "architecture": self.architecture,
                "context_text": self.context_text,
                "context_ids": list(self.context_ids) if self.context_ids is not None else None,
                "canvas_ids": list(self.canvas_ids) if self.canvas_ids is not None else None,
                "decision_position": self.decision_position,
                "generation_step": self.generation_step,
                "legal_action_ids": list(self.legal_action_ids),
                "decision_kind": self.decision_kind,
                "abstract_state_role": self.abstract_state_role,
                "grammar_state_hash": self.grammar_state_hash,
                "policy_checkpoint_sha": self.policy_checkpoint_sha,
                "tokenizer_sha": self.tokenizer_sha,
                "decode_config_hash": self.decode_config_hash,
                "verifier_bundle_hash": self.verifier_bundle_hash,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in ("legal_action_ids", "context_ids", "canvas_ids"):
            value = data[name]
            data[name] = list(value) if value is not None else None
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionStateV2:
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown decision state fields: {sorted(unknown)}")
        payload = {key: value[key] for key in value if key != "state_id"}
        state = cls(**payload)
        recorded = value.get("state_id")
        if recorded and recorded != state.state_id:
            raise ValueError("decision state_id does not match its state (tampered)")
        return state


# --------------------------------------------------------------------------- #
# Action evidence.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActionOutcomeV2:
    """Append-only verifier evidence for one legal candidate at one state.

    ``verifier_vectors`` is a tuple of complete ordered G0-G12 vectors (one per
    rollout); each inner vector is a tuple of ``(gate, status)`` pairs in G0..G12
    order. It may be empty when evidence is incomplete (e.g. migrated from V1),
    but any present vector must carry the full ordered gate set.
    """

    state_id: str
    action_id: int
    legal: bool
    rollout_policy_sha: str
    continuation_seeds: tuple[int, ...] = ()
    outcome_hashes: tuple[str, ...] = ()
    verifier_vectors: tuple[tuple[tuple[str, str], ...], ...] = ()
    reward_vectors: tuple[tuple[tuple[str, float], ...], ...] = ()
    mean_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence_confidence: float = 0.0
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError("unsupported action outcome version")
        if not self.state_id:
            raise ValueError("action outcome requires a state_id")
        if not self.rollout_policy_sha:
            raise ValueError("action outcome requires a rollout_policy_sha")
        object.__setattr__(self, "action_id", int(self.action_id))
        object.__setattr__(self, "continuation_seeds", tuple(int(s) for s in self.continuation_seeds))
        object.__setattr__(self, "outcome_hashes", tuple(str(h) for h in self.outcome_hashes))
        object.__setattr__(
            self, "verifier_vectors", tuple(_normalize_gate_vector(v) for v in self.verifier_vectors)
        )
        object.__setattr__(
            self, "reward_vectors", tuple(_normalize_reward_vector(v) for v in self.reward_vectors)
        )
        object.__setattr__(self, "evidence_ids", tuple(str(e) for e in self.evidence_ids))
        if not 0.0 <= float(self.evidence_confidence) <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if low > high:
                raise ValueError("confidence_interval must be ordered (low <= high)")
            object.__setattr__(self, "confidence_interval", (float(low), float(high)))
        if self.mean_value is not None:
            object.__setattr__(self, "mean_value", float(self.mean_value))

    @property
    def content_key(self) -> str:
        """Content identity used to deduplicate append-only evidence (order-free)."""
        return content_sha(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "action_id": self.action_id,
            "legal": self.legal,
            "rollout_policy_sha": self.rollout_policy_sha,
            "continuation_seeds": list(self.continuation_seeds),
            "outcome_hashes": list(self.outcome_hashes),
            "verifier_vectors": [[list(pair) for pair in vector] for vector in self.verifier_vectors],
            "reward_vectors": [[list(pair) for pair in vector] for vector in self.reward_vectors],
            "mean_value": self.mean_value,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "evidence_ids": list(self.evidence_ids),
            "evidence_confidence": self.evidence_confidence,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionOutcomeV2:
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown action outcome fields: {sorted(unknown)}")
        ci = value.get("confidence_interval")
        return cls(
            state_id=str(value["state_id"]),
            action_id=int(value["action_id"]),
            legal=bool(value["legal"]),
            rollout_policy_sha=str(value["rollout_policy_sha"]),
            continuation_seeds=tuple(value.get("continuation_seeds") or ()),
            outcome_hashes=tuple(value.get("outcome_hashes") or ()),
            verifier_vectors=tuple(
                tuple(tuple(pair) for pair in vector) for vector in value.get("verifier_vectors") or ()
            ),
            reward_vectors=tuple(
                tuple(tuple(pair) for pair in vector) for vector in value.get("reward_vectors") or ()
            ),
            mean_value=value.get("mean_value"),
            confidence_interval=tuple(ci) if ci else None,
            evidence_ids=tuple(value.get("evidence_ids") or ()),
            evidence_confidence=float(value.get("evidence_confidence", 0.0)),
        )


def _normalize_gate_vector(vector: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    pairs = tuple((str(gate), str(status)) for gate, status in vector)
    if tuple(gate for gate, _ in pairs) != GATE_ORDER:
        raise ValueError("verifier vector must carry the complete ordered G0-G12 gate set")
    for _, status in pairs:
        if status not in _GATE_STATUSES:
            raise ValueError(f"gate status must be one of {sorted(_GATE_STATUSES)}, got {status!r}")
    return pairs


def _normalize_reward_vector(vector: Sequence[Any]) -> tuple[tuple[str, float], ...]:
    return tuple((str(name), float(score)) for name, score in vector)


def append_action_outcomes(
    existing: Iterable[ActionOutcomeV2], new: Iterable[ActionOutcomeV2]
) -> tuple[ActionOutcomeV2, ...]:
    """Append-only merge that deduplicates by content identity, not array order.

    Earlier evidence is never mutated; identical content appended twice collapses
    to one row. The result is ordered deterministically by ``(action_id, key)``.
    """
    merged: dict[str, ActionOutcomeV2] = {}
    for outcome in (*existing, *new):
        merged[outcome.content_key] = outcome
    return tuple(sorted(merged.values(), key=lambda o: (o.action_id, o.content_key)))


# --------------------------------------------------------------------------- #
# Objective materialization.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObjectiveView:
    """A pure, versioned materialization of one state's action table."""

    good_action_ids: tuple[int, ...]
    bad_action_ids: tuple[int, ...]
    ambiguous_action_ids: tuple[int, ...]
    unobserved_action_ids: tuple[int, ...]
    weights: tuple[tuple[int, float], ...]
    materializer_id: str
    materializer_config_hash: str
    trainable: bool = True
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("good_action_ids", "bad_action_ids", "ambiguous_action_ids", "unobserved_action_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name)))
        partitions = [
            set(self.good_action_ids),
            set(self.bad_action_ids),
            set(self.ambiguous_action_ids),
            set(self.unobserved_action_ids),
        ]
        seen: set[int] = set()
        for partition in partitions:
            if partition & seen:
                raise ValueError("objective-view action partitions must be disjoint")
            seen |= partition
        object.__setattr__(
            self, "weights", tuple((int(a), float(w)) for a, w in self.weights)
        )
        if any(action not in seen for action, _ in self.weights):
            raise ValueError("objective-view weights reference an unpartitioned action")

    def to_dict(self) -> dict[str, Any]:
        return {
            "good_action_ids": list(self.good_action_ids),
            "bad_action_ids": list(self.bad_action_ids),
            "ambiguous_action_ids": list(self.ambiguous_action_ids),
            "unobserved_action_ids": list(self.unobserved_action_ids),
            "weights": [list(pair) for pair in self.weights],
            "materializer_id": self.materializer_id,
            "materializer_config_hash": self.materializer_config_hash,
            "trainable": self.trainable,
            "version": self.version,
        }


def _observed_reward(outcome: ActionOutcomeV2) -> dict[str, float] | None:
    """Mean reward vector across an outcome's rollouts, or None if unobserved."""
    if not outcome.reward_vectors:
        return None
    totals: dict[str, float] = {name: 0.0 for name in _METRICS}
    for vector in outcome.reward_vectors:
        row = dict(vector)
        for name in _METRICS:
            totals[name] += float(row.get(name, 0.0))
    count = len(outcome.reward_vectors)
    return {name: totals[name] / count for name in _METRICS}


def materialize_pareto(
    state: DecisionStateV2, outcomes: Sequence[ActionOutcomeV2]
) -> ObjectiveView:
    """Exact Pareto pass/fail over the ordered reward metrics.

    ``good`` = observed, reward-bearing, non-dominated actions; ``bad`` = observed
    actions that are dominated or carry no reward evidence; ``ambiguous`` =
    observed actions with no reward vector to score; ``unobserved`` = legal actions
    with no evidence at all.
    """
    by_action: dict[int, dict[str, float]] = {}
    observed: set[int] = set()
    for outcome in outcomes:
        if outcome.state_id != state.state_id:
            raise ValueError("action outcome does not belong to this state")
        observed.add(outcome.action_id)
        reward = _observed_reward(outcome)
        if reward is not None:
            by_action[outcome.action_id] = reward

    def dominates(left: dict[str, float], right: dict[str, float]) -> bool:
        no_worse = all(left[name] >= right[name] for name in _METRICS)
        better = any(left[name] > right[name] for name in _METRICS)
        return no_worse and better

    scored = list(by_action.items())
    good = [
        action
        for action, reward in scored
        if not any(other != action and dominates(by_action[other], reward) for other, _ in scored)
    ]
    good_set = set(good)
    bad = [action for action in observed if action not in good_set and action in by_action]
    ambiguous = [action for action in observed if action not in by_action]
    unobserved = [action for action in state.legal_action_ids if action not in observed]
    return ObjectiveView(
        good_action_ids=good,
        bad_action_ids=bad,
        ambiguous_action_ids=ambiguous,
        unobserved_action_ids=unobserved,
        weights=tuple((action, 1.0) for action in sorted(good_set)),
        materializer_id="pareto_v2",
        materializer_config_hash=content_sha({"metrics": list(_METRICS), "rule": "pareto_nondominated"}),
        trainable=True,
    )


def materialize_constraint_shadow(
    state: DecisionStateV2, outcomes: Sequence[ActionOutcomeV2]
) -> ObjectiveView:
    """Legality-only diagnostic view. Explicitly non-semantic and NOT trainable.

    A semantic trainer must refuse ``trainable=False`` views: legality is not a
    preference label (the E284 lesson).
    """
    for outcome in outcomes:
        if outcome.state_id != state.state_id:
            raise ValueError("action outcome does not belong to this state")
    observed = {outcome.action_id for outcome in outcomes}
    return ObjectiveView(
        good_action_ids=(),
        bad_action_ids=(),
        ambiguous_action_ids=tuple(sorted(observed)),
        unobserved_action_ids=tuple(a for a in state.legal_action_ids if a not in observed),
        weights=(),
        materializer_id="constraint_shadow_diagnostic_v2",
        materializer_config_hash=content_sha({"rule": "legality_only", "semantic": False}),
        trainable=False,
    )


# --------------------------------------------------------------------------- #
# V1 migration (one-way, lossless, evidence-honest).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MigratedDecision:
    """A V1 event migrated to V2 form, with an explicit completeness flag."""

    state: DecisionStateV2
    outcomes: tuple[ActionOutcomeV2, ...]
    complete: bool
    note: str


def migrate_v1_event(event: DecisionEventV1) -> MigratedDecision:
    """Migrate one :class:`DecisionEventV1` to V2 form without fabricating evidence.

    A V1 counterfactual becomes a state with *partial* action evidence (the good /
    bad partition it recorded) and is marked incomplete because V1 kept no
    replayable rollout/verifier vectors. A V1 constraint shadow becomes a legality
    diagnostic. Neither path invents rollout, seed, or G-vector evidence.
    """
    is_counterfactual = event.evidence_kind == "counterfactual"
    bundle = (
        verifier_bundle_hash(SEMANTIC_VERIFIER_V1, _METRICS)
        if is_counterfactual
        else verifier_bundle_hash(_CONSTRAINT_SHADOW_VERIFIER, ())
    )
    state = DecisionStateV2(
        group_id=event.group_id,
        architecture="twotower",
        context_text=event.context_text,
        canvas_ids=event.canvas_ids,
        decision_position=event.position,
        legal_action_ids=event.legal_token_ids,
        decision_kind=event.decision_kind,
        abstract_state_role=event.decision_kind,
        grammar_state_hash=content_sha(
            {"decision_kind": event.decision_kind, "legal_token_ids": list(event.legal_token_ids)}
        ),
        policy_checkpoint_sha=event.policy_checkpoint_sha,
        tokenizer_sha=event.tokenizer_sha,
        decode_config_hash=event.decode_config_hash,
        verifier_bundle_hash=bundle,
        split=event.split,
    )
    outcomes: list[ActionOutcomeV2] = []
    if is_counterfactual:
        # Record the V1 good/bad partition as legal actions with NO replayable
        # verifier/rollout evidence (empty vectors) -> incomplete by construction.
        for action in event.good_token_ids:
            outcomes.append(
                ActionOutcomeV2(
                    state_id=state.state_id,
                    action_id=action,
                    legal=True,
                    rollout_policy_sha=event.policy_checkpoint_sha,
                    continuation_seeds=(event.seed,),
                    evidence_ids=(f"v1:{event.event_id}",),
                    evidence_confidence=event.evidence_confidence,
                )
            )
        for action in event.bad_token_ids:
            outcomes.append(
                ActionOutcomeV2(
                    state_id=state.state_id,
                    action_id=action,
                    legal=action in set(event.legal_token_ids),
                    rollout_policy_sha=event.policy_checkpoint_sha,
                    continuation_seeds=(event.seed,),
                    evidence_ids=(f"v1:{event.event_id}",),
                    evidence_confidence=event.evidence_confidence,
                )
            )
        note = "v1 counterfactual migrated with partial evidence (no replayable G-vectors)"
    else:
        note = "v1 constraint shadow migrated as a legality diagnostic (non-semantic)"
    return MigratedDecision(
        state=state,
        outcomes=tuple(outcomes),
        complete=False,
        note=note,
    )


# --------------------------------------------------------------------------- #
# Atomic, duplicate-safe corpus IO (mirrors write_decision_events).
# --------------------------------------------------------------------------- #
def write_action_outcomes(path: Path | str, outcomes: Iterable[ActionOutcomeV2]) -> int:
    """Atomically write action outcomes as sorted, deduplicated JSONL."""
    path = Path(path)
    rows = append_action_outcomes((), outcomes)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        for outcome in rows:
            handle.write(json.dumps(outcome.to_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise
    return len(rows)


def decision_state_manifest(
    states: Sequence[DecisionStateV2],
    outcomes: Sequence[ActionOutcomeV2],
    views: Sequence[ObjectiveView] = (),
) -> dict[str, Any]:
    """Separately fingerprint states, action evidence, and objective views.

    Rows are canonically ordered first, so a fingerprint changes only when the
    underlying content changes, never when input row order changes.
    """
    ordered_states = sorted(states, key=lambda state: state.state_id)
    ordered_outcomes = append_action_outcomes((), outcomes)
    ordered_views = sorted(views, key=lambda view: content_sha(view.to_dict()))
    return {
        "schema_version": SCHEMA_VERSION,
        "state_count": len(ordered_states),
        "outcome_count": len(ordered_outcomes),
        "view_count": len(ordered_views),
        "state_fingerprint": content_sha([state.to_dict() for state in ordered_states]),
        "outcome_fingerprint": content_sha([outcome.to_dict() for outcome in ordered_outcomes]),
        "view_fingerprint": content_sha([view.to_dict() for view in ordered_views]),
    }
