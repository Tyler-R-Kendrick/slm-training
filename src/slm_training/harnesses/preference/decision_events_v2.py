"""DecisionEventV2 (LDI0-02 / SLM-116): state + action-verdict tables.

V1 (`local_decisions.DecisionEventV1`) froze one `good_token_ids` /
`bad_token_ids` partition *into* the event, so grammar-state support could be
complete while the sampled action partition used by the objective was
unsupported or incompatible (the E284 blocker). V2 splits three concerns:

* `DecisionStateV2` — a canonical, replayable **state identity**. `state_id` is a
  hash of the exact model state + immutable runtime identities **only** — never
  sampled good/bad labels, rollout outcomes, file position, or candidate order.
  Reordering or augmenting the action evidence never changes it.
* `ActionOutcomeV2` — append-only, content-deduplicated **evidence** for one
  `(state_id, action_id)`: legality, rollout seeds/output hashes, the complete
  ordered G0–G12 verifier vector, named independent-judge evidence, and reward
  vectors. A scalar value may be *derived* but never replaces the raw evidence.
* `ObjectiveView` — a pure, versioned **materializer** output that turns one
  state/action table into good/bad/ambiguous/unobserved partitions. Multiple
  materializers can produce different views from the same evidence without
  touching it.

V1 stays readable and is migrated one-way (never fabricating rollout/verifier
evidence); a migrated semantic counterfactual becomes a V2 state with **partial**
action evidence marked incomplete, and a V1 constraint shadow stays a legality
diagnostic that semantic trainers cannot consume.

Torch-free. No trainer change, no rollout policy, no adapter, no quality claim.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    Split,
    _ids,
    _sha,
    split_for_group,
)

SCHEMA_VERSION_V2 = 2
Architecture = Literal["twotower", "causal"]

# Named verifier metrics carried through a materializer (Pareto axes).
PARETO_METRICS = (
    "placeholder_fidelity",
    "component_recall",
    "structural_similarity",
    "reward",
)

# Fields that define the exact model decision state. `state_id` hashes exactly
# these — never labels, outcomes, ordinal position, or candidate order.
_STATE_IDENTITY_FIELDS = (
    "architecture",
    "context_text",
    "context_ids",
    "canvas_ids",
    "decision_position",
    "generation_step",
    "legal_action_ids",
    "decision_kind",
    "abstract_state_role",
    "grammar_state_hash",
    "policy_checkpoint_sha",
    "tokenizer_sha",
    "decode_config_hash",
    "verifier_bundle_hash",
)

_REQUIRED_IDENTITY_SHAS = (
    "policy_checkpoint_sha",
    "tokenizer_sha",
    "decode_config_hash",
    "verifier_bundle_hash",
    "grammar_state_hash",
)


def _opt_ids(values: Iterable[int] | None) -> tuple[int, ...] | None:
    return None if values is None else tuple(int(v) for v in values)


# --------------------------------------------------------------------------- #
# Decision state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DecisionStateV2:
    """Canonical, replayable identity for one exact model decision site."""

    group_id: str
    architecture: Architecture
    context_text: str
    context_ids: tuple[int, ...] | None
    canvas_ids: tuple[int, ...] | None
    decision_position: int
    generation_step: int | None
    legal_action_ids: tuple[int, ...]
    decision_kind: str
    abstract_state_role: str
    grammar_state_hash: str
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    verifier_bundle_hash: str
    split: Split
    state_id: str = ""
    schema_version: int = SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        object.__setattr__(self, "legal_action_ids", _ids(self.legal_action_ids))
        object.__setattr__(self, "canvas_ids", _opt_ids(self.canvas_ids))
        object.__setattr__(self, "context_ids", _opt_ids(self.context_ids))
        if self.schema_version != SCHEMA_VERSION_V2:
            raise ValueError("unsupported decision state schema version")
        if self.architecture not in ("twotower", "causal"):
            raise ValueError("architecture must be 'twotower' or 'causal'")
        if not self.group_id or not self.context_text:
            raise ValueError("group_id and context_text are required")
        # Causal surfaces must store exact input ids so the supervised action is
        # recoverable without retokenization; TwoTower must store the canvas.
        if self.architecture == "causal" and self.context_ids is None:
            raise ValueError("causal state requires context_ids (no retokenization)")
        if self.architecture == "twotower" and self.canvas_ids is None:
            raise ValueError("twotower state requires canvas_ids")
        if self.decision_position < 0:
            raise ValueError("decision_position must be non-negative")
        if not self.legal_action_ids:
            raise ValueError("legal_action_ids must be non-empty")
        for name in _REQUIRED_IDENTITY_SHAS:
            if not str(getattr(self, name)):
                raise ValueError(f"missing runtime identity: {name}")
        if self.split != split_for_group(self.group_id):
            raise ValueError("split must be derived deterministically from group_id")
        expected = self.canonical_state_id()
        if not self.state_id:
            object.__setattr__(self, "state_id", expected)
        elif self.state_id != expected:
            raise ValueError("state_id is not the canonical hash of the model state")

    def canonical_state_id(self) -> str:
        payload = {}
        for name in _STATE_IDENTITY_FIELDS:
            value = getattr(self, name)
            payload[name] = list(value) if isinstance(value, tuple) else value
        return _sha(payload)

    def replay_inputs(self) -> dict[str, Any]:
        """Exact inputs needed to recompute this decision (no retokenization)."""
        return {
            "architecture": self.architecture,
            "context_ids": list(self.context_ids) if self.context_ids else None,
            "canvas_ids": list(self.canvas_ids) if self.canvas_ids else None,
            "decision_position": self.decision_position,
            "generation_step": self.generation_step,
            "legal_action_ids": list(self.legal_action_ids),
            "tokenizer_sha": self.tokenizer_sha,
            "policy_checkpoint_sha": self.policy_checkpoint_sha,
            "decode_config_hash": self.decode_config_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionStateV2":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown decision state fields: {sorted(unknown)}")
        data = dict(value)
        for name in ("context_ids", "canvas_ids", "legal_action_ids"):
            if data.get(name) is not None:
                data[name] = tuple(data[name])
        return cls(**data)


# --------------------------------------------------------------------------- #
# Action outcome evidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ActionOutcomeV2:
    """Append-only evidence for one observed `(state_id, action_id)` candidate."""

    state_id: str
    action_id: int
    legal: bool
    rollout_policy_sha: str
    continuation_seeds: tuple[int, ...]
    outcome_hashes: tuple[str, ...]
    verifier_vectors: tuple[dict[str, Any], ...]
    reward_vectors: tuple[dict[str, float], ...]
    mean_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence_confidence: float = 1.0
    migrated_incomplete: bool = False
    schema_version: int = SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        object.__setattr__(self, "continuation_seeds", tuple(int(s) for s in self.continuation_seeds))
        object.__setattr__(self, "outcome_hashes", tuple(str(h) for h in self.outcome_hashes))
        object.__setattr__(self, "verifier_vectors", tuple(self.verifier_vectors))
        object.__setattr__(self, "reward_vectors", tuple(self.reward_vectors))
        object.__setattr__(self, "evidence_ids", tuple(str(e) for e in self.evidence_ids))
        if self.schema_version != SCHEMA_VERSION_V2:
            raise ValueError("unsupported action outcome schema version")
        if self.action_id < 0:
            raise ValueError("action_id must be non-negative")
        if not str(self.state_id):
            raise ValueError("action outcome requires a state_id")
        # One recorded output hash per rollout seed — seed/evidence agreement.
        if len(self.outcome_hashes) != len(self.continuation_seeds):
            raise ValueError("outcome_hashes and continuation_seeds count disagree")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1]")
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            if lo > hi:
                raise ValueError("confidence_interval must be ordered")
            object.__setattr__(self, "confidence_interval", (float(lo), float(hi)))

    def content_id(self) -> str:
        """Content identity for order-independent dedup/merge."""
        return _sha(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionOutcomeV2":
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown action outcome fields: {sorted(unknown)}")
        data = dict(value)
        for name in ("continuation_seeds", "outcome_hashes", "verifier_vectors", "reward_vectors", "evidence_ids"):
            if data.get(name) is not None:
                data[name] = tuple(data[name])
        if data.get("confidence_interval") is not None:
            data["confidence_interval"] = tuple(data["confidence_interval"])
        return cls(**data)


def merge_action_evidence(
    outcomes: Iterable[ActionOutcomeV2],
) -> list[ActionOutcomeV2]:
    """Append-only, order-independent merge: dedup by content identity."""
    by_content: dict[str, ActionOutcomeV2] = {}
    for outcome in outcomes:
        by_content.setdefault(outcome.content_id(), outcome)
    return [by_content[key] for key in sorted(by_content)]


# --------------------------------------------------------------------------- #
# Objective materializers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectiveView:
    """A materialized objective partition over one state/action table."""

    state_id: str
    good_action_ids: tuple[int, ...]
    bad_action_ids: tuple[int, ...]
    ambiguous_action_ids: tuple[int, ...]
    unobserved_action_ids: tuple[int, ...]
    weights: dict[str, float]
    materializer_id: str
    materializer_config_hash: str
    semantic: bool
    schema_version: int = SCHEMA_VERSION_V2

    def fingerprint(self) -> str:
        return _sha(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verified(outcome: ActionOutcomeV2) -> bool:
    """A candidate is verified iff every recorded judge probe passed."""
    if not outcome.verifier_vectors:
        return False
    return all(bool(vec.get("ok")) or bool(vec.get("verified")) for vec in outcome.verifier_vectors)


def _metric_means(outcome: ActionOutcomeV2) -> dict[str, float]:
    if not outcome.reward_vectors:
        return {name: 0.0 for name in PARETO_METRICS}
    means: dict[str, float] = {}
    for name in PARETO_METRICS:
        vals = [float(vec.get(name, 0.0)) for vec in outcome.reward_vectors]
        means[name] = sum(vals) / len(vals) if vals else 0.0
    return means


def _partition_frame(state: DecisionStateV2, observed: set[int]):
    legal = set(state.legal_action_ids)
    unobserved = tuple(sorted(legal - observed))
    return legal, unobserved


def _materializer_config_hash(materializer_id: str, config: dict[str, Any]) -> str:
    return _sha({"materializer_id": materializer_id, "config": config or {}})


def materialize_pareto(
    state: DecisionStateV2,
    outcomes: list[ActionOutcomeV2],
    config: dict[str, Any] | None = None,
) -> ObjectiveView:
    """Exact Pareto pass/fail over named verifier metrics (semantic)."""
    config = config or {}
    observed = {o.action_id for o in outcomes if o.legal}
    verified = {o.action_id: _metric_means(o) for o in outcomes if o.legal and _verified(o)}
    good: list[int] = []
    for action, metrics in verified.items():
        dominated = any(
            other != action
            and all(verified[other][m] >= metrics[m] for m in PARETO_METRICS)
            and any(verified[other][m] > metrics[m] for m in PARETO_METRICS)
            for other in verified
        )
        if not dominated:
            good.append(action)
    good_set = set(good)
    bad = sorted(a for a in observed if a not in good_set and a not in verified)
    ambiguous = sorted(a for a in verified if a not in good_set)
    _, unobserved = _partition_frame(state, observed)
    return ObjectiveView(
        state_id=state.state_id,
        good_action_ids=tuple(sorted(good_set)),
        bad_action_ids=tuple(bad),
        ambiguous_action_ids=tuple(ambiguous),
        unobserved_action_ids=unobserved,
        weights={},
        materializer_id="pareto_pass_fail_v1",
        materializer_config_hash=_materializer_config_hash("pareto_pass_fail_v1", config),
        semantic=True,
    )


def materialize_thresholded(
    state: DecisionStateV2,
    outcomes: list[ActionOutcomeV2],
    config: dict[str, Any] | None = None,
) -> ObjectiveView:
    """Thresholded scalar value with a confidence requirement (semantic)."""
    config = config or {}
    threshold = float(config.get("threshold", 0.5))
    min_conf = float(config.get("min_confidence", 0.0))
    observed = {o.action_id for o in outcomes if o.legal}
    good: list[int] = []
    bad: list[int] = []
    ambiguous: list[int] = []
    for outcome in outcomes:
        if not outcome.legal:
            continue
        if outcome.mean_value is None or outcome.evidence_confidence < min_conf:
            ambiguous.append(outcome.action_id)
        elif outcome.mean_value >= threshold:
            good.append(outcome.action_id)
        else:
            bad.append(outcome.action_id)
    _, unobserved = _partition_frame(state, observed)
    return ObjectiveView(
        state_id=state.state_id,
        good_action_ids=tuple(sorted(set(good))),
        bad_action_ids=tuple(sorted(set(bad) - set(good))),
        ambiguous_action_ids=tuple(sorted(set(ambiguous) - set(good) - set(bad))),
        unobserved_action_ids=unobserved,
        weights={},
        materializer_id="thresholded_value_v1",
        materializer_config_hash=_materializer_config_hash("thresholded_value_v1", config),
        semantic=True,
    )


def materialize_single_best_worst(
    state: DecisionStateV2,
    outcomes: list[ActionOutcomeV2],
    config: dict[str, Any] | None = None,
) -> ObjectiveView:
    """Single-best / single-worst control (semantic)."""
    config = config or {}
    scored = [(o.action_id, o.mean_value) for o in outcomes if o.legal and o.mean_value is not None]
    observed = {o.action_id for o in outcomes if o.legal}
    good: tuple[int, ...] = ()
    bad: tuple[int, ...] = ()
    if scored:
        best = max(scored, key=lambda pair: pair[1])
        worst = min(scored, key=lambda pair: pair[1])
        good = (best[0],)
        bad = (worst[0],) if worst[0] != best[0] else ()
    ambiguous = tuple(sorted(observed - set(good) - set(bad)))
    _, unobserved = _partition_frame(state, observed)
    return ObjectiveView(
        state_id=state.state_id,
        good_action_ids=good,
        bad_action_ids=bad,
        ambiguous_action_ids=ambiguous,
        unobserved_action_ids=unobserved,
        weights={},
        materializer_id="single_best_worst_v1",
        materializer_config_hash=_materializer_config_hash("single_best_worst_v1", config),
        semantic=True,
    )


def materialize_set_valued(
    state: DecisionStateV2,
    outcomes: list[ActionOutcomeV2],
    config: dict[str, Any] | None = None,
) -> ObjectiveView:
    """Set-valued good/bad partitions from verified vs failed evidence (semantic)."""
    config = config or {}
    observed = {o.action_id for o in outcomes if o.legal}
    good = sorted(o.action_id for o in outcomes if o.legal and _verified(o))
    bad = sorted(o.action_id for o in outcomes if o.legal and not _verified(o))
    good_set = set(good)
    bad = [a for a in bad if a not in good_set]
    _, unobserved = _partition_frame(state, observed)
    return ObjectiveView(
        state_id=state.state_id,
        good_action_ids=tuple(good),
        bad_action_ids=tuple(bad),
        ambiguous_action_ids=(),
        unobserved_action_ids=unobserved,
        weights={},
        materializer_id="set_valued_v1",
        materializer_config_hash=_materializer_config_hash("set_valued_v1", config),
        semantic=True,
    )


def materialize_constraint_shadow(
    state: DecisionStateV2,
    outcomes: list[ActionOutcomeV2],
    config: dict[str, Any] | None = None,
) -> ObjectiveView:
    """Legality diagnostic view — explicitly NON-semantic and non-trainable.

    Partitions by declared legality only (compiler-forced), never by verified
    semantics. `semantic=False` marks it un-consumable by semantic trainers.
    """
    config = config or {}
    legal_actions = sorted(o.action_id for o in outcomes if o.legal)
    illegal_actions = sorted(o.action_id for o in outcomes if not o.legal)
    observed = {o.action_id for o in outcomes}
    _, unobserved = _partition_frame(state, observed)
    return ObjectiveView(
        state_id=state.state_id,
        good_action_ids=tuple(legal_actions),
        bad_action_ids=tuple(illegal_actions),
        ambiguous_action_ids=(),
        unobserved_action_ids=unobserved,
        weights={},
        materializer_id="constraint_shadow_v1",
        materializer_config_hash=_materializer_config_hash("constraint_shadow_v1", config),
        semantic=False,
    )


MATERIALIZERS = {
    "pareto_pass_fail_v1": materialize_pareto,
    "thresholded_value_v1": materialize_thresholded,
    "single_best_worst_v1": materialize_single_best_worst,
    "set_valued_v1": materialize_set_valued,
    "constraint_shadow_v1": materialize_constraint_shadow,
}


def materialize(
    state: DecisionStateV2,
    outcomes: Iterable[ActionOutcomeV2],
    materializer_id: str,
    config: dict[str, Any] | None = None,
) -> ObjectiveView:
    if materializer_id not in MATERIALIZERS:
        raise ValueError(f"unknown materializer: {materializer_id}")
    merged = merge_action_evidence(outcomes)
    validate_state_action_table(state, merged)
    return MATERIALIZERS[materializer_id](state, merged, config)


def assert_semantic_trainable(view: ObjectiveView) -> ObjectiveView:
    """Guard: a semantic trainer must never consume a non-semantic view."""
    if not view.semantic:
        raise ValueError(
            f"materializer {view.materializer_id!r} is a non-semantic diagnostic; "
            "semantic trainers must not consume it"
        )
    return view


# --------------------------------------------------------------------------- #
# Fail-closed validation
# --------------------------------------------------------------------------- #


def validate_state_action_table(
    state: DecisionStateV2, outcomes: Iterable[ActionOutcomeV2]
) -> None:
    """Reject an incoherent state/action table (fail closed)."""
    legal_set = set(state.legal_action_ids)
    for outcome in outcomes:
        if outcome.state_id != state.state_id:
            raise ValueError("state identities disagree across rows")
        # Semantic (legal) outcomes must address the declared legal set.
        if outcome.legal and outcome.action_id not in legal_set:
            raise ValueError("legal action outcome is outside the declared legal set")
    # `DecisionStateV2.__post_init__` already enforced identity presence, causal
    # retokenization safety, and deterministic split derivation.


def check_split_homogeneity(states: Iterable[DecisionStateV2]) -> None:
    """A single exact state may not straddle train and held-out splits."""
    by_state: dict[str, str] = {}
    for state in states:
        prior = by_state.setdefault(state.state_id, state.split)
        if prior != state.split:
            raise ValueError(
                f"state {state.state_id[:12]}… appears in both {prior} and {state.split}"
            )


# --------------------------------------------------------------------------- #
# V1 → V2 migration (one-way, never fabricating evidence)
# --------------------------------------------------------------------------- #


def migrate_v1_event(
    event: DecisionEventV1,
) -> tuple[DecisionStateV2, list[ActionOutcomeV2]]:
    """Migrate a V1 event to a V2 state + **partial** action evidence.

    A V1 semantic counterfactual becomes a state with action evidence marked
    ``migrated_incomplete`` (no rollout seeds/verifier vectors are fabricated). A
    V1 constraint shadow stays a legality diagnostic (its good/bad are the
    compiler-forced legal/raw tokens, not verified semantics).
    """
    state = DecisionStateV2(
        group_id=event.group_id,
        architecture="twotower",  # V1 is canvas-based
        context_text=event.context_text,
        context_ids=None,
        canvas_ids=event.canvas_ids,
        decision_position=event.position,
        generation_step=None,
        legal_action_ids=event.legal_token_ids,
        decision_kind=event.decision_kind,
        abstract_state_role=event.evidence_kind,
        # V1 carried no grammar/verifier bundle hash; use the decode-config hash
        # as a stable, non-fabricated stand-in and mark the evidence incomplete.
        grammar_state_hash=event.decode_config_hash,
        policy_checkpoint_sha=event.policy_checkpoint_sha,
        tokenizer_sha=event.tokenizer_sha,
        decode_config_hash=event.decode_config_hash,
        verifier_bundle_hash=event.decode_config_hash,
        split=event.split,
    )
    is_shadow = event.evidence_kind == "constraint_shadow"
    outcomes: list[ActionOutcomeV2] = []
    for action_id in event.good_token_ids:
        outcomes.append(
            _migrated_outcome(state, action_id, legal=True, event=event, verified=not is_shadow)
        )
    for action_id in event.bad_token_ids:
        # V1 "bad" tokens: illegal (raw) under a constraint shadow, legal-but-
        # rejected under a counterfactual.
        outcomes.append(
            _migrated_outcome(state, action_id, legal=not is_shadow, event=event, verified=False)
        )
    return state, merge_action_evidence(outcomes)


def _migrated_outcome(
    state: DecisionStateV2,
    action_id: int,
    *,
    legal: bool,
    event: DecisionEventV1,
    verified: bool,
) -> ActionOutcomeV2:
    return ActionOutcomeV2(
        state_id=state.state_id,
        action_id=int(action_id),
        legal=legal,
        rollout_policy_sha=event.policy_checkpoint_sha,
        continuation_seeds=(),
        outcome_hashes=(),
        # Preserve the V1 verdict as a single labelled vector; do NOT fabricate a
        # G0–G12 rollout the V1 record never had.
        verifier_vectors=(
            {"source": "migrated_v1", "evidence_kind": event.evidence_kind, "ok": verified},
        ),
        reward_vectors=(),
        mean_value=1.0 if verified else 0.0,
        evidence_ids=(event.event_id,),
        evidence_confidence=event.evidence_confidence,
        migrated_incomplete=True,
    )


# --------------------------------------------------------------------------- #
# Persistence + manifest
# --------------------------------------------------------------------------- #


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return len(rows)


def write_decision_states(path: Path | str, states: Iterable[DecisionStateV2]) -> int:
    unique = {s.state_id: s for s in states}
    rows = [unique[key].to_dict() for key in sorted(unique)]
    return _atomic_write_jsonl(Path(path), rows)


def write_action_outcomes(path: Path | str, outcomes: Iterable[ActionOutcomeV2]) -> int:
    merged = merge_action_evidence(outcomes)
    rows = sorted((o.to_dict() for o in merged), key=lambda r: (r["state_id"], r["action_id"], _sha(r)))
    return _atomic_write_jsonl(Path(path), rows)


def load_decision_states(path: Path | str) -> list[DecisionStateV2]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [DecisionStateV2.from_dict(json.loads(line)) for line in handle if line.strip()]


def load_action_outcomes(path: Path | str) -> list[ActionOutcomeV2]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [ActionOutcomeV2.from_dict(json.loads(line)) for line in handle if line.strip()]


def decision_v2_manifest(
    states: Iterable[DecisionStateV2],
    outcomes: Iterable[ActionOutcomeV2],
    *,
    dataset_id: str,
    views: Iterable[ObjectiveView] = (),
) -> dict[str, Any]:
    """Manifest fingerprinting states, action evidence, and objective views separately."""
    state_rows = {s.state_id: s for s in states}
    if not state_rows:
        raise ValueError("decision state corpus must not be empty")
    check_split_homogeneity(state_rows.values())
    identities = {
        (s.policy_checkpoint_sha, s.tokenizer_sha, s.decode_config_hash)
        for s in state_rows.values()
    }
    if len(identities) != 1:
        raise ValueError("decision state corpus mixes policy identities")
    checkpoint_sha, tokenizer_sha, decode_hash = identities.pop()
    merged = merge_action_evidence(outcomes)
    # Every outcome must reference a known state (fail closed).
    for outcome in merged:
        if outcome.state_id not in state_rows:
            raise ValueError("action outcome references an unknown state_id")
    state_dicts = [state_rows[key].to_dict() for key in sorted(state_rows)]
    outcome_ids = sorted(o.content_id() for o in merged)
    view_dicts = sorted((v.to_dict() for v in views), key=_sha)
    splits = {
        split: sum(s.split == split for s in state_rows.values())
        for split in ("train", "held_out")
    }
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "kind": "decision_event_v2_corpus",
        "dataset_id": dataset_id,
        "immutable": True,
        "state_count": len(state_rows),
        "action_outcome_count": len(merged),
        "states_fingerprint": _sha(state_dicts),
        "action_evidence_fingerprint": _sha(outcome_ids),
        "objective_views_fingerprint": _sha(view_dicts) if view_dicts else None,
        "policy_checkpoint_sha": checkpoint_sha,
        "tokenizer_sha": tokenizer_sha,
        "decode_config_hash": decode_hash,
        "splits": splits,
        "split_groups": {
            split: len({s.group_id for s in state_rows.values() if s.split == split})
            for split in ("train", "held_out")
        },
        "incomplete_evidence": sum(o.migrated_incomplete for o in merged),
        "note": (
            "Fixture/wiring evidence only; no trainer change, no rollout policy, "
            "no model-quality claim."
        ),
    }


def write_decision_v2_manifest(path: Path | str, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SCHEMA_VERSION_V2",
    "ActionOutcomeV2",
    "DecisionStateV2",
    "MATERIALIZERS",
    "ObjectiveView",
    "assert_semantic_trainable",
    "check_split_homogeneity",
    "decision_v2_manifest",
    "load_action_outcomes",
    "load_decision_states",
    "materialize",
    "merge_action_evidence",
    "migrate_v1_event",
    "validate_state_action_table",
    "write_action_outcomes",
    "write_decision_states",
    "write_decision_v2_manifest",
]
