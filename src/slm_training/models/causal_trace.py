"""Torch-free capture core for exact causal decision-state tracing (LDI1-01).

The causal OpenUI plug-in decodes under a hard grammar constraint. To recover every
supervised decision from exact prefix token IDs and model logits — not from decoded
strings or later retokenization — the capture is split into a **torch-free core**
(this module) and a thin torch wiring on the plug-in.

The core drives a greedy constrained decode through two injected callables:

* ``forward_logits(prefix_ids) -> Sequence[float]`` — the model's next-token logits
  for a prefix (abstracts ``model(input_ids).logits[:, -1, :]``);
* ``allowed_ids(prefix_ids) -> Sequence[int]`` — the grammar-legal token id set for a
  prefix (abstracts the plug-in's ``_allowed_ids``).

Because both are injected, the whole capture algorithm — raw argmax, legal masking,
constrained selection, constraint-shadow detection, per-step records, honest EOS/stop,
and bounded selection policies — is deterministic and unit-testable with fixture
logits, without torch, a real tokenizer, or the grammar. Integer token IDs are the
authority; decoded surfaces are diagnostic only.

A ``constraint_shadow`` observation is emitted when the raw (pre-mask) argmax is
illegal while the constrained selection is legal. It is a non-semantic evidence class:
it measures where the hard constraint changes the policy and must never pass semantic
corpus admission or feed a semantic trainer.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.harnesses.distill.trace_store import TraceStore
from slm_training.harnesses.preference.decision_events_v2 import (
    ActionOutcomeV2,
    DecisionStateV2,
    ObjectiveView,
    materialize_constraint_shadow,
    verifier_bundle_hash,
)
from slm_training.harnesses.preference.local_decisions import split_for_group
from slm_training.lineage.records import content_sha

ForwardLogits = Callable[[tuple[int, ...]], Sequence[float]]
AllowedIds = Callable[[tuple[int, ...]], Sequence[int]]
RoleOf = Callable[[tuple[int, ...]], str | None]

CAUSAL_DECISION_KIND = "causal_decision"
CAUSAL_TRACE_ROW_KIND = "causal_decision"
_CONSTRAINT_SHADOW_VERIFIER = "constraint_shadow_legality_only"
_CAPTURE_VERIFIER = "causal_capture_no_semantic_verifier"

__all__ = [
    "AllowedIds",
    "CAUSAL_DECISION_KIND",
    "CAUSAL_TRACE_ROW_KIND",
    "CaptureResult",
    "CausalTracedGeneration",
    "CausalTraceError",
    "CausalTraceIdentity",
    "CausalTraceWriter",
    "ForwardLogits",
    "GeneratedOutcome",
    "RawStepObservation",
    "RoleOf",
    "TracePolicy",
    "TraceSelection",
    "build_decision_state",
    "capture_raw_steps",
    "causal_trace_row",
    "emit_causal_decision",
    "fold_policy_identity",
    "legal_set_reference",
    "load_causal_decision_states",
]


class CausalTraceError(ValueError):
    """Raised when a capture request is internally inconsistent."""


def legal_set_reference(legal_token_ids: Sequence[int]) -> str:
    """Content-addressed reference for a legal token set (order-independent)."""
    return content_sha(sorted({int(token) for token in legal_token_ids}))


class TraceSelection(str, Enum):
    """Bounded selection policies for which decisions are recorded."""

    EVERY = "every"
    CONSTRAINT_SHADOW_ONLY = "constraint_shadow_only"
    MARGIN_THRESHOLD = "margin_threshold"
    SAMPLED_POSITIONS = "sampled_positions"
    NAMED_ROLES = "named_roles"


@dataclass(frozen=True)
class TracePolicy:
    """A bounded selection policy plus the per-step telemetry width (``top_k``)."""

    selection: TraceSelection = TraceSelection.EVERY
    margin_threshold: float = 0.0
    sampled_positions: tuple[int, ...] = ()
    named_roles: tuple[str, ...] = ()
    top_k: int = 5

    def __post_init__(self) -> None:
        if int(self.top_k) < 1:
            raise ValueError("top_k must be >= 1")
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "margin_threshold", float(self.margin_threshold))
        object.__setattr__(
            self, "sampled_positions", tuple(int(p) for p in self.sampled_positions)
        )
        object.__setattr__(
            self, "named_roles", tuple(str(r) for r in self.named_roles)
        )

    def records(self, obs: RawStepObservation) -> bool:
        """Whether this observation is retained under the selection policy."""
        if self.selection is TraceSelection.EVERY:
            return True
        if self.selection is TraceSelection.CONSTRAINT_SHADOW_ONLY:
            return obs.constraint_shadow
        if self.selection is TraceSelection.MARGIN_THRESHOLD:
            return obs.legal_margin <= self.margin_threshold
        if self.selection is TraceSelection.SAMPLED_POSITIONS:
            return obs.generated_ordinal in self.sampled_positions
        if self.selection is TraceSelection.NAMED_ROLES:
            return obs.grammar_role in self.named_roles
        raise CausalTraceError(f"unknown selection policy: {self.selection}")


@dataclass(frozen=True)
class RawStepObservation:
    """One committed next-token decision, recoverable from exact prefix IDs.

    ``decision_index`` counts only non-forced steps (a step with two or more legal
    candidates is a real reversible decision); ``generated_ordinal`` counts every
    emitted token. A ``forced`` step (a single legal candidate) is a deterministic
    grammar deduction, not a decision, and carries the index of the decision it
    precedes.
    """

    decision_index: int
    generated_ordinal: int
    prefix_token_ids: tuple[int, ...]
    raw_argmax_id: int
    selected_token_id: int
    legal_token_ids: tuple[int, ...]
    raw_topk: tuple[tuple[int, float, float], ...]
    legal_topk: tuple[tuple[int, float], ...]
    constraint_shadow: bool
    forced: bool
    grammar_role: str | None = None

    @property
    def legal_set_reference(self) -> str:
        return legal_set_reference(self.legal_token_ids)

    @property
    def legal_margin(self) -> float:
        """Post-mask top1 − top2 legal probability (top1 when a single candidate)."""
        if not self.legal_topk:
            return 0.0
        if len(self.legal_topk) == 1:
            return self.legal_topk[0][1]
        return self.legal_topk[0][1] - self.legal_topk[1][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "generated_ordinal": self.generated_ordinal,
            "prefix_token_ids": list(self.prefix_token_ids),
            "raw_argmax_id": self.raw_argmax_id,
            "selected_token_id": self.selected_token_id,
            "legal_token_ids": list(self.legal_token_ids),
            "legal_set_reference": self.legal_set_reference,
            "raw_topk": [list(row) for row in self.raw_topk],
            "legal_topk": [list(row) for row in self.legal_topk],
            "constraint_shadow": self.constraint_shadow,
            "forced": self.forced,
            "grammar_role": self.grammar_role,
        }


@dataclass(frozen=True)
class CaptureResult:
    """The outcome of a traced constrained decode.

    ``observations`` holds only the policy-retained decisions; ``generated_token_ids``
    is the complete emitted continuation (every selected token, unaffected by the
    selection policy) so the terminal program can be materialized exactly;
    ``stop_reason`` records why decoding stopped.
    """

    observations: tuple[RawStepObservation, ...]
    generated_token_ids: tuple[int, ...]
    stop_reason: str

    @property
    def constraint_shadow_count(self) -> int:
        return sum(1 for obs in self.observations if obs.constraint_shadow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [obs.to_dict() for obs in self.observations],
            "generated_token_ids": list(self.generated_token_ids),
            "stop_reason": self.stop_reason,
        }


def _log_softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    shifted = [float(value) - peak for value in logits]
    log_denom = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_denom for value in shifted]


def _raw_topk(
    logits: Sequence[float], log_probs: Sequence[float], top_k: int
) -> tuple[tuple[int, float, float], ...]:
    order = sorted(range(len(logits)), key=lambda i: (logits[i], -i), reverse=True)
    return tuple(
        (int(i), float(logits[i]), float(log_probs[i])) for i in order[:top_k]
    )


def _legal_topk(
    logits: Sequence[float], legal: Sequence[int], top_k: int
) -> tuple[tuple[int, float], ...]:
    peak = max(logits[token] for token in legal)
    weights = {int(token): math.exp(float(logits[token]) - peak) for token in legal}
    total = sum(weights.values())
    order = sorted(weights, key=lambda token: (weights[token], -token), reverse=True)
    return tuple((token, weights[token] / total) for token in order[:top_k])


def capture_raw_steps(
    *,
    forward_logits: ForwardLogits,
    allowed_ids: AllowedIds,
    eos_id: int,
    max_new_tokens: int,
    initial_prefix: Sequence[int] = (),
    policy: TracePolicy | None = None,
    role_of: RoleOf | None = None,
) -> CaptureResult:
    """Greedy constrained decode that records exact per-step decision evidence.

    At each step the raw (pre-mask) argmax and the constrained selection (greedy over
    the legal set) are both computed from the same logits, so a ``constraint_shadow``
    — a raw winner outside the legal set overridden by a legal selection — is exact.
    Decoding stops honestly when EOS is selected (``eos``), when the legal set is empty
    (``no_legal_continuation``), or when ``max_new_tokens`` is reached
    (``max_new_tokens``). The returned ``observations`` are only those retained by
    ``policy``, but selection never changes which token is emitted — the full emitted
    continuation is returned in ``generated_token_ids``.
    """
    policy = policy or TracePolicy()
    if int(max_new_tokens) < 0:
        raise CausalTraceError("max_new_tokens must be non-negative")
    prefix = tuple(int(token) for token in initial_prefix)
    eos = int(eos_id)
    observations: list[RawStepObservation] = []
    generated: list[int] = []
    decisions_seen = 0
    stop_reason = "max_new_tokens"
    for ordinal in range(int(max_new_tokens)):
        logits = [float(value) for value in forward_logits(prefix)]
        if not logits:
            raise CausalTraceError("forward_logits returned an empty logit vector")
        if not all(math.isfinite(value) for value in logits):
            raise CausalTraceError("forward_logits returned non-finite logits")
        legal = tuple(int(token) for token in allowed_ids(prefix))
        if not legal:
            stop_reason = "no_legal_continuation"
            break
        legal_set = set(legal)
        if any(token < 0 or token >= len(logits) for token in legal_set):
            raise CausalTraceError("legal token id out of logit range")
        log_probs = _log_softmax(logits)
        raw_argmax = max(range(len(logits)), key=lambda i: (logits[i], -i))
        selected = max(legal, key=lambda token: (logits[token], -token))
        forced = len(legal_set) == 1
        obs = RawStepObservation(
            decision_index=decisions_seen,
            generated_ordinal=ordinal,
            prefix_token_ids=prefix,
            raw_argmax_id=int(raw_argmax),
            selected_token_id=int(selected),
            legal_token_ids=legal,
            raw_topk=_raw_topk(logits, log_probs, policy.top_k),
            legal_topk=_legal_topk(logits, legal, policy.top_k),
            constraint_shadow=(raw_argmax not in legal_set) and (selected in legal_set),
            forced=forced,
            grammar_role=role_of(prefix) if role_of is not None else None,
        )
        if policy.records(obs):
            observations.append(obs)
        if not forced:
            decisions_seen += 1
        generated.append(int(selected))
        prefix = (*prefix, int(selected))
        if selected == eos:
            stop_reason = "eos"
            break
    return CaptureResult(
        observations=tuple(observations),
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
    )


def fold_policy_identity(base_checkpoint_sha: str, adapter_identity: str) -> str:
    """Fold base-checkpoint and active-adapter identity into one policy fingerprint.

    ``DecisionStateV2`` carries a single ``policy_checkpoint_sha`` with no dedicated
    adapter field, so both the base checkpoint and the active adapter are folded here.
    Because the state id hashes ``policy_checkpoint_sha``, an adapter-enabled and an
    adapter-disabled capture over the same prefix receive **different** state
    identities — the acceptance requirement that adapter fingerprints be part of state
    identity.
    """
    return content_sha(
        {"base_checkpoint_sha": str(base_checkpoint_sha), "adapter": str(adapter_identity)}
    )


@dataclass(frozen=True)
class CausalTraceIdentity:
    """The trajectory-level identity stamped onto every captured causal state."""

    group_id: str
    context_text: str
    policy_checkpoint_sha: str
    tokenizer_sha: str
    decode_config_hash: str
    base_model_revision: str = ""
    adapter_identity: str = ""

    def __post_init__(self) -> None:
        for name in (
            "group_id",
            "policy_checkpoint_sha",
            "tokenizer_sha",
            "decode_config_hash",
        ):
            if not str(getattr(self, name)):
                raise CausalTraceError(f"causal trace identity field {name!r} must be non-empty")


def build_decision_state(
    obs: RawStepObservation, identity: CausalTraceIdentity
) -> DecisionStateV2:
    """Materialize a causal ``DecisionStateV2`` from one raw observation.

    Integer prefix ids are the state authority (``context_ids``); the grammar state is
    the content-addressed legal set. A ``constraint_shadow`` step is named as such and
    tagged with the legality-only verifier so downstream code cannot mistake it for a
    semantic verdict.
    """
    shadow = obs.constraint_shadow
    if shadow:
        decision_kind = "constraint_shadow"
        verifier = _CONSTRAINT_SHADOW_VERIFIER
    elif obs.forced:
        decision_kind = "forced_deduction"
        verifier = _CAPTURE_VERIFIER
    else:
        decision_kind = CAUSAL_DECISION_KIND
        verifier = _CAPTURE_VERIFIER
    role = obs.grammar_role or decision_kind
    return DecisionStateV2(
        group_id=identity.group_id,
        architecture="causal",
        context_text=identity.context_text,
        decision_position=obs.generated_ordinal,
        legal_action_ids=obs.legal_token_ids,
        decision_kind=decision_kind,
        abstract_state_role=role,
        grammar_state_hash=obs.legal_set_reference,
        policy_checkpoint_sha=identity.policy_checkpoint_sha,
        tokenizer_sha=identity.tokenizer_sha,
        decode_config_hash=identity.decode_config_hash,
        verifier_bundle_hash=verifier_bundle_hash(verifier, ()),
        split=split_for_group(identity.group_id),
        context_ids=obs.prefix_token_ids,
        generation_step=obs.generated_ordinal,
    )


def _constraint_shadow_outcome(
    state: DecisionStateV2, obs: RawStepObservation, identity: CausalTraceIdentity
) -> ActionOutcomeV2:
    """A legality-only outcome for the selected (legal) action of a shadow step.

    It carries **no** reward or verifier vectors — legality is the only evidence — so
    the view it feeds is non-trainable and cannot supervise a semantic objective.
    """
    return ActionOutcomeV2(
        state_id=state.state_id,
        action_id=obs.selected_token_id,
        legal=True,
        rollout_policy_sha=identity.policy_checkpoint_sha,
    )


def emit_causal_decision(
    obs: RawStepObservation, identity: CausalTraceIdentity
) -> tuple[DecisionStateV2, tuple[ActionOutcomeV2, ...], ObjectiveView | None]:
    """Emit the schema-compatible evidence for one raw observation.

    A ``constraint_shadow`` step yields a legality-only outcome and a **non-trainable**
    ``constraint_shadow`` view (via ``materialize_constraint_shadow``); every other step
    yields the replayable state alone (no rollout evidence is invented). The returned
    view is ``None`` unless a shadow was observed.
    """
    state = build_decision_state(obs, identity)
    if not obs.constraint_shadow:
        return state, (), None
    outcome = _constraint_shadow_outcome(state, obs, identity)
    view = materialize_constraint_shadow(state, (outcome,))
    return state, (outcome,), view


def causal_trace_row(
    state: DecisionStateV2,
    outcomes: Sequence[ActionOutcomeV2],
    obs: RawStepObservation,
    identity: CausalTraceIdentity,
) -> dict[str, Any]:
    """A ``TraceStore`` row for one captured causal decision.

    Identity hashes are lifted to the row top level so a consumer can fail closed on a
    mismatched checkpoint/tokenizer without parsing the full state.
    """
    return {
        "kind": CAUSAL_TRACE_ROW_KIND,
        "state": state.to_dict(),
        "outcomes": [outcome.to_dict() for outcome in outcomes],
        "raw_observation": obs.to_dict(),
        "constraint_shadow": obs.constraint_shadow,
        "policy_checkpoint_sha": identity.policy_checkpoint_sha,
        "tokenizer_sha": identity.tokenizer_sha,
        "decode_config_hash": identity.decode_config_hash,
    }


class CausalTraceWriter:
    """Append captured causal decisions to a ``TraceStore`` and track a manifest."""

    def __init__(self, store: TraceStore, identity: CausalTraceIdentity) -> None:
        self._store = store
        self._identity = identity
        self._state_count = 0
        self._shadow_count = 0
        self._legal_set_refs: dict[str, int] = {}
        self.trajectory_ids: list[str] = []

    def record(self, obs: RawStepObservation) -> str:
        state, outcomes, _view = emit_causal_decision(obs, self._identity)
        trajectory_id = self._store.append(
            causal_trace_row(state, outcomes, obs, self._identity)
        )
        self.trajectory_ids.append(trajectory_id)
        self._state_count += 1
        if obs.constraint_shadow:
            self._shadow_count += 1
        ref = obs.legal_set_reference
        self._legal_set_refs[ref] = self._legal_set_refs.get(ref, 0) + 1
        return trajectory_id

    def record_all(self, result: CaptureResult) -> list[str]:
        return [self.record(obs) for obs in result.observations]

    def manifest(self) -> dict[str, Any]:
        traces_path = Path(self._store.traces_path)
        total_bytes = traces_path.stat().st_size if traces_path.exists() else 0
        duplicate_set_reuse = sum(count - 1 for count in self._legal_set_refs.values())
        return {
            "kind": "causal_trace_manifest",
            "group_id": self._identity.group_id,
            "policy_checkpoint_sha": self._identity.policy_checkpoint_sha,
            "tokenizer_sha": self._identity.tokenizer_sha,
            "decode_config_hash": self._identity.decode_config_hash,
            "base_model_revision": self._identity.base_model_revision,
            "adapter_identity": self._identity.adapter_identity,
            "state_count": self._state_count,
            "constraint_shadow_count": self._shadow_count,
            "unique_legal_sets": len(self._legal_set_refs),
            "duplicate_set_reuse": duplicate_set_reuse,
            "total_bytes": total_bytes,
            "bytes_per_state": (
                total_bytes / self._state_count if self._state_count else 0.0
            ),
        }

    def write_manifest(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        tmp = Path(raw)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.manifest(), handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def load_causal_decision_states(
    store: TraceStore,
    *,
    expected_checkpoint_sha: str,
    expected_tokenizer_sha: str,
) -> list[DecisionStateV2]:
    """Load captured causal states, failing closed on a mismatched identity.

    Loading against a checkpoint or tokenizer other than the one that produced the
    trace raises before any state is returned — rollouts from different checkpoints
    must never be mixed. ``DecisionStateV2.from_dict`` additionally re-verifies each
    state id (tamper check).
    """
    states: list[DecisionStateV2] = []
    for row in store.iter_kind(CAUSAL_TRACE_ROW_KIND):
        if row.get("policy_checkpoint_sha") != expected_checkpoint_sha:
            raise ValueError("causal decision states do not match the policy checkpoint")
        if row.get("tokenizer_sha") != expected_tokenizer_sha:
            raise ValueError("causal decision states do not match the checkpoint tokenizer")
        state = DecisionStateV2.from_dict(row["state"])
        # The nested state must agree with the row envelope and the requested identity,
        # not merely be internally consistent.
        if state.policy_checkpoint_sha != expected_checkpoint_sha:
            raise ValueError("causal decision state does not match the policy checkpoint")
        if state.tokenizer_sha != expected_tokenizer_sha:
            raise ValueError("causal decision state does not match the checkpoint tokenizer")
        states.append(state)
    return states


@dataclass(frozen=True)
class GeneratedOutcome:
    """A forced-action replay outcome, handed to the shared counterfactual owner.

    The plug-in never runs a judge: it returns the replay-side fields only (action,
    seed, finish reason, raw program, canonical program when valid). The counterfactual
    owner scores it with ``semantic_outcome`` and partitions with
    ``label_pareto_candidates``. ``to_candidate`` yields the pre-judge candidate dict
    that owner consumes.
    """

    action_id: int
    continuation_seed: int
    finish_reason: str
    raw_program: str
    canonical_program: str | None
    selected: bool = True
    completion_source: str = "policy"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "continuation_seed": self.continuation_seed,
            "finish_reason": self.finish_reason,
            "raw_program": self.raw_program,
            "canonical_program": self.canonical_program,
            "selected": self.selected,
            "completion_source": self.completion_source,
        }

    def to_candidate(self) -> dict[str, Any]:
        canonical = self.canonical_program
        return {
            "token_id": self.action_id,
            "selected": self.selected,
            "completion_source": self.completion_source,
            "raw_text": self.raw_program,
            "text": canonical if canonical is not None else self.raw_program,
            "finalization_changed": canonical is not None and canonical != self.raw_program,
        }


@dataclass(frozen=True)
class CausalTracedGeneration:
    """The result of a traced constrained generation on the plug-in."""

    text: str
    result: CaptureResult
    valid: bool
