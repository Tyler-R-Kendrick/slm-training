"""DSH4-01: operator decision-state export reusing legal-set/trace contracts.

SLM-386 extends the repository's existing legal-set trace/export contracts
with certified DSH3-06 operator candidates instead of building a duplicate
distillation data stack. Every field below is either a typed compiler
identity/digest, a bounded verifier outcome, or a numeric score -- no natural
language reasoning or chain-of-thought is ever stored.

Reused, not duplicated:

* ``OperatorStateV1`` / ``ApplicationProvenanceV1`` / ``OperatorApplicationV1``
  / ``ActionEffectV1`` (``slm_training.dsl.operators.contracts`` and
  ``slm_training.dsl.operators.registry``) -- compiler state, provenance, and
  effect identity.
* ``ReferenceTableV1`` (DSH3-03, ``slm_training.dsl.operators.references``) --
  the inference-visible compiler-fact context bound to the same state.
* ``OperatorLegalSetV1`` / ``enumerate_operator_legal_set`` (DSH3-06,
  ``slm_training.dsl.operators.legal_set``) -- the complete legal
  action/operator set, coverage, and bounded rejection evidence. This module
  never re-implements enumeration; it always calls the DSH3-06 function.
* ``TeacherTraceManifest`` (SPV2-03,
  ``slm_training.harnesses.distill.legal_set_teacher_trace``) -- the
  provenance envelope binding a scoring run. Only ``manifest_id`` is carried
  on each trace, exactly as ``LegalSetTeacherTrace`` already does.
* ``SupervisionSource`` (EFS3-01, ``slm_training.evals.solver_state_supervision``)
  -- gold-state vs on-policy topology/decoder state provenance.
* ``OperatorLibraryV1.replay`` / ``OperatorReplayError`` (DSH3-02,
  ``slm_training.dsl.operators.registry``) -- the only replay authority; this
  module classifies its (static, unformatted) messages into stable codes but
  never re-derives replay identity itself.

Stop rule (issue DSH4-01): if the exact state/action identity of an accepted
trajectory cannot be replayed to a verifier-equivalent canonical result, the
exporter must be repaired before any teacher is queried.
``export_for_teacher_query`` enforces this: it raises
``OperatorDecisionStateError`` and returns nothing queryable when replay does
not verify.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    OperatorApplicationV1,
    _fingerprint,
    _require_identifier,
)
from slm_training.dsl.operators.legal_set import (
    LegalSetCoverage,
    OperatorLegalSetV1,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorReplayError,
    OperatorStateV1,
)
from slm_training.dsl.pack import DslPack
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.legal_set_teacher_trace import TeacherTraceManifest

__all__ = [
    "CaptureStage",
    "OperatorDecisionStateError",
    "OperatorDecisionStateTraceV1",
    "OperatorReplayDivergenceV1",
    "OperatorTrajectoryReplayResultV1",
    "OperatorTrajectoryStepV1",
    "OperatorTrajectoryV1",
    "build_operator_trajectory",
    "capture_operator_decision_state",
    "export_for_teacher_query",
    "replay_operator_trajectory",
    "write_operator_decision_state_traces",
]

_SCHEMA_VERSION = "dsh4-01.v1"

# ``OperatorLibraryV1.replay`` (registry.py) raises ``OperatorReplayError``
# with one of these exact, unformatted messages. Classifying them into stable
# codes here does not re-derive replay identity -- ``library.replay`` remains
# the sole authority; this only makes its outcomes greppable/countable.
_REPLAY_ERROR_CODES: dict[str, str] = {
    "before state digest does not match replay state": (
        "operator.replay.before_state_digest_mismatch"
    ),
    "before AST digest does not match replay state": (
        "operator.replay.before_ast_digest_mismatch"
    ),
    "recorded operator fingerprint is unavailable": (
        "operator.replay.unknown_operator_fingerprint"
    ),
    "replayed application identity differs from recorded evidence": (
        "operator.replay.application_identity_mismatch"
    ),
}
_REPLAY_CODE_UNCLASSIFIED = "operator.replay.unclassified"


class CaptureStage(str, Enum):
    """When, relative to repair/fallback mutation, a trace was captured.

    Only ``PRE_REPAIR`` is accepted today: DSH4-01 traces are built directly
    from ``enumerate_operator_legal_set`` over the caller-supplied state, and
    this module never calls a repair or decode-fallback path. The enum stays
    open (rather than a bare bool) so a future capture stage cannot silently
    satisfy today's acceptance bar without an explicit new member.
    """

    PRE_REPAIR = "pre_repair"


class OperatorDecisionStateError(ValueError):
    """Fail-closed error for a decision-state export that cannot be trusted."""


@dataclass(frozen=True)
class OperatorDecisionStateTraceV1:
    """One certified operator decision-state export (DSH4-01).

    Binds one DSH3-06 ``OperatorLegalSetV1`` -- the exact state fingerprint,
    complete legal action/operator set, coverage, and bounded verifier
    (rejection) outcomes -- plus the DSH3-03 inference-visible reference
    table, to acceptance/score evidence and an SPV2-03 manifest binding,
    captured before any repair or fallback mutation runs.
    """

    trace_id: str
    manifest_id: str
    supervision_source: SupervisionSource
    state: OperatorStateV1
    reference_table: ReferenceTableV1
    provenance: ApplicationProvenanceV1
    legal_set: OperatorLegalSetV1
    accepted_application_ids: tuple[str, ...]
    current_scores: Mapping[str, float] | None
    approximate: bool
    capture_stage: CaptureStage
    timestamp: str
    schema: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_identifier(self.trace_id, "trace_id")
        _require_identifier(self.manifest_id, "manifest_id")
        if self.capture_stage is not CaptureStage.PRE_REPAIR:
            raise OperatorDecisionStateError(
                "operator decision-state traces must be captured before "
                "repair/fallback mutation (capture_stage=pre_repair)"
            )
        if self.reference_table.state_digest != self.state.state_digest:
            raise OperatorDecisionStateError(
                "reference table is stale for the bound compiler state"
            )
        if self.reference_table.request_id != self.provenance.request_id:
            raise OperatorDecisionStateError(
                "reference table and provenance request differ"
            )
        if self.legal_set.reference_table_fingerprint != self.reference_table.fingerprint:
            raise OperatorDecisionStateError(
                "legal set is not bound to the exported reference table"
            )
        expected_state_fingerprint = _fingerprint(
            {
                "schema": "operator_legal_state/v1",
                "state_digest": self.state.state_digest,
                "ast_digest": self.state.ast_digest,
            }
        )
        if self.legal_set.state_fingerprint != expected_state_fingerprint:
            raise OperatorDecisionStateError(
                "legal set is stale for the exact bound compiler state "
                "fingerprint -- repair the exporter before querying any teacher"
            )
        known_ids = {action.application_id for action in self.legal_set.operator_actions}
        if len(set(self.accepted_application_ids)) != len(self.accepted_application_ids):
            raise OperatorDecisionStateError("accepted application ids must be unique")
        unknown_accepted = set(self.accepted_application_ids) - known_ids
        if unknown_accepted:
            raise OperatorDecisionStateError(
                "accepted application id is absent from the exported legal set"
            )
        if self.current_scores is not None:
            unknown_scored = set(self.current_scores) - known_ids
            if unknown_scored:
                raise OperatorDecisionStateError(
                    "scored application id is absent from the exported legal set"
                )

    @property
    def coverage(self) -> LegalSetCoverage:
        """Explicit exact-vs-approximate legal-set enumeration coverage."""
        return self.legal_set.coverage

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trace_id": self.trace_id,
            "manifest_id": self.manifest_id,
            "supervision_source": self.supervision_source.value,
            "capture_stage": self.capture_stage.value,
            "approximate": self.approximate,
            "coverage": self.coverage.value,
            "timestamp": self.timestamp,
            "state": {
                "schema": self.state.schema,
                "pack_id": self.state.pack_id,
                "state_digest": self.state.state_digest,
                "ast_digest": self.state.ast_digest,
                "source": self.state.source,
            },
            "reference_table": self.reference_table.to_dict(),
            "provenance": self.provenance.to_dict(),
            "legal_set": self.legal_set.to_dict(),
            "accepted_application_ids": list(self.accepted_application_ids),
            "current_scores": (
                dict(self.current_scores) if self.current_scores is not None else None
            ),
        }


def capture_operator_decision_state(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    provenance: ApplicationProvenanceV1,
    manifest: TeacherTraceManifest,
    supervision_source: SupervisionSource,
    trace_id: str,
    accepted_application_ids: tuple[str, ...] = (),
    current_scores: Mapping[str, float] | None = None,
    approximate: bool = False,
    ordinary_nonoperator_actions: tuple[str, ...] = (),
    max_combinations_per_operator: int = 10_000,
    max_rejection_samples_per_operator: int = 8,
) -> OperatorDecisionStateTraceV1:
    """Capture one DSH4-01 decision-state export for ``state``.

    Calls DSH3-06 ``enumerate_operator_legal_set`` directly on the supplied
    ``state`` -- no repair or fallback mutation runs anywhere in this path --
    so the returned trace is structurally captured pre-repair/pre-fallback.
    """
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=state,
        reference_table=reference_table,
        provenance=provenance,
        ordinary_nonoperator_actions=ordinary_nonoperator_actions,
        max_combinations_per_operator=max_combinations_per_operator,
        max_rejection_samples_per_operator=max_rejection_samples_per_operator,
    )
    return OperatorDecisionStateTraceV1(
        trace_id=trace_id,
        manifest_id=manifest.manifest_id,
        supervision_source=supervision_source,
        state=state,
        reference_table=reference_table,
        provenance=provenance,
        legal_set=legal_set,
        accepted_application_ids=tuple(accepted_application_ids),
        current_scores=(
            dict(current_scores) if current_scores is not None else None
        ),
        approximate=approximate,
        capture_stage=CaptureStage.PRE_REPAIR,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------- #
# Accepted-trajectory replay
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OperatorTrajectoryStepV1:
    """One accepted operator application within a trajectory."""

    state_before: OperatorStateV1
    application: OperatorApplicationV1

    def __post_init__(self) -> None:
        if not self.application.succeeded:
            raise OperatorDecisionStateError(
                "trajectory steps must be accepted (proof-bearing) applications"
            )
        if (
            self.application.before_state_digest != self.state_before.state_digest
            or self.application.before_ast_digest != self.state_before.ast_digest
        ):
            raise OperatorDecisionStateError(
                "trajectory step application does not match its declared before state"
            )


@dataclass(frozen=True)
class OperatorTrajectoryV1:
    """An ordered, chained sequence of accepted operator applications."""

    trajectory_id: str
    steps: tuple[OperatorTrajectoryStepV1, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.trajectory_id, "trajectory_id")
        if not self.steps:
            raise OperatorDecisionStateError("a trajectory requires at least one step")
        for previous, following in zip(self.steps, self.steps[1:]):
            after = previous.application
            before = following.state_before
            if (
                after.after_state_digest != before.state_digest
                or after.after_ast_digest != before.ast_digest
            ):
                raise OperatorDecisionStateError(
                    "trajectory steps are not chained: after-state does not "
                    "match the next step's before state"
                )


def build_operator_trajectory(
    trajectory_id: str,
    steps: tuple[tuple[OperatorStateV1, OperatorApplicationV1], ...],
) -> OperatorTrajectoryV1:
    """Convenience constructor from ``(state_before, application)`` pairs."""
    return OperatorTrajectoryV1(
        trajectory_id=trajectory_id,
        steps=tuple(
            OperatorTrajectoryStepV1(state_before=state_before, application=application)
            for state_before, application in steps
        ),
    )


@dataclass(frozen=True)
class OperatorReplayDivergenceV1:
    """The first step at which replay diverged from the recorded trajectory."""

    step_index: int
    recorded_application_id: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "recorded_application_id": self.recorded_application_id,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class OperatorTrajectoryReplayResultV1:
    """Outcome of replaying one accepted trajectory to a canonical result."""

    trajectory_id: str
    verified: bool
    steps_replayed: int
    canonical_final_state: OperatorStateV1 | None
    first_divergence: OperatorReplayDivergenceV1 | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "verified": self.verified,
            "steps_replayed": self.steps_replayed,
            "canonical_final_state_digest": (
                self.canonical_final_state.state_digest
                if self.canonical_final_state is not None
                else None
            ),
            "first_divergence": (
                self.first_divergence.to_dict()
                if self.first_divergence is not None
                else None
            ),
        }


def _classify_replay_error(exc: OperatorReplayError) -> str:
    return _REPLAY_ERROR_CODES.get(str(exc), _REPLAY_CODE_UNCLASSIFIED)


def replay_operator_trajectory(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    trajectory: OperatorTrajectoryV1,
) -> OperatorTrajectoryReplayResultV1:
    """Replay an accepted trajectory to a canonical verified result.

    Calls only ``OperatorLibraryV1.replay`` (the sole DSH3-02 replay
    authority) at each step. Stops and reports the first divergent action
    instead of raising, so callers can decide what to do with a partial
    replay; ``export_for_teacher_query`` is the fail-closed gate that turns a
    non-verified result into a hard error before any teacher call.
    """
    state = trajectory.steps[0].state_before
    for index, step in enumerate(trajectory.steps):
        try:
            replayed = library.replay(pack, state, step.application)
        except OperatorReplayError as exc:
            return OperatorTrajectoryReplayResultV1(
                trajectory_id=trajectory.trajectory_id,
                verified=False,
                steps_replayed=index,
                canonical_final_state=None,
                first_divergence=OperatorReplayDivergenceV1(
                    step_index=index,
                    recorded_application_id=step.application.application_id,
                    code=_classify_replay_error(exc),
                    detail=str(exc),
                ),
            )
        assert replayed.state is not None  # library.replay only returns on success
        state = replayed.state
    return OperatorTrajectoryReplayResultV1(
        trajectory_id=trajectory.trajectory_id,
        verified=True,
        steps_replayed=len(trajectory.steps),
        canonical_final_state=state,
        first_divergence=None,
    )


def export_for_teacher_query(
    trace: OperatorDecisionStateTraceV1,
    trajectory: OperatorTrajectoryV1 | None,
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
) -> dict[str, Any]:
    """Gate a decision-state trace before any teacher scoring call.

    Stop rule (DSH4-01): when ``trajectory`` is supplied, it must replay to a
    verifier-equivalent canonical result. If it does not, this raises
    ``OperatorDecisionStateError`` and returns nothing -- the caller must
    repair the exporter, never query a teacher over unverified state/action
    identity. ``trajectory is None`` exports the decision-state trace alone
    (no accepted-trajectory claim to verify).
    """
    if trajectory is not None:
        result = replay_operator_trajectory(pack=pack, library=library, trajectory=trajectory)
        if not result.verified:
            divergence = result.first_divergence
            raise OperatorDecisionStateError(
                "operator decision-state export blocked: accepted trajectory "
                f"{trajectory.trajectory_id!r} diverges at step "
                f"{divergence.step_index if divergence else '?'} "
                f"({divergence.code if divergence else 'unknown'}); "
                "repair the exporter before querying any teacher"
            )
    return trace.to_dict()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_operator_decision_state_traces(
    path: str | Path, traces: list[OperatorDecisionStateTraceV1]
) -> None:
    """Write decision-state traces as JSONL (write-only export contract).

    ``OperatorLegalSetV1`` (DSH3-06) has no ``from_dict`` -- reconstructing a
    legal action requires re-resolving opaque references against a live
    ``ReferenceTableV1``, which is a pack-bound operation, not a generic JSON
    round trip. This module keeps that same boundary rather than inventing a
    parallel deserialization path for objects the upstream contract does not
    support round-tripping either.
    """
    handle = Path(path).open("w", encoding="utf-8")
    for trace in traces:
        handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
    handle.close()
