"""Operator-policy corpus rows from collapse hard negatives (DSH3-24/SLM-399).

Turns a replay-authoritative :class:`CollapsedInstructionV1`
(``slm_training.dsl.operators.collapse``) into leak-free supervision rows:
one row per collapsed step, each carrying the SLM-397 sanitized
``OperatorPolicyInputV1`` view over a **freshly re-enumerated** live legal
set (never the planner's own recorded membership), the accepted
operator/argument rows, and — when this step participates in one of the
collapse's own adjacent-swap hard negatives — a typed ``CONFLICT`` or
``DIFFERENT_RESULT`` negative. No new synthetic corpus and no additional
hard-negative mining: this module only re-projects evidence
``collapse_conversation_trace`` already computed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from slm_training.dsl.operators.collapse import (
    CollapsedInstructionV1,
    HardNegativeOutcome,
)
from slm_training.dsl.operators.conversation import (
    ConversationTraceV1,
    OperatorAuthorityResolver,
)
from slm_training.dsl.operators.registry import OperatorApplyResultV1
from slm_training.dsl.operators.contracts import (
    _fingerprint,
    _require_digest,
    _require_identifier,
)
from slm_training.dsl.operators.legal_set import enumerate_operator_legal_set
from slm_training.models.operator_policy_view import (
    build_operator_policy_input,
    validate_no_forbidden_fields,
)

__all__ = [
    "CollapseNegativeAblationManifestV1",
    "FrozenCollapseNegativeExampleV1",
    "OperatorPolicyCorpusQualityReportV1",
    "OperatorPolicyHardNegativeV1",
    "OperatorPolicyMaterializationV1",
    "OperatorPolicyRowV1",
    "OperatorTerminationRowV1",
    "RowRejectionKind",
    "build_operator_policy_corpus",
    "build_operator_policy_rows",
    "freeze_collapse_negative_ablation",
    "materialize_operator_policy_selection",
    "build_operator_termination_rows",
]


class RowRejectionKind:
    """String constants for row-rejection reasons (not a closed enum: new
    reasons may be added without a schema bump, each is always a stable,
    lowercase, machine-readable identifier)."""

    ACCEPTED_ACTION_NOT_LIVE = "accepted_action_not_live"
    ACCEPTED_ARGUMENT_NOT_REFERENCEABLE = "accepted_argument_not_referenceable"


@dataclass(frozen=True)
class OperatorPolicyHardNegativeV1:
    """One collapse-derived adjacent-swap hard negative, re-projected onto
    this step's freshly re-enumerated live action rows."""

    outcome: HardNegativeOutcome
    alternate_operator_id: str
    alternate_action_row: int | None
    conflict_code: str | None
    observed_final_state_digest: str | None
    schema: str = "operator_policy_hard_negative/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.alternate_operator_id, "alternate_operator_id")
        if self.alternate_action_row is not None and self.alternate_action_row < 0:
            raise ValueError("alternate_action_row must be non-negative")
        if self.outcome is HardNegativeOutcome.CONFLICT:
            if not self.conflict_code or self.observed_final_state_digest is not None:
                raise ValueError("conflict hard negative evidence is incomplete")
        else:
            if (
                self.observed_final_state_digest is None
                or self.conflict_code is not None
            ):
                raise ValueError(
                    "different-result hard negative evidence is incomplete"
                )
            _require_digest(
                self.observed_final_state_digest, "observed_final_state_digest"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "outcome": self.outcome.value,
            "alternate_operator_id": self.alternate_operator_id,
            "alternate_action_row": self.alternate_action_row,
            "conflict_code": self.conflict_code,
            "observed_final_state_digest": self.observed_final_state_digest,
        }


@dataclass(frozen=True)
class FrozenCollapseNegativeExampleV1:
    """Replay-backed negative kept outside the model-facing policy input."""

    row_id: str
    source_trace_id: str
    split: str
    outcome: HardNegativeOutcome
    evidence: dict[str, Any]
    schema: str = "frozen_collapse_negative_example/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "source_trace_id": self.source_trace_id,
            "split": self.split,
            "outcome": self.outcome.value,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CollapseNegativeAblationManifestV1:
    """Frozen, split-safe input contract for SLM-405's five matched arms.

    The manifest deliberately records hard-negative outcomes as evaluator/training
    labels only.  ``OperatorPolicyInputV1`` is not copied here and remains the
    sole model-facing input boundary.
    """

    examples: tuple[FrozenCollapseNegativeExampleV1, ...]
    positive_row_ids: dict[str, tuple[str, ...]]
    ready: bool
    stop_reason: str | None
    schema: str = "collapse_negative_ablation_manifest/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": [
                "no_negatives",
                "conflict_only",
                "different_result_only",
                "equal_mix",
                "curriculum_mix",
            ],
            "examples": [example.to_dict() for example in self.examples],
            "positive_row_ids": {
                split: list(row_ids)
                for split, row_ids in sorted(self.positive_row_ids.items())
            },
            "ready": self.ready,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True)
class OperatorPolicyMaterializationV1:
    """Evaluator-only proof that a selected typed row was compiler-applied.

    This deliberately retains runtime evidence outside ``OperatorPolicyInputV1``:
    models see only the sanitized view while evaluators reconstruct the same
    fresh legal set and apply the chosen live action through the compiler.
    """

    row_id: str
    selected_action_row: int
    selected_argument_rows: tuple[tuple[str, int], ...]
    application_id: str
    result: OperatorApplyResultV1
    schema: str = "operator_policy_materialization/v1"

    def __post_init__(self) -> None:
        if not self.result.succeeded or self.result.application is None:
            raise ValueError("materialized operator policy selection did not apply")
        if self.result.application.application_id != self.application_id:
            raise ValueError("materialized application identity disagrees")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "selected_action_row": self.selected_action_row,
            "selected_argument_rows": [list(item) for item in self.selected_argument_rows],
            "application_id": self.application_id,
            "application": self.result.application.to_dict(),
            "result_ast": self.result.state.source if self.result.state is not None else None,
        }


@dataclass(frozen=True)
class OperatorPolicyRowV1:
    """One leak-free operator-policy supervision row.

    ``policy_input`` is the SLM-397 sanitized ``OperatorPolicyInputV1.
    to_dict()`` payload — recursively forbidden-field-checked at
    construction here, not just when it was originally built, so a stale
    or hand-edited row can never smuggle a forbidden field back in.
    ``accepted_action_row`` / ``accepted_argument_rows`` point into that
    same payload's row-local numbering; both are validated against it.
    """

    row_id: str
    collapse_id: str
    turn_id: str
    step_index: int
    request_id: str
    state_digest: str
    ast_digest: str
    branch_digest: str
    reference_table_fingerprint: str
    legal_set_fingerprint: str
    policy_input_digest: str
    policy_input: dict[str, Any]
    accepted_operator_id: str
    accepted_action_row: int
    accepted_argument_rows: tuple[tuple[str, int], ...]
    accepted_application_id: str
    hard_negative: OperatorPolicyHardNegativeV1 | None
    split: str
    schema: str = "operator_policy_row/v1"

    def __post_init__(self) -> None:
        _require_digest(self.collapse_id, "collapse_id")
        _require_digest(self.turn_id, "turn_id")
        _require_digest(self.state_digest, "state_digest")
        _require_digest(self.ast_digest, "ast_digest")
        _require_digest(self.branch_digest, "branch_digest")
        _require_digest(self.reference_table_fingerprint, "reference_table_fingerprint")
        _require_digest(self.legal_set_fingerprint, "legal_set_fingerprint")
        _require_digest(self.policy_input_digest, "policy_input_digest")
        _require_digest(self.accepted_application_id, "accepted_application_id")
        _require_identifier(self.accepted_operator_id, "accepted_operator_id")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.split not in {"train", "dev"}:
            raise ValueError("unsupported split")
        if self.policy_input_digest != _fingerprint(self.policy_input):
            raise ValueError("policy_input_digest does not match policy_input content")
        validate_no_forbidden_fields(self.policy_input)

        action_rows = self.policy_input.get("action_rows", [])
        if not 0 <= self.accepted_action_row < len(action_rows):
            raise ValueError("accepted_action_row is outside policy_input.action_rows")
        accepted_action = action_rows[self.accepted_action_row]
        if accepted_action["operator_id"] != self.accepted_operator_id:
            raise ValueError("accepted_action_row does not name accepted_operator_id")
        slots_by_id = {
            slot["slot_id"]: slot for slot in accepted_action["argument_slots"]
        }
        n_refs = len(self.policy_input.get("reference_rows", []))
        for slot_id, row in self.accepted_argument_rows:
            slot = slots_by_id.get(slot_id)
            if slot is None:
                raise ValueError(
                    f"accepted argument names an unknown slot: {slot_id!r}"
                )
            if row not in slot["candidate_rows"]:
                raise ValueError(
                    "accepted argument row is not a live candidate for its slot"
                )
            if not 0 <= row < n_refs:
                raise ValueError(
                    "accepted argument row is outside policy_input.reference_rows"
                )
        if (
            self.hard_negative is not None
            and self.hard_negative.alternate_action_row is not None
        ):
            if not 0 <= self.hard_negative.alternate_action_row < len(action_rows):
                raise ValueError(
                    "hard negative alternate_action_row is outside action_rows"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "collapse_id": self.collapse_id,
            "turn_id": self.turn_id,
            "step_index": self.step_index,
            "request_id": self.request_id,
            "state_digest": self.state_digest,
            "ast_digest": self.ast_digest,
            "branch_digest": self.branch_digest,
            "reference_table_fingerprint": self.reference_table_fingerprint,
            "legal_set_fingerprint": self.legal_set_fingerprint,
            "policy_input_digest": self.policy_input_digest,
            "policy_input": self.policy_input,
            "accepted_operator_id": self.accepted_operator_id,
            "accepted_action_row": self.accepted_action_row,
            "accepted_argument_rows": [
                list(item) for item in self.accepted_argument_rows
            ],
            "accepted_application_id": self.accepted_application_id,
            "hard_negative": (
                self.hard_negative.to_dict() if self.hard_negative is not None else None
            ),
            "split": self.split,
        }


@dataclass(frozen=True)
class OperatorTerminationRowV1:
    """Replayable terminal/nonterminal control-plane label over a fresh view."""

    row_id: str
    state_digest: str
    legal_set_fingerprint: str
    policy_input: dict[str, Any]
    policy_input_digest: str
    stop: bool
    remaining_distance: int
    proof_ids: tuple[str, ...]
    split: str
    schema: str = "operator_termination_row/v1"

    def __post_init__(self) -> None:
        _require_digest(self.state_digest, "state_digest")
        _require_digest(self.legal_set_fingerprint, "legal_set_fingerprint")
        _require_digest(self.policy_input_digest, "policy_input_digest")
        if self.remaining_distance < 0:
            raise ValueError("remaining_distance must be non-negative")
        if self.split not in {"train", "dev"}:
            raise ValueError("unsupported split")
        if self.policy_input_digest != _fingerprint(self.policy_input):
            raise ValueError("policy_input_digest does not match policy_input")
        validate_no_forbidden_fields(self.policy_input)
        if self.stop and self.policy_input.get("coverage") != "complete":
            raise ValueError("PARTIAL coverage cannot masquerade as STOP")

    def model_input(self) -> dict[str, Any]:
        """The sanitized live view only; distance/proofs are training diagnostics."""
        return self.policy_input


def _accepted_action_row(
    legal_set, operator_id: str, application_id: str
) -> int | None:
    for row, entry in enumerate(legal_set.entries):
        if entry.operator_id != operator_id:
            continue
        if any(
            action.application_id == application_id for action in entry.legal_actions
        ):
            return row
        return None
    return None


def _action_row_for_operator(legal_set, operator_id: str) -> int | None:
    for row, entry in enumerate(legal_set.entries):
        if entry.operator_id == operator_id:
            return row
    return None


def build_operator_policy_rows(
    *,
    trace: ConversationTraceV1,
    collapse: CollapsedInstructionV1,
    authority_resolver: OperatorAuthorityResolver,
    split: str = "train",
    max_combinations_per_operator: int = 64,
) -> tuple[tuple[OperatorPolicyRowV1, ...], tuple[dict[str, Any], ...]]:
    """Build one row per collapsed step. Returns ``(rows, rejected)``.

    Re-enumerates live legal-set membership fresh at each step's actual
    trace state — it never copies membership from ``collapse`` itself,
    matching the "re-enumerate current live membership rather than copying
    planner membership" requirement. A step whose recorded application is
    absent from that fresh live set is rejected, not silently forced in.
    """
    if max_combinations_per_operator <= 0:
        raise ValueError("max_combinations_per_operator must be positive")
    if collapse.turn_ids != tuple(turn.turn_id for turn in trace.turns):
        raise ValueError("collapse turn lineage differs from the source trace")
    negatives_by_left_step = {
        negative.swapped_step_indices[0]: negative
        for negative in collapse.hard_negatives
    }
    rows: list[OperatorPolicyRowV1] = []
    rejected: list[dict[str, Any]] = []
    for step_index, (operator_id, application) in enumerate(
        zip(collapse.operator_ids, collapse.applications, strict=True)
    ):
        turn = trace.turns[step_index]
        input_node = trace.node(turn.input_state_id)
        state = input_node.state
        reference_table = input_node.reference_table
        turn_pack, library = authority_resolver(input_node)

        legal_set = enumerate_operator_legal_set(
            pack=turn_pack,
            library=library,
            state=state,
            reference_table=reference_table,
            provenance=application.provenance,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        accepted_action_row = _accepted_action_row(
            legal_set, operator_id, application.application_id
        )
        if accepted_action_row is None:
            rejected.append(
                {
                    "collapse_id": collapse.collapse_id,
                    "step_index": step_index,
                    "reason": RowRejectionKind.ACCEPTED_ACTION_NOT_LIVE,
                }
            )
            continue

        row_by_ref = {
            (entry.ref.KIND, entry.ref.opaque_id): row
            for row, entry in enumerate(reference_table.entries)
        }
        try:
            accepted_argument_rows = tuple(
                (
                    argument.slot_id,
                    row_by_ref[(argument.value.KIND, argument.value.opaque_id)],
                )
                for argument in application.arguments
            )
        except KeyError:
            rejected.append(
                {
                    "collapse_id": collapse.collapse_id,
                    "step_index": step_index,
                    "reason": RowRejectionKind.ACCEPTED_ARGUMENT_NOT_REFERENCEABLE,
                }
            )
            continue

        policy_input = build_operator_policy_input(reference_table, legal_set, library)
        reference_row_map, action_row_map = policy_input.canonical_row_maps()
        policy_input_dict = policy_input.to_dict()

        hard_negative = None
        negative = negatives_by_left_step.get(step_index)
        if negative is not None:
            alternate_operator_id = collapse.operator_ids[step_index + 1]
            hard_negative = OperatorPolicyHardNegativeV1(
                outcome=negative.outcome,
                alternate_operator_id=alternate_operator_id,
                alternate_action_row=(
                    None
                    if (alternate_row := _action_row_for_operator(
                        legal_set, alternate_operator_id
                    ))
                    is None
                    else action_row_map[alternate_row]
                ),
                conflict_code=negative.conflict_code,
                observed_final_state_digest=negative.observed_final_state_digest,
            )

        identity = {
            "schema": "operator_policy_row_identity/v1",
            "collapse_id": collapse.collapse_id,
            "step_index": step_index,
            "state_digest": state.state_digest,
            "reference_table_fingerprint": reference_table.fingerprint,
        }
        rows.append(
            OperatorPolicyRowV1(
                row_id=f"row_{_fingerprint(identity)[:24]}",
                collapse_id=collapse.collapse_id,
                turn_id=turn.turn_id,
                step_index=step_index,
                request_id=reference_table.request_id,
                state_digest=state.state_digest,
                ast_digest=state.ast_digest,
                branch_digest=reference_table.branch_digest,
                reference_table_fingerprint=reference_table.fingerprint,
                legal_set_fingerprint=legal_set.fingerprint,
                policy_input_digest=_fingerprint(policy_input_dict),
                policy_input=policy_input_dict,
                accepted_operator_id=operator_id,
                accepted_action_row=action_row_map[accepted_action_row],
                accepted_argument_rows=tuple(
                    (slot_id, reference_row_map[row])
                    for slot_id, row in accepted_argument_rows
                ),
                accepted_application_id=application.application_id,
                hard_negative=hard_negative,
                split=split,
            )
        )
    return tuple(rows), tuple(rejected)


def freeze_collapse_negative_ablation(
    rows: tuple[OperatorPolicyRowV1, ...] | list[OperatorPolicyRowV1],
) -> CollapseNegativeAblationManifestV1:
    """Freeze replay-proven negative labels for a matched SLM-405 ablation.

    The two comparison types must appear in both train and dev, and a source
    trace may belong to exactly one split.  Until then the manifest is an
    explicit stop record, not a permissive partial experiment.
    """
    source_splits: dict[str, str] = {}
    positives: dict[str, list[str]] = {}
    examples: list[FrozenCollapseNegativeExampleV1] = []
    for row in sorted(rows, key=lambda item: (item.split, item.collapse_id, item.row_id)):
        previous = source_splits.setdefault(row.collapse_id, row.split)
        if previous != row.split:
            raise ValueError("source trace appears in more than one split")
        positives.setdefault(row.split, []).append(row.row_id)
        if row.hard_negative is None:
            continue
        examples.append(
            FrozenCollapseNegativeExampleV1(
                row_id=row.row_id,
                source_trace_id=row.collapse_id,
                split=row.split,
                outcome=row.hard_negative.outcome,
                evidence=row.hard_negative.to_dict(),
            )
        )
    counts = {
        split: {
            outcome: sum(
                example.split == split and example.outcome.value == outcome
                for example in examples
            )
            for outcome in ("conflict", "different_result")
        }
        for split in ("train", "dev")
    }
    missing = [
        f"{split}:{outcome}"
        for split, values in counts.items()
        for outcome, count in values.items()
        if count == 0
    ]
    return CollapseNegativeAblationManifestV1(
        examples=tuple(examples),
        positive_row_ids={
            split: tuple(row_ids) for split, row_ids in sorted(positives.items())
        },
        ready=not missing,
        stop_reason=(
            None
            if not missing
            else "missing replay-verified matched negative strata: " + ", ".join(missing)
        ),
    )


def materialize_operator_policy_selection(
    *,
    trace: ConversationTraceV1,
    row: OperatorPolicyRowV1,
    authority_resolver: OperatorAuthorityResolver,
    selected_action_row: int,
    selected_argument_rows: tuple[tuple[str, int], ...],
    max_combinations_per_operator: int = 64,
) -> OperatorPolicyMaterializationV1:
    """Replay one selected sanitized row through its fresh compiler legal set.

    Selection indices are row-local policy output. They are translated back to
    runtime refs only here, after exact view/fingerprint validation, and then
    applied through ``OperatorLibraryV1.apply``. This is the single boundary
    between learned selection and CAP2 replay evidence.
    """
    if max_combinations_per_operator <= 0:
        raise ValueError("max_combinations_per_operator must be positive")
    if not 0 <= selected_action_row < len(row.policy_input["action_rows"]):
        raise ValueError("selected action row is outside policy input")
    try:
        turn = next(turn for turn in trace.turns if turn.turn_id == row.turn_id)
    except StopIteration as exc:
        raise ValueError("policy row turn is absent from trace") from exc
    if turn.application is None:
        raise ValueError("policy row turn has no operator application")
    node = trace.node(turn.input_state_id)
    pack, library = authority_resolver(node)
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=node.state,
        reference_table=node.reference_table,
        provenance=turn.application.provenance,
        max_combinations_per_operator=max_combinations_per_operator,
    )
    policy_input = build_operator_policy_input(node.reference_table, legal_set, library)
    if legal_set.fingerprint != row.legal_set_fingerprint:
        raise ValueError("policy row legal set is stale")
    if policy_input.to_dict() != row.policy_input:
        raise ValueError("policy row view differs from fresh legal set")
    reference_map, action_map = policy_input.canonical_row_maps()
    live_action_row = next(
        (live for live, canonical in action_map.items() if canonical == selected_action_row),
        None,
    )
    if live_action_row is None:
        raise ValueError("selected action row is not live")
    entry = legal_set.entries[live_action_row]
    selected = tuple(sorted(selected_argument_rows))
    if len(set(selected)) != len(selected):
        raise ValueError("selected argument rows are duplicated")
    candidate = next(
        (
            action
            for action in entry.legal_actions
            if tuple(
                sorted(
                    (
                        argument.slot_id,
                        reference_map[
                            next(
                                index
                                for index, item in enumerate(node.reference_table.entries)
                                if item.ref == argument.value
                            )
                        ],
                    )
                    for argument in action.arguments
                )
            )
            == selected
        ),
        None,
    )
    if candidate is None:
        raise ValueError("selected arguments are not a live legal action")
    result = library.apply(
        pack,
        node.state,
        candidate.operator_id,
        candidate.arguments,
        turn.application.provenance,
    )
    if not result.succeeded or result.application is None:
        raise ValueError("compiler rejected selected live legal action")
    return OperatorPolicyMaterializationV1(
        row_id=row.row_id,
        selected_action_row=selected_action_row,
        selected_argument_rows=selected,
        application_id=candidate.application_id,
        result=result,
    )


def build_operator_termination_rows(
    *,
    trace: ConversationTraceV1,
    authority_resolver: OperatorAuthorityResolver,
    split: str = "train",
    max_combinations_per_operator: int = 64,
) -> tuple[OperatorTerminationRowV1, ...]:
    """Re-enumerate every trace state and label only its replay endpoint STOP.

    The endpoint may retain legal actions; STOP is a control-plane target, not
    evidence that the compiler's action domain is empty.
    """
    if max_combinations_per_operator <= 0:
        raise ValueError("max_combinations_per_operator must be positive")
    states = [trace.node(turn.input_state_id) for turn in trace.turns]
    states.append(trace.current)
    rows: list[OperatorTerminationRowV1] = []
    for index, node in enumerate(states):
        pack, library = authority_resolver(node)
        source_digest = hashlib.sha256(node.state.source.encode("utf-8")).hexdigest()
        provenance = replace(
            trace.root_provenance,
            source_artifact_digest=source_digest,
            request_id=node.reference_table.request_id,
        )
        legal_set = enumerate_operator_legal_set(
            pack=pack,
            library=library,
            state=node.state,
            reference_table=node.reference_table,
            provenance=provenance,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        policy_input = build_operator_policy_input(
            node.reference_table, legal_set, library
        ).to_dict()
        stop = node.state_id == trace.current_state_id
        if stop and legal_set.coverage is not legal_set.coverage.COMPLETE:
            raise ValueError("terminal state has partial legal-set coverage")
        identity = {
            "schema": "operator_termination_row_identity/v1",
            "state": node.state.state_digest,
        }
        rows.append(
            OperatorTerminationRowV1(
                row_id=f"term_{_fingerprint(identity)[:24]}",
                state_digest=node.state.state_digest,
                legal_set_fingerprint=legal_set.fingerprint,
                policy_input=policy_input,
                policy_input_digest=_fingerprint(policy_input),
                stop=stop,
                remaining_distance=len(states) - index - 1,
                proof_ids=tuple(
                    action.application_id for action in legal_set.operator_actions
                ),
                split=split,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class OperatorPolicyCorpusQualityReportV1:
    """Aggregate denominators over one or more collapses' rows. Wiring
    evidence only — see the DSH3-24 design doc's stop-rule disposition."""

    total_steps: int
    accepted_rows: int
    rejected_rows: int
    rejection_counts: dict[str, int]
    hard_negative_counts: dict[str, int]
    positive_only_rows: int
    schema: str = "operator_policy_corpus_quality_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "total_steps": self.total_steps,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
            "hard_negative_counts": dict(sorted(self.hard_negative_counts.items())),
            "positive_only_rows": self.positive_only_rows,
        }


def build_operator_policy_corpus(
    *,
    trace: ConversationTraceV1,
    collapse: CollapsedInstructionV1,
    authority_resolver: OperatorAuthorityResolver,
    split: str = "train",
    max_combinations_per_operator: int = 64,
) -> tuple[tuple[OperatorPolicyRowV1, ...], OperatorPolicyCorpusQualityReportV1]:
    """``build_operator_policy_rows`` plus its aggregate quality report."""
    rows, rejected = build_operator_policy_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        split=split,
        max_combinations_per_operator=max_combinations_per_operator,
    )
    rejection_counts: dict[str, int] = {}
    for entry in rejected:
        reason = str(entry["reason"])
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    hard_negative_counts: dict[str, int] = {}
    positive_only = 0
    for row in rows:
        if row.hard_negative is None:
            positive_only += 1
        else:
            outcome = row.hard_negative.outcome.value
            hard_negative_counts[outcome] = hard_negative_counts.get(outcome, 0) + 1
    report = OperatorPolicyCorpusQualityReportV1(
        total_steps=len(collapse.applications),
        accepted_rows=len(rows),
        rejected_rows=len(rejected),
        rejection_counts=rejection_counts,
        hard_negative_counts=hard_negative_counts,
        positive_only_rows=positive_only,
    )
    return rows, report
