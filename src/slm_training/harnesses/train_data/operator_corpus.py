"""Verified symbolic operator QA artifacts derived from admitted train roots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from slm_training.dsl.operators import (
    ApplicationProvenanceV1,
    ConversationTraceV1,
    LegalOperatorActionV1,
    OperatorLibraryV1,
    OperatorPreferenceSequenceV1,
    OperatorPreferenceStepV1,
    OperatorStateV1,
    branch_fingerprint,
    build_openui_local_operator_context,
    build_openui_local_operator_library,
    collapse_conversation_trace,
    create_conversation_trace,
    enumerate_operator_legal_set,
    fork_conversation,
    replay_conversation_trace,
    append_operator_turn,
)
from slm_training.dsl.operators.contracts import _fingerprint, _require_digest
from slm_training.dsl.pack import DslPack, get_pack
from slm_training.dsl.schema import ExampleRecord


class OperatorExampleKind(str, Enum):
    SINGLE_TURN = "single_turn"
    NEXT_TURN = "next_turn"
    SIBLING_FORK = "sibling_fork"


class OperatorTargetView(str, Enum):
    OPERATOR_ONLY = "operator_only"
    RESULT_AST_ONLY = "result_ast_only"
    DUAL = "dual"
    HISTORY_ONLY = "history_only"


@dataclass(frozen=True)
class OperatorCorpusConfig:
    max_roots: int = 8
    actions_per_state: int = 4
    max_combinations_per_operator: int = 64
    sibling_forks: bool = True

    def __post_init__(self) -> None:
        if (
            self.max_roots <= 0
            or self.actions_per_state <= 0
            or self.max_combinations_per_operator <= 0
        ):
            raise ValueError("operator corpus bounds must be positive")


@dataclass(frozen=True)
class CollapsedOperatorExampleV1:
    example_id: str
    source_record_id: str
    question: dict[str, Any]
    answer: dict[str, Any]
    collapse: dict[str, Any]
    conversation_trace: dict[str, Any]
    nl_available: bool = False
    nl_unavailable_reason: str = "CERT_CAP1_unavailable"
    schema: str = "symbolic_collapsed_operator_example/v1"

    def __post_init__(self) -> None:
        _require_digest(self.example_id, "example_id")
        if not self.source_record_id:
            raise ValueError("collapsed operator source identity is required")
        if set(self.question) != {"opcode", "state_ast", "required_order"}:
            raise ValueError("collapsed operator question keys are not closed")
        if self.question["opcode"] != "APPLY_OPERATOR_SEQUENCE":
            raise ValueError("collapsed operator opcode is not symbolic")
        if set(self.answer) != {"operators", "result_ast"}:
            raise ValueError("collapsed operator answer keys are not closed")
        applications = self.collapse.get("applications", ())
        expected_order = list(range(len(applications)))
        if len(applications) < 2:
            raise ValueError("collapsed operator example requires multiple turns")
        if self.question["required_order"] != expected_order:
            raise ValueError("collapsed operator order must remain explicit")
        if len(self.answer["operators"]) != len(applications):
            raise ValueError("collapsed operator target sequence length drifted")
        if self.collapse.get("required_order") != expected_order:
            raise ValueError("collapse artifact order disagrees with the question")
        if self.collapse.get("final_state_id") != self.conversation_trace.get(
            "current_state_id"
        ):
            raise ValueError("collapsed operator final state is not trace-authoritative")
        current_id = self.conversation_trace.get("current_state_id")
        current_nodes = [
            node
            for node in self.conversation_trace.get("state_nodes", ())
            if node.get("state_id") == current_id
        ]
        if (
            len(current_nodes) != 1
            or self.answer["result_ast"] != current_nodes[0]["state"]["source"]
        ):
            raise ValueError("collapsed operator result AST drifted from the trace")
        if self.nl_available:
            raise ValueError("NL collapse requires the unavailable CERT_CAP1 path")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "example_id": self.example_id,
            "source_record_id": self.source_record_id,
            "question": self.question,
            "answer": self.answer,
            "collapse": self.collapse,
            "conversation_trace": self.conversation_trace,
            "nl_available": self.nl_available,
            "nl_unavailable_reason": self.nl_unavailable_reason,
        }


@dataclass(frozen=True)
class SymbolicOperatorExampleV1:
    example_id: str
    kind: OperatorExampleKind
    target_view: OperatorTargetView
    question: dict[str, Any]
    answer: dict[str, Any]
    source_record_id: str
    semantic_family: str
    operator_family: str
    argument_kinds: tuple[str, ...]
    state_size: int
    scope_complexity: int
    outcome: str
    before_ast: str
    after_ast: str
    legal_set_fingerprint: str
    legal_action: dict[str, Any] | None
    application: dict[str, Any] | None
    canonical_preference: dict[str, Any] | None
    conversation_trace: dict[str, Any]
    schema: str = "symbolic_operator_example/v1"

    def __post_init__(self) -> None:
        if not self.example_id or not self.source_record_id:
            raise ValueError("operator example identity is required")
        if self.state_size <= 0 or self.scope_complexity <= 0:
            raise ValueError("operator state strata must be positive")
        if self.outcome not in {"success", "fork"}:
            raise ValueError("operator example outcome is not symbolic")
        if self.kind is OperatorExampleKind.SIBLING_FORK:
            if self.target_view is not OperatorTargetView.HISTORY_ONLY:
                raise ValueError("fork examples require the history-only view")
            if any(
                value is not None
                for value in (
                    self.legal_action,
                    self.application,
                    self.canonical_preference,
                )
            ):
                raise ValueError("fork examples cannot claim an AST application")
        elif (
            self.target_view is OperatorTargetView.HISTORY_ONLY
            or self.legal_action is None
            or self.application is None
            or self.canonical_preference is None
        ):
            raise ValueError("operator turns require action/application evidence")
        expected_question = {
            "opcode",
            "view",
            "state_ast",
            "legal_set_fingerprint",
            "trace_fingerprint",
        }
        if set(self.question) != expected_question:
            raise ValueError("symbolic question keys are not closed")
        allowed_answers = {
            OperatorTargetView.OPERATOR_ONLY: {"operator"},
            OperatorTargetView.RESULT_AST_ONLY: {"result_ast"},
            OperatorTargetView.DUAL: {"operator", "result_ast"},
            OperatorTargetView.HISTORY_ONLY: {
                "operation",
                "state_id",
                "branch_digest",
            },
        }
        if set(self.answer) != allowed_answers[self.target_view]:
            raise ValueError("symbolic answer keys do not match its target view")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "example_id": self.example_id,
            "kind": self.kind.value,
            "target_view": self.target_view.value,
            "question": self.question,
            "answer": self.answer,
            "source_record_id": self.source_record_id,
            "semantic_family": self.semantic_family,
            "operator_family": self.operator_family,
            "argument_kinds": list(self.argument_kinds),
            "state_size": self.state_size,
            "scope_complexity": self.scope_complexity,
            "outcome": self.outcome,
            "before_ast": self.before_ast,
            "after_ast": self.after_ast,
            "legal_set_fingerprint": self.legal_set_fingerprint,
            "legal_action": self.legal_action,
            "application": self.application,
            "canonical_preference": self.canonical_preference,
            "conversation_trace": self.conversation_trace,
        }


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _walk_elements(value: Any, depth: int = 1) -> Iterable[tuple[dict, int]]:
    if isinstance(value, dict):
        if value.get("type") == "element":
            yield value, depth
        for child in value.values():
            yield from _walk_elements(child, depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_elements(child, depth + 1)


def _state_strata(pack: DslPack, source: str) -> tuple[int, int, tuple[dict, ...]]:
    parsed = pack.backend.parse(source)
    elements = tuple(_walk_elements(getattr(parsed, "root", None)))
    if not elements:
        raise ValueError("operator corpus source has no AST elements")
    return (
        len(elements),
        max(depth for _, depth in elements),
        tuple(value for value, _ in elements[:2]),
    )


def _provenance(
    pack: DslPack, state: OperatorStateV1, request_id: str
) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id=pack.pack_id,
        compiler_id="openui.symbolic_operator_corpus",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id=request_id,
    )


def _authority(
    base_pack: DslPack,
    state: OperatorStateV1,
    *,
    request_id: str,
    branch_digest: str,
    seed: int,
    templates: tuple[dict, ...],
    values: tuple[str, ...],
) -> tuple[DslPack, OperatorLibraryV1, Any]:
    context = build_openui_local_operator_context(
        base_pack,
        state,
        request_id=request_id,
        branch_digest=branch_digest,
        seed=seed,
        templates=templates,
        values=values,
    )
    library = build_openui_local_operator_library(context)
    return replace(base_pack, operator_library=library), library, context


def _balanced_actions(
    actions: tuple[LegalOperatorActionV1, ...], limit: int
) -> tuple[LegalOperatorActionV1, ...]:
    groups: dict[tuple[str, tuple[str, ...]], list[LegalOperatorActionV1]] = (
        defaultdict(list)
    )
    for action in actions:
        groups[
            (
                action.operator_id,
                tuple(argument.value.KIND.value for argument in action.arguments),
            )
        ].append(action)
    for values in groups.values():
        values.sort(key=lambda item: item.semantic_id)
    selected: list[LegalOperatorActionV1] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop(0))
    return tuple(selected)


def _answer(
    view: OperatorTargetView, action: str, result_ast: str
) -> dict[str, str]:
    if view is OperatorTargetView.OPERATOR_ONLY:
        return {"operator": action}
    if view is OperatorTargetView.RESULT_AST_ONLY:
        return {"result_ast": result_ast}
    return {"operator": action, "result_ast": result_ast}


def _preference(
    library: OperatorLibraryV1,
    action: LegalOperatorActionV1,
    application,
    state: OperatorStateV1,
) -> OperatorPreferenceSequenceV1:
    step = OperatorPreferenceStepV1.from_application(
        declaration=library.lookup(action.operator_id),
        semantic_action_id=action.semantic_id,
        application=application,
    )
    return OperatorPreferenceSequenceV1(
        initial_state_fingerprint=state.state_digest,
        initial_ast_fingerprint=state.ast_digest,
        steps=(step,),
    )


def _turn_examples(
    *,
    source_record: ExampleRecord,
    kind: OperatorExampleKind,
    state: OperatorStateV1,
    output_state: OperatorStateV1,
    state_size: int,
    scope_complexity: int,
    legal_set,
    action: LegalOperatorActionV1,
    application,
    preference: OperatorPreferenceSequenceV1,
    input_trace: ConversationTraceV1,
    output_trace: ConversationTraceV1,
) -> list[SymbolicOperatorExampleV1]:
    semantic_family = str(
        (source_record.meta or {}).get("program_family_id")
        or (source_record.meta or {}).get("source_family")
        or source_record.source
    )
    base_question = {
        "opcode": (
            "APPLY_OPERATOR"
            if kind is OperatorExampleKind.SINGLE_TURN
            else "NEXT_OPERATOR"
        ),
        "view": None,
        "state_ast": state.source,
        "legal_set_fingerprint": legal_set.fingerprint,
        "trace_fingerprint": input_trace.fingerprint,
    }
    out = []
    for view in (
        OperatorTargetView.OPERATOR_ONLY,
        OperatorTargetView.RESULT_AST_ONLY,
        OperatorTargetView.DUAL,
    ):
        question = {**base_question, "view": view.value}
        example_id = _fingerprint(
            {
                "schema": "symbolic_operator_example_id/v1",
                "kind": kind.value,
                "view": view.value,
                "source_record_id": source_record.id,
                "application_id": application.application_id,
                "trace_fingerprint": output_trace.fingerprint,
            }
        )
        out.append(
            SymbolicOperatorExampleV1(
                example_id=example_id,
                kind=kind,
                target_view=view,
                question=question,
                answer=_answer(view, action.serialized, output_state.source),
                source_record_id=source_record.id,
                semantic_family=semantic_family,
                operator_family=action.operator_id,
                argument_kinds=tuple(
                    argument.value.KIND.value for argument in action.arguments
                ),
                state_size=state_size,
                scope_complexity=scope_complexity,
                outcome="success",
                before_ast=state.source,
                after_ast=output_state.source,
                legal_set_fingerprint=legal_set.fingerprint,
                legal_action=action.to_dict(),
                application=application.to_dict(),
                canonical_preference=preference.to_dict(),
                conversation_trace=output_trace.to_dict(),
            )
        )
    return out


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def build_symbolic_operator_corpus(
    *,
    records: Iterable[ExampleRecord],
    output_dir: Path,
    version: str,
    version_stamp: dict[str, Any],
    config: OperatorCorpusConfig,
) -> dict[str, Any]:
    """Build only after every generated transition and trace replays exactly."""
    base_pack = get_pack("openui")
    examples: list[SymbolicOperatorExampleV1] = []
    collapsed_records: list[CollapsedOperatorExampleV1] = []
    legal_sets: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    legal_successes = 0
    rejected_combinations = 0

    def record_legal_set(source_record_id: str, phase: str, legal_set) -> None:
        nonlocal legal_successes, rejected_combinations
        legal_sets.append(
            {
                "source_record_id": source_record_id,
                "phase": phase,
                "legal_set": legal_set.to_dict(),
            }
        )
        for entry in legal_set.entries:
            legal_successes += len(entry.legal_actions)
            rejected_combinations += sum(
                count for _, count in entry.rejection_counts
            )
            if entry.verdict.value != "supported" or entry.rejection_counts:
                gaps.append(
                    {
                        "source_record_id": source_record_id,
                        "phase": phase,
                        "operator_id": entry.operator_id,
                        "verdict": entry.verdict.value,
                        "coverage": entry.coverage.value,
                        "rejection_counts": dict(entry.rejection_counts),
                        "rejection_samples": [
                            sample.to_dict()
                            for sample in entry.rejection_samples
                        ],
                        "evaluated_combinations": entry.evaluated_combinations,
                        "total_combinations": entry.total_combinations,
                    }
                )

    roots = [
        record
        for record in sorted(records, key=lambda item: item.id)
        if record.target_kind == "document"
    ][: config.max_roots]
    if not roots:
        raise ValueError("operator corpus requires an admitted document root")
    for root_index, record in enumerate(roots):
        state = OperatorStateV1.from_source(base_pack, record.openui)
        state_size, scope_complexity, templates = _state_strata(
            base_pack, state.source
        )
        request_id = f"operator-{_sha(record.id)[:20]}"
        branch = branch_fingerprint(state.state_digest, _sha(record.id))
        values = tuple(
            sorted(
                {
                    "clear",
                    "column",
                    "outline",
                    "row",
                    *(record.placeholders or ()),
                }
            )
        )
        pack, library, context = _authority(
            base_pack,
            state,
            request_id=request_id,
            branch_digest=branch,
            seed=root_index,
            templates=templates,
            values=values,
        )
        provenance = _provenance(pack, state, request_id)
        legal_set = enumerate_operator_legal_set(
            pack=pack,
            library=library,
            state=state,
            reference_table=context.reference_table,
            provenance=provenance,
            max_combinations_per_operator=config.max_combinations_per_operator,
        )
        record_legal_set(record.id, "single_turn", legal_set)
        trace = create_conversation_trace(
            pack=pack,
            root_state=state,
            root_reference_table=context.reference_table,
            provenance=provenance,
        )
        authorities = {trace.current_state_id: (pack, library)}
        selected = _balanced_actions(
            legal_set.operator_actions, config.actions_per_state
        )
        first_trace: ConversationTraceV1 | None = None
        first_output_authority = None
        first_action: LegalOperatorActionV1 | None = None
        for action_index, action in enumerate(selected):
            result = library.apply(
                pack,
                state,
                action.operator_id,
                action.arguments,
                provenance,
            )
            if not result.succeeded or result.state is None:
                raise ValueError("legal operator action failed corpus apply")
            replayed = library.replay(pack, state, result.application)
            if replayed.state != result.state:
                raise ValueError("operator corpus application replay drifted")
            output_pack, output_library, output_context = _authority(
                base_pack,
                result.state,
                request_id=request_id,
                branch_digest=branch,
                seed=10_000 + root_index * 100 + action_index,
                templates=templates,
                values=values,
            )
            action_trace = append_operator_turn(
                trace,
                pack=pack,
                library=library,
                application=result.application,
                output_reference_table=output_context.reference_table,
            )
            if replay_conversation_trace(
                pack=base_pack,
                authority_resolver=lambda node: authorities[node.state_id],
                trace=action_trace,
            ) != action_trace.current:
                raise ValueError("operator corpus single-turn trace drifted")
            preference = _preference(
                library, action, result.application, state
            )
            examples.extend(
                _turn_examples(
                    source_record=record,
                    kind=OperatorExampleKind.SINGLE_TURN,
                    state=state,
                    output_state=result.state,
                    state_size=state_size,
                    scope_complexity=scope_complexity,
                    legal_set=legal_set,
                    action=action,
                    application=result.application,
                    preference=preference,
                    input_trace=trace,
                    output_trace=action_trace,
                )
            )
            if first_trace is None:
                first_trace = action_trace
                first_action = action
                first_output_authority = (
                    output_pack,
                    output_library,
                    output_context,
                    result.state,
                )

        if (
            first_trace is not None
            and first_output_authority is not None
            and first_action is not None
        ):
            output_pack, output_library, output_context, output_state = (
                first_output_authority
            )
            authorities[first_trace.current_state_id] = (
                output_pack,
                output_library,
            )
            next_provenance = _provenance(
                output_pack, output_state, request_id
            )
            next_legal_set = enumerate_operator_legal_set(
                pack=output_pack,
                library=output_library,
                state=output_state,
                reference_table=output_context.reference_table,
                provenance=next_provenance,
                max_combinations_per_operator=config.max_combinations_per_operator,
            )
            record_legal_set(record.id, "next_turn", next_legal_set)
            next_actions = _balanced_actions(
                next_legal_set.operator_actions, 1
            )
            if next_actions:
                next_action = next_actions[0]
                next_result = output_library.apply(
                    output_pack,
                    output_state,
                    next_action.operator_id,
                    next_action.arguments,
                    next_provenance,
                )
                if not next_result.succeeded or next_result.state is None:
                    raise ValueError("next legal operator failed corpus apply")
                next_pack, next_library, next_context = _authority(
                    base_pack,
                    next_result.state,
                    request_id=request_id,
                    branch_digest=branch,
                    seed=20_000 + root_index,
                    templates=templates,
                    values=values,
                )
                next_trace = append_operator_turn(
                    first_trace,
                    pack=output_pack,
                    library=output_library,
                    application=next_result.application,
                    output_reference_table=next_context.reference_table,
                )
                authorities[next_trace.current_state_id] = (
                    next_pack,
                    next_library,
                )
                if replay_conversation_trace(
                    pack=base_pack,
                    authority_resolver=lambda node: authorities[node.state_id],
                    trace=next_trace,
                ) != next_trace.current:
                    raise ValueError("operator corpus multi-turn trace drifted")
                collapse_decision = collapse_conversation_trace(
                    pack=base_pack,
                    authority_resolver=lambda node: authorities[node.state_id],
                    trace=next_trace,
                )
                if collapse_decision.collapse is None:
                    raise ValueError(
                        "verified multi-turn trace failed symbolic collapse"
                    )
                collapsed = collapse_decision.collapse
                collapsed_records.append(
                    CollapsedOperatorExampleV1(
                        example_id=_fingerprint(
                            {
                                "schema": "symbolic_collapsed_operator_example_id/v1",
                                "source_record_id": record.id,
                                "collapse_id": collapsed.collapse_id,
                            }
                        ),
                        source_record_id=record.id,
                        question={
                            "opcode": "APPLY_OPERATOR_SEQUENCE",
                            "state_ast": state.source,
                            "required_order": list(
                                range(len(collapsed.applications))
                            ),
                        },
                        answer={
                            "operators": [
                                first_action.serialized,
                                next_action.serialized,
                            ],
                            "result_ast": next_result.state.source,
                        },
                        collapse=collapsed.to_dict(),
                        conversation_trace=next_trace.to_dict(),
                    )
                )
                next_state_size, next_scope_complexity, _ = _state_strata(
                    base_pack, output_state.source
                )
                examples.extend(
                    _turn_examples(
                        source_record=record,
                        kind=OperatorExampleKind.NEXT_TURN,
                        state=output_state,
                        output_state=next_result.state,
                        state_size=next_state_size,
                        scope_complexity=next_scope_complexity,
                        legal_set=next_legal_set,
                        action=next_action,
                        application=next_result.application,
                        preference=_preference(
                            output_library,
                            next_action,
                            next_result.application,
                            output_state,
                        ),
                        input_trace=first_trace,
                        output_trace=next_trace,
                    )
                )

        if config.sibling_forks:
            fork_trace = fork_conversation(
                trace,
                branch_nonce_digest=_sha(f"{record.id}:sibling"),
                reference_seed=30_000 + root_index,
                provenance=provenance,
            )
            if replay_conversation_trace(
                pack=base_pack,
                authority_resolver=lambda node: authorities[node.state_id],
                trace=fork_trace,
            ) != fork_trace.current:
                raise ValueError("operator corpus fork trace drifted")
            examples.append(
                SymbolicOperatorExampleV1(
                    example_id=_fingerprint(
                        {
                            "schema": "symbolic_operator_example_id/v1",
                            "kind": "sibling_fork",
                            "source_record_id": record.id,
                            "trace_fingerprint": fork_trace.fingerprint,
                        }
                    ),
                    kind=OperatorExampleKind.SIBLING_FORK,
                    target_view=OperatorTargetView.HISTORY_ONLY,
                    question={
                        "opcode": "FORK",
                        "view": "history_only",
                        "state_ast": state.source,
                        "legal_set_fingerprint": legal_set.fingerprint,
                        "trace_fingerprint": trace.fingerprint,
                    },
                    answer={
                        "operation": "FORK",
                        "state_id": fork_trace.current_state_id,
                        "branch_digest": fork_trace.current.branch_digest,
                    },
                    source_record_id=record.id,
                    semantic_family=str(
                        (record.meta or {}).get("program_family_id")
                        or (record.meta or {}).get("source_family")
                        or record.source
                    ),
                    operator_family="history.fork",
                    argument_kinds=(),
                    state_size=state_size,
                    scope_complexity=scope_complexity,
                    outcome="fork",
                    before_ast=state.source,
                    after_ast=state.source,
                    legal_set_fingerprint=legal_set.fingerprint,
                    legal_action=None,
                    application=None,
                    canonical_preference=None,
                    conversation_trace=fork_trace.to_dict(),
                )
            )

    if not examples:
        raise ValueError("operator corpus produced no symbolic examples")
    examples.sort(key=lambda item: item.example_id)
    collapsed_records.sort(key=lambda item: item.example_id)
    records_path = output_dir / "operator_records.jsonl"
    collapsed_path = output_dir / "operator_collapsed_records.jsonl"
    report_path = output_dir / "operator_coverage.json"
    counts = {
        "kind": Counter(item.kind.value for item in examples),
        "target_view": Counter(item.target_view.value for item in examples),
        "operator_family": Counter(item.operator_family for item in examples),
        "argument_kinds": Counter(
            ",".join(item.argument_kinds) or "none" for item in examples
        ),
        "state_size": Counter(str(item.state_size) for item in examples),
        "scope_complexity": Counter(
            str(item.scope_complexity) for item in examples
        ),
        "outcome": Counter(item.outcome for item in examples),
        "semantic_family": Counter(item.semantic_family for item in examples),
    }
    report = {
        "schema": "symbolic_operator_corpus_report/v1",
        "version": version,
        "record_count": len(examples),
        "collapsed_record_count": len(collapsed_records),
        "root_count": len(roots),
        "config": {
            "max_roots": config.max_roots,
            "actions_per_state": config.actions_per_state,
            "max_combinations_per_operator": (
                config.max_combinations_per_operator
            ),
            "sibling_forks": config.sibling_forks,
        },
        "strata": {
            key: dict(sorted(value.items())) for key, value in counts.items()
        },
        "application_coverage": {
            "legal_successes": legal_successes,
            "rejected_combinations": rejected_combinations,
            "emitted_illegal_targets": 0,
        },
        "coverage_gaps": gaps,
        "legal_sets": legal_sets,
        "invalid_family_count": 0,
        "collapse": {
            "symbolic_only": True,
            "nl_available": False,
            "nl_unavailable_reason": "CERT_CAP1_unavailable",
            "hard_negative_count": sum(
                len(item.collapse["hard_negatives"])
                for item in collapsed_records
            ),
        },
        "version_stamp": version_stamp,
    }
    _write_jsonl(records_path, (item.to_dict() for item in examples))
    _write_jsonl(
        collapsed_path, (item.to_dict() for item in collapsed_records)
    )
    _write_json(report_path, report)
    return {
        "records_path": str(records_path.as_posix()),
        "report_path": str(report_path.as_posix()),
        "collapsed_records_path": str(collapsed_path.as_posix()),
        "record_count": len(examples),
        "collapsed_record_count": len(collapsed_records),
        "root_count": len(roots),
        "content_fingerprint": _fingerprint(
            {
                "schema": "symbolic_operator_corpus_content/v1",
                "examples": [item.to_dict() for item in examples],
                "collapsed_records": [
                    item.to_dict() for item in collapsed_records
                ],
            }
        ),
        "report": report,
    }


__all__ = [
    "CollapsedOperatorExampleV1",
    "OperatorCorpusConfig",
    "OperatorExampleKind",
    "OperatorTargetView",
    "SymbolicOperatorExampleV1",
    "build_symbolic_operator_corpus",
]
