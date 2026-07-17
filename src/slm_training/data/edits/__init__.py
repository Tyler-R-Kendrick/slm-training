"""Verified statement-level edit and trajectory oracles for OpenUI programs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from slm_training.data.progspec import ProgramSpec, emit_record
from slm_training.data.verify import VerificationContext, stamp_record, verify_record
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord

_STATEMENT_RE = re.compile(r"^\s*([a-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|[a-z_][A-Za-z0-9_]*')


@dataclass(frozen=True)
class Statement:
    """One named OpenUI statement."""

    name: str
    expression: str

    @property
    def source(self) -> str:
        return f"{self.name} = {self.expression}"


@dataclass(frozen=True)
class ProgramDocument:
    """Canonical OpenUI represented as ordered, independently editable statements."""

    statements: tuple[Statement, ...]

    @classmethod
    def from_openui(cls, source: str) -> ProgramDocument:
        validated_source = source.strip()
        validate(validated_source)
        statements: list[Statement] = []
        seen: set[str] = set()
        for line in validated_source.splitlines():
            if not line.strip():
                continue
            match = _STATEMENT_RE.fullmatch(line)
            if match is None:
                raise ValueError(f"unsupported multiline statement: {line!r}")
            name, expression = match.groups()
            if name in seen:
                raise ValueError(f"duplicate statement: {name}")
            seen.add(name)
            statements.append(Statement(name, expression))
        if "root" not in seen:
            raise ValueError("program must define root")
        return cls(tuple(statements))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(statement.name for statement in self.statements)

    def statement(self, name: str) -> Statement:
        for statement in self.statements:
            if statement.name == name:
                return statement
        raise ValueError(f"unknown statement: {name}")

    def to_openui(self) -> str:
        source = "\n".join(statement.source for statement in self.statements)
        validate(source)
        return source.strip()


class EditKind(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    RENAME = "rename"
    REORDER = "reorder"
    NOOP = "noop"


class EditIntent(str, Enum):
    """Formal edit taxonomy; every intent compiles to the primitives above."""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    REORDER = "reorder"
    WRAP = "wrap"
    UNWRAP = "unwrap"
    SPLIT = "split"
    MERGE = "merge"
    DUPLICATE = "duplicate"
    RENAME_PRESERVING_REFS = "rename_preserving_refs"
    CHANGE_CONTENT = "change_content"
    CHANGE_PROP = "change_prop"
    CHANGE_LAYOUT = "change_layout"
    CHANGE_RESPONSIVE = "change_responsive"
    ADD_OR_MODIFY_STATE = "add_or_modify_state"
    ADD_OR_MODIFY_QUERY = "add_or_modify_query"
    ADD_OR_MODIFY_MUTATION = "add_or_modify_mutation"
    ADD_OR_MODIFY_ACTION = "add_or_modify_action"
    APPLY_TO_ALL_MATCHING = "apply_to_all_matching"
    NOOP_ALREADY_SATISFIED = "noop_already_satisfied"
    UNSUPPORTED_REQUEST = "unsupported_request"


@dataclass(frozen=True)
class EditOperation:
    """One minimal statement-AST operation with replay preconditions."""

    kind: EditKind
    name: str
    before: str | None = None
    after: str | None = None
    target: str | None = None
    index: int | None = None
    previous_index: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operation name must be non-empty")
        required: dict[EditKind, tuple[str, ...]] = {
            EditKind.ADD: ("after",),
            EditKind.REMOVE: ("before",),
            EditKind.REPLACE: ("before", "after"),
            EditKind.RENAME: ("target",),
            EditKind.REORDER: ("index", "previous_index"),
            EditKind.NOOP: (),
        }
        for field_name in required[self.kind]:
            if getattr(self, field_name) is None:
                raise ValueError(f"{self.kind.value} requires {field_name}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind.value, "name": self.name}
        for key in ("before", "after", "target", "index", "previous_index"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass(frozen=True)
class EditPatch:
    """A replayable structural delta; high-level edits compile to these primitives."""

    operations: tuple[EditOperation, ...]
    instruction: str = ""
    intent: EditIntent | None = None
    collect_unreachable: bool = True
    unsupported_reason: str | None = None

    @property
    def ast_operation_count(self) -> int:
        return sum(operation.kind is not EditKind.NOOP for operation in self.operations)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operations": [operation.to_dict() for operation in self.operations],
            "ast_operation_count": self.ast_operation_count,
            "collect_unreachable": self.collect_unreachable,
        }
        if self.instruction:
            result["instruction"] = self.instruction
        if self.intent is not None:
            result["intent"] = self.intent.value
        if self.unsupported_reason:
            result["unsupported_reason"] = self.unsupported_reason
        return result


def unsupported_patch(reason: str, *, instruction: str = "") -> EditPatch:
    """Represent an unsupported request without silently changing the program."""
    if not reason.strip():
        raise ValueError("unsupported reason must be non-empty")
    return EditPatch(
        (EditOperation(EditKind.NOOP, "root"),),
        instruction=instruction,
        intent=EditIntent.UNSUPPORTED_REQUEST,
        unsupported_reason=reason,
    )


def _references(expression: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(expression)
        if not token.startswith('"') and token not in {"true", "false", "null"}
    }


def _rename_reference(expression: str, old: str, new: str) -> str:
    return _TOKEN_RE.sub(
        lambda match: new if match.group() == old else match.group(), expression
    )


def _collect_reachable(statements: list[Statement]) -> list[Statement]:
    by_name = {statement.name: statement for statement in statements}
    reachable: set[str] = set()
    pending = ["root"]
    while pending:
        name = pending.pop()
        if name in reachable or name not in by_name:
            continue
        reachable.add(name)
        pending.extend(_references(by_name[name].expression) & by_name.keys())
    return [statement for statement in statements if statement.name in reachable]


def apply_patch(before: str, patch: EditPatch) -> str:
    """Apply a delta with strict preconditions, then validate and canonicalize it."""
    statements = list(ProgramDocument.from_openui(before).statements)

    def index_of(name: str) -> int:
        try:
            return next(
                i for i, statement in enumerate(statements) if statement.name == name
            )
        except StopIteration as exc:
            raise ValueError(f"unknown statement: {name}") from exc

    for operation in patch.operations:
        if operation.kind is EditKind.NOOP:
            continue
        if operation.kind is EditKind.ADD:
            if operation.name in {statement.name for statement in statements}:
                raise ValueError(f"statement already exists: {operation.name}")
            position = len(statements) if operation.index is None else operation.index
            if not 0 <= position <= len(statements):
                raise ValueError(f"invalid insertion index: {position}")
            assert operation.after is not None
            statements.insert(position, Statement(operation.name, operation.after))
        elif operation.kind is EditKind.REMOVE:
            if operation.name == "root":
                raise ValueError("root cannot be removed")
            position = index_of(operation.name)
            current = statements[position]
            if current.expression != operation.before:
                raise ValueError(f"remove precondition failed: {operation.name}")
            statements.pop(position)
        elif operation.kind is EditKind.REPLACE:
            position = index_of(operation.name)
            current = statements[position]
            if current.expression != operation.before:
                raise ValueError(f"replace precondition failed: {operation.name}")
            assert operation.after is not None
            statements[position] = Statement(operation.name, operation.after)
        elif operation.kind is EditKind.RENAME:
            if operation.name == "root" or operation.target == "root":
                raise ValueError("root cannot be renamed")
            position = index_of(operation.name)
            assert operation.target is not None
            if operation.target in {statement.name for statement in statements}:
                raise ValueError(f"statement already exists: {operation.target}")
            statements[position] = Statement(
                operation.target, statements[position].expression
            )
            statements = [
                Statement(
                    statement.name,
                    _rename_reference(
                        statement.expression, operation.name, operation.target
                    ),
                )
                for statement in statements
            ]
        elif operation.kind is EditKind.REORDER:
            current_index = index_of(operation.name)
            if current_index != operation.previous_index:
                raise ValueError(f"reorder precondition failed: {operation.name}")
            assert operation.index is not None
            if not 0 <= operation.index < len(statements):
                raise ValueError(f"invalid reorder index: {operation.index}")
            statements.insert(operation.index, statements.pop(current_index))

    if patch.collect_unreachable:
        statements = _collect_reachable(statements)
    return ProgramDocument(tuple(statements)).to_openui()


def diff_programs(
    before: str,
    after: str,
    *,
    instruction: str = "",
    renames: Mapping[str, str] | None = None,
    intent: EditIntent | None = None,
) -> EditPatch:
    """Return a deterministic statement-AST delta from ``before`` to ``after``."""
    target = ProgramDocument.from_openui(after)
    working_source = ProgramDocument.from_openui(before).to_openui()
    operations: list[EditOperation] = []

    for old, new in (renames or {}).items():
        operation = EditOperation(EditKind.RENAME, old, target=new)
        working_source = apply_patch(
            working_source,
            EditPatch((operation,), collect_unreachable=False),
        )
        operations.append(operation)

    working = ProgramDocument.from_openui(working_source)
    target_by_name = {statement.name: statement for statement in target.statements}
    working_by_name = {statement.name: statement for statement in working.statements}
    residual_start = len(operations)

    for index in range(len(working.statements) - 1, -1, -1):
        statement = working.statements[index]
        if statement.name not in target_by_name:
            operations.append(
                EditOperation(
                    EditKind.REMOVE,
                    statement.name,
                    before=statement.expression,
                    previous_index=index,
                )
            )
    for index, statement in enumerate(target.statements):
        if statement.name not in working_by_name:
            operations.append(
                EditOperation(
                    EditKind.ADD,
                    statement.name,
                    after=statement.expression,
                    index=index,
                )
            )
    for statement in target.statements:
        current = working_by_name.get(statement.name)
        if current is not None and current.expression != statement.expression:
            operations.append(
                EditOperation(
                    EditKind.REPLACE,
                    statement.name,
                    before=current.expression,
                    after=statement.expression,
                )
            )

    structural = EditPatch(
        tuple(operations[residual_start:]), collect_unreachable=False
    )
    intermediate = ProgramDocument.from_openui(apply_patch(working_source, structural))
    order = list(intermediate.names)
    for target_index, name in enumerate(target.names):
        current_index = order.index(name)
        if current_index == target_index:
            continue
        operations.append(
            EditOperation(
                EditKind.REORDER,
                name,
                index=target_index,
                previous_index=current_index,
            )
        )
        order.insert(target_index, order.pop(current_index))

    patch = EditPatch(tuple(operations), instruction=instruction, intent=intent)
    if apply_patch(before, patch) != target.to_openui():
        raise ValueError("constructed patch does not reproduce target")
    return patch


def invert_patch(before: str, patch: EditPatch) -> EditPatch:
    """Build the true inverse, including statements removed by reachability GC."""
    after = apply_patch(before, patch)
    return diff_programs(
        after, before, instruction=f"undo: {patch.instruction}".strip()
    )


def minimal_statement_patch(before: str, patch: EditPatch) -> str:
    """Serialize only target statements added or changed by a structural delta."""
    before_document = ProgramDocument.from_openui(before)
    after_document = ProgramDocument.from_openui(apply_patch(before, patch))
    before_by_name = {
        statement.name: statement.expression for statement in before_document.statements
    }
    return "\n".join(
        statement.source
        for statement in after_document.statements
        if before_by_name.get(statement.name) != statement.expression
    )


RenderVerifier = Callable[[str], bool | None]


@dataclass(frozen=True)
class EditTransition:
    before: str
    instruction: str
    patch: EditPatch
    after: str
    inverse: EditPatch
    statement_patch: str
    render_verified: bool

    @property
    def ast_operation_count(self) -> int:
        return self.patch.ast_operation_count


def _assert_program(
    source: str,
    *,
    render_verifier: RenderVerifier,
) -> bool:
    canonical = ProgramDocument.from_openui(source).to_openui()
    record = ExampleRecord(
        id="edit_verification",
        prompt="verify edit transition",
        openui=canonical,
        placeholders=extract_placeholders(canonical),
        source="program",
    )
    report = verify_record(record)
    if not report.ok:
        raise ValueError(f"program failed {report.failing_gate.value}")
    if render_verifier(canonical) is False:
        raise ValueError("program failed render verification")
    return True


def build_transition(
    before: str,
    instruction: str,
    patch: EditPatch,
    *,
    render_verifier: RenderVerifier,
) -> EditTransition:
    """Apply and independently verify one reversible edit transition."""
    if not instruction.strip():
        raise ValueError("instruction must be non-empty")
    canonical_before = ProgramDocument.from_openui(before).to_openui()
    after = apply_patch(canonical_before, patch)
    record = ExampleRecord(
        id="edit_transition",
        prompt=instruction,
        openui=after,
        placeholders=extract_placeholders(after),
        source="program",
    )
    context = VerificationContext(
        source_kind="program",
        patch_before=canonical_before,
        patch=patch,
        patch_after=after,
        patch_applier=apply_patch,
    )
    report = verify_record(record, context)
    if not report.ok:
        assert report.failing_gate is not None
        raise ValueError(f"transition failed {report.failing_gate.value}")
    rendered = _assert_program(after, render_verifier=render_verifier)
    inverse = invert_patch(canonical_before, patch)
    if apply_patch(after, inverse) != canonical_before:
        raise ValueError("inverse patch does not restore the original program")
    return EditTransition(
        before=canonical_before,
        instruction=instruction,
        patch=patch,
        after=after,
        inverse=inverse,
        statement_patch=minimal_statement_patch(canonical_before, patch),
        render_verified=rendered,
    )


def emit_transition_records(
    spec: ProgramSpec, transition: EditTransition
) -> tuple[ExampleRecord, ExampleRecord, ExampleRecord]:
    """Emit GENERATE, APPLY_PATCH, and PATCH derivatives for one transition."""
    from slm_training.data.quality import (
        render_semantic_contract_prompt,
        semantic_contract_for_openui,
    )

    if (
        ProgramDocument.from_openui(spec.canonical_openui).to_openui()
        != transition.before
    ):
        raise ValueError("transition must start from the supplied ProgramSpec")
    edit_meta = {
        "instruction": transition.instruction,
        "before": transition.before,
        "after": transition.after,
        "patch": transition.patch.to_dict(),
        "inverse": transition.inverse.to_dict(),
        "statement_patch": transition.statement_patch,
        "ast_operation_count": transition.ast_operation_count,
        "render_verified": transition.render_verified,
    }
    semantic_contract = semantic_contract_for_openui(transition.after)
    records: list[ExampleRecord] = []
    modes = (
        (
            "GENERATE",
            "generation",
            render_semantic_contract_prompt(semantic_contract),
            {"semantic_contract": semantic_contract},
        ),
        (
            "APPLY_PATCH",
            "edit",
            "Apply this structural patch to the current program:\n"
            + json.dumps(transition.patch.to_dict(), sort_keys=True),
            {},
        ),
        (
            "PATCH",
            "patch",
            f"Current program:\n{transition.before}\n\nChange: {transition.instruction}",
            {},
        ),
    )
    for mode, task, prompt, task_meta in modes:
        record = emit_record(
            spec,
            prompt=prompt,
            task=task,
            openui=transition.after,
            source="edit_patch",
            meta={"edit": {**edit_meta, "mode": mode}, **task_meta},
        )
        context = VerificationContext(
            source_kind="program",
            patch_before=transition.before,
            patch=transition.patch,
            patch_after=transition.after,
            patch_applier=apply_patch,
        )
        records.append(stamp_record(record, context))
    return tuple(records)  # type: ignore[return-value]


def emit_trajectory_records(
    spec: ProgramSpec, trajectory: EditTrajectory
) -> tuple[ExampleRecord, ...]:
    """Emit each verified multi-turn transition with its bounded conversation state."""
    if (
        ProgramDocument.from_openui(spec.canonical_openui).to_openui()
        != trajectory.initial
    ):
        raise ValueError("trajectory must start from the supplied ProgramSpec")
    records: list[ExampleRecord] = []
    history: list[str] = []
    for turn, transition in enumerate(trajectory.transitions, start=1):
        prior = "\n".join(history[-3:]) or "(none)"
        prompt = (
            f"Recent edit history:\n{prior}\n\n"
            f"Current program:\n{transition.before}\n\n"
            f"User: {transition.instruction}"
        )
        edit_meta = {
            "mode": "PATCH",
            "turn": turn,
            "instruction": transition.instruction,
            "before": transition.before,
            "after": transition.after,
            "patch": transition.patch.to_dict(),
            "inverse": transition.inverse.to_dict(),
            "statement_patch": transition.statement_patch,
            "ast_operation_count": transition.ast_operation_count,
            "render_verified": transition.render_verified,
            "history_summary": prior,
        }
        record = emit_record(
            spec,
            prompt=prompt,
            task="edit",
            openui=transition.after,
            source="edit_trajectory",
            meta={"edit": edit_meta},
        )
        records.append(
            stamp_record(
                record,
                VerificationContext(
                    source_kind="program",
                    patch_before=transition.before,
                    patch=transition.patch,
                    patch_after=transition.after,
                    patch_applier=apply_patch,
                ),
            )
        )
        history.append(f"{turn}. {transition.instruction}")
    return tuple(records)


@dataclass
class EditTrajectory:
    """A coherent multi-turn state machine with deterministic undo and redo."""

    initial: str
    render_verifier: RenderVerifier
    transitions: list[EditTransition] = field(default_factory=list)
    focus: str | None = None
    _redo: list[EditTransition] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.initial = ProgramDocument.from_openui(self.initial).to_openui()
        _assert_program(self.initial, render_verifier=self.render_verifier)

    @property
    def current(self) -> str:
        return self.transitions[-1].after if self.transitions else self.initial

    def apply(
        self,
        instruction: str,
        patch: EditPatch,
        *,
        focus: str | None = None,
    ) -> EditTransition:
        transition = build_transition(
            self.current,
            instruction,
            patch,
            render_verifier=self.render_verifier,
        )
        self.transitions.append(transition)
        self._redo.clear()
        if focus is not None:
            if focus not in ProgramDocument.from_openui(self.current).names:
                raise ValueError(f"focus does not exist after edit: {focus}")
            self.focus = focus
        return transition

    def undo(self) -> str:
        if not self.transitions:
            raise ValueError("nothing to undo")
        transition = self.transitions.pop()
        self._redo.append(transition)
        _assert_program(self.current, render_verifier=self.render_verifier)
        return self.current

    def redo(self) -> str:
        if not self._redo:
            raise ValueError("nothing to redo")
        transition = self._redo.pop()
        if transition.before != self.current:
            raise ValueError("redo history no longer matches current program")
        self.transitions.append(transition)
        _assert_program(self.current, render_verifier=self.render_verifier)
        return self.current

    def rollback_last(
        self,
        instruction: str,
        *,
        revert_names: Sequence[str],
    ) -> EditTransition:
        """Apply an interpreted partial rollback while keeping unrelated changes."""
        if not self.transitions:
            raise ValueError("nothing to roll back")
        selected = set(revert_names)
        operations = tuple(
            operation
            for operation in self.transitions[-1].inverse.operations
            if operation.name in selected or operation.target in selected
        )
        if not operations:
            raise ValueError("rollback selection matched no operations")
        return self.apply(
            instruction,
            EditPatch(operations, instruction=instruction),
            focus=self.focus,
        )

    def resolve_reference(self, reference: str) -> str:
        """Resolve exact binder names and ordinary follow-up pronouns."""
        names = ProgramDocument.from_openui(self.current).names
        normalized = reference.strip().lower()
        if reference in names:
            return reference
        if normalized in {"it", "that", "this", "that one", "this one"}:
            if self.focus is None:
                raise ValueError("reference has no focused statement")
            return self.focus
        matches = [
            name
            for name in names
            if name != "root" and name.replace("_", " ").lower() in normalized
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"ambiguous or unknown reference: {reference}")

    def summary(self, *, max_turns: int = 3) -> str:
        turns = self.transitions[-max_turns:]
        return "\n".join(
            f"{index + 1}. {turn.instruction} ({turn.ast_operation_count} AST ops)"
            for index, turn in enumerate(turns)
        )


__all__ = [
    "EditIntent",
    "EditKind",
    "EditOperation",
    "EditPatch",
    "EditTrajectory",
    "EditTransition",
    "ProgramDocument",
    "Statement",
    "apply_patch",
    "build_transition",
    "diff_programs",
    "emit_transition_records",
    "emit_trajectory_records",
    "invert_patch",
    "minimal_statement_patch",
    "unsupported_patch",
]
