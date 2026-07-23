"""Conservative three-way merge for verified conversation branch edits."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Any

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    AstOperatorV1,
    CompilerCoverage,
    OperatorApplicationV1,
    OperatorRef,
    _fingerprint,
    _require_digest,
)
from slm_training.dsl.operators.conversation import ConversationStateNodeV1
from slm_training.dsl.operators.registry import OperatorLibraryV1, OperatorStateV1
from slm_training.dsl.pack import DslPack


class MergeConflictKind(str, Enum):
    SAME_NODE_INCOMPATIBLE_EDIT = "same_node_incompatible_edit"
    DELETE_MODIFY = "delete_modify"
    ROLE_CARDINALITY = "role_cardinality"
    CHILD_ORDER = "child_order"
    SCOPE_BINDER = "scope_binder"
    STALE_REF = "stale_ref"
    UNSUPPORTED_EFFECT = "unsupported_effect"


BranchAuthorityResolver = Callable[
    [ConversationStateNodeV1], tuple[DslPack, OperatorLibraryV1]
]


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _state_dict(state: OperatorStateV1) -> dict[str, str]:
    return {
        "schema": state.schema,
        "pack_id": state.pack_id,
        "source": state.source,
        "state_digest": state.state_digest,
        "ast_digest": state.ast_digest,
    }


@dataclass(frozen=True)
class BranchEditV1:
    """One verified single-application edge from a fork of the merge base."""

    input_node: ConversationStateNodeV1
    output_node: ConversationStateNodeV1
    application: OperatorApplicationV1
    schema: str = "branch_edit/v1"

    def __post_init__(self) -> None:
        if self.output_node.parent_state_id != self.input_node.state_id:
            raise ValueError("branch edit output must name its exact input")
        if self.output_node.branch_digest != self.input_node.branch_digest:
            raise ValueError("branch edit cannot cross branches")
        if not self.application.succeeded:
            raise ValueError("branch edit requires a successful application")
        if (
            self.application.before_state_digest
            != self.input_node.state.state_digest
            or self.application.before_ast_digest
            != self.input_node.state.ast_digest
            or self.application.after_state_digest
            != self.output_node.state.state_digest
            or self.application.after_ast_digest
            != self.output_node.state.ast_digest
        ):
            raise ValueError("branch edit application digests do not match its edge")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "input_state_id": self.input_node.state_id,
            "output_state_id": self.output_node.state_id,
            "branch_digest": self.input_node.branch_digest,
            "application": self.application.to_dict(),
        }


@dataclass(frozen=True)
class BranchMergeConflictV1:
    kind: MergeConflictKind
    base_state_id: str
    branch_state_ids: tuple[str, str]
    application_ids: tuple[str, str]
    target_fingerprints: tuple[str, ...] = ()
    schema: str = "branch_merge_conflict/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_id, "base_state_id")
        if tuple(sorted(self.branch_state_ids)) != self.branch_state_ids:
            raise ValueError("branch state IDs must be canonical")
        if tuple(sorted(self.application_ids)) != self.application_ids:
            raise ValueError("application IDs must be canonical")
        for value in (
            *self.branch_state_ids,
            *self.application_ids,
            *self.target_fingerprints,
        ):
            _require_digest(value, "merge conflict fingerprint")

    @property
    def conflict_id(self) -> str:
        return _fingerprint(self.to_dict(include_conflict_id=False))

    def to_dict(self, *, include_conflict_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "kind": self.kind.value,
            "base_state_id": self.base_state_id,
            "branch_state_ids": list(self.branch_state_ids),
            "application_ids": list(self.application_ids),
            "target_fingerprints": list(self.target_fingerprints),
        }
        if include_conflict_id:
            value["conflict_id"] = self.conflict_id
        return value


@dataclass(frozen=True)
class BranchMergeArtifactV1:
    pack_id: str
    base_state_id: str
    branch_state_ids: tuple[str, str]
    application_ids: tuple[str, str]
    merged_branch_digest: str
    merged_state: OperatorStateV1
    schema: str = "branch_merge/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_id, "base_state_id")
        _require_digest(self.merged_branch_digest, "merged_branch_digest")
        if tuple(sorted(self.branch_state_ids)) != self.branch_state_ids:
            raise ValueError("branch state IDs must be canonical")
        if tuple(sorted(self.application_ids)) != self.application_ids:
            raise ValueError("application IDs must be canonical")
        for value in (*self.branch_state_ids, *self.application_ids):
            _require_digest(value, "merge artifact fingerprint")
        if self.merged_state.pack_id != self.pack_id:
            raise ValueError("merged state belongs to another pack")

    @property
    def merge_id(self) -> str:
        return _fingerprint(self.to_dict(include_merge_id=False))

    def to_dict(self, *, include_merge_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "base_state_id": self.base_state_id,
            "branch_state_ids": list(self.branch_state_ids),
            "application_ids": list(self.application_ids),
            "merged_branch_digest": self.merged_branch_digest,
            "merged_state": _state_dict(self.merged_state),
        }
        if include_merge_id:
            value["merge_id"] = self.merge_id
        return value


@dataclass(frozen=True)
class BranchMergeDecisionV1:
    merge: BranchMergeArtifactV1 | None = None
    conflict: BranchMergeConflictV1 | None = None
    schema: str = "branch_merge_decision/v1"

    def __post_init__(self) -> None:
        if (self.merge is None) == (self.conflict is None):
            raise ValueError("exactly one merge or conflict is required")

    @property
    def succeeded(self) -> bool:
        return self.merge is not None

    @property
    def decision_id(self) -> str:
        return _fingerprint(self.to_dict(include_decision_id=False))

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "merge": self.merge.to_dict() if self.merge is not None else None,
            "conflict": (
                self.conflict.to_dict() if self.conflict is not None else None
            ),
        }
        if include_decision_id:
            value["decision_id"] = self.decision_id
        return value


def _declaration(
    library: OperatorLibraryV1, application: OperatorApplicationV1
) -> AstOperatorV1 | None:
    return next(
        (
            declaration
            for declaration in library.declarations
            if declaration.fingerprint == application.operator_fingerprint
        ),
        None,
    )


def _refs(effect: ActionEffectV1) -> tuple[OperatorRef, ...]:
    return (
        *effect.consumed_roles,
        *effect.produced_roles,
        *effect.consumed_binders,
        *effect.produced_binders,
        *(delta.target for delta in effect.scope_deltas),
        *(delta.target for delta in effect.cardinality_deltas),
        *(delta.target for delta in effect.property_deltas),
        *(delta.target for delta in effect.topology_deltas),
    )


def _base_targets(
    base: ConversationStateNodeV1, branch: ConversationStateNodeV1
) -> dict[OperatorRef, str]:
    if (
        branch.reference_table.request_id
        != base.reference_table.request_id
    ):
        return {}
    base_by_semantic = {
        entry.descriptor.semantic_fingerprint: entry
        for entry in base.reference_table.entries
    }
    if branch.branch_digest == base.branch_digest:
        if (
            {entry.descriptor for entry in branch.reference_table.entries}
            != {entry.descriptor for entry in base.reference_table.entries}
            or branch.reference_table.runtime_symbols
            != base.reference_table.runtime_symbols
        ):
            return {}
        return {
            entry.ref: entry.descriptor.fingerprint
            for entry in branch.reference_table.entries
            if (
                entry.descriptor.semantic_fingerprint in base_by_semantic
                and entry.descriptor
                == base_by_semantic[
                    entry.descriptor.semantic_fingerprint
                ].descriptor
            )
        }
    semantic_map = {
        semantic: _fingerprint(
            {
                "schema": "conversation_fork_semantic_ref/v1",
                "source_semantic_fingerprint": semantic,
                "branch_digest": branch.branch_digest,
            }
        )
        for semantic in base_by_semantic
    }
    expected = {
        replace(
            entry.descriptor,
            semantic_fingerprint=semantic_map[semantic],
            parent_fingerprint=(
                semantic_map.get(
                    entry.descriptor.parent_fingerprint,
                    _fingerprint(
                        {
                            "schema": "conversation_fork_parent_ref/v1",
                            "source_parent_fingerprint": (
                                entry.descriptor.parent_fingerprint
                            ),
                            "branch_digest": branch.branch_digest,
                        }
                    ),
                )
                if entry.descriptor.parent_fingerprint is not None
                else None
            ),
            parent_order_digest=(
                _fingerprint(
                    {
                        "schema": "conversation_fork_parent_order/v1",
                        "source_parent_order_digest": (
                            entry.descriptor.parent_order_digest
                        ),
                        "branch_digest": branch.branch_digest,
                    }
                )
                if entry.descriptor.parent_order_digest is not None
                else None
            ),
        ): entry.descriptor.fingerprint
        for semantic, entry in base_by_semantic.items()
    }
    if {entry.descriptor for entry in branch.reference_table.entries} != set(
        expected
    ):
        return {}
    descriptor_fingerprints = {
        entry.descriptor.fingerprint: cloned.fingerprint
        for entry in base.reference_table.entries
        for cloned in expected
        if expected[cloned] == entry.descriptor.fingerprint
    }
    expected_runtime_symbols = tuple(
        replace(
            symbol,
            symbol_fingerprint=_fingerprint(
                {
                    "schema": "conversation_fork_runtime_symbol/v1",
                    "source_symbol_fingerprint": symbol.symbol_fingerprint,
                    "branch_digest": branch.branch_digest,
                }
            ),
            ref_fingerprint=descriptor_fingerprints[symbol.ref_fingerprint],
        )
        for symbol in base.reference_table.runtime_symbols
    )
    if branch.reference_table.runtime_symbols != expected_runtime_symbols:
        return {}
    return {
        entry.ref: expected[entry.descriptor]
        for entry in branch.reference_table.entries
        if entry.descriptor in expected
    }


def _effect_targets(
    effect: ActionEffectV1, lineage: Mapping[OperatorRef, str]
) -> tuple[dict[str, set[str]], bool]:
    categories: dict[str, set[str]] = {}

    def add(category: str, ref: OperatorRef) -> None:
        target = lineage.get(ref)
        if target is None:
            raise KeyError(ref)
        categories.setdefault(target, set()).add(category)

    try:
        for ref in (*effect.consumed_roles, *effect.produced_roles):
            add("role", ref)
        for ref in (*effect.consumed_binders, *effect.produced_binders):
            add("binder", ref)
        for delta in (
            *effect.scope_deltas,
            *effect.cardinality_deltas,
            *effect.property_deltas,
            *effect.topology_deltas,
        ):
            add(delta.kind.value, delta.target)
    except KeyError:
        return {}, False
    return categories, True


def _conflict_kind(
    left_categories: Mapping[str, set[str]],
    right_categories: Mapping[str, set[str]],
    *,
    left_operator_id: str,
    right_operator_id: str,
) -> tuple[MergeConflictKind, tuple[str, ...]] | None:
    overlap = set(left_categories) & set(right_categories)
    if not overlap:
        return None
    combined = {
        category
        for target in overlap
        for category in (
            left_categories[target] | right_categories[target]
        )
    }
    if any("remove" in value or "delete" in value for value in (
        left_operator_id,
        right_operator_id,
    )):
        kind = MergeConflictKind.DELETE_MODIFY
    elif combined & {"scope", "binder"}:
        kind = MergeConflictKind.SCOPE_BINDER
    elif combined & {"cardinality", "role"}:
        kind = MergeConflictKind.ROLE_CARDINALITY
    elif "topology" in combined:
        kind = MergeConflictKind.CHILD_ORDER
    else:
        kind = MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT
    return kind, tuple(sorted(overlap))


class _StructuralConflict(ValueError):
    def __init__(self, kind: MergeConflictKind) -> None:
        self.kind = kind
        super().__init__(kind.value)


_MISSING = object()


def _merge_value(base: Any, left: Any, right: Any) -> Any:
    if left == right:
        return left
    if left == base:
        return right
    if right == base:
        return left
    if _MISSING in (base, left, right):
        raise _StructuralConflict(MergeConflictKind.DELETE_MODIFY)
    if type(base) is not type(left) or type(base) is not type(right):
        raise _StructuralConflict(
            MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT
        )
    if is_dataclass(base) and not isinstance(base, type):
        derived_fields = {
            "source",
            "serialized",
            "placeholders",
            "meta",
            "policy_errors",
        }
        changes = {
            field.name: (
                (
                    None
                    if field.name == "serialized"
                    else getattr(base, field.name)
                )
                if field.name in derived_fields
                else _merge_value(
                    getattr(base, field.name),
                    getattr(left, field.name),
                    getattr(right, field.name),
                )
            )
            for field in fields(base)
        }
        return replace(base, **changes)
    if isinstance(base, Mapping):
        keys = set(base) | set(left) | set(right)
        return type(base)(
            (
                key,
                merged,
            )
            for key in sorted(keys, key=str)
            if (
                merged := _merge_value(
                    base.get(key, _MISSING),
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                )
            )
            is not _MISSING
        )
    if isinstance(base, (tuple, list)):
        if len(base) != len(left) or len(base) != len(right):
            raise _StructuralConflict(MergeConflictKind.CHILD_ORDER)
        values = (
            _merge_value(base[index], left[index], right[index])
            for index in range(len(base))
        )
        return type(base)(values)
    raise _StructuralConflict(MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT)


def _identity(
    base: ConversationStateNodeV1,
    left: BranchEditV1,
    right: BranchEditV1,
) -> tuple[tuple[str, str], tuple[str, str]]:
    return (
        tuple(sorted((left.output_node.state_id, right.output_node.state_id))),
        tuple(
            sorted(
                (
                    left.application.application_id,
                    right.application.application_id,
                )
            )
        ),
    )


def _conflict(
    kind: MergeConflictKind,
    base: ConversationStateNodeV1,
    left: BranchEditV1,
    right: BranchEditV1,
    targets: tuple[str, ...] = (),
) -> BranchMergeDecisionV1:
    branch_ids, application_ids = _identity(base, left, right)
    return BranchMergeDecisionV1(
        conflict=BranchMergeConflictV1(
            kind=kind,
            base_state_id=base.state_id,
            branch_state_ids=branch_ids,
            application_ids=application_ids,
            target_fingerprints=tuple(sorted(targets)),
        )
    )


def merge_conversation_branches(
    *,
    pack: DslPack,
    base: ConversationStateNodeV1,
    left: BranchEditV1,
    right: BranchEditV1,
    authority_resolver: BranchAuthorityResolver,
) -> BranchMergeDecisionV1:
    """Merge exactly two verified one-step branch edits or return a typed conflict."""
    if (
        left.input_node.state != base.state
        or right.input_node.state != base.state
        or left.input_node.branch_digest == right.input_node.branch_digest
    ):
        return _conflict(
            MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
        )
    declarations: list[AstOperatorV1] = []
    for edit in (left, right):
        branch_pack, library = authority_resolver(edit.input_node)
        if branch_pack.pack_id != pack.pack_id:
            return _conflict(
                MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
            )
        if (
            edit.application.provenance.request_id
            != edit.input_node.reference_table.request_id
            or edit.application.provenance.source_artifact_digest
            != _source_digest(edit.input_node.state.source)
        ):
            return _conflict(MergeConflictKind.STALE_REF, base, left, right)
        if any(
            argument.value not in {
                entry.ref for entry in edit.input_node.reference_table.entries
            }
            for argument in edit.application.arguments
        ):
            return _conflict(MergeConflictKind.STALE_REF, base, left, right)
        try:
            replayed = library.replay(
                branch_pack, edit.input_node.state, edit.application
            )
        except Exception:  # noqa: BLE001 - external authority becomes typed refusal
            return _conflict(
                MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
            )
        if replayed.state != edit.output_node.state:
            return _conflict(
                MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
            )
        declaration = _declaration(library, edit.application)
        if declaration is None:
            return _conflict(
                MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
            )
        declarations.append(declaration)

    effects = (left.application.effect, right.application.effect)
    if any(
        effect is None
        or effect.compiler_coverage is not CompilerCoverage.EXACT
        or not _refs(effect)
        for effect in effects
    ):
        return _conflict(
            MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
        )
    assert effects[0] is not None and effects[1] is not None
    left_targets, left_fresh = _effect_targets(
        effects[0], _base_targets(base, left.input_node)
    )
    right_targets, right_fresh = _effect_targets(
        effects[1], _base_targets(base, right.input_node)
    )
    if not left_fresh or not right_fresh:
        return _conflict(MergeConflictKind.STALE_REF, base, left, right)

    overlap = _conflict_kind(
        left_targets,
        right_targets,
        left_operator_id=declarations[0].operator_id,
        right_operator_id=declarations[1].operator_id,
    )
    mutually_commuting = (
        declarations[1].operator_id in declarations[0].commutes_with
        and declarations[0].operator_id in declarations[1].commutes_with
    )
    if overlap is not None and not mutually_commuting:
        return _conflict(overlap[0], base, left, right, overlap[1])

    try:
        merged_ast = _merge_value(
            pack.backend.parse(base.state.source),
            pack.backend.parse(left.output_node.state.source),
            pack.backend.parse(right.output_node.state.source),
        )
        merged_state = OperatorStateV1.from_source(
            pack, pack.backend.serialize(merged_ast)
        )
    except _StructuralConflict as exc:
        targets = overlap[1] if overlap is not None else ()
        return _conflict(exc.kind, base, left, right, targets)
    except Exception:  # noqa: BLE001 - invalid merged AST is a typed refusal
        return _conflict(
            MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
        )

    branch_ids, application_ids = _identity(base, left, right)
    merged_branch_digest = _fingerprint(
        {
            "schema": "conversation_merge_branch/v1",
            "base_state_id": base.state_id,
            "branch_state_ids": branch_ids,
            "application_ids": application_ids,
        }
    )
    return BranchMergeDecisionV1(
        merge=BranchMergeArtifactV1(
            pack_id=pack.pack_id,
            base_state_id=base.state_id,
            branch_state_ids=branch_ids,
            application_ids=application_ids,
            merged_branch_digest=merged_branch_digest,
            merged_state=merged_state,
        )
    )


def replay_branch_merge(
    *,
    pack: DslPack,
    base: ConversationStateNodeV1,
    left: BranchEditV1,
    right: BranchEditV1,
    authority_resolver: BranchAuthorityResolver,
    recorded: BranchMergeDecisionV1,
) -> BranchMergeDecisionV1:
    """Recompute the complete decision and require exact provenance identity."""
    replayed = merge_conversation_branches(
        pack=pack,
        base=base,
        left=left,
        right=right,
        authority_resolver=authority_resolver,
    )
    if replayed.decision_id != recorded.decision_id:
        raise ValueError("branch merge replay differs from recorded decision")
    return replayed


__all__ = [
    "BranchAuthorityResolver",
    "BranchEditV1",
    "BranchMergeArtifactV1",
    "BranchMergeConflictV1",
    "BranchMergeDecisionV1",
    "MergeConflictKind",
    "merge_conversation_branches",
    "replay_branch_merge",
]
