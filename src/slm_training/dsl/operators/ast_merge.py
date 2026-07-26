"""Shared conservative base-relative structural AST merge primitive.

Extracted from ``merge.py`` (DSH3-19/DSH3-20) so that DSH5-05's N-way
transaction composition — a fold of pairwise base-relative merges over one
or more independently-prepared action outputs — and DSH5-04/merge.py's
existing two-way conversation-branch merge share exactly one merge
algorithm, never two parallel implementations. ``merge.py`` itself now
imports :func:`merge_ast_value`/:class:`StructuralMergeConflict`/
:class:`MergeConflictKind` from here rather than defining them locally; its
own public behavior and tests are unchanged.

This module is deliberately free of any dependency on ``conversation.py``
(and, transitively, everything ``conversation.py`` itself never imports):
``transaction_executor.py`` (DSH5-05) needs this exact merge primitive
without ever importing the conversation module, because ``merge.py`` (which
used to be the only place this primitive lived) itself imports
``conversation.py`` for unrelated reasons (``BranchEditV1``'s node types) —
routing DSH5-05's reuse through ``merge.py`` directly would close an import
cycle back through ``conversation.py`` once conversation turns learn to
carry a transaction. See ``docs/design/dsh5-05-transaction-executor.md``
("Layering: why a shared ast_merge module") for the full account.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Any, Mapping


class MergeConflictKind(str, Enum):
    SAME_NODE_INCOMPATIBLE_EDIT = "same_node_incompatible_edit"
    DELETE_MODIFY = "delete_modify"
    ROLE_CARDINALITY = "role_cardinality"
    CHILD_ORDER = "child_order"
    SCOPE_BINDER = "scope_binder"
    STALE_REF = "stale_ref"
    UNSUPPORTED_EFFECT = "unsupported_effect"
    REFERENCE_REBUILD_FAILED = "reference_rebuild_failed"


class StructuralMergeConflict(ValueError):
    """A base-relative structural merge found a real, unresolvable overlap."""

    def __init__(self, kind: MergeConflictKind) -> None:
        self.kind = kind
        super().__init__(kind.value)


_MISSING = object()

#: Fields every merge participant carries but that are never themselves
#: structurally merged: raw source text and re-derived caches are either
#: kept from ``base`` verbatim or cleared so a caller always re-serializes
#: from the merged structural fields instead of a stale cached string.
_DERIVED_FIELDS = frozenset({"source", "serialized", "placeholders", "meta", "policy_errors"})


def merge_ast_value(base: Any, left: Any, right: Any) -> Any:
    """Conservative base-relative 3-way merge of one parsed AST value.

    ``base`` is the common ancestor; ``left``/``right`` are two values each
    independently derived from ``base`` (never from one another). Where only
    one side changed a field/element relative to ``base``, that side's value
    wins; where both changed it identically, the (agreeing) value wins; where
    both changed it *differently*, raises :class:`StructuralMergeConflict`
    naming the closest :class:`MergeConflictKind` the mismatch resembles,
    rather than guessing.

    Folding this pairwise primitive over more than two independently
    prepared outputs (``result = base; for output in outputs: result =
    merge_ast_value(base, result, output)``) generalizes it to N-way
    composition without ever re-deriving an output from another output —
    every call still diffs against the one fixed ``base``. This is the
    composition primitive DSH5-05's transaction executor folds over; see
    that module's ``_compose_prepared_outputs``.
    """
    if left == right:
        return left
    if left == base:
        return right
    if right == base:
        return left
    if _MISSING in (base, left, right):
        raise StructuralMergeConflict(MergeConflictKind.DELETE_MODIFY)
    if type(base) is not type(left) or type(base) is not type(right):
        raise StructuralMergeConflict(MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT)
    if is_dataclass(base) and not isinstance(base, type):
        changes = {
            field.name: (
                (None if field.name == "serialized" else getattr(base, field.name))
                if field.name in _DERIVED_FIELDS
                else merge_ast_value(
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
            (key, merged)
            for key in sorted(keys, key=str)
            if (
                merged := merge_ast_value(
                    base.get(key, _MISSING),
                    left.get(key, _MISSING),
                    right.get(key, _MISSING),
                )
            )
            is not _MISSING
        )
    if isinstance(base, (tuple, list)):
        if len(base) != len(left) or len(base) != len(right):
            raise StructuralMergeConflict(MergeConflictKind.CHILD_ORDER)
        values = (
            merge_ast_value(base[index], left[index], right[index])
            for index in range(len(base))
        )
        return type(base)(values)
    raise StructuralMergeConflict(MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT)


__all__ = [
    "MergeConflictKind",
    "StructuralMergeConflict",
    "merge_ast_value",
]
