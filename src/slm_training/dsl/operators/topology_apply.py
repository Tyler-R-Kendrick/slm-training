"""Verified operator edits as topology-diffusion edit actions (DSH3-16, wiring-only).

Maps applicable, compiler-verified ``ActionEffectV1`` transitions from a live
``OperatorLegalSetV1`` onto typed-topology expand / contract / move / rewrite
edits, so a topology-diffusion decode loop can re-materialize only the nodes an
edit actually touched instead of recomputing the whole typed tree after every
ordinary lowering step.

Authority boundaries (non-negotiable, per AGENTS.md):

- Only **exact** legal actions are applied directly: the action must be a
  member of the live ``OperatorLegalSetV1`` and its effect must carry
  ``CompilerCoverage.EXACT``. Anything else returns ``DEFER`` and the caller
  keeps the ordinary production path (ordinary lowering + full
  re-materialization). Exact fallback therefore preserves completeness.
- Exact-singleton force emission (``OperatorLegalSetV1.forced_action``) and
  head ``DEFER`` dispositions are untouched -- this module never ranks, never
  scores, and never runs a model forward.
- The new source of truth after every edit remains the compiler-validated
  ``OperatorApplyResultV1.state``; the typed tree is a *view* whose edits are
  certified against the effect's declared target region. A diff that reaches
  outside the effect's declared region, or any stale request/branch/state
  identity, fails closed (raises) rather than returning uncertified topology.

Scope: wiring/fixture-scale integration plus instrumentation. It is not a ship
claim, not a systems claim, and serialized target compression is never reported
here as inference speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping

from slm_training.dsl.operators.contracts import (
    CompilerCoverage,
    NodeRef,
    _fingerprint,
)
from slm_training.dsl.operators.legal_set import (
    LegalOperatorActionV1,
    OperatorLegalSetV1,
)
from slm_training.dsl.operators.local import (
    ADD_CHILD,
    REMOVE_NODE,
    REORDER_CHILDREN,
    REPLACE_NODE,
    SET_PROPERTY,
    UNSET_PROPERTY,
    NodeLocationV1,
    OpenUILocalOperatorContextV1,
)
from slm_training.dsl.operators.registry import (
    OperatorApplyResultV1,
    OperatorStateV1,
)
from slm_training.dsl.operators.topology import (
    CONTRACT_SUBTREE,
    DUPLICATE_SUBTREE,
    EXPAND_TEMPLATE,
    MOVE_NODE,
    REPARENT_NODE,
    UNWRAP_NODE,
    WRAP_NODE,
)

__all__ = [
    "TOPOLOGY_APPLY_SCHEMA_VERSION",
    "StaleTopologyStateError",
    "TopologyApplyConfigV1",
    "TopologyApplyMode",
    "TopologyApplySession",
    "TopologyApplyStatsV1",
    "TopologyEditKind",
    "TopologyEditV1",
    "TypedTopologyNodeV1",
    "map_operator_to_edit_kind",
    "materialize_topology_tree",
]

TOPOLOGY_APPLY_SCHEMA_VERSION = "topology_apply/v1"

TopologyApplyMode = Literal["off", "topology_apply"]


class StaleTopologyStateError(ValueError):
    """A request/branch/state identity crossed a lifecycle boundary."""


class TopologyEditKind(str, Enum):
    """Topology-diffusion edit action a verified operator transition maps to."""

    EXPAND = "expand"
    CONTRACT = "contract"
    MOVE = "move"
    REWRITE = "rewrite"


_OPERATOR_EDIT_KINDS: Mapping[str, TopologyEditKind] = {
    MOVE_NODE: TopologyEditKind.MOVE,
    REPARENT_NODE: TopologyEditKind.MOVE,
    REORDER_CHILDREN: TopologyEditKind.MOVE,
    DUPLICATE_SUBTREE: TopologyEditKind.EXPAND,
    ADD_CHILD: TopologyEditKind.EXPAND,
    EXPAND_TEMPLATE: TopologyEditKind.EXPAND,
    WRAP_NODE: TopologyEditKind.EXPAND,
    REMOVE_NODE: TopologyEditKind.CONTRACT,
    CONTRACT_SUBTREE: TopologyEditKind.CONTRACT,
    UNWRAP_NODE: TopologyEditKind.CONTRACT,
    REPLACE_NODE: TopologyEditKind.REWRITE,
    SET_PROPERTY: TopologyEditKind.REWRITE,
    UNSET_PROPERTY: TopologyEditKind.REWRITE,
}

# Argument slots whose node refs declare the region an edit may touch.
_TARGET_SLOTS = ("node", "parent", "new_parent")


def map_operator_to_edit_kind(operator_id: str) -> TopologyEditKind | None:
    """Map one operator id to its topology-diffusion edit kind, or ``None``.

    ``None`` means the operator has no certified direct mapping and the caller
    must keep the ordinary lowering path (DEFER). The mapping never decides
    legality -- it only classifies transitions the compiler already verified.
    """
    return _OPERATOR_EDIT_KINDS.get(operator_id)


@dataclass(frozen=True)
class TopologyApplyConfigV1:
    """Pinned, default-off configuration for topology-apply integration.

    ``mode="off"`` (the default) makes ``TopologyApplySession.apply_direct`` a
    true no-op: it always returns ``False`` (DEFER) and the ordinary lowering
    path is completely untouched.
    """

    mode: TopologyApplyMode = "off"
    schema: str = TOPOLOGY_APPLY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in ("off", "topology_apply"):
            raise ValueError(f"unknown topology-apply mode: {self.mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "mode": self.mode}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopologyApplyConfigV1":
        return cls(mode=str(data.get("mode", "off")))  # type: ignore[arg-type]


@dataclass(frozen=True)
class TypedTopologyNodeV1:
    """Immutable typed view of one element subtree of a statement-bindings AST.

    ``digest`` is the canonical fingerprint of the underlying element subtree,
    so an unchanged subtree is re-used (zero node passes) across edits and a
    changed subtree is rebuilt exactly. ``roles`` aligns with ``children``.
    """

    type_name: str
    digest: str
    roles: tuple[str, ...] = ()
    children: tuple["TypedTopologyNodeV1", ...] = ()
    schema: str = "typed_topology_node/v1"

    @property
    def subtree_size(self) -> int:
        return 1 + sum(child.subtree_size for child in self.children)


@dataclass(frozen=True)
class TopologyEditV1:
    """One certified direct edit applied to the typed topology tree."""

    kind: TopologyEditKind
    operator_id: str
    application_id: str
    target_paths: tuple[tuple[Any, ...], ...]
    materialized_nodes: int
    schema: str = "topology_edit/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "operator_id": self.operator_id,
            "application_id": self.application_id,
            "target_paths": [list(path) for path in self.target_paths],
            "materialized_nodes": self.materialized_nodes,
        }


@dataclass
class TopologyApplyStatsV1:
    """DecodeStats-like work counters for one topology-apply session."""

    active_nodes: int = 0
    node_passes: int = 0
    remask_phases: int = 0
    rewrite_phases: int = 0
    model_forwards: int = 0
    compiler_expansions: int = 0
    verifier_calls: int = 0
    direct_applies: int = 0
    deferred_applies: int = 0
    schema: str = "topology_apply_stats/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "active_nodes": self.active_nodes,
            "node_passes": self.node_passes,
            "remask_phases": self.remask_phases,
            "rewrite_phases": self.rewrite_phases,
            "model_forwards": self.model_forwards,
            "compiler_expansions": self.compiler_expansions,
            "verifier_calls": self.verifier_calls,
            "direct_applies": self.direct_applies,
            "deferred_applies": self.deferred_applies,
        }


def _is_element(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "element"


def _element_digest(element: Mapping[str, Any]) -> str:
    return _fingerprint({"schema": "typed_topology_node/v1", "element": element})


def _materialize(element: Mapping[str, Any]) -> tuple[TypedTopologyNodeV1, int]:
    """Build the typed view of one element subtree; returns (node, passes)."""
    roles: list[str] = []
    children: list[TypedTopologyNodeV1] = []
    passes = 1
    for name, value in (element.get("props") or {}).items():
        if not isinstance(value, list):
            continue
        for item in value:
            if _is_element(item):
                child, child_passes = _materialize(item)
                roles.append(name)
                children.append(child)
                passes += child_passes
    node = TypedTopologyNodeV1(
        type_name=str(element.get("typeName") or ""),
        digest=_element_digest(element),
        roles=tuple(roles),
        children=tuple(children),
    )
    return node, passes


def materialize_topology_tree(
    bindings: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[TypedTopologyNodeV1, ...], int]:
    """Fully materialize a statement-bindings AST; returns (names, roots, passes)."""
    names: list[str] = []
    roots: list[TypedTopologyNodeV1] = []
    passes = 0
    for name in sorted(bindings):
        element = bindings[name]
        if not _is_element(element):
            raise ValueError(f"statement {name!r} is not an element subtree")
        node, node_passes = _materialize(element)
        names.append(name)
        roots.append(node)
        passes += node_passes
    return tuple(names), tuple(roots), passes


def _element_at(bindings: Mapping[str, Any], path: tuple[Any, ...]) -> Mapping[str, Any]:
    value: Any = bindings
    for segment in path:
        value = value[segment]
    if not _is_element(value):
        raise ValueError("topology path does not resolve to an element")
    return value


def _rematerialize(
    node: TypedTopologyNodeV1,
    element: Mapping[str, Any],
    path: tuple[Any, ...],
    rebuilt: list[tuple[Any, ...]],
) -> tuple[TypedTopologyNodeV1, int]:
    """Digest-guided minimal re-materialization of ``node`` against ``element``.

    Unchanged subtrees (identical canonical digest) are reused with zero node
    passes; only changed wrappers are rebuilt. Every rebuilt wrapper records
    its path in ``rebuilt`` for the effect-region certification check.
    """
    if node.digest == _element_digest(element):
        return node, 0
    rebuilt.append(path)
    roles: list[str] = []
    children: list[TypedTopologyNodeV1] = []
    passes = 1
    old_by_role: dict[str, list[TypedTopologyNodeV1]] = {}
    for role, child in zip(node.roles, node.children, strict=True):
        old_by_role.setdefault(role, []).append(child)
    consumed: dict[str, int] = {}
    for name, value in (element.get("props") or {}).items():
        if not isinstance(value, list):
            continue
        for index, item in enumerate(value):
            if not _is_element(item):
                continue
            old_index = consumed.get(name, 0)
            consumed[name] = old_index + 1
            old_children = old_by_role.get(name) or []
            child_path = (*path, "props", name, index)
            if old_index < len(old_children):
                child, child_passes = _rematerialize(
                    old_children[old_index], item, child_path, rebuilt
                )
            else:
                child, child_passes = _materialize(item)
                rebuilt.append(child_path)
            roles.append(name)
            children.append(child)
            passes += child_passes
    return (
        TypedTopologyNodeV1(
            type_name=str(element.get("typeName") or ""),
            digest=_element_digest(element),
            roles=tuple(roles),
            children=tuple(children),
        ),
        passes,
    )


def _is_prefix(prefix: tuple[Any, ...], path: tuple[Any, ...]) -> bool:
    return len(prefix) <= len(path) and path[: len(prefix)] == prefix


class TopologyApplySession:
    """Request/branch-scoped typed-topology lifecycle for direct verified edits.

    The session is bound to exactly one ``request_id`` and one
    ``branch_digest`` at a time. ``begin``/``begin_branch`` invalidate every
    previously held node; any apply whose recorded before-state, request, or
    legal-set identity does not match the live session fails closed with
    ``StaleTopologyStateError`` -- no stale topology state can cross a request
    or branch boundary.
    """

    def __init__(
        self,
        config: TopologyApplyConfigV1 | None = None,
        *,
        request_id: str,
        branch_digest: str,
    ) -> None:
        self.config = config or TopologyApplyConfigV1()
        self.request_id = request_id
        self.branch_digest = branch_digest
        self.stats = TopologyApplyStatsV1()
        self.edits: list[TopologyEditV1] = []
        self._state_digest: str | None = None
        self._ast_digest: str | None = None
        self._root_names: tuple[str, ...] = ()
        self._roots: tuple[TypedTopologyNodeV1, ...] = ()

    @property
    def state_digest(self) -> str | None:
        return self._state_digest

    @property
    def root_names(self) -> tuple[str, ...]:
        return self._root_names

    @property
    def roots(self) -> tuple[TypedTopologyNodeV1, ...]:
        return self._roots

    @property
    def enabled(self) -> bool:
        return self.config.mode == "topology_apply"

    def begin(self, state: OperatorStateV1, bindings: Mapping[str, Any]) -> None:
        """(Re)initialize the session tree from an authoritative state."""
        self._root_names, self._roots, passes = materialize_topology_tree(bindings)
        self._state_digest = state.state_digest
        self._ast_digest = state.ast_digest
        self.stats.active_nodes = passes
        self.stats.node_passes += passes

    def begin_branch(
        self,
        branch_digest: str,
        state: OperatorStateV1,
        bindings: Mapping[str, Any],
    ) -> None:
        """Open a new branch, invalidating every node of the previous branch."""
        self.branch_digest = branch_digest
        self._state_digest = None
        self._ast_digest = None
        self._root_names = ()
        self._roots = ()
        self.begin(state, bindings)

    def _legal_state_fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": "operator_legal_state/v1",
                "state_digest": self._state_digest,
                "ast_digest": self._ast_digest,
            }
        )

    def _check_fresh(
        self,
        *,
        result: OperatorApplyResultV1,
        action: LegalOperatorActionV1,
        legal_set: OperatorLegalSetV1,
    ) -> None:
        if self._state_digest is None or self._ast_digest is None:
            raise StaleTopologyStateError("topology apply before session begin")
        application = result.application
        if not result.succeeded or result.state is None:
            raise ValueError("cannot direct-apply a rejected operator application")
        if application.provenance.request_id != self.request_id:
            raise StaleTopologyStateError(
                "operator application belongs to another request"
            )
        if (
            application.before_state_digest != self._state_digest
            or application.before_ast_digest != self._ast_digest
        ):
            raise StaleTopologyStateError(
                "operator application was recorded against a stale state"
            )
        if legal_set.state_fingerprint != self._legal_state_fingerprint():
            raise StaleTopologyStateError("legal set is stale for the session state")
        if action.application_id not in {
            candidate.application_id for candidate in legal_set.operator_actions
        }:
            raise ValueError("action is not a member of the live legal set")

    def _target_paths(
        self,
        action: LegalOperatorActionV1,
        context: OpenUILocalOperatorContextV1,
    ) -> tuple[tuple[Any, ...], ...]:
        kind = map_operator_to_edit_kind(action.operator_id)
        paths: list[tuple[Any, ...]] = []
        for argument in action.arguments:
            if argument.slot_id not in _TARGET_SLOTS:
                continue
            if not isinstance(argument.value, NodeRef):
                continue
            payload = context.local.payload(argument.value)
            if not isinstance(payload, NodeLocationV1):
                raise ValueError("legal action target does not resolve to a node")
            path = tuple(payload.path)
            if (
                (kind is TopologyEditKind.MOVE or action.operator_id == REMOVE_NODE)
                and len(path) > 1
                and isinstance(path[-1], int)
            ):
                # Moves/removals shift list siblings; the certified region is
                # the owning parent element, not only the touched child.
                path = path[:-3]
            if path not in paths:
                paths.append(path)
        if not paths:
            raise ValueError("legal action declares no node target region")
        return tuple(paths)

    def apply_direct(
        self,
        *,
        result: OperatorApplyResultV1,
        action: LegalOperatorActionV1,
        legal_set: OperatorLegalSetV1,
        context: OpenUILocalOperatorContextV1,
        new_bindings: Mapping[str, Any],
    ) -> bool:
        """Apply one exact legal action's verified edit to the typed tree.

        Returns ``True`` when the edit was applied directly, ``False`` when the
        session deferred to ordinary lowering (full re-materialization from the
        authoritative ``new_bindings``). Never weakens singleton/DEFER
        authority: callers still commit forced singletons and head DEFERs
        through the ordinary path before reaching this method.
        """
        if not self.enabled:
            return False
        self._check_fresh(result=result, action=action, legal_set=legal_set)
        application = result.application
        assert result.state is not None and application.effect is not None
        effect = application.effect
        kind = map_operator_to_edit_kind(action.operator_id)
        if (
            kind is None
            or effect.compiler_coverage is not CompilerCoverage.EXACT
            or effect.request_ids != frozenset({self.request_id})
        ):
            return self._defer(result.state, new_bindings)

        target_paths = self._target_paths(action, context)
        new_names = tuple(sorted(new_bindings))
        if new_names != self._root_names or any(
            not _is_element(new_bindings[name]) for name in new_names
        ):
            return self._defer(result.state, new_bindings)

        rebuilt: list[tuple[Any, ...]] = []
        passes = 0
        roots: list[TypedTopologyNodeV1] = []
        for name, root in zip(self._root_names, self._roots, strict=True):
            new_root, root_passes = _rematerialize(
                root, _element_at(new_bindings, (name,)), (name,), rebuilt
            )
            roots.append(new_root)
            passes += root_passes
        for path in rebuilt:
            if not any(
                _is_prefix(path, target) or _is_prefix(target, path)
                for target in target_paths
            ):
                raise StaleTopologyStateError(
                    "verified edit reached outside its declared effect region"
                )

        self._roots = tuple(roots)
        self._state_digest = result.state.state_digest
        self._ast_digest = result.state.ast_digest
        active = sum(root.subtree_size for root in self._roots)
        self.stats.active_nodes = active
        self.stats.node_passes += passes
        if kind is TopologyEditKind.REWRITE:
            self.stats.rewrite_phases += 1
        else:
            self.stats.remask_phases += 1
        self.stats.direct_applies += 1
        self.edits.append(
            TopologyEditV1(
                kind=kind,
                operator_id=action.operator_id,
                application_id=action.application_id,
                target_paths=target_paths,
                materialized_nodes=passes,
            )
        )
        return True

    def _defer(
        self,
        state: OperatorStateV1,
        new_bindings: Mapping[str, Any],
    ) -> bool:
        """Ordinary lowering fallback: full re-materialization, always exact."""
        self.begin(state, new_bindings)
        self.stats.deferred_applies += 1
        return False
