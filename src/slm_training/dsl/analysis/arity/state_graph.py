"""Bounded OpenUI grammar/scope state graph and exact continuation quotient.

CAP1-01 uses the existing choice-codec owner (``ChoiceDecodeState``) as the
source of legal next actions. It explores a finite prefix graph, collapses
deterministic forced suffixes, fingerprints each node with only
compiler-deterministic information, and minimizes the acyclic graph bottom-up.

A profile declares the finite bounds. If any bound is hit or the underlying
owner reports incomplete coverage, the resulting report is marked ``UNKNOWN``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from slm_training.dsl.analysis.arity.profiles import AnalysisProfile
from slm_training.models.choice_tokenizer import (
    LIST_CLOSE,
    OBJ_CLOSE,
    OPEN_PREFIX,
    SLOT_PREFIX,
    ChoiceDecodeState,
    ChoiceTokenizer,
    _component_contracts,
    is_choice_tokenizer,
)

STATE_GRAPH_VERSION = "cap1-01-v1"


@dataclass(frozen=True)
class StateFingerprint:
    """Canonical, hashable summary of a grammar/scope state under a profile.

    Includes only information that determines future compiler behavior:
    pushdown signature, active terminals, schema context, remaining budget, and
    structural counters. Excludes surface spelling, absolute binder names,
    object identity, and model scores.
    """

    profile_id: str
    representation: str
    version: str
    signature: tuple[object, ...]
    schema_context: tuple[tuple[str, int, str | None], ...]
    active_terminals: tuple[str, ...]
    remaining_decisions: int
    component_count: int
    list_depth: int
    object_depth: int
    flags: tuple[str, ...] = ()

    def explain(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "representation": self.representation,
            "version": self.version,
            "signature": repr(self.signature),
            "schema_context": list(self.schema_context),
            "active_terminals": list(self.active_terminals),
            "remaining_decisions": self.remaining_decisions,
            "component_count": self.component_count,
            "list_depth": self.list_depth,
            "object_depth": self.object_depth,
            "flags": list(self.flags),
        }

    def digest(self) -> str:
        payload = json.dumps(
            {
                "profile_id": self.profile_id,
                "representation": self.representation,
                "version": self.version,
                "signature": repr(self.signature),
                "schema_context": self.schema_context,
                "active_terminals": self.active_terminals,
                "remaining_decisions": self.remaining_decisions,
                "component_count": self.component_count,
                "list_depth": self.list_depth,
                "object_depth": self.object_depth,
                "flags": self.flags,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload, usedforsecurity=False).hexdigest()[:32]

    def __repr__(self) -> str:
        return f"FP({self.digest()})"


@dataclass(frozen=True)
class GraphEdge:
    """One replayable transition between fingerprinted states."""

    label: str
    witness: tuple[int, ...]
    target: StateFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "witness": list(self.witness),
            "target": self.target.digest(),
        }


@dataclass
class GraphNode:
    """Mutable exploration node holding the concrete owner state."""

    fingerprint: StateFingerprint
    state: ChoiceDecodeState
    remaining: int
    component_count: int
    edges: list[GraphEdge] = field(default_factory=list)
    terminal: bool = False
    invalid: bool = False
    unknown: bool = False
    pruned: bool = False


@dataclass(frozen=True)
class StateGraphReport:
    """Exact or partial report for a bounded state-graph run."""

    profile: AnalysisProfile
    state_graph_version: str
    dsl: str
    slot_contract: tuple[str, ...]
    exact: bool
    status: str
    raw_states: int
    minimized_states: int
    transition_count: int
    terminal_count: int
    invalid_count: int
    unknown_count: int
    branching_histogram: tuple[tuple[int, int], ...]
    forced_decision_histogram: tuple[tuple[int, int], ...]
    work_counters: dict[str, int]
    nodes: tuple[dict[str, Any], ...]
    minimized_classes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "state_graph_version": self.state_graph_version,
            "dsl": self.dsl,
            "slot_contract": list(self.slot_contract),
            "exact": self.exact,
            "status": self.status,
            "raw_states": self.raw_states,
            "minimized_states": self.minimized_states,
            "transition_count": self.transition_count,
            "terminal_count": self.terminal_count,
            "invalid_count": self.invalid_count,
            "unknown_count": self.unknown_count,
            "branching_histogram": {
                str(k): v for k, v in self.branching_histogram
            },
            "forced_decision_histogram": {
                str(k): v for k, v in self.forced_decision_histogram
            },
            "work_counters": dict(self.work_counters),
            "nodes": list(self.nodes),
            "minimized_classes": list(self.minimized_classes),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def one_line_summary(self) -> str:
        return (
            f"{self.profile.profile_id}: {self.status} "
            f"raw={self.raw_states} min={self.minimized_states} "
            f"edges={self.transition_count} "
            f"T={self.terminal_count} I={self.invalid_count} U={self.unknown_count}"
        )


class StateGraph:
    """Explore a bounded state graph over the choice-codec owner."""

    def __init__(
        self,
        profile: AnalysisProfile,
        tokenizer: ChoiceTokenizer,
        slot_contract: tuple[str, ...] = (),
    ) -> None:
        if not is_choice_tokenizer(tokenizer):
            raise TypeError("StateGraph requires a ChoiceTokenizer")
        self.profile = profile
        self.tokenizer = tokenizer
        self.slot_contract = slot_contract
        self._terminal_fp = StateFingerprint(
            profile_id=profile.profile_id,
            representation=profile.representation,
            version=STATE_GRAPH_VERSION,
            signature=(),
            schema_context=(),
            active_terminals=("EOS",),
            remaining_decisions=-1,
            component_count=-1,
            list_depth=-1,
            object_depth=-1,
            flags=("terminal",),
        )
        self._invalid_fp = StateFingerprint(
            profile_id=profile.profile_id,
            representation=profile.representation,
            version=STATE_GRAPH_VERSION,
            signature=(),
            schema_context=(),
            active_terminals=(),
            remaining_decisions=-1,
            component_count=-1,
            list_depth=-1,
            object_depth=-1,
            flags=("invalid",),
        )
        self._unknown_fp = StateFingerprint(
            profile_id=profile.profile_id,
            representation=profile.representation,
            version=STATE_GRAPH_VERSION,
            signature=(),
            schema_context=(),
            active_terminals=(),
            remaining_decisions=-1,
            component_count=-1,
            list_depth=-1,
            object_depth=-1,
            flags=("unknown",),
        )
        self.nodes: dict[StateFingerprint, GraphNode] = {
            self._terminal_fp: GraphNode(
                fingerprint=self._terminal_fp,
                state=ChoiceDecodeState(tokenizer, slot_count=0),
                remaining=-1,
                component_count=-1,
                terminal=True,
            ),
            self._invalid_fp: GraphNode(
                fingerprint=self._invalid_fp,
                state=ChoiceDecodeState(tokenizer, slot_count=0),
                remaining=-1,
                component_count=-1,
                invalid=True,
            ),
            self._unknown_fp: GraphNode(
                fingerprint=self._unknown_fp,
                state=ChoiceDecodeState(tokenizer, slot_count=0),
                remaining=-1,
                component_count=-1,
                unknown=True,
            ),
        }
        self._work: dict[str, int] = {
            "states_seen": 0,
            "edges_explored": 0,
            "forced_tokens_collapsed": 0,
            "pruned_actions": 0,
        }
        self._any_pruned = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explore(self) -> StateGraphReport:
        """Run deterministic BFS/DFS exploration and minimization."""
        start_state = ChoiceDecodeState(
            self.tokenizer, slot_count=len(self.slot_contract)
        )
        start_fp = self._fingerprint(
            start_state, self.profile.max_semantic_decisions, 0
        )
        self.nodes[start_fp] = GraphNode(
            fingerprint=start_fp,
            state=start_state,
            remaining=self.profile.max_semantic_decisions,
            component_count=0,
        )

        queue: list[StateFingerprint] = [start_fp]
        while queue:
            fp = queue.pop(0)
            node = self.nodes[fp]
            if node.edges or node.terminal or node.invalid or node.unknown:
                continue
            self._expand(node)
            for edge in node.edges:
                if edge.target not in self.nodes:
                    raise RuntimeError(
                        f"target {edge.target} was not materialized for {edge.label}"
                    )
                queue.append(edge.target)

        return self._build_report()

    def replay(self, fp: StateFingerprint, witness: tuple[int, ...]) -> StateFingerprint:
        """Replay a witness and return the resulting fingerprint."""
        node = self.nodes[fp]
        state = node.state.clone()
        remaining = node.remaining
        component_count = node.component_count
        for token_id in witness:
            if not state.advance_id(token_id):
                raise ValueError(
                    f"replay failed at token {token_id} from {fp}"
                )
            remaining -= 1
            if self._is_component(token_id):
                component_count += 1
        return self._fingerprint(state, remaining, component_count)

    # ------------------------------------------------------------------
    # Exploration internals
    # ------------------------------------------------------------------

    def _expand(self, node: GraphNode) -> None:
        self._work["states_seen"] += 1
        allowed, pruned_here = self._filtered_allowed_ids(
            node.state, node.remaining, node.component_count
        )
        if pruned_here:
            node.pruned = True
            self._any_pruned = True
            self._work["pruned_actions"] += pruned_here

        # Terminal action: EOS is always a separate edge when legal.
        if self.tokenizer.eos_id in allowed and node.state.can_end():
            node.terminal = True
            node.edges.append(
                GraphEdge(
                    label="EOS",
                    witness=(self.tokenizer.eos_id,),
                    target=self._terminal_fp,
                )
            )
            self._work["edges_explored"] += 1
            allowed = allowed - {self.tokenizer.eos_id}

        if not allowed:
            if not node.terminal:
                node.invalid = True
            return

        for token_id in sorted(allowed):
            result = self._apply_action(
                node.state, token_id, node.remaining, node.component_count
            )
            if result is None:
                continue
            new_state, new_remaining, suffix, new_component_count = result
            label = self.tokenizer.id_to_token.get(token_id, str(token_id))
            if suffix:
                label = f"{label}+[{len(suffix)} forced]"
                self._work["forced_tokens_collapsed"] += len(suffix)

            # Hard profile checks after the full action+suffix.
            bound_violation = self._check_bounds(new_state, new_remaining, new_component_count)
            if bound_violation:
                node.unknown = True
                self._any_pruned = True
                self._work["pruned_actions"] += 1
                continue

            target_fp = self._fingerprint(
                new_state, new_remaining, new_component_count
            )
            if target_fp not in self.nodes:
                self.nodes[target_fp] = GraphNode(
                    fingerprint=target_fp,
                    state=new_state,
                    remaining=new_remaining,
                    component_count=new_component_count,
                )
            witness = (token_id,) + suffix
            node.edges.append(
                GraphEdge(label=label, witness=witness, target=target_fp)
            )
            self._work["edges_explored"] += 1

    def _apply_action(
        self,
        state: ChoiceDecodeState,
        token_id: int,
        remaining: int,
        component_count: int,
    ) -> tuple[ChoiceDecodeState, int, tuple[int, ...], int] | None:
        """Advance one semantic decision and collapse any forced suffix."""
        new_state = state.clone()
        if not new_state.advance_id(token_id):
            return None
        consumed = 1
        new_component_count = component_count + (
            1 if self._is_component(token_id) else 0
        )
        suffix: list[int] = []
        while consumed < remaining:
            allowed, pruned = self._filtered_allowed_ids(
                new_state,
                remaining - consumed,
                new_component_count,
                record_pruned=False,
            )
            non_eos = allowed - {self.tokenizer.eos_id}
            if len(non_eos) != 1:
                break
            forced_id = next(iter(non_eos))
            probe = new_state.clone()
            if not probe.advance_id(forced_id):
                break
            # A forced action that the profile would prune ends the suffix here;
            # the successor node will be marked unknown when expanded.
            if self._would_prune(probe, remaining - consumed - 1, new_component_count):
                break
            new_state = probe
            suffix.append(forced_id)
            consumed += 1
            new_component_count += 1 if self._is_component(forced_id) else 0
        return new_state, remaining - consumed, tuple(suffix), new_component_count

    def _filtered_allowed_ids(
        self,
        state: ChoiceDecodeState,
        remaining: int,
        component_count: int,
        *,
        record_pruned: bool = True,
    ) -> tuple[set[int], int]:
        """Return allowed ids after applying profile constraints.

        Returns the allowed set plus the count of actions truncated by a hard
        bound (as opposed to silently removed by an intentional language
        restriction such as ``allowed_component_subset``).
        """
        truncated = 0
        if remaining <= 0:
            return set(), 0
        allowed = set(state.allowed_ids(remaining))
        specials = {
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            self.tokenizer.mask_id,
            self.tokenizer.unk_id,
        }
        allowed -= specials

        subset = set(self.profile.allowed_component_subset)
        result: set[int] = set()
        for token_id in allowed:
            token = self.tokenizer.id_to_token.get(token_id, "")
            reason = self._prune_reason(
                token, token_id, component_count, subset
            )
            if reason == "component_not_in_subset":
                # Intentional language restriction: not part of the bounded Q.
                continue
            if reason and record_pruned:
                truncated += 1
                continue
            result.add(token_id)
        return result, truncated

    def _prune_reason(
        self,
        token: str,
        token_id: int,
        component_count: int,
        subset: set[str],
    ) -> str | None:
        if token.startswith(OPEN_PREFIX):
            component = token[len(OPEN_PREFIX) :]
            if subset and component not in subset:
                return "component_not_in_subset"
            if component_count >= self.profile.max_components:
                return "max_components"
        if token.startswith(SLOT_PREFIX):
            try:
                slot = int(token[len(SLOT_PREFIX) :])
            except ValueError:
                return "invalid_slot"
            if slot >= self.profile.max_literal_slots:
                return "max_literal_slots"
        return None

    def _would_prune(
        self, state: ChoiceDecodeState, remaining: int, component_count: int
    ) -> bool:
        """Quick check: would expanding this state hit a profile boundary?"""
        if remaining < 0:
            return True
        if component_count > self.profile.max_components:
            return True
        if len(state.section_types) > self.profile.max_live_bindings:
            return True
        return False

    def _check_bounds(
        self,
        state: ChoiceDecodeState,
        remaining: int,
        component_count: int,
    ) -> str | None:
        if remaining < 0:
            return "max_semantic_decisions"
        if component_count > self.profile.max_components:
            return "max_components"
        if len(state.section_types) > self.profile.max_live_bindings:
            return "max_live_bindings"
        list_depth = sum(
            1 for f in state.frames if f.close == LIST_CLOSE
        )
        object_depth = sum(
            1 for f in state.frames if f.close == OBJ_CLOSE
        )
        if list_depth > self.profile.max_list_items:
            return "max_list_items"
        if object_depth > self.profile.max_object_members:
            return "max_object_members"
        return None

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    def _fingerprint(
        self,
        state: ChoiceDecodeState,
        remaining: int,
        component_count: int,
        *,
        flags: tuple[str, ...] = (),
    ) -> StateFingerprint:
        active, _ = self._filtered_allowed_ids(
            state, max(1, remaining), component_count, record_pruned=False
        )
        active_terminals = tuple(
            sorted(self.tokenizer.id_to_token.get(tid, str(tid)) for tid in active)
        )
        return StateFingerprint(
            profile_id=self.profile.profile_id,
            representation=self.profile.representation,
            version=STATE_GRAPH_VERSION,
            signature=state.signature(),
            schema_context=self._schema_context(state),
            active_terminals=active_terminals,
            remaining_decisions=remaining,
            component_count=component_count,
            list_depth=sum(1 for f in state.frames if f.close == LIST_CLOSE),
            object_depth=sum(1 for f in state.frames if f.close == OBJ_CLOSE),
            flags=flags,
        )

    def _schema_context(
        self, state: ChoiceDecodeState
    ) -> tuple[tuple[str, int, str | None], ...]:
        contracts = _component_contracts()
        ctx: list[tuple[str, int, str | None]] = []
        for frame in state.frames:
            if frame.kind != "component":
                continue
            component = frame.expr_type.split(":", 1)[-1]
            contract = contracts.get(component)
            if contract is None:
                continue
            schemas, _required = contract
            if frame.arg_index >= len(schemas):
                schema_type: str | None = None
            else:
                schema = schemas[frame.arg_index]
                schema_type = str(schema.get("type") or schema.get("$ref"))
            ctx.append((component, frame.arg_index, schema_type))
        return tuple(ctx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_component(self, token_id: int) -> bool:
        token = self.tokenizer.id_to_token.get(token_id, "")
        return token.startswith(OPEN_PREFIX)

    # ------------------------------------------------------------------
    # Minimization and report
    # ------------------------------------------------------------------

    def _minimize(self) -> dict[StateFingerprint, str]:
        """Bottom-up partition refinement over the acyclic budget DAG."""
        class_map: dict[StateFingerprint, str] = {
            self._terminal_fp: "TERMINAL",
            self._invalid_fp: "INVALID",
            self._unknown_fp: "UNKNOWN",
        }
        # Sort by remaining budget descending so successors are classified first.
        for fp in sorted(
            self.nodes,
            key=lambda x: x.remaining_decisions,
            reverse=True,
        ):
            node = self.nodes[fp]
            if node.unknown or node.pruned:
                class_map[fp] = "UNKNOWN"
                continue
            if not node.edges:
                class_map[fp] = "INVALID"
                continue
            signature_key = self._node_signature_key(node, class_map)
            class_map[fp] = self._class_id(signature_key)
        return class_map

    def _node_signature_key(
        self, node: GraphNode, class_map: dict[StateFingerprint, str]
    ) -> tuple[bool, frozenset[tuple[str, str]]]:
        edge_classes = frozenset(
            (edge.label, class_map.get(edge.target, "UNKNOWN"))
            for edge in node.edges
        )
        return (node.terminal, edge_classes)

    def _class_id(self, key: tuple[bool, frozenset[tuple[str, str]]]) -> str:
        payload = json.dumps(
            {"terminal": key[0], "edges": sorted(key[1])},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload, usedforsecurity=False).hexdigest()[:16]

    def _build_report(self) -> StateGraphReport:
        class_map = self._minimize()
        sink_fingerprints = {
            self._terminal_fp,
            self._invalid_fp,
            self._unknown_fp,
        }
        explored_nodes = {
            fp: node
            for fp, node in self.nodes.items()
            if fp not in sink_fingerprints
        }
        raw_states = len(explored_nodes)
        classes = {
            cls for fp, cls in class_map.items() if fp not in sink_fingerprints
        }
        minimized_states = len(classes)
        transition_count = sum(
            len(n.edges) for n in explored_nodes.values()
        )
        terminal_count = sum(1 for n in explored_nodes.values() if n.terminal)
        invalid_count = sum(1 for n in explored_nodes.values() if n.invalid)
        unknown_count = sum(
            1 for n in explored_nodes.values() if n.unknown or n.pruned
        )
        branching = Counter(len(n.edges) for n in explored_nodes.values())
        forced = Counter(
            len(edge.witness) - 1
            for n in explored_nodes.values()
            for edge in n.edges
        )

        exact = (
            not self._any_pruned
            and self.profile.required_coverage == "complete"
            and unknown_count == 0
        )

        node_dicts: list[dict[str, Any]] = []
        for fp in sorted(explored_nodes, key=lambda x: x.digest()):
            node = explored_nodes[fp]
            node_dicts.append(
                {
                    "fingerprint": fp.digest(),
                    "explain": fp.explain(),
                    "class": class_map.get(fp, "UNKNOWN"),
                    "terminal": node.terminal,
                    "invalid": node.invalid,
                    "unknown": node.unknown or node.pruned,
                    "edges": [edge.to_dict() for edge in node.edges],
                }
            )

        class_members: dict[str, list[str]] = {}
        for fp, cls in class_map.items():
            class_members.setdefault(cls, []).append(fp.digest())
        minimized_classes = tuple(
            {"class": cls, "members": sorted(members)}
            for cls, members in sorted(class_members.items())
        )

        return StateGraphReport(
            profile=self.profile,
            state_graph_version=STATE_GRAPH_VERSION,
            dsl=self.profile.dsl,
            slot_contract=self.slot_contract,
            exact=exact,
            status="EXACT" if exact else "UNKNOWN",
            raw_states=raw_states,
            minimized_states=minimized_states,
            transition_count=transition_count,
            terminal_count=terminal_count,
            invalid_count=invalid_count,
            unknown_count=unknown_count,
            branching_histogram=tuple(sorted(branching.items())),
            forced_decision_histogram=tuple(sorted(forced.items())),
            work_counters=dict(self._work),
            nodes=tuple(node_dicts),
            minimized_classes=minimized_classes,
        )


__all__ = [
    "StateFingerprint",
    "GraphEdge",
    "GraphNode",
    "StateGraphReport",
    "StateGraph",
    "STATE_GRAPH_VERSION",
]
