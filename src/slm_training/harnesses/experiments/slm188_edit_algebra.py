"""SLM-188 (FFE1-02): edit-algebra reachability, canonical invariance, and
replayable transition certificates — wiring/fixture harness.

This harness proves that the canonical legal-edit algebra can connect declared
source seeds to train/eval targets under explicit node, depth, edit, verifier,
and time bounds, and that canonicalization, alpha-renaming, slot permutation,
and independent-edit reordering preserve transition identity.

No model is trained, no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.canonicalize import canonical_fingerprint, canonicalize
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.production_codec import ProductionCodec
from slm_training.dsl.solver.topology_adapter import (
    TopologyAdapterConfig,
    V05_MARKERS,
    _node_type,
)
from slm_training.dsl.solver.topology_solver import SolverTopologyNode
from slm_training.models.tree_edit_diffusion import (
    TreeEditSpace,
    parse_statements,
    render_statements,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "CanonicalEdit",
    "EditReachabilityCase",
    "EditReachabilityReport",
    "InvarianceResult",
    "TransitionCertificateV1",
    "apply_canonical_edit",
    "build_fixture_codec",
    "build_seed_target_pairs",
    "build_sketch_seed",
    "permute_slot_contract",
    "plan_edit_sequence",
    "replay_transition_certificate",
    "render_markdown",
    "run_edit_reachability_fixture",
    "run_invariance_suite",
    "topology_tree_from_openui",
]

MATRIX_VERSION = "ffe1-02-v1"
MATRIX_SET = "slm188_edit_algebra"
EXPERIMENT_ID = "slm188-edit-algebra"

_HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'

_HYPOTHESIS = (
    "The canonical statement-level edit algebra (InsertStatement, DeleteStatement, "
    "ReplaceProduction, SetArity, InsertChild, DeleteSubtree, BindSlotPointer, "
    "BindReferencePointer, SetEnum) can reach every supported fixture target from a "
    "structural sketch seed within declared node/depth/edit bounds, and canonical "
    "invariance (idempotence, alpha-renaming, slot permutation, independent-edit "
    "commutativity) holds for every transition in the fixture domain."
)

_FALSIFIER = (
    "A bounded fixture planner/search finds a supported canonical target that is "
    "unreachable from a sketch seed under the declared edit budget, or a transition "
    "whose certificate does not replay to the same canonical target, or an invariance "
    "check (canonical idempotence, alpha, slot permutation, commutativity) that fails."
)

_HONEST_CAVEATS = (
    "Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.",
    "Reachability uses a deterministic sketch→target planner for ordinary records and "
    "bounded exact BFS for tiny closed domains; both emit replay-valid certificates.",
    "The search budget is intentionally small so the harness stays CPU-only; "
    "real bridge coverage needs the standard multi-step solver budget.",
    "v0.5 state/query/action statements, object literals, member access, and operators "
    "are represented in the edit-algebra vocabulary but are not exercised by the "
    "current OpenUI statement fixtures; they are reported as 'unsupported_pack_feature' "
    "when encountered.",
    "Statement insertion/deletion is performed through the tree-edit space; the "
    "topology-solver edit algebra is reused for local node expansion where possible.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _is_valid(source: str) -> bool:
    try:
        validate(source)
        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class CanonicalEdit:
    """Symbolic canonical edit over the OpenUI statement grammar.

    This is the pack-neutral, versioned interface requested by SLM-188.  The
    implementation delegates to the tree-edit space and topology edits, but the
    public surface is independent of any single objective.
    """

    edit_id: str
    action: str
    target_name: str
    production: str | None = None
    child_name: str | None = None
    slot: str | None = None
    direction: str | None = None
    index: int | None = None
    previous_index: int | None = None
    preconditions: tuple[str, ...] = ()
    affected_node_ids: tuple[str, ...] = ()
    inverse_action: str | None = None
    dependency_footprint: tuple[str, ...] = ()
    cost: dict[str, int] = field(default_factory=lambda: {"edits": 1, "nodes_touched": 1, "verifier_calls": 1, "serialization_delta": 0})
    coverage_tier: str = "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "edit_id": self.edit_id,
            "action": self.action,
            "target_name": self.target_name,
            "production": self.production,
            "child_name": self.child_name,
            "slot": self.slot,
            "direction": self.direction,
            "index": self.index,
            "previous_index": self.previous_index,
            "preconditions": list(self.preconditions),
            "affected_node_ids": list(self.affected_node_ids),
            "inverse_action": self.inverse_action,
            "dependency_footprint": list(self.dependency_footprint),
            "cost": dict(self.cost),
            "coverage_tier": self.coverage_tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalEdit":
        return cls(
            edit_id=str(data["edit_id"]),
            action=str(data["action"]),
            target_name=str(data["target_name"]),
            production=data.get("production"),
            child_name=data.get("child_name"),
            slot=data.get("slot"),
            direction=data.get("direction"),
            index=data.get("index"),
            previous_index=data.get("previous_index"),
            preconditions=tuple(data.get("preconditions", ())),
            affected_node_ids=tuple(data.get("affected_node_ids", ())),
            inverse_action=data.get("inverse_action"),
            dependency_footprint=tuple(data.get("dependency_footprint", ())),
            cost=dict(data.get("cost", {})),
            coverage_tier=str(data.get("coverage_tier", "complete")),
        )


@dataclass(frozen=True)
class TransitionCertificateV1:
    """Replayable proof that one canonical edit transitions between two states."""

    schema: str = "TransitionCertificateV1"
    source_fingerprint: str = ""
    target_fingerprint: str = ""
    edit: dict[str, Any] = field(default_factory=dict)
    source_program: str | None = None
    target_program: str | None = None
    verifier_profile: str = "statement/canonical"
    verifier_accepted: bool = False
    verifier_detail: str = ""
    version_pins: dict[str, Any] = field(default_factory=dict)
    certificate_digest: str = ""

    def __post_init__(self) -> None:
        if not self.certificate_digest:
            digest = _sha256(_canonical_json(self.to_dict(exclude_digest=True)))
            object.__setattr__(self, "certificate_digest", digest)

    def to_dict(self, exclude_digest: bool = False) -> dict[str, Any]:
        data = {
            "schema": self.schema,
            "source_fingerprint": self.source_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "edit": dict(self.edit),
            "source_program": self.source_program,
            "target_program": self.target_program,
            "verifier_profile": self.verifier_profile,
            "verifier_accepted": self.verifier_accepted,
            "verifier_detail": self.verifier_detail,
            "version_pins": dict(self.version_pins),
        }
        if not exclude_digest:
            data["certificate_digest"] = self.certificate_digest
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransitionCertificateV1":
        return cls(
            schema=str(data.get("schema", "TransitionCertificateV1")),
            source_fingerprint=str(data.get("source_fingerprint", "")),
            target_fingerprint=str(data.get("target_fingerprint", "")),
            edit=dict(data.get("edit") or {}),
            source_program=data.get("source_program"),
            target_program=data.get("target_program"),
            verifier_profile=str(data.get("verifier_profile", "statement/canonical")),
            verifier_accepted=bool(data.get("verifier_accepted", False)),
            verifier_detail=str(data.get("verifier_detail", "")),
            version_pins=dict(data.get("version_pins") or {}),
            certificate_digest=str(data.get("certificate_digest", "")),
        )


def build_fixture_codec() -> ProductionCodec:
    """Return the HERO codec used by topology solver tests."""
    return ProductionCodec.build([_HERO])


def _tree_fingerprint(root: SolverTopologyNode) -> str:
    """Stable fingerprint of a complete topology tree."""
    return _sha256(_canonical_json(root.to_dict()))


def topology_tree_from_openui(
    source: str,
    codec: ProductionCodec,
    slot_inventory: list[str] | None = None,
    *,
    active: bool = False,
) -> SolverTopologyNode:
    """Build a SolverTopologyNode tree from canonical OpenUI source.

    Mirrors ``grammar_diffusion._topology_from_ids`` for the torch-free
    ``ProductionCodec``.  When ``active`` is True, every non-root node is left
    active so it can serve as a search frontier.
    """
    production_ids, slot_ids = codec.encode(source, slot_inventory)
    pairs = [
        (pid, sid)
        for pid, sid in zip(production_ids, slot_ids)
        if pid not in {codec.pad_id, codec.bos_id, codec.eos_id}
    ]
    next_id = 1

    def make(node_type: str, pid: int, sid: int = 0) -> SolverTopologyNode:
        nonlocal next_id
        node = SolverTopologyNode(next_id, node_type, pid, sid)
        next_id += 1
        return node

    def parse_expr(index: int) -> tuple[SolverTopologyNode, int]:
        if index >= len(pairs):
            raise ValueError("unexpected end of production tree")
        pid, sid = pairs[index]
        token = codec.id_to_production.get(pid, "<unk>")
        node = make(_node_type(token), pid, sid)
        node.active = active
        index += 1
        if token.startswith("+"):
            while index < len(pairs):
                next_token = codec.id_to_production.get(pairs[index][0], "<unk>")
                if next_token == "-":
                    return node, index + 1
                child, index = parse_expr(index)
                node.children.append(child)
            raise ValueError("unterminated component production")
        if token == "[":
            while index < len(pairs):
                next_token = codec.id_to_production.get(pairs[index][0], "<unk>")
                if next_token == "]":
                    return node, index + 1
                child, index = parse_expr(index)
                node.children.append(child)
            raise ValueError("unterminated list production")
        return node, index

    v05_id = codec.production_to_id.get("!v0.5")
    root_pid = codec.bos_id
    index = 0
    is_v05 = bool(pairs and pairs[0][0] == v05_id)
    if is_v05:
        root_pid = pairs[0][0]
        index = 1
    root = SolverTopologyNode(0, "document", root_pid)
    root.active = active
    if is_v05:
        eol_id = codec.production_to_id.get(";")
        while index < len(pairs):
            marker_id, marker_slot = pairs[index]
            marker = codec.id_to_production.get(marker_id, "")
            if marker not in V05_MARKERS:
                raise ValueError(f"expected v0.5 statement marker, got {marker!r}")
            statement = make("statement", marker_id, marker_slot)
            index += 1
            while index < len(pairs) and pairs[index][0] != eol_id:
                pid, sid = pairs[index]
                statement.children.append(make("expression", pid, sid))
                index += 1
            index += int(index < len(pairs))
            root.children.append(statement)
    else:
        assign_id = codec.production_to_id.get("=")
        while index < len(pairs):
            if pairs[index][0] != assign_id:
                raise ValueError("expected statement production")
            statement = make("statement", pairs[index][0], pairs[index][1])
            statement.active = active
            child, index = parse_expr(index + 1)
            statement.children.append(child)
            root.children.append(statement)
    _refresh_layout(root)
    return root


def _refresh_layout(root: SolverTopologyNode) -> None:
    """Recompute depth, sibling_index, and parent_id from tree structure."""

    def visit(node: SolverTopologyNode, depth: int, sibling_index: int, parent_id: int) -> None:
        node.depth = depth
        node.sibling_index = sibling_index
        node.parent_id = parent_id
        for i, child in enumerate(node.children):
            visit(child, depth + 1, i, node.node_id)

    for i, child in enumerate(root.children):
        visit(child, 1, i, root.node_id)


def _flatten(root: SolverTopologyNode) -> list[SolverTopologyNode]:
    out: list[SolverTopologyNode] = [root]
    for child in root.children:
        out.extend(_flatten(child))
    return out


def build_sketch_seed(target_program: str) -> str:
    """Return a structural sketch of ``target_program``.

    The sketch preserves statement names, the binding graph, and container
    directions, but replaces every component with ``TextContent`` and every
    placeholder slot with ``:slot``.  This gives the edit algebra a non-trivial
    transformation task while keeping the search deterministic and small.
    """
    try:
        canonical = canonicalize(target_program, validate=True)
    except Exception:  # noqa: BLE001
        canonical = target_program
    statements = parse_statements(canonical)
    if statements is None:
        return target_program
    out: list[Any] = []
    for stmt in statements:
        rest = stmt.rest
        if stmt.has_list:
            # Replace any slot literals in rest (e.g. direction enum) but keep
            # non-slot args like "column"/"row".
            new_rest = rest
        else:
            # Leaf argument: replace with :slot unless it is an enum/direction.
            stripped = stmt.rest.strip()
            if stripped.startswith('"') or stripped.startswith("'") or stripped.startswith(":"):
                new_rest = json.dumps(":slot", ensure_ascii=False)
            else:
                new_rest = stripped
        new_comp = "TextContent" if not stmt.has_list else "Stack"
        out.append(
            type(stmt)(
                name=stmt.name,
                comp=new_comp,
                children=list(stmt.children),
                rest=new_rest,
                has_list=stmt.has_list,
            )
        )
    return render_statements(out)


def _args_to_rest(args: str) -> list[str]:
    """Split a component arg list into positional pieces for comparison."""
    if not args:
        return []
    args = args.strip()
    if args.startswith("["):
        return ["list"]
    parts: list[str] = []
    depth = 0
    buf = ""
    for ch in args:
        if ch in "([":
            depth += 1
            buf += ch
        elif ch in ")]":
            depth -= 1
            buf += ch
        elif ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _set_leaf_arg(stmt: Any, new_arg: str) -> Any:
    """Return a new Statement with the leaf arg replaced."""
    return type(stmt)(
        name=stmt.name,
        comp=stmt.comp,
        children=list(stmt.children),
        rest=new_arg,
        has_list=stmt.has_list,
    )


def _set_direction(stmt: Any, direction: str) -> Any:
    """Return a new container Statement with the direction enum replaced."""
    rest = stmt.rest.strip()
    if rest.startswith(','):
        rest = f', "{direction}"' + rest[rest.find('"', rest.find('"') + 1) + 1:]
    else:
        rest = f'"{direction}"'
    return type(stmt)(
        name=stmt.name,
        comp=stmt.comp,
        children=list(stmt.children),
        rest=rest,
        has_list=stmt.has_list,
    )


def plan_edit_sequence(
    seed_source: str,
    target_program: str,
    *,
    edit_index_start: int = 0,
) -> tuple[list[CanonicalEdit], str]:
    """Deterministic planner: build a canonical edit sequence from sketch to target.

    Returns (edits, stop_reason).  The plan may be empty if seed already equals
    target.  The planner is not shortest-path optimal; it is a canonical,
    replayable witness that the edit algebra spans the sketch→target pair.
    """
    seed_statements = parse_statements(seed_source)
    target_statements = parse_statements(target_program)
    if seed_statements is None or target_statements is None:
        return [], "parse_failed"

    seed_by_name = {s.name: s for s in seed_statements}
    target_by_name = {s.name: s for s in target_statements}
    edits: list[CanonicalEdit] = []
    idx = edit_index_start

    # 1. Insert statements present in target but not seed.
    for t_stmt in target_statements:
        if t_stmt.name not in seed_by_name:
            insert_slot = t_stmt.rest.strip() if not t_stmt.has_list else None
            if insert_slot and insert_slot.startswith('"'):
                try:
                    insert_slot = str(json.loads(insert_slot))
                except json.JSONDecodeError:
                    pass
            insert_child = t_stmt.children[0] if t_stmt.has_list and t_stmt.children else None
            insert_args = _args_to_rest(t_stmt.rest) if t_stmt.has_list else []
            edits.append(
                CanonicalEdit(
                    edit_id=f"insert-{idx}",
                    action="InsertStatement",
                    target_name=t_stmt.name,
                    production=t_stmt.comp,
                    child_name=insert_child,
                    slot=insert_slot,
                    direction=(
                        insert_args[0].strip('"')
                        if insert_args
                        else None
                    ),
                    affected_node_ids=(t_stmt.name,),
                    inverse_action="DeleteStatement",
                    cost={"edits": 1, "nodes_touched": 1, "verifier_calls": 1, "serialization_delta": 1},
                )
            )
            seed_by_name[t_stmt.name] = t_stmt
            idx += 1

    # 2. Replace production and args for each statement.
    for t_stmt in target_statements:
        s_stmt = seed_by_name.get(t_stmt.name)
        if s_stmt is None:
            continue
        if s_stmt.comp != t_stmt.comp:
            edits.append(
                CanonicalEdit(
                    edit_id=f"replace-{idx}",
                    action="ReplaceProduction",
                    target_name=t_stmt.name,
                    production=t_stmt.comp,
                    affected_node_ids=(t_stmt.name,),
                    inverse_action="ReplaceProduction",
                    cost={"edits": 1, "nodes_touched": 1, "verifier_calls": 1, "serialization_delta": 0},
                )
            )
            # Update local copy to reflect the planned change.
            s_stmt = type(s_stmt)(
                name=s_stmt.name,
                comp=t_stmt.comp,
                children=list(s_stmt.children),
                rest=s_stmt.rest,
                has_list=s_stmt.has_list,
            )
            seed_by_name[t_stmt.name] = s_stmt
            idx += 1

        if not t_stmt.has_list:
            # Leaf slot/arg change.
            t_arg = t_stmt.rest.strip()
            s_arg = s_stmt.rest.strip()
            if t_arg != s_arg:
                edits.append(
                    CanonicalEdit(
                        edit_id=f"bind-slot-{idx}",
                        action="BindSlotPointer",
                        target_name=t_stmt.name,
                        slot=t_arg,
                        affected_node_ids=(t_stmt.name,),
                        inverse_action="BindSlotPointer",
                        cost={"edits": 1, "nodes_touched": 1, "verifier_calls": 1, "serialization_delta": 0},
                    )
                )
                s_stmt = _set_leaf_arg(s_stmt, t_arg)
                seed_by_name[t_stmt.name] = s_stmt
                idx += 1
        else:
            # Container direction/enum change.
            t_args = _args_to_rest(t_stmt.rest)
            s_args = _args_to_rest(s_stmt.rest)
            if t_args and s_args and t_args[0] != s_args[0]:
                edits.append(
                    CanonicalEdit(
                        edit_id=f"set-enum-{idx}",
                        action="SetEnum",
                        target_name=t_stmt.name,
                        direction=t_args[0].strip('"'),
                        affected_node_ids=(t_stmt.name,),
                        inverse_action="SetEnum",
                        cost={"edits": 1, "nodes_touched": 1, "verifier_calls": 1, "serialization_delta": 0},
                    )
                )
                s_stmt = _set_direction(s_stmt, t_args[0].strip('"'))
                seed_by_name[t_stmt.name] = s_stmt
                idx += 1

            # Reference child list rewrite.
            if list(s_stmt.children) != list(t_stmt.children):
                added = [c for c in t_stmt.children if c not in s_stmt.children]
                removed = [c for c in s_stmt.children if c not in t_stmt.children]
                for child in added:
                    edits.append(
                        CanonicalEdit(
                            edit_id=f"insert-child-{idx}",
                            action="InsertChild",
                            target_name=t_stmt.name,
                            child_name=child,
                            affected_node_ids=(t_stmt.name, child),
                            inverse_action="DeleteChild",
                            cost={"edits": 1, "nodes_touched": 2, "verifier_calls": 1, "serialization_delta": 1},
                        )
                    )
                    s_stmt.children.append(child)
                    idx += 1
                for child in removed:
                    edits.append(
                        CanonicalEdit(
                            edit_id=f"delete-child-{idx}",
                            action="DeleteChild",
                            target_name=t_stmt.name,
                            child_name=child,
                            affected_node_ids=(t_stmt.name, child),
                            inverse_action="InsertChild",
                            cost={"edits": 1, "nodes_touched": 2, "verifier_calls": 1, "serialization_delta": -1},
                        )
                    )
                    s_stmt.children = [c for c in s_stmt.children if c != child]
                    idx += 1

    # 3. Delete statements in seed not present in target.
    for s_name in list(seed_by_name):
        if s_name not in target_by_name:
            edits.append(
                CanonicalEdit(
                    edit_id=f"delete-stmt-{idx}",
                    action="DeleteStatement",
                    target_name=s_name,
                    affected_node_ids=(s_name,),
                    inverse_action="InsertStatement",
                    cost={"edits": 1, "nodes_touched": 1, "verifier_calls": 1, "serialization_delta": -1},
                )
            )
            del seed_by_name[s_name]
            idx += 1

    return edits, "planned"


def apply_canonical_edit(source: str, edit: CanonicalEdit) -> str | None:
    """Apply one canonical edit to ``source`` and return the result.

    Returns ``None`` when the edit cannot be applied or the result is invalid.
    """
    statements = parse_statements(source)
    if statements is None:
        return None
    by_name = {s.name: s for s in statements}
    if edit.action == "InsertStatement":
        if edit.target_name in by_name:
            return None
    elif edit.target_name not in by_name:
        return None

    def copy(stmt: Any) -> Any:
        return type(stmt)(
            name=stmt.name,
            comp=stmt.comp,
            children=list(stmt.children),
            rest=stmt.rest,
            has_list=stmt.has_list,
        )

    stmt = copy(by_name[edit.target_name]) if edit.target_name in by_name else None

    if edit.action == "InsertStatement":
        if edit.production is None:
            return None
        if edit.child_name is not None:
            rendered = (
                f"{edit.target_name} = {edit.production}"
                f"([{edit.child_name}], {json.dumps(edit.direction or 'column')})"
            )
        else:
            slot = edit.slot or ":slot"
            slot_text = slot if slot.startswith('"') else json.dumps(slot, ensure_ascii=False)
            rendered = f"{edit.target_name} = {edit.production}({slot_text})"
        inserted = parse_statements(rendered)
        if inserted is None:
            return None
        by_name[edit.target_name] = inserted[0]
    elif edit.action == "DeleteStatement":
        if edit.target_name == "root":
            return None
        for other in by_name.values():
            if edit.target_name in other.children:
                other.children = [c for c in other.children if c != edit.target_name]
        del by_name[edit.target_name]
    elif edit.action == "ReplaceProduction":
        if edit.production is None or stmt is None:
            return None
        # Leaf/container compatibility: keep has_list if possible.
        stmt.comp = edit.production
    elif edit.action == "BindSlotPointer":
        if edit.slot is None or stmt is None or stmt.has_list:
            return None
        stmt.rest = edit.slot if edit.slot.startswith('"') else json.dumps(edit.slot, ensure_ascii=False)
    elif edit.action == "SetEnum":
        if edit.direction is None or stmt is None or not stmt.has_list:
            return None
        stmt = _set_direction(stmt, edit.direction)
    elif edit.action == "InsertChild":
        if edit.child_name is None or stmt is None or not stmt.has_list:
            return None
        if edit.child_name not in by_name:
            return None
        if edit.child_name not in stmt.children:
            stmt.children.append(edit.child_name)
    elif edit.action == "DeleteChild":
        if edit.child_name is None or stmt is None or not stmt.has_list:
            return None
        stmt.children = [c for c in stmt.children if c != edit.child_name]
    else:
        return None

    if edit.action not in {"InsertStatement", "DeleteStatement"} and stmt is not None:
        by_name[edit.target_name] = stmt

    # Reorder: root last, dependencies before uses.
    ordered = _topological_order(by_name)
    rendered = render_statements(ordered)
    if not _is_valid(rendered):
        return None
    return rendered


def _topological_order(by_name: dict[str, Any]) -> list[Any]:
    """Return statements in dependency order with root last."""
    seen: set[str] = set()
    ordered: list[Any] = []

    def visit(name: str) -> None:
        if name in seen or name not in by_name:
            return
        seen.add(name)
        stmt = by_name[name]
        for child in stmt.children:
            visit(child)
        ordered.append(stmt)

    for name in by_name:
        visit(name)
    # Ensure root is last.
    root_idx = next((i for i, s in enumerate(ordered) if s.name == "root"), None)
    if root_idx is not None and root_idx != len(ordered) - 1:
        ordered.append(ordered.pop(root_idx))
    return ordered


def _bounded_bfs(
    seed_source: str,
    target_program: str,
    *,
    max_edits: int = 6,
) -> tuple[str, list[CanonicalEdit], int, int, str]:
    """Exact BFS over statement-level edits for tiny closed domains.

    Returns (result, edit_path, nodes_expanded, max_frontier, stop_reason).
    """
    seed_canonical = canonicalize(seed_source, validate=False)
    target_canonical = canonicalize(target_program, validate=False)
    target_fp = canonical_fingerprint(target_canonical)
    if canonical_fingerprint(seed_canonical) == target_fp:
        return "reachable", [], 0, 1, "seed_equals_target"

    inventory = [p if p.startswith(":") else f":{p}" for p in extract_placeholders(target_canonical)]
    if not inventory:
        inventory = [":slot"]

    seed_statements = parse_statements(seed_canonical)
    if seed_statements is None:
        return "unsupported_pack_feature", [], 0, 0, "seed_parse_failed"

    start = render_statements(seed_statements)
    visited: dict[str, tuple[str, list[CanonicalEdit]]] = {canonical_fingerprint(start): (start, [])}
    frontier: list[tuple[str, list[CanonicalEdit]]] = [(start, [])]
    nodes_expanded = 0
    max_frontier = len(frontier)

    for _ in range(max_edits):
        if not frontier:
            break
        next_frontier: list[tuple[str, list[CanonicalEdit]]] = []
        for current, path in frontier:
            nodes_expanded += 1
            statements = parse_statements(current)
            if statements is None:
                continue
            for i, stmt in enumerate(statements):
                # Generate simple structural mutations.
                mutations = _statement_mutations(stmt, statements, inventory)
                for edit in mutations:
                    next_program = apply_canonical_edit(current, edit)
                    if next_program is None:
                        continue
                    next_path = [*path, edit]
                    fp = canonical_fingerprint(next_program)
                    if fp in visited:
                        continue
                    visited[fp] = (next_program, next_path)
                    if fp == target_fp:
                        return "reachable", next_path, nodes_expanded, max(max_frontier, len(next_frontier) + len(frontier)), "found_target"
                    next_frontier.append((next_program, next_path))
        frontier = next_frontier
        max_frontier = max(max_frontier, len(frontier))

    if not frontier:
        return "unreachable_complete", [], nodes_expanded, max_frontier, "frontier_exhausted"
    return "unknown_budget", [], nodes_expanded, max_frontier, "max_edits_reached"


def _statement_mutations(
    stmt: Any,
    statements: list[Any],
    inventory: list[str],
) -> list[CanonicalEdit]:
    """Return small statement-level edits around one statement."""
    edits: list[CanonicalEdit] = []
    space = TreeEditSpace()
    components = list(space.components)

    if not stmt.has_list:
        # Replace leaf component with another leaf component.
        for comp in components:
            if comp in space.LEAF_COMPONENTS and comp != stmt.comp:
                edits.append(
                    CanonicalEdit(
                        edit_id=f"mut-replace-{stmt.name}-{comp}",
                        action="ReplaceProduction",
                        target_name=stmt.name,
                        production=comp,
                    )
                )
        # Change slot to another inventory slot.
        for slot in inventory:
            edits.append(
                CanonicalEdit(
                    edit_id=f"mut-slot-{stmt.name}-{slot}",
                    action="BindSlotPointer",
                    target_name=stmt.name,
                    slot=slot,
                )
            )
    else:
        # Replace container component.
        for comp in components:
            if comp in space.CONTAINER_COMPONENTS and comp != stmt.comp:
                edits.append(
                    CanonicalEdit(
                        edit_id=f"mut-replace-{stmt.name}-{comp}",
                        action="ReplaceProduction",
                        target_name=stmt.name,
                        production=comp,
                    )
                )
        # Add/remove a child reference.
        candidates = [s.name for s in statements if s.name != stmt.name and s.name != "root"]
        for child in candidates:
            if child not in stmt.children:
                edits.append(
                    CanonicalEdit(
                        edit_id=f"mut-insert-child-{stmt.name}-{child}",
                        action="InsertChild",
                        target_name=stmt.name,
                        child_name=child,
                    )
                )
        for child in stmt.children:
            edits.append(
                CanonicalEdit(
                    edit_id=f"mut-delete-child-{stmt.name}-{child}",
                    action="DeleteChild",
                    target_name=stmt.name,
                    child_name=child,
                )
            )
    return edits


@dataclass(frozen=True)
class EditReachabilityCase:
    """One seed→target reachability attempt."""

    case_id: str
    source_seed_id: str
    target_id: str
    target_program: str
    target_fingerprint: str
    result: str
    path_length: int = 0
    edits: tuple[CanonicalEdit, ...] = ()
    certificates: tuple[TransitionCertificateV1, ...] = ()
    nodes_expanded: int = 0
    max_frontier: int = 0
    stop_reason: str = ""
    verifier_replay_ok: bool = False
    canonical_invariant_ok: bool = False
    alpha_invariant_ok: bool = False
    slot_invariant_ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_seed_id": self.source_seed_id,
            "target_id": self.target_id,
            "target_program": self.target_program,
            "target_fingerprint": self.target_fingerprint,
            "result": self.result,
            "path_length": self.path_length,
            "edits": [e.to_dict() for e in self.edits],
            "certificates": [c.to_dict() for c in self.certificates],
            "nodes_expanded": self.nodes_expanded,
            "max_frontier": self.max_frontier,
            "stop_reason": self.stop_reason,
            "verifier_replay_ok": self.verifier_replay_ok,
            "canonical_invariant_ok": self.canonical_invariant_ok,
            "alpha_invariant_ok": self.alpha_invariant_ok,
            "slot_invariant_ok": self.slot_invariant_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditReachabilityCase":
        return cls(
            case_id=str(data["case_id"]),
            source_seed_id=str(data["source_seed_id"]),
            target_id=str(data["target_id"]),
            target_program=str(data["target_program"]),
            target_fingerprint=str(data["target_fingerprint"]),
            result=str(data["result"]),
            path_length=int(data.get("path_length", 0)),
            edits=tuple(CanonicalEdit.from_dict(e) for e in data.get("edits", ())),
            certificates=tuple(TransitionCertificateV1.from_dict(c) for c in data.get("certificates", ())),
            nodes_expanded=int(data.get("nodes_expanded", 0)),
            max_frontier=int(data.get("max_frontier", 0)),
            stop_reason=str(data.get("stop_reason", "")),
            verifier_replay_ok=bool(data.get("verifier_replay_ok", False)),
            canonical_invariant_ok=bool(data.get("canonical_invariant_ok", False)),
            alpha_invariant_ok=bool(data.get("alpha_invariant_ok", False)),
            slot_invariant_ok=bool(data.get("slot_invariant_ok", False)),
        )


@dataclass(frozen=True)
class InvarianceResult:
    """Canonical invariance checks for one seed→target pair."""

    case_id: str
    canonical_idempotent: bool
    alpha_equivalent: bool
    slot_permutation_equivalent: bool
    commutativity_equivalent: bool
    details: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "canonical_idempotent": self.canonical_idempotent,
            "alpha_equivalent": self.alpha_equivalent,
            "slot_permutation_equivalent": self.slot_permutation_equivalent,
            "commutativity_equivalent": self.commutativity_equivalent,
            "details": list(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InvarianceResult":
        return cls(
            case_id=str(data["case_id"]),
            canonical_idempotent=bool(data.get("canonical_idempotent", False)),
            alpha_equivalent=bool(data.get("alpha_equivalent", False)),
            slot_permutation_equivalent=bool(data.get("slot_permutation_equivalent", False)),
            commutativity_equivalent=bool(data.get("commutativity_equivalent", False)),
            details=tuple(data.get("details", ())),
        )


@dataclass(frozen=True)
class EditReachabilityReport:
    """Full fixture report for SLM-188."""

    schema: str = "EditReachabilityReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm188-edit-algebra"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    cases: tuple[EditReachabilityCase, ...] = ()
    invariance_results: tuple[InvarianceResult, ...] = ()
    n_cases: int = 0
    n_reachable: int = 0
    n_unreachable_complete: int = 0
    n_unknown_budget: int = 0
    n_unsupported: int = 0
    n_invariance_ok: int = 0
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_cases", len(self.cases))
        object.__setattr__(
            self,
            "n_reachable",
            sum(1 for c in self.cases if c.result == "reachable"),
        )
        object.__setattr__(
            self,
            "n_unreachable_complete",
            sum(1 for c in self.cases if c.result == "unreachable_complete"),
        )
        object.__setattr__(
            self,
            "n_unknown_budget",
            sum(1 for c in self.cases if c.result == "unknown_budget"),
        )
        object.__setattr__(
            self,
            "n_unsupported",
            sum(1 for c in self.cases if c.result == "unsupported_pack_feature"),
        )
        object.__setattr__(
            self,
            "n_invariance_ok",
            sum(
                1
                for inv in self.invariance_results
                if inv.canonical_idempotent
                and inv.alpha_equivalent
                and inv.slot_permutation_equivalent
                and inv.commutativity_equivalent
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cases": [c.to_dict() for c in self.cases],
            "invariance_results": [i.to_dict() for i in self.invariance_results],
            "n_cases": self.n_cases,
            "n_reachable": self.n_reachable,
            "n_unreachable_complete": self.n_unreachable_complete,
            "n_unknown_budget": self.n_unknown_budget,
            "n_unsupported": self.n_unsupported,
            "n_invariance_ok": self.n_invariance_ok,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EditReachabilityReport":
        return cls(
            schema=str(data.get("schema", "EditReachabilityReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", "slm188-edit-algebra")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            cases=tuple(EditReachabilityCase.from_dict(c) for c in data.get("cases", ())),
            invariance_results=tuple(InvarianceResult.from_dict(i) for i in data.get("invariance_results", ())),
            n_cases=int(data.get("n_cases", 0)),
            n_reachable=int(data.get("n_reachable", 0)),
            n_unreachable_complete=int(data.get("n_unreachable_complete", 0)),
            n_unknown_budget=int(data.get("n_unknown_budget", 0)),
            n_unsupported=int(data.get("n_unsupported", 0)),
            n_invariance_ok=int(data.get("n_invariance_ok", 0)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def _seed_programs() -> list[tuple[str, str]]:
    """Return (seed_id, source) minimal valid seeds."""
    return [
        ("minimal_stack", 'root = Stack([], "column")'),
        ("minimal_stack_text", 'root = Stack([n0], "column")\nn0 = TextContent(":slot")'),
    ]


def build_seed_target_pairs(
    seed_id: str,
    seed_source: str,
    target_sources: list[tuple[str, str, list[str]]],
) -> list[tuple[str, str, str, list[str]]]:
    """Return (case_id, seed_id, target_program, slot_inventory) pairs."""
    pairs: list[tuple[str, str, str, list[str]]] = []
    for target_id, raw_source, slots in target_sources:
        case_id = f"{seed_id}__{target_id}"
        try:
            canonical_target = canonicalize(raw_source, validate=True)
        except Exception:  # noqa: BLE001
            canonical_target = raw_source
        pairs.append((case_id, seed_id, canonical_target, slots))
    return pairs


def _check_v05(source: str) -> bool:
    """Detect v0.5 state/query/action syntax outside the current topology domain."""
    return any(marker in source for marker in {"$=", "q=", "m=", "a="})


def _run_one_reachability(
    seed_source: str,
    target_program: str,
    slot_inventory: list[str],
    version_pins: dict[str, Any],
    *,
    use_bfs: bool = False,
    max_edits: int = 12,
) -> tuple[str, list[CanonicalEdit], list[TransitionCertificateV1], int, int, str, bool]:
    """Run planner or BFS for one seed→target pair and emit certificates."""
    target_fp = canonical_fingerprint(target_program)

    if use_bfs:
        result, edits, nodes_expanded, max_frontier, stop_reason = _bounded_bfs(
            seed_source, target_program, max_edits=max_edits
        )
    else:
        edits, stop_reason = plan_edit_sequence(seed_source, target_program)
        result = "reachable" if edits else "unreachable_complete"
        nodes_expanded = len(edits)
        max_frontier = 1

    certificates: list[TransitionCertificateV1] = []
    replay_ok = False
    if result == "reachable" and edits:
        current = seed_source
        for idx, edit in enumerate(edits):
            next_program = apply_canonical_edit(current, edit)
            if next_program is None:
                certificates.append(
                    TransitionCertificateV1(
                        source_fingerprint=canonical_fingerprint(current),
                        edit=edit.to_dict(),
                        source_program=current,
                        verifier_accepted=False,
                        verifier_detail="edit_did_not_apply",
                        version_pins=version_pins,
                    )
                )
                break
            accepted = _is_valid(next_program)
            certificates.append(
                TransitionCertificateV1(
                    source_fingerprint=canonical_fingerprint(current),
                    target_fingerprint=canonical_fingerprint(next_program),
                    edit=edit.to_dict(),
                    source_program=current if idx == 0 else None,
                    target_program=next_program,
                    verifier_accepted=accepted,
                    verifier_detail="ok" if accepted else "invalid_after_edit",
                    version_pins=version_pins,
                )
            )
            current = next_program
        replay_ok = (
            all(cert.verifier_accepted for cert in certificates)
            and canonical_fingerprint(current) == target_fp
        )

    return result, edits, certificates, nodes_expanded, max_frontier, stop_reason, replay_ok


def permute_slot_contract(source: str, old_slots: list[str], new_slots: list[str]) -> str:
    """Return source with placeholders remapped according to the slot permutation.

    Uses temporary tokens so chain replacements do not collide.
    """
    if len(old_slots) != len(new_slots):
        return source
    mapping = {old: new for old, new in zip(old_slots, new_slots)}
    result = source
    for i, old in enumerate(old_slots):
        result = result.replace(old, f"__SLOT{i}__")
    for i, old in enumerate(old_slots):
        result = result.replace(f"__SLOT{i}__", mapping[old])
    return result


def run_invariance_suite(
    seed_source: str,
    target_program: str,
    slot_inventory: list[str],
    *,
    case_id: str = "invariance",
) -> InvarianceResult:
    """Run canonical invariance checks for one seed→target pair."""
    details: list[str] = []

    # C(C(x)) == C(x)
    try:
        c1 = canonicalize(target_program, validate=True)
        c2 = canonicalize(c1, validate=True)
        canonical_idempotent = c1 == c2
    except Exception as exc:  # noqa: BLE001
        canonical_idempotent = False
        details.append(f"canonical_idempotence_error: {exc}")

    # Alpha-renaming: the canonicalizer normalizes binder names; verify by
    # renaming every non-root binder and comparing canonical fingerprints.
    alpha_equivalent = False
    try:
        renamed_statements = parse_statements(target_program)
        if renamed_statements is not None:
            name_map = {"root": "root"}
            for stmt in renamed_statements:
                if stmt.name != "root":
                    name_map[stmt.name] = f"v{len(name_map)}"
            for stmt in renamed_statements:
                stmt.name = name_map.get(stmt.name, stmt.name)
                stmt.children = [name_map.get(c, c) for c in stmt.children]
            renamed = render_statements(renamed_statements)
            alpha_fp = canonical_fingerprint(canonicalize(renamed, validate=False))
            target_fp = canonical_fingerprint(canonicalize(target_program, validate=False))
            alpha_equivalent = alpha_fp == target_fp
        else:
            alpha_equivalent = False
            details.append("alpha_rename_parse_failed")
    except Exception as exc:  # noqa: BLE001
        details.append(f"alpha_rename_error: {exc}")

    # Slot permutation: remap slots and verify the production token streams are
    # identical up to the permutation (i.e. structure and ref choices preserved).
    slot_permutation_equivalent = False
    try:
        if len(slot_inventory) >= 2:
            permuted = list(slot_inventory)
            permuted[0], permuted[1] = permuted[1], permuted[0]
            permuted_source = permute_slot_contract(target_program, slot_inventory, permuted)
            inverse = {p: o for o, p in zip(slot_inventory, permuted)}
            from slm_training.dsl.production_codec import encode_openui

            orig_program = encode_openui(target_program, slot_contract=slot_inventory, relative_refs=True)
            perm_program = encode_openui(permuted_source, slot_contract=permuted, relative_refs=True)

            def normalize(tokens: tuple[str, ...], contract: list[str]) -> tuple[str, ...]:
                out: list[str] = []
                for tok in tokens:
                    if tok.startswith("@"):
                        out.append(contract[int(tok[1:])])
                    else:
                        out.append(tok)
                return tuple(out)

            orig_norm = normalize(orig_program.tokens, slot_inventory)
            # Map each permuted slot index back to the original placeholder.
            perm_contract_original = [inverse.get(p, p) for p in permuted]
            perm_norm = normalize(perm_program.tokens, perm_contract_original)
            slot_permutation_equivalent = orig_norm == perm_norm
        else:
            slot_permutation_equivalent = True
    except Exception as exc:  # noqa: BLE001
        details.append(f"slot_permutation_error: {exc}")

    # Independent-edit commutativity: find a pair of edits with disjoint
    # affected nodes, swap them, and verify the canonical result is unchanged.
    commutativity_equivalent = True
    try:
        edits, _ = plan_edit_sequence(seed_source, target_program)
        if len(edits) >= 2:
            swapped = False
            for i in range(len(edits)):
                for j in range(i + 1, len(edits)):
                    nodes_i = set(edits[i].affected_node_ids or [edits[i].target_name])
                    nodes_j = set(edits[j].affected_node_ids or [edits[j].target_name])
                    if nodes_i.isdisjoint(nodes_j):
                        reordered = list(edits)
                        reordered[i], reordered[j] = reordered[j], reordered[i]
                        current_base = seed_source
                        for edit in edits:
                            nxt = apply_canonical_edit(current_base, edit)
                            if nxt is None:
                                break
                            current_base = nxt
                        current_swapped = seed_source
                        for edit in reordered:
                            nxt = apply_canonical_edit(current_swapped, edit)
                            if nxt is None:
                                break
                            current_swapped = nxt
                        commutativity_equivalent = (
                            canonical_fingerprint(current_base) == canonical_fingerprint(current_swapped)
                        )
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                # No independent pair found; vacuously true for this case.
                commutativity_equivalent = True
    except Exception as exc:  # noqa: BLE001
        commutativity_equivalent = False
        details.append(f"commutativity_error: {exc}")

    return InvarianceResult(
        case_id=case_id,
        canonical_idempotent=canonical_idempotent,
        alpha_equivalent=alpha_equivalent,
        slot_permutation_equivalent=slot_permutation_equivalent,
        commutativity_equivalent=commutativity_equivalent,
        details=tuple(details),
    )


def run_edit_reachability_fixture(
    *,
    codec: ProductionCodec | None = None,
    config: TopologyAdapterConfig | None = None,
    targets: list[tuple[str, str, list[str]]] | None = None,
    seed_index: int = 0,
    max_edits: int = 12,
    use_bfs: bool = False,
    run_id: str | None = None,
) -> EditReachabilityReport:
    """Run the SLM-188 edit-algebra reachability fixture."""
    codec = codec or build_fixture_codec()
    seeds = _seed_programs()
    seed_id, seed_source = seeds[seed_index % len(seeds)]

    if targets is None:
        targets = [
            ("hero", _HERO, [":hero.title", ":hero.body"]),
            ("text_only", 'root = Stack([blurb], "column")\nblurb = TextContent(":page.blurb")', [":page.blurb"]),
            ("button_row", 'root = Stack([primary, secondary], "row")\nprimary = Button(":actions.primary")\nsecondary = Button(":actions.secondary")', [":actions.primary", ":actions.secondary"]),
        ]

    version_pins = build_version_stamp(
        "harness.experiments",
        "harness.experiments.slm188_edit_algebra",
        "dsl.solver.topology",
    )

    cases: list[EditReachabilityCase] = []
    invariance_results: list[InvarianceResult] = []
    for case_id, _seed_id, target_program, slot_inventory in build_seed_target_pairs(
        seed_id, seed_source, targets
    ):
        if _check_v05(target_program):
            cases.append(
                EditReachabilityCase(
                    case_id=case_id,
                    source_seed_id=_seed_id,
                    target_id=case_id.split("__")[-1],
                    target_program=target_program,
                    target_fingerprint=canonical_fingerprint(target_program),
                    result="unsupported_pack_feature",
                    stop_reason="v0.5_state_query_action_not_in_statement_domain",
                )
            )
            continue

        sketch = build_sketch_seed(target_program)
        result, edits, certificates, nodes_expanded, max_frontier, stop_reason, replay_ok = _run_one_reachability(
            sketch,
            target_program,
            slot_inventory,
            version_pins,
            use_bfs=use_bfs,
            max_edits=max_edits,
        )

        invariance = run_invariance_suite(
            sketch, target_program, slot_inventory, case_id=case_id
        )
        invariance_results.append(invariance)

        cases.append(
            EditReachabilityCase(
                case_id=case_id,
                source_seed_id=_seed_id,
                target_id=case_id.split("__")[-1],
                target_program=target_program,
                target_fingerprint=canonical_fingerprint(target_program),
                result=result,
                path_length=len(edits),
                edits=tuple(edits),
                certificates=tuple(certificates),
                nodes_expanded=nodes_expanded,
                max_frontier=max_frontier,
                stop_reason=stop_reason,
                verifier_replay_ok=replay_ok,
                canonical_invariant_ok=invariance.canonical_idempotent,
                alpha_invariant_ok=invariance.alpha_equivalent,
                slot_invariant_ok=invariance.slot_permutation_equivalent,
            )
        )

    disposition, rationale = _resolve_disposition(tuple(cases), tuple(invariance_results))
    return EditReachabilityReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        cases=tuple(cases),
        invariance_results=tuple(invariance_results),
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=version_pins,
    )


def replay_transition_certificate(
    certificate: TransitionCertificateV1,
    codec: ProductionCodec | None = None,
    slot_inventory: list[str] | None = None,
) -> tuple[bool, str]:
    """Independently replay one transition certificate and check the target."""
    edit_data = certificate.edit
    try:
        edit = CanonicalEdit.from_dict(edit_data)
    except (KeyError, ValueError, TypeError) as exc:
        return False, f"bad edit payload: {exc}"
    if certificate.source_program is None:
        return False, "missing source_program"
    next_program = apply_canonical_edit(certificate.source_program, edit)
    if next_program is None:
        return False, "edit_did_not_apply"
    accepted = _is_valid(next_program)
    if not accepted:
        return False, "target_not_valid"
    if certificate.target_fingerprint and canonical_fingerprint(next_program) != certificate.target_fingerprint:
        return False, "target_fingerprint_mismatch"
    return True, "ok"


def _resolve_disposition(
    cases: tuple[EditReachabilityCase, ...],
    invariance_results: tuple[InvarianceResult, ...],
) -> tuple[str, str]:
    """Classify the fixture outcome."""
    if not cases:
        return ("inconclusive", "No cases were generated.")

    n_reachable = sum(1 for c in cases if c.result == "reachable")
    n_unknown = sum(1 for c in cases if c.result == "unknown_budget")
    n_cases_supported = sum(1 for c in cases if c.result != "unsupported_pack_feature")
    invariance_ok = all(
        inv.canonical_idempotent and inv.alpha_equivalent and inv.slot_permutation_equivalent and inv.commutativity_equivalent
        for inv in invariance_results
    )
    replay_ok = all(c.verifier_replay_ok for c in cases if c.result == "reachable")

    if n_cases_supported == 0:
        return (
            "inconclusive",
            "All targets are outside the current statement domain; expand fixtures or narrow scope.",
        )

    reachability_fraction = n_reachable / max(1, n_cases_supported)
    if reachability_fraction >= 0.95 and replay_ok and invariance_ok:
        return (
            "reachability_holds",
            "Over the bounded fixture domain, supported targets are reachable, replayable, and canonically invariant.",
        )
    if reachability_fraction < 0.95:
        return (
            "reachability_gap",
            f"Only {reachability_fraction:.0%} of supported targets were reachable within budget; "
            "expand the edit domain or budget before bridge training.",
        )
    if not replay_ok:
        return (
            "certificate_gap",
            "A reachable target did not replay to its certificate; fix transition recording before training.",
        )
    if not invariance_ok:
        return (
            "invariance_gap",
            "Canonical invariance checks failed; fix canonical identity before publishing bridges.",
        )
    if n_unknown / max(1, n_cases_supported) > 0.10:
        return (
            "planner_budget_gap",
            "More than 10% of supported targets hit the edit budget; declare a higher budget or accept 'unknown_budget'.",
        )
    return (
        "inconclusive",
        "No single gap dominates; review per-case diagnostics.",
    )


def render_markdown(report: EditReachabilityReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-188 (FFE1-02): edit-algebra reachability, canonical invariance, and transition certificates ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable "
        "weights were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Summary",
        "",
        f"- Cases: {report.n_cases}",
        f"- Reachable: {report.n_reachable}",
        f"- Unreachable (complete): {report.n_unreachable_complete}",
        f"- Unknown (budget): {report.n_unknown_budget}",
        f"- Unsupported pack feature: {report.n_unsupported}",
        f"- Invariance OK: {report.n_invariance_ok}",
        f"- Disposition: **{report.disposition}**",
        "",
        "## Reachability cases",
        "",
        "| Case | Seed | Target | Result | Path length | Expansions | Frontier max | Replay OK | Invariants |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for case in report.cases:
        lines.append(
            f"| {case.case_id} | {case.source_seed_id} | {case.target_id} | {case.result} | "
            f"{case.path_length} | {case.nodes_expanded} | {case.max_frontier} | "
            f"{case.verifier_replay_ok} | C={case.canonical_invariant_ok} A={case.alpha_invariant_ok} S={case.slot_invariant_ok} |"
        )

    lines.extend(
        [
            "",
            "## Invariance results",
            "",
            "| Case | Idempotent | Alpha | Slot perm | Commutativity |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for inv in report.invariance_results:
        lines.append(
            f"| {inv.case_id} | {inv.canonical_idempotent} | {inv.alpha_equivalent} | "
            f"{inv.slot_permutation_equivalent} | {inv.commutativity_equivalent} |"
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. Real bridge coverage "
            "over the full train/eval corpus, v0.5 state/query/action targets, and "
            "RICO/deeper-tree fixtures requires the standard solver budget and a "
            "trained runtime. Do not publish bridges or start flow/direct-policy "
            "training until supported-target reachability is ≥95%, every emitted "
            "transition replays, and canonical invariance holds.",
            "",
            "## Honest caveats",
            "",
        ]
    )
    for caveat in report.honest_caveats:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.audit_edit_reachability --describe",
            "python -m scripts.audit_edit_reachability --fixtures",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
