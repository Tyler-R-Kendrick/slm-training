"""Compiler-drafted semantic completions for constrained TwoTower decode."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Literal

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.force_emit import force_next_token_id
from slm_training.dsl.grammar.fastpath.token_map import (
    allowed_id_set,
    apply_literal_frame,
    decode_prefix,
    token_surface_piece as _token_piece,
)

Coverage = Literal["complete", "partial", "none"]


class ConstraintStage(str, Enum):
    """Hard-constraint stage that admitted or excluded a considered action.

    Stages name the *owner* of a decision inside ``build_completion_forest``;
    they are not a proof that the exclusion holds in every completion. Only a
    ``complete``-coverage forest turns an exclusion into an exact fact (VSS0-01,
    ``verified-scope-solver.md``): under ``partial``/``none`` coverage this
    evidence is diagnostic, not exhaustive support proof.
    """

    GRAMMAR = "grammar"
    SCHEMA = "schema"
    BINDING = "binding"
    SLOT_CONTRACT = "slot_contract"
    DATAFLOW = "dataflow"
    LITERAL_FRAME = "literal_frame"
    MIN_CONTENT = "min_content"
    TERMINAL = "terminal"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class ConstraintEvidence:
    """Reason-coded record for one considered action at one constraint stage.

    Immutable and JSON-serializable (``as_dict``/``from_dict``). Evidence is
    emitted only for actions the compiler actually enumerated; it never asserts
    anything about un-enumerated vocabulary. ``admitted=False`` records the stage
    that excluded a considered candidate; ``admitted=True`` records an accepted
    path (or the EOS/coverage decision).
    """

    candidate_id: int | None
    path_token_ids: tuple[int, ...]
    stage: ConstraintStage
    admitted: bool
    reason_code: str
    details: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "path_token_ids": list(self.path_token_ids),
            "stage": self.stage.value,
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "details": [list(item) for item in self.details],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintEvidence:
        return cls(
            candidate_id=(
                None if data["candidate_id"] is None else int(data["candidate_id"])
            ),
            path_token_ids=tuple(int(token) for token in data["path_token_ids"]),
            stage=ConstraintStage(data["stage"]),
            admitted=bool(data["admitted"]),
            reason_code=str(data["reason_code"]),
            details=tuple(
                (str(key), str(value)) for key, value in data.get("details", ())
            ),
        )


@dataclass(frozen=True)
class ConstraintEvidenceSummary:
    """Aggregate stage counts plus the forest coverage for an explained forest."""

    coverage: Coverage
    considered: int
    admitted: int
    excluded: int
    stage_excluded: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "considered": self.considered,
            "admitted": self.admitted,
            "excluded": self.excluded,
            "stage_excluded": [list(item) for item in self.stage_excluded],
        }


@dataclass(frozen=True)
class CompletionPath:
    """One compiler-valid semantic action plus its maximal forced suffix."""

    token_ids: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class CompletionForest:
    """All known next actions for a prefix and their coverage guarantee.

    ``evidence``/``evidence_summary`` are populated only when
    ``build_completion_forest`` is called with ``explain=True``; they default to
    empty so a default forest is byte-for-byte identical to earlier releases.
    """

    paths: tuple[CompletionPath, ...]
    coverage: Coverage
    terminals: tuple[str, ...] = ()
    evidence: tuple[ConstraintEvidence, ...] = ()
    evidence_summary: ConstraintEvidenceSummary | None = None

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(path.token_ids[0] for path in self.paths if path.token_ids)

    def evidence_as_json(self) -> dict[str, Any]:
        """Deterministic JSON-ready view of the evidence and its summary."""
        return {
            "coverage": self.coverage,
            "summary": (
                self.evidence_summary.as_dict()
                if self.evidence_summary is not None
                else None
            ),
            "evidence": [item.as_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class CompilerDecision:
    """One gold decision point classified by its tokenizer semantic kind."""

    position: int
    kind: str
    token_kind: str
    candidate_ids: tuple[int, ...]

    @property
    def is_semantic_role(self) -> bool:
        """Whether this branch selects an AST role rather than serialization."""
        return self.token_kind in {"component", "bind", "state", "builtin"}


def _literal_frame_is_open(tokenizer: Any, token_ids: list[int]) -> bool:
    """Whether lexer-native token ids end inside a framed literal body."""
    opened = False
    for token_id in token_ids:
        raw = str(tokenizer.id_to_token.get(int(token_id), ""))
        if raw in {"LIT_STR", "LIT_NUM"} and not opened:
            opened = True
        elif raw == "LIT_END" and opened:
            opened = False
    return opened


def _semantic_kind(tokenizer: Any, token_id: int) -> str:
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
        try:
            kind = kind_of(int(token_id))
            return str(getattr(kind, "value", kind))
        except Exception:  # noqa: BLE001
            pass
    piece = _token_piece(tokenizer, token_id)
    if piece[:1].isupper() and piece.isidentifier():
        return "component"
    if piece[:1].islower() and piece.isidentifier():
        return "binder"
    if piece.startswith(":") or piece.startswith("<SYM_"):
        return "symbol"
    return "structural"


def _grammar_terminal_kind(
    tokenizer: Any,
    token_id: int,
    terminals: tuple[str, ...],
    state: Any | None = None,
    declaration_scope: str | None = None,
) -> str:
    """Classify structural choices by the active Lark terminal."""
    semantic = _semantic_kind(tokenizer, token_id)
    if semantic not in {"struct", "structural"}:
        return semantic
    matches = [
        terminal
        for terminal in terminals
        if token_id
        in (allowed_id_set(tokenizer, frozenset({terminal})) or set())
    ]
    if not matches:
        return semantic
    terminal = min(matches)
    kind = f"grammar_{terminal.lower().strip('_').replace('$', 'end_')}"
    if terminal == "RSQB":
        occupancy = _active_list_occupancy(state)
        context = [part for part in (declaration_scope, occupancy) if part]
        if context:
            kind = "_".join((kind, *context))
    return kind


def _active_list_occupancy(state: Any) -> str | None:
    """Read empty/populated state for the innermost open Lark list frame."""
    parser = getattr(state, "_ip", None)
    parser_state = getattr(parser, "parser_state", None)
    values = list(getattr(parser_state, "value_stack", ()) or ())
    nested = 0
    for index in range(len(values) - 1, -1, -1):
        token_type = str(getattr(values[index], "type", ""))
        if token_type == "RSQB":
            nested += 1
        elif token_type == "LSQB":
            if nested:
                nested -= 1
            else:
                return "empty" if index == len(values) - 1 else "populated"
    return None


def _at_declaration_value(tokenizer: Any, prefix_ids: list[int]) -> bool:
    """Whether the layout grammar is awaiting a component declaration value."""
    try:
        return (
            len(prefix_ids) >= 2
            and int(prefix_ids[-2]) in set(tokenizer.kind_ids("bind"))
            and int(prefix_ids[-1]) == int(tokenizer.token_to_id["="])
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _binder_scope(
    tokenizer: Any, prefix_ids: list[int]
) -> tuple[list[int], list[int], int | None]:
    """Return declarations, references, and the active declaration slot."""
    bind_ids = set(tokenizer.kind_ids("bind"))
    equal_id = int(tokenizer.token_to_id["="])
    newline_id = tokenizer.token_to_id.get("NL")
    declarations: list[int] = []
    references: list[int] = []
    declaration_positions: list[int] = []
    for index, token_id in enumerate(prefix_ids):
        token_id = int(token_id)
        if token_id not in bind_ids:
            continue
        if index + 1 < len(prefix_ids) and int(prefix_ids[index + 1]) == equal_id:
            declarations.append(token_id)
            declaration_positions.append(index)
        elif token_id not in references:
            references.append(token_id)
    active = None
    if declaration_positions:
        last = declaration_positions[-1]
        if newline_id is None or newline_id not in prefix_ids[last + 1 :]:
            active = declarations[-1]
    return declarations, references, active


def _references_resolved(tokenizer: Any, prefix_ids: list[int]) -> bool:
    """Whether every generated binder reference has a declaration."""
    declarations, references, _active = _binder_scope(tokenizer, prefix_ids)
    return set(references) <= set(declarations)


def _binder_component_types(tokenizer: Any, prefix_ids: list[int]) -> dict[int, str]:
    """Return binder-to-component types certified by completed declarations."""
    bind_ids = set(tokenizer.kind_ids("bind"))
    component_ids = set(tokenizer.kind_ids("component"))
    equal_id = int(tokenizer.token_to_id["="])
    return {
        int(prefix_ids[index]): _token_piece(tokenizer, int(prefix_ids[index + 2]))
        for index in range(max(0, len(prefix_ids) - 2))
        if int(prefix_ids[index]) in bind_ids
        and int(prefix_ids[index + 1]) == equal_id
        and int(prefix_ids[index + 2]) in component_ids
    }


def _forward_binder_component_requirements(
    tokenizer: Any,
    prefix_ids: list[int],
    schema: dict[str, Any],
) -> dict[int, frozenset[str]]:
    """Propagate typed component use-site constraints to later declarations."""
    bind_ids = set(tokenizer.kind_ids("bind"))
    equal_id = int(tokenizer.token_to_id["="])
    engine = OpenUIIncrementalEngine()
    requirements: dict[int, frozenset[str]] = {}
    for index, raw_token_id in enumerate(prefix_ids):
        token_id = int(raw_token_id)
        is_declaration = (
            token_id in bind_ids
            and index + 1 < len(prefix_ids)
            and int(prefix_ids[index + 1]) == equal_id
        )
        if token_id in bind_ids and not is_declaration:
            allowed = (
                _schema_array_item_components(engine, schema)
                if _active_array_position(engine) == "item_start"
                else _schema_slot_components(engine, schema)
            )
            if allowed:
                prior = requirements.get(token_id)
                requirements[token_id] = (
                    allowed if prior is None else frozenset(prior & allowed)
                )
        piece = _token_piece(tokenizer, token_id)
        if piece and not engine.advance(piece):
            break
    return requirements


def _binder_dependencies(tokenizer: Any, prefix_ids: list[int]) -> dict[int, set[int]]:
    """Return declaration-reference edges present in the generated prefix."""
    bind_ids = set(tokenizer.kind_ids("bind"))
    equal_id = int(tokenizer.token_to_id["="])
    newline_id = tokenizer.token_to_id.get("NL")
    dependencies: dict[int, set[int]] = {}
    active: int | None = None
    for index, raw_token_id in enumerate(prefix_ids):
        token_id = int(raw_token_id)
        if newline_id is not None and token_id == int(newline_id):
            active = None
            continue
        if token_id not in bind_ids:
            continue
        if index + 1 < len(prefix_ids) and int(prefix_ids[index + 1]) == equal_id:
            active = token_id
            dependencies.setdefault(active, set())
        elif active is not None:
            dependencies.setdefault(active, set()).add(token_id)
    return dependencies


def _binder_reference_would_cycle(
    source: int, target: int, dependencies: dict[int, set[int]]
) -> bool:
    """Whether adding ``source -> target`` closes a declaration cycle."""
    pending = [target]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current == source:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(dependencies.get(current, ()))
    return False


def _active_array_start_index(tokenizer: Any, prefix_ids: list[int]) -> int | None:
    """Return the token index of the innermost live array."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[tuple[str, int]] = []
    for index, token_id in enumerate(prefix_ids):
        piece = _token_piece(tokenizer, int(token_id))
        if piece in {"(", "[", "{"}:
            stack.append((piece, index))
        elif piece in pairs and stack and stack[-1][0] == pairs[piece]:
            stack.pop()
    array = next(
        ((piece, index) for piece, index in reversed(stack) if piece == "["),
        None,
    )
    return array[1] if array is not None else None


def _active_array_direct_references(
    tokenizer: Any, prefix_ids: list[int]
) -> frozenset[int]:
    """Return binder references already used as direct items of the live array."""
    array_start = _active_array_start_index(tokenizer, prefix_ids)
    if array_start is None:
        return frozenset()
    pairs = {")": "(", "]": "[", "}": "{"}
    bind_ids = set(tokenizer.kind_ids("bind"))
    nested: list[str] = []
    references: set[int] = set()
    for raw_token_id in prefix_ids[array_start + 1 :]:
        token_id = int(raw_token_id)
        piece = _token_piece(tokenizer, token_id)
        if piece in {"(", "[", "{"}:
            nested.append(piece)
        elif piece in pairs and nested and nested[-1] == pairs[piece]:
            nested.pop()
        elif not nested and token_id in bind_ids:
            references.add(token_id)
    return frozenset(references)


def _active_declaration_scope(tokenizer: Any, prefix_ids: list[int]) -> str | None:
    """Classify the live declaration by typed root/bound binder identity."""
    _declarations, _references, active = _binder_scope(tokenizer, prefix_ids)
    if active is None:
        return None
    return "root" if active == tokenizer.bind_id(0) else "bound"


def active_declaration_binder_id(
    tokenizer: Any, prefix_ids: list[int]
) -> int | None:
    """Return the grammar-native binder for the active declaration."""
    _declarations, _references, active = _binder_scope(tokenizer, prefix_ids)
    return active


def emitted_component_count(tokenizer: Any, prefix_ids: list[int]) -> int:
    """Count component instantiations already emitted in ``prefix_ids``.

    Uses the tokenizer's compiler-derived ``component`` symbol space (no AST
    parse, no Node bridge). This is the content measure the minimum-content
    decode contract (A4) checks: an empty/underfull layout has too few
    components to satisfy a prompt that names them.
    """
    try:
        component_ids = set(tokenizer.kind_ids("component"))
    except Exception:  # noqa: BLE001 - tokenizer without kind_ids → no gate
        return 0
    return sum(1 for token_id in prefix_ids if int(token_id) in component_ids)


def binder_reference_arities(
    tokenizer: Any, token_ids: list[int] | tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """Return declaration binder and reference count from grammar token roles."""
    bind_ids = set(tokenizer.kind_ids("bind"))
    equal_id = int(tokenizer.token_to_id["="])
    newline_id = tokenizer.token_to_id.get("NL")
    statements: list[list[int]] = []
    current: list[int] = []
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if newline_id is not None and token_id == int(newline_id):
            if current:
                statements.append(current)
            current = []
        else:
            current.append(token_id)
    if current:
        statements.append(current)

    arities: list[tuple[int, int]] = []
    for statement in statements:
        declaration_at = next(
            (
                index
                for index, token_id in enumerate(statement[:-1])
                if token_id in bind_ids and statement[index + 1] == equal_id
            ),
            None,
        )
        if declaration_at is None:
            continue
        references = sum(
            token_id in bind_ids
            for token_id in statement[declaration_at + 2 :]
        )
        arities.append((statement[declaration_at], references))
    return tuple(arities)


def active_declaration_reference_count(
    tokenizer: Any, prefix_ids: list[int]
) -> int | None:
    """Count binder references emitted in the active declaration statement."""
    active = active_declaration_binder_id(tokenizer, prefix_ids)
    if active is None:
        return None
    arities = binder_reference_arities(tokenizer, prefix_ids)
    return next((count for binder, count in reversed(arities) if binder == active), 0)


def root_declaration_reference_arity_target(
    tokenizer: Any, token_ids: list[int] | tuple[int, ...]
) -> tuple[int, int] | None:
    """Return lexer root-reference count and available bound declarations."""
    try:
        root = int(tokenizer.bind_id(0))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    arities = binder_reference_arities(tokenizer, token_ids)
    target = next((count for binder, count in arities if binder == root), None)
    if target is None:
        return None
    bound = sum(int(binder) != root for binder, _count in arities)
    return int(target), max(int(target), bound)


def active_parent_component_ids(
    tokenizer: Any, prefix_ids: list[int]
) -> tuple[int, ...]:
    """Return known component types that reference the active declaration."""
    _declarations, _references, active = _binder_scope(tokenizer, prefix_ids)
    if active is None:
        return ()
    bind_ids = set(tokenizer.kind_ids("bind"))
    component_ids = set(tokenizer.kind_ids("component"))
    equal_id = int(tokenizer.token_to_id["="])
    newline_id = tokenizer.token_to_id.get("NL")
    statements: list[list[int]] = []
    current: list[int] = []
    for raw_token_id in prefix_ids:
        token_id = int(raw_token_id)
        if newline_id is not None and token_id == int(newline_id):
            if current:
                statements.append(current)
            current = []
        else:
            current.append(token_id)
    if current:
        statements.append(current)

    parents: set[int] = set()
    for statement in statements:
        equal_at = next(
            (
                index + 1
                for index, token_id in enumerate(statement[:-1])
                if token_id in bind_ids and statement[index + 1] == equal_id
            ),
            None,
        )
        if equal_at is None:
            continue
        owner_component = next(
            (token_id for token_id in statement[equal_at + 1 :] if token_id in component_ids),
            None,
        )
        if owner_component is None:
            continue
        references = {
            token_id
            for token_id in statement[equal_at + 1 :]
            if token_id in bind_ids
        }
        if active in references:
            parents.add(owner_component)
    return tuple(sorted(parents))


def semantic_component_edges(
    root: Any, tokenizer: Any
) -> tuple[tuple[int, int], ...]:
    """Extract parent/child component-type edges from a resolved OpenUI AST."""
    token_to_id = tokenizer.token_to_id
    edges: list[tuple[int, int]] = []

    def direct_elements(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            if value.get("type") == "element":
                return [value]
            found: list[dict[str, Any]] = []
            for child in value.values():
                found.extend(direct_elements(child))
            return found
        if isinstance(value, list):
            found = []
            for child in value:
                found.extend(direct_elements(child))
            return found
        return []

    def visit(node: Any) -> None:
        if not isinstance(node, dict) or node.get("type") != "element":
            return
        parent_id = token_to_id.get(str(node.get("typeName") or ""))
        props = node.get("props")
        children = direct_elements(props) if isinstance(props, dict) else []
        for child in children:
            child_id = token_to_id.get(str(child.get("typeName") or ""))
            if parent_id is not None and child_id is not None:
                edges.append((int(parent_id), int(child_id)))
            visit(child)

    visit(root)
    return tuple(edges)


def _known_terminal_coverage(tokenizer: Any, terminals: frozenset[str]) -> bool:
    """Whether token mapping exhausts the model vocabulary for these terminals."""
    broad = {
        "NAME",
        "COMPONENT",
        "STATE_NAME",
        "BUILTIN",
        "STRING",
        "NUMBER",
        "BOOL",
        "NULL",
    }
    # Lexical/BPE candidates are only a subset of semantic names. They remain
    # valid drafts, but only lexer-native symbol/action ids are exhaustive.
    if terminals & broad and not callable(getattr(tokenizer, "kind_of", None)):
        return False
    known = {
        "$END",
        "COMMENT",
        "WS_INLINE",
        "_NL",
        "NL",
        "NAME",
        "COMPONENT",
        "STATE_NAME",
        "BUILTIN",
        "STRING",
        "NUMBER",
        "BOOL",
        "NULL",
        "EQUAL",
        "LPAR",
        "RPAR",
        "LSQB",
        "RSQB",
        "LBRACE",
        "RBRACE",
        "COMMA",
        "DOT",
        "COLON",
        "QMARK",
        "PLUS",
        "MINUS",
        "STAR",
        "SLASH",
        "PERCENT",
        "BANG",
        "MORETHAN",
        "LESSTHAN",
        "__ANON_0",
        "__ANON_1",
        "__ANON_2",
        "__ANON_3",
        "__ANON_4",
        "__ANON_5",
    }
    vocab = getattr(tokenizer, "token_to_id", {})
    return all(term in known or term in vocab for term in terminals)


@lru_cache(maxsize=1)
def _official_schema() -> dict[str, Any] | None:
    try:
        from slm_training.dsl import lang_core

        return lang_core.library_schema()
    except Exception:  # noqa: BLE001
        pass
    return None


def _schema_accepts_symbol(
    value_schema: dict[str, Any], root_schema: dict[str, Any]
) -> bool:
    """Whether a positional property can consume a declared text symbol."""
    reference = value_schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        target = (root_schema.get("$defs") or {}).get(reference.rsplit("/", 1)[-1])
        return isinstance(target, dict) and _schema_accepts_symbol(target, root_schema)
    if value_schema.get("type") == "string":
        return not value_schema.get("enum") and "const" not in value_schema
    return any(
        isinstance(branch, dict) and _schema_accepts_symbol(branch, root_schema)
        for keyword in ("anyOf", "oneOf", "allOf")
        for branch in value_schema.get(keyword) or ()
    ) or (
        value_schema.get("type") == "array"
        and isinstance(value_schema.get("items"), dict)
        and _schema_accepts_symbol(value_schema["items"], root_schema)
    )


def _schema_requires_symbol(
    value_schema: dict[str, Any],
    root_schema: dict[str, Any],
    seen_refs: frozenset[str] = frozenset(),
) -> bool:
    """Whether every valid nonempty value needs a declared text symbol."""
    reference = value_schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        name = reference.rsplit("/", 1)[-1]
        if name in seen_refs:
            return False
        target = (root_schema.get("$defs") or {}).get(name)
        return isinstance(target, dict) and _schema_requires_symbol(
            target, root_schema, seen_refs | {name}
        )
    if value_schema.get("type") == "string":
        return not value_schema.get("enum") and "const" not in value_schema
    if value_schema.get("type") == "array":
        items = value_schema.get("items")
        return isinstance(items, dict) and _schema_requires_symbol(
            items, root_schema, seen_refs
        )
    branches = [
        branch
        for keyword in ("anyOf", "oneOf")
        for branch in value_schema.get(keyword) or ()
        if isinstance(branch, dict)
    ]
    if branches:
        return all(
            _schema_requires_symbol(branch, root_schema, seen_refs)
            for branch in branches
        )
    all_of = [
        branch
        for branch in value_schema.get("allOf") or ()
        if isinstance(branch, dict)
    ]
    if all_of and any(
        _schema_requires_symbol(branch, root_schema, seen_refs)
        for branch in all_of
    ):
        return True
    properties = value_schema.get("properties") or {}
    return any(
        isinstance(properties.get(name), dict)
        and _schema_requires_symbol(properties[name], root_schema, seen_refs)
        for name in value_schema.get("required") or ()
    )


@lru_cache(maxsize=None)
def _component_requires_available_content(component: str) -> bool:
    """Whether a component opens text or child-node content."""
    schema = _official_schema() or {}
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    return _schema_requires_symbol(definition, schema) or any(
        isinstance(property_schema, dict)
        and (
            _schema_accepts_symbol(property_schema, schema)
            or bool(_schema_component_refs(property_schema, schema))
            or (
                property_name == "children"
                and property_schema.get("type") == "array"
            )
        )
        for property_name, property_schema in properties.items()
    )


@lru_cache(maxsize=2048)
def _generated_ast_is_complete(prefix_text: str) -> bool:
    """Ask the official AST parser whether the current document is complete."""
    try:
        from slm_training.dsl import lang_core

        program = lang_core.parse(prefix_text)
        return isinstance(program.root, dict) and bool(program.root)
    except Exception:  # noqa: BLE001
        try:
            from slm_training.dsl.grammar.backends import get_backend

            program = get_backend("openui-lark").validate(prefix_text)
            return isinstance(program.root, dict) and bool(program.root)
        except Exception:  # noqa: BLE001
            return False


def _active_call(state: Any) -> tuple[str, int, int] | None:
    """Read the active call frame from Lark's parser value stack.

    The grammar owns delimiter/quote handling.  This deliberately does not
    rescan source text: reduced nested expressions are already represented by
    Lark trees, while the live comma token identifies the current positional
    argument.
    """
    parser = getattr(state, "_ip", None)
    parser_state = getattr(parser, "parser_state", None)
    values = list(getattr(parser_state, "value_stack", ()) or ())
    frames: list[tuple[int, str]] = []
    for index, value in enumerate(values):
        if str(getattr(value, "data", "")) != "call_name":
            continue
        children = list(getattr(value, "children", ()) or ())
        if children:
            frames.append((index, str(children[0])))

    opening = {"LPAR", "LSQB", "LBRACE"}
    closing = {"RPAR", "RSQB", "RBRACE"}
    for call_index, component in reversed(frames):
        lpar = next(
            (
                index
                for index in range(call_index + 1, len(values))
                if str(getattr(values[index], "type", "")) == "LPAR"
            ),
            None,
        )
        if lpar is None:
            continue
        depth = 0
        separators = 0
        current_started = False
        closed = False
        for value in values[lpar + 1 :]:
            token_type = str(getattr(value, "type", ""))
            data = str(getattr(value, "data", ""))
            if token_type in opening:
                depth += 1
                current_started = True
            elif token_type in closing:
                if depth == 0:
                    if token_type == "RPAR":
                        closed = True
                        break
                else:
                    depth -= 1
                    current_started = True
            elif depth == 0 and token_type == "COMMA":
                separators += 1
                current_started = False
            elif depth == 0 and data.startswith("__arg_list_star_"):
                # Lark reduces ("," expr)* into one generated tree whose
                # children are the completed expressions after separators.
                separators += len(list(getattr(value, "children", ()) or ()))
                current_started = True
            elif depth == 0:
                current_started = True
        if not closed:
            return component, separators, separators + int(current_started)
    return None


def _schema_enum_sequences(
    tokenizer: Any, state: Any, schema: dict[str, Any]
) -> tuple[tuple[int, ...], ...] | None:
    active = _active_call(state)
    if active is None:
        return None
    component, index, arg_count = active
    if arg_count > index:
        return None
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return set()
    values = (properties.get(names[index]) or {}).get("enum")
    if not values:
        return None
    sequences: set[tuple[int, ...]] = set()
    for value in values:
        encoded = tokenizer.encode(json.dumps(value), add_special=False)
        if encoded:
            sequences.add(tuple(int(token_id) for token_id in encoded))
    return tuple(sorted(sequences))


def _schema_slot_type(
    state: Any, schema: dict[str, Any]
) -> str | None:
    active = _active_call(state)
    if active is None:
        return None
    component, index, _ = active
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return None
    value = properties.get(names[index]) or {}
    return str(value.get("type")) if value.get("type") else None


def _schema_slot_name(state: Any, schema: dict[str, Any]) -> str | None:
    """Return the generated AST property for the active positional argument."""
    active = _active_call(state)
    if active is None:
        return None
    component, index, _ = active
    definition = (schema.get("$defs") or {}).get(component) or {}
    names = list(definition.get("properties") or {})
    return str(names[index]) if index < len(names) else None


def _schema_component_refs(
    value_schema: dict[str, Any], root_schema: dict[str, Any]
) -> frozenset[str]:
    """Resolve component names admitted by a local schema reference/union."""
    names: set[str] = set()
    reference = value_schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        name = reference.rsplit("/", 1)[-1]
        if name in set(root_schema.get("properties") or {}):
            names.add(name)
        else:
            target = (root_schema.get("$defs") or {}).get(name)
            if isinstance(target, dict):
                names.update(_schema_component_refs(target, root_schema))
    for keyword in ("anyOf", "oneOf", "allOf"):
        for branch in value_schema.get(keyword) or ():
            if isinstance(branch, dict):
                names.update(_schema_component_refs(branch, root_schema))
    items = value_schema.get("items")
    if isinstance(items, dict):
        names.update(_schema_component_refs(items, root_schema))
    return frozenset(names)


def _schema_array_item_components(
    state: Any, schema: dict[str, Any]
) -> frozenset[str]:
    """Component types admitted by the active array property."""
    active = _active_call(state)
    if active is None:
        return frozenset()
    component, index, _ = active
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return frozenset()
    items = (properties.get(names[index]) or {}).get("items") or {}
    return _schema_component_refs(items, schema)


def _schema_slot_components(
    state: Any, schema: dict[str, Any]
) -> frozenset[str]:
    """Component types admitted directly by the active positional property."""
    active = _active_call(state)
    if active is None:
        return frozenset()
    component, index, _ = active
    definition = (schema.get("$defs") or {}).get(component) or {}
    properties = definition.get("properties") or {}
    names = list(properties)
    if index >= len(names):
        return frozenset()
    value = properties.get(names[index]) or {}
    if value.get("type") == "array":
        return frozenset()
    return _schema_component_refs(value, schema)


def _active_array_is_empty(state: Any) -> bool:
    parser = getattr(state, "_ip", None)
    parser_state = getattr(parser, "parser_state", None)
    values = list(getattr(parser_state, "value_stack", ()) or ())
    return bool(values) and str(getattr(values[-1], "type", "")) == "LSQB"


def _active_array_position(state: Any) -> str | None:
    """Return the schema array's item position from incremental parser state."""
    parser = getattr(state, "_ip", None)
    parser_state = getattr(parser, "parser_state", None)
    values = list(getattr(parser_state, "value_stack", ()) or ())
    call_at = next(
        (
            index
            for index in range(len(values) - 1, -1, -1)
            if str(getattr(values[index], "type", "")) == "LPAR"
        ),
        -1,
    )
    brackets: list[int] = []
    for index, value in enumerate(values[call_at + 1 :], start=call_at + 1):
        token_type = str(getattr(value, "type", ""))
        if token_type == "LSQB":
            brackets.append(index)
        elif token_type == "RSQB" and brackets:
            brackets.pop()
    if not brackets:
        return None
    tail_type = str(getattr(values[-1], "type", ""))
    return "item_start" if tail_type in {"LSQB", "COMMA"} else "item_end"


def _schema_call_arity(
    state: Any, schema: dict[str, Any]
) -> tuple[int, int, int, bool] | None:
    """Return required, maximum, supplied args and current-slot progress."""
    active = _active_call(state)
    if active is None:
        return None
    component, index, arg_count = active
    definition = (schema.get("$defs") or {}).get(component) or {}
    names = list(definition.get("properties") or {})
    if not names:
        return None
    required = set(definition.get("required") or ())
    minimum = max((names.index(name) + 1 for name in required if name in names), default=0)
    return minimum, len(names), arg_count, arg_count > index


_SCHEMA_TYPE_TERMINALS: dict[str, frozenset[str]] = {
    "string": frozenset({"STRING"}),
    "number": frozenset({"NUMBER"}),
    "integer": frozenset({"NUMBER"}),
    "boolean": frozenset({"BOOL"}),
    "null": frozenset({"NULL"}),
    "array": frozenset({"LSQB"}),
    "object": frozenset({"LBRACE"}),
}


def _schema_type_terminals(schema_type: str | None) -> frozenset[str] | None:
    """Map generated JSON-schema value types to grammar start terminals."""
    if schema_type is None:
        return None
    return _SCHEMA_TYPE_TERMINALS.get(schema_type)


def _decision_kind(
    tokenizer: Any,
    token_id: int,
    prefix_ids: list[int],
    terminals: tuple[str, ...],
    state: Any,
    schema: dict[str, Any] | None,
) -> str:
    """Build a semantic decision signature from parser/schema roles."""
    scope = _active_declaration_scope(tokenizer, prefix_ids)
    kind = _grammar_terminal_kind(
        tokenizer, token_id, terminals, state, scope
    )
    if kind == "component" and _at_declaration_value(tokenizer, prefix_ids):
        return f"component_{scope}" if scope else kind
    if kind != "bind":
        return kind
    last = prefix_ids[-1] if prefix_ids else None
    at_statement_start = (
        len(prefix_ids) <= 1 or tokenizer.id_to_token.get(last) == "NL"
    )
    if at_statement_start:
        target_scope = "root" if token_id == tokenizer.bind_id(0) else "bound"
        return f"bind_declaration_{target_scope}"
    parts = ["bind_reference"]
    if scope:
        parts.append(scope)
    slot = _schema_slot_name(state, schema) if schema else None
    if slot:
        parts.append("".join(char if char.isalnum() else "_" for char in slot))
    return "_".join(parts)


def build_completion_forest(
    tokenizer: Any,
    prefix_ids: list[int],
    *,
    state: Any | None = None,
    slot_contract: list[str] | None = None,
    max_path_tokens: int = 8,
    min_content: int = 0,
    enforce_schema_component_types: bool = False,
    explain: bool = False,
) -> CompletionForest:
    """Enumerate every mapped, globally extendable action at ``prefix_ids``.

    Lark supplies parser reachability, the lexer-native tokenizer supplies the
    compiler-derived component/binder/symbol spaces, and the optional slot
    contract restricts active placeholder symbols. Each branch is extended
    through its maximal deterministic grammar suffix.

    ``min_content`` (A4 minimum-content decode contract): when > 0, EOS is not
    admitted until at least ``min_content`` components have been emitted, so a
    grammatically valid but empty/underfull layout is not a legal completion
    while the grammar still offers a way to add content. The gate never creates
    a dead end — it only withholds EOS when a non-EOS continuation remains.

    ``enforce_schema_component_types`` restricts singular component properties
    and propagates typed-array use-site requirements to later binder
    declarations. It is opt-in while the decode lever is evaluated.

    ``explain`` (VSS0-02): when True the returned forest also carries reason-coded
    :class:`ConstraintEvidence` for every considered action plus a
    :class:`ConstraintEvidenceSummary`. Explanation is purely observational — the
    returned ``paths``, ``candidate_ids``, ``coverage``, and ``terminals`` are
    byte-for-byte identical to the default path, nothing is collected or
    allocated when ``explain`` is False, and evidence about *considered*
    candidates is not, on its own, an exhaustive support proof (see
    ``verified-scope-solver.md``): only a ``complete`` coverage certifies an
    exclusion as exact.
    """
    evidence: list[ConstraintEvidence] | None = [] if explain else None

    def _snapshot() -> set[int] | None:
        return set(candidates) if evidence is not None else None

    def _record_excluded(
        stage: ConstraintStage, reason: str, before: set[int] | None
    ) -> None:
        if evidence is None or before is None:
            return
        for token_id in sorted(before - candidates):
            evidence.append(
                ConstraintEvidence(int(token_id), (int(token_id),), stage, False, reason)
            )

    def _record_one(
        stage: ConstraintStage, reason: str, token_id: int, *, admitted: bool
    ) -> None:
        if evidence is None:
            return
        evidence.append(
            ConstraintEvidence(int(token_id), (int(token_id),), stage, admitted, reason)
        )

    def _finalize(paths: list[CompletionPath], coverage: Coverage) -> CompletionForest:
        terminals_tuple = tuple(sorted(str(term) for term in terminals))
        if evidence is None:
            return CompletionForest(tuple(paths), coverage, terminals_tuple)
        evidence.append(
            ConstraintEvidence(
                None,
                (),
                ConstraintStage.COVERAGE,
                coverage == "complete",
                f"coverage_{coverage}",
            )
        )
        admitted_ids = {
            item.candidate_id
            for item in evidence
            if item.admitted and item.candidate_id is not None
        }
        excluded = [item for item in evidence if not item.admitted]
        excluded_ids = {
            item.candidate_id for item in excluded if item.candidate_id is not None
        } - admitted_ids
        stage_counts = Counter(item.stage.value for item in excluded)
        summary = ConstraintEvidenceSummary(
            coverage=coverage,
            considered=len(admitted_ids | excluded_ids),
            admitted=len(admitted_ids),
            excluded=len(excluded_ids),
            stage_excluded=tuple(sorted(stage_counts.items())),
        )
        return CompletionForest(
            tuple(paths), coverage, terminals_tuple, tuple(evidence), summary
        )

    engine = getattr(state, "engine", None) if state is not None else None
    if not isinstance(engine, OpenUIIncrementalEngine):
        engine = OpenUIIncrementalEngine()
    if state is not None:
        prefix_text = state.sync_ids(tokenizer, prefix_ids)
    else:
        prefix_text = decode_prefix(tokenizer, prefix_ids)
    if not engine.set_prefix(prefix_text) and prefix_text.strip():
        if evidence is not None:
            summary = ConstraintEvidenceSummary("none", 0, 0, 0, ())
            return CompletionForest((), "none", (), (), summary)
        return CompletionForest((), "none")

    terminals = engine.next_terminals()
    candidates = allowed_id_set(tokenizer, terminals) or set()
    before_stage = _snapshot()
    if prefix_ids and tokenizer.id_to_token.get(int(prefix_ids[-1])) == "NL":
        newline_id = tokenizer.token_to_id.get("NL")
        if newline_id is not None:
            candidates.discard(int(newline_id))
    _record_excluded(ConstraintStage.GRAMMAR, "grammar_newline_repeat", before_stage)
    ast_complete = _generated_ast_is_complete(prefix_text)
    references_resolved = _references_resolved(tokenizer, prefix_ids)
    # A4: withhold EOS while the layout has fewer than ``min_content`` components,
    # but only when the grammar still offers a non-EOS continuation (never create
    # a dead end that would force a fallback on an otherwise-valid document).
    content_met = True
    if min_content > 0:
        other_candidates = candidates - {int(tokenizer.eos_id)}
        if other_candidates and emitted_component_count(tokenizer, prefix_ids) < min_content:
            content_met = False
    if "$END" in terminals and ast_complete and references_resolved and content_met:
        candidates.add(int(tokenizer.eos_id))
    else:
        candidates.discard(int(tokenizer.eos_id))
    if evidence is not None and "$END" in terminals:
        # Distinguish *why* EOS is withheld so a certificate builder can tell a
        # min-content floor apart from a grammar/dataflow rejection.
        eos_id = int(tokenizer.eos_id)
        if not ast_complete:
            _record_one(ConstraintStage.TERMINAL, "eos_withheld_incomplete_ast", eos_id, admitted=False)
        elif not references_resolved:
            _record_one(ConstraintStage.DATAFLOW, "eos_withheld_unresolved_reference", eos_id, admitted=False)
        elif not content_met:
            _record_one(ConstraintStage.MIN_CONTENT, "eos_withheld_min_content", eos_id, admitted=False)
        else:
            _record_one(ConstraintStage.TERMINAL, "eos_admitted", eos_id, admitted=True)
    before_stage = _snapshot()
    if "$END" in terminals and ast_complete:
        # Lark accepts postfix operators after any expression. Once the
        # generated AST has a complete document, retain only the grammar's
        # document-continuation terminals; this derives the boundary from the
        # parser and AST rather than enumerating punctuation or components.
        continuation_terminals = frozenset(
            {"$END", "_NL", "NAME", "STATE_NAME", "COMMENT", "WS_INLINE"}
        )
        continuation_ids = allowed_id_set(tokenizer, continuation_terminals) or set()
        candidates &= continuation_ids | {int(tokenizer.eos_id)}
    _record_excluded(ConstraintStage.TERMINAL, "terminal_document_continuation", before_stage)
    needs_schema = bool(terminals & {"COMPONENT", "STRING"}) or _active_call(engine) is not None
    schema = _official_schema() if needs_schema else None
    before_stage = _snapshot()
    if schema is not None and "COMPONENT" in terminals:
        component_names = set(schema.get("properties") or {})
        candidates = {
            token_id
            for token_id in candidates
            if _semantic_kind(tokenizer, token_id) != "component"
            or _token_piece(tokenizer, token_id) in component_names
        }
    _record_excluded(ConstraintStage.SCHEMA, "schema_component_not_in_library", before_stage)
    if enforce_schema_component_types and schema is not None and "COMPONENT" in terminals:
        slot_components = _schema_slot_components(engine, schema)
        if slot_components:
            before_stage = _snapshot()
            bind_ids = set(tokenizer.kind_ids("bind"))
            binder_components = _binder_component_types(tokenizer, prefix_ids)
            typed_binders = {
                binder
                for binder, component in binder_components.items()
                if component in slot_components
            }
            unknown_binders = bind_ids - set(binder_components)
            pending_requirements = _forward_binder_component_requirements(
                tokenizer, prefix_ids, schema
            )
            unknown_binders = {
                binder
                for binder in unknown_binders
                if pending_requirements.get(binder) is None
                or bool(pending_requirements[binder] & slot_components)
            }
            candidates = {
                token_id
                for token_id in candidates
                if (
                    _semantic_kind(tokenizer, token_id) == "component"
                    and _token_piece(tokenizer, token_id) in slot_components
                )
                or token_id in typed_binders
                or token_id in unknown_binders
            }
            _record_excluded(
                ConstraintStage.SCHEMA,
                "schema_slot_component_type",
                before_stage,
            )
    _declarations, _references, active_declaration = _binder_scope(
        tokenizer, prefix_ids
    )
    if (
        enforce_schema_component_types
        and schema is not None
        and "COMPONENT" in terminals
        and active_declaration is not None
    ):
        required_components = _forward_binder_component_requirements(
            tokenizer, prefix_ids, schema
        ).get(active_declaration)
        if required_components is not None:
            before_stage = _snapshot()
            candidates = {
                token_id
                for token_id in candidates
                if _semantic_kind(tokenizer, token_id) != "component"
                or _token_piece(tokenizer, token_id) in required_components
            }
            _record_excluded(
                ConstraintStage.SCHEMA,
                "schema_forward_binder_component_type",
                before_stage,
            )
    if schema is not None and slot_contract and "COMPONENT" in terminals:
        from slm_training.models.grammar import contract_allowed_token_ids

        if contract_allowed_token_ids(tokenizer, prefix_ids, slot_contract) == set():
            before_stage = _snapshot()
            candidates = {
                token_id
                for token_id in candidates
                if _semantic_kind(tokenizer, token_id) != "component"
                or not _component_requires_available_content(
                    _token_piece(tokenizer, token_id)
                )
            }
            _record_excluded(
                ConstraintStage.SLOT_CONTRACT,
                "component_requires_unavailable_symbol",
                before_stage,
            )
    enum_sequences = (
        _schema_enum_sequences(tokenizer, engine, schema) if schema else None
    )
    before_stage = _snapshot()
    if enum_sequences is not None:
        candidates = {sequence[0] for sequence in enum_sequences if sequence}
    _record_excluded(ConstraintStage.SCHEMA, "schema_enum_restricted", before_stage)
    schema_type = _schema_slot_type(engine, schema) if schema else None
    schema_slot = _schema_slot_name(engine, schema) if schema else None
    type_terminals = _schema_type_terminals(schema_type)
    arity = _schema_call_arity(engine, schema) if schema else None
    current_started = arity[3] if arity is not None else False
    if type_terminals is not None and enum_sequences is None and not current_started:
        before_stage = _snapshot()
        typed_ids = allowed_id_set(tokenizer, type_terminals) or set()
        candidates &= typed_ids
        _record_excluded(ConstraintStage.SCHEMA, "schema_type_mismatch", before_stage)
        if schema_type == "string" and slot_contract:
            before_stage = _snapshot()
            try:
                from slm_training.dsl.placeholders import CONTENT_PROPS
                from slm_training.models.grammar import contract_allowed_token_ids

                contract_ids = set(
                    contract_allowed_token_ids(tokenizer, prefix_ids, slot_contract)
                    or set()
                )
                kind_ids = getattr(tokenizer, "kind_ids", None)
                if callable(kind_ids):
                    candidates -= set(kind_ids("sym"))
                candidates |= contract_ids
                if schema_slot in CONTENT_PROPS:
                    candidates = contract_ids
            except Exception:  # noqa: BLE001
                pass
            _record_excluded(ConstraintStage.SLOT_CONTRACT, "slot_contract_restricted", before_stage)

    array_item_components = (
        _schema_array_item_components(engine, schema)
        if schema_type == "array" and schema is not None
        else frozenset()
    )
    active_array_position = (
        _active_array_position(engine)
        if schema_type == "array" and current_started
        else None
    )
    if active_array_position is not None and schema is not None:
        active = _active_call(engine)
        if active is not None:
            component, index, _arg_count = active
            definition = (schema.get("$defs") or {}).get(component) or {}
            names = list(definition.get("properties") or {})
            value = (
                (definition.get("properties") or {}).get(names[index]) or {}
                if index < len(names)
                else {}
            )
            items = value.get("items") or {}
            item_terminals = _schema_type_terminals(items.get("type"))
            if item_terminals is not None:
                before_stage = _snapshot()
                if active_array_position == "item_start":
                    typed_ids = allowed_id_set(tokenizer, item_terminals) or set()
                    close_ids = allowed_id_set(
                        tokenizer, frozenset({"RSQB"})
                    ) or set()
                    candidates &= typed_ids | close_ids
                else:
                    candidates &= allowed_id_set(
                        tokenizer, frozenset({"COMMA", "RSQB"})
                    ) or set()
                _record_excluded(
                    ConstraintStage.SCHEMA,
                    "schema_array_item_type",
                    before_stage,
                )
    if (
        schema_type == "array"
        and current_started
        and (schema_slot == "children" or array_item_components)
    ):
        before_stage = _snapshot()
        node_terminals = frozenset({"NAME", "COMPONENT", "COMMA", "RSQB", "RPAR"})
        node_ids = allowed_id_set(tokenizer, node_terminals) or set()
        candidates &= node_ids
        if array_item_components:
            bind_ids = set(tokenizer.kind_ids("bind"))
            binder_components = _binder_component_types(tokenizer, prefix_ids)
            typed_binders = {
                binder
                for binder, component in binder_components.items()
                if component in array_item_components
            }
            from slm_training.models.grammar import contract_allowed_token_ids

            unused_symbols = contract_allowed_token_ids(
                tokenizer, prefix_ids, slot_contract
            )
            unknown_binders = (
                bind_ids - set(binder_components)
                if unused_symbols is None or bool(unused_symbols)
                else set()
            )
            if enforce_schema_component_types:
                pending_requirements = _forward_binder_component_requirements(
                    tokenizer, prefix_ids, schema
                )
                unknown_binders = {
                    binder
                    for binder in unknown_binders
                    if pending_requirements.get(binder) is None
                    or bool(pending_requirements[binder] & array_item_components)
                }
            candidates = {
                token_id
                for token_id in candidates
                if (
                    _semantic_kind(tokenizer, token_id) != "component"
                    or _token_piece(tokenizer, token_id) in array_item_components
                )
                and (
                    token_id not in bind_ids
                    or token_id in typed_binders
                    or token_id in unknown_binders
                )
            }
        if _active_array_is_empty(engine):
            candidates -= allowed_id_set(tokenizer, frozenset({"RSQB"})) or set()
        _record_excluded(ConstraintStage.SCHEMA, "schema_array_children", before_stage)

        before_stage = _snapshot()
        candidates -= _active_array_direct_references(tokenizer, prefix_ids)
        _record_excluded(
            ConstraintStage.BINDING,
            "binding_array_reference_reuse",
            before_stage,
        )

    if arity is not None:
        before_stage = _snapshot()
        minimum, maximum, arg_count, current_started = arity
        separator_ids = allowed_id_set(tokenizer, frozenset({"COMMA", "RPAR"})) or set()
        if current_started and "RPAR" in terminals:
            candidates &= separator_ids
        if arg_count < minimum:
            rpar_ids = allowed_id_set(tokenizer, frozenset({"RPAR"})) or set()
            candidates -= rpar_ids
        if arg_count >= maximum and not (
            schema_type == "array" and active_array_position is not None
        ):
            comma_ids = allowed_id_set(tokenizer, frozenset({"COMMA"})) or set()
            candidates -= comma_ids
        _record_excluded(ConstraintStage.SCHEMA, "schema_arity", before_stage)

    # Apply tokenizer framing after parser/schema filtering. LIT_STR renders as
    # a quote, so the surface parser sees an empty completed string while the
    # lexer-native token stream still requires BYTE* + LIT_END.
    before_stage = _snapshot()
    candidates = apply_literal_frame(tokenizer, prefix_ids, candidates) or set()
    _record_excluded(ConstraintStage.LITERAL_FRAME, "literal_frame", before_stage)

    inventory_complete = not (needs_schema and schema is None)
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
        before_stage = _snapshot()
        try:
            from slm_training.models.dsl_tokenizer import TokenKind

            bind_ids = set(tokenizer.kind_ids(TokenKind.BIND))
            state_ids = set(tokenizer.kind_ids(TokenKind.STATE))
            builtin_ids = set(tokenizer.kind_ids(TokenKind.BUILTIN))
            sym_ids = set(tokenizer.kind_ids(TokenKind.SYM))
            declarations, references, active_declaration = _binder_scope(
                tokenizer, prefix_ids
            )
            last = prefix_ids[-1] if prefix_ids else None
            # LTR/compiler prefixes include BOS, which is not a source token.
            # Treat BOS-only as the first statement so the root binder remains
            # available to the symbolic tree.
            at_statement_start = len(prefix_ids) <= 1 or tokenizer.id_to_token.get(last) == "NL"
            if at_statement_start:
                declared = set(declarations)
                unresolved = [token_id for token_id in references if token_id not in declared]
                used = declared | set(references)
                next_slot = next(
                    (
                        slot
                        for slot in range(tokenizer.bind_slots)
                        if tokenizer.bind_id(slot) not in used
                    ),
                    max(0, tokenizer.bind_slots - 1),
                )
                next_id = unresolved[0] if unresolved else tokenizer.bind_id(next_slot)
                candidates -= bind_ids
                candidates.add(int(next_id))
            elif declarations:
                declared = set(declarations)
                reusable = declared - {int(active_declaration or -1), tokenizer.bind_id(0)}
                if active_declaration is not None:
                    dependencies = _binder_dependencies(tokenizer, prefix_ids)
                    reusable = {
                        token_id
                        for token_id in reusable
                        if not _binder_reference_would_cycle(
                            int(active_declaration), token_id, dependencies
                        )
                    }
                unresolved = set(references) - declared
                used = declared | set(references)
                next_slot = next(
                    (
                        slot
                        for slot in range(1, tokenizer.bind_slots)
                        if tokenizer.bind_id(slot) not in used
                    ),
                    None,
                )
                forward = (
                    {int(tokenizer.bind_id(next_slot))}
                    if next_slot is not None
                    else set()
                )
                scope = reusable | unresolved | forward
                candidates = (candidates - bind_ids) | (candidates & scope)
            else:
                candidates -= bind_ids
            # The selected 0.2.x layout contract excludes state/effect actions.
            candidates -= state_ids | builtin_ids
            if candidates & sym_ids and not (
                slot_contract and schema_type == "string"
            ):
                candidates -= sym_ids
        except Exception:  # noqa: BLE001
            inventory_complete = False
        _record_excluded(ConstraintStage.BINDING, "binding_scope", before_stage)

    if _at_declaration_value(tokenizer, prefix_ids):
        before_stage = _snapshot()
        candidates = {
            token_id
            for token_id in candidates
            if _semantic_kind(tokenizer, token_id) == "component"
        }
        _record_excluded(
            ConstraintStage.SCHEMA, "declaration_value_requires_component", before_stage
        )

    specials = {
        int(tokenizer.pad_id),
        int(tokenizer.mask_id),
        int(tokenizer.bos_id),
        int(tokenizer.unk_id),
    }
    if evidence is not None:
        for token_id in sorted(candidates & specials):
            _record_one(ConstraintStage.GRAMMAR, "special_token_excluded", token_id, admitted=False)
    paths: list[CompletionPath] = []
    max_path_tokens = max(1, int(max_path_tokens))
    candidate_sequences = (
        enum_sequences
        if enum_sequences is not None
        else tuple((candidate,) for candidate in sorted(candidates - specials))
    )
    for sequence in candidate_sequences:
        if not sequence:
            continue
        candidate = int(sequence[0])
        if candidate == int(tokenizer.eos_id):
            paths.append(CompletionPath((candidate,), "eos"))
            continue
        branch = OpenUIIncrementalEngine(engine.grammar_path)
        if not branch.set_prefix(prefix_text):
            _record_one(ConstraintStage.GRAMMAR, "branch_prefix_rejected", candidate, admitted=False)
            continue
        branch_text = prefix_text
        admitted = True
        opened_literal_frame = _literal_frame_is_open(tokenizer, prefix_ids)
        crossed_literal_boundary = False
        for token_id in sequence:
            raw = str(tokenizer.id_to_token.get(int(token_id), ""))
            if raw in {"LIT_STR", "LIT_NUM"}:
                opened_literal_frame = True
                crossed_literal_boundary = True
                continue
            if raw == "LIT_END":
                opened_literal_frame = False
                crossed_literal_boundary = True
                continue
            piece = _token_piece(tokenizer, int(token_id))
            probe = branch.probe_chunk(piece)
            if probe is None:
                admitted = branch.set_prefix(branch_text + piece)
            elif probe:
                admitted = branch.advance(piece)
            else:
                admitted = False
            if not admitted:
                break
            branch_text += piece
        # InteractiveParser accepted the edge and exposes at least one follow
        # terminal, which is the exact CFG reachability guarantee we need.
        if not admitted or not branch.next_terminals():
            _record_one(ConstraintStage.GRAMMAR, "branch_unreachable", candidate, admitted=False)
            continue
        drafted = [int(token_id) for token_id in sequence]
        while (
            not opened_literal_frame
            and not crossed_literal_boundary
            and len(drafted) < max_path_tokens
        ):
            forced = force_next_token_id(branch, tokenizer, branch_text)
            if forced is None or forced in specials:
                break
            drafted.append(int(forced))
            branch_text += _token_piece(tokenizer, forced)
        paths.append(
            CompletionPath(
                tuple(drafted),
                _decision_kind(
                    tokenizer,
                    candidate,
                    prefix_ids,
                    tuple(sorted(str(term) for term in terminals)),
                    engine,
                    schema,
                ),
            )
        )
        if evidence is not None:
            evidence.append(
                ConstraintEvidence(
                    candidate, tuple(drafted), ConstraintStage.GRAMMAR, True, "admitted"
                )
            )

    if not paths:
        coverage: Coverage = "partial" if terminals else "none"
    elif inventory_complete and _known_terminal_coverage(tokenizer, terminals):
        coverage = "complete"
    else:
        coverage = "partial"
    return _finalize(paths, coverage)


def gold_compiler_decisions(
    tokenizer: Any,
    token_ids: list[int] | tuple[int, ...],
    *,
    slot_contract: list[str] | None = None,
    max_path_tokens: int = 8,
) -> tuple[CompilerDecision, ...]:
    """Replay a gold stream and classify every Lark-derived branch decision."""
    try:
        from slm_training.models.choice_tokenizer import (
            ChoiceDecodeState,
            is_choice_tokenizer,
        )

        if is_choice_tokenizer(tokenizer):
            ids = tuple(int(token_id) for token_id in token_ids)
            top_level_components: list[int] = []
            probe = ChoiceDecodeState(
                tokenizer, slot_count=len(slot_contract or ())
            )
            probe_cursor = 1 if ids and ids[0] == int(tokenizer.bos_id) else 0
            while (
                probe_cursor < len(ids)
                and ids[probe_cursor]
                not in {int(tokenizer.pad_id), int(tokenizer.eos_id)}
            ):
                if (
                    _semantic_kind(tokenizer, ids[probe_cursor]) == "component"
                    and not probe.frames
                ):
                    top_level_components.append(probe_cursor)
                if not probe.advance_id(ids[probe_cursor]):
                    break
                probe_cursor += 1
            structural_root = (
                top_level_components[-1] if top_level_components else None
            )
            state = ChoiceDecodeState(
                tokenizer, slot_count=len(slot_contract or ())
            )
            stop_ids = {int(tokenizer.pad_id), int(tokenizer.eos_id)}
            cursor = 1 if ids and ids[0] == int(tokenizer.bos_id) else 0
            decisions: list[CompilerDecision] = []
            while cursor < len(ids) and ids[cursor] not in stop_ids:
                remaining = len(ids) - cursor
                candidates = tuple(sorted(state.allowed_ids(remaining)))
                gold = ids[cursor]
                token_kind = _semantic_kind(tokenizer, gold)
                kind = token_kind
                if token_kind == "component" and not state.frames:
                    kind = (
                        "component_root"
                        if state.current_marker == "r="
                        or (
                            state.mode != "v05"
                            and cursor == structural_root
                        )
                        else "component_bound"
                    )
                if len(candidates) > 1:
                    decisions.append(
                        CompilerDecision(cursor, kind, token_kind, candidates)
                    )
                if not state.advance_id(gold):
                    break
                cursor += 1
            return tuple(decisions)
    except Exception:  # noqa: BLE001
        pass

    ids = tuple(int(token_id) for token_id in token_ids)
    stop_ids = {int(tokenizer.pad_id), int(tokenizer.eos_id)}
    cursor = 1 if ids and ids[0] == int(tokenizer.bos_id) else 0
    decisions: list[CompilerDecision] = []
    while cursor < len(ids) and ids[cursor] not in stop_ids:
        forest = build_completion_forest(
            tokenizer,
            list(ids[:cursor]),
            slot_contract=slot_contract,
            max_path_tokens=max_path_tokens,
        )
        remaining = ids[cursor:]
        matches = [
            path
            for path in forest.paths
            if path.token_ids and remaining[: len(path.token_ids)] == path.token_ids
        ]
        if not matches:
            cursor += 1
            continue
        path = max(matches, key=lambda candidate: len(candidate.token_ids))
        if len(set(forest.candidate_ids)) > 1:
            kind = path.kind
            decisions.append(
                CompilerDecision(
                    cursor,
                    kind,
                    _semantic_kind(tokenizer, ids[cursor]),
                    tuple(sorted(set(forest.candidate_ids))),
                )
            )
        cursor += len(path.token_ids)
    return tuple(decisions)


def gold_compiler_decision_positions(
    tokenizer: Any,
    token_ids: list[int] | tuple[int, ...],
    *,
    slot_contract: list[str] | None = None,
    max_path_tokens: int = 8,
) -> tuple[int, ...]:
    """Compatibility view of grammar-derived gold decision positions."""
    return tuple(
        decision.position
        for decision in gold_compiler_decisions(
            tokenizer,
            token_ids,
            slot_contract=slot_contract,
            max_path_tokens=max_path_tokens,
        )
    )


__all__ = [
    "active_declaration_binder_id",
    "active_declaration_reference_count",
    "root_declaration_reference_arity_target",
    "active_parent_component_ids",
    "binder_reference_arities",
    "CompletionForest",
    "CompletionPath",
    "CompilerDecision",
    "ConstraintEvidence",
    "ConstraintEvidenceSummary",
    "ConstraintStage",
    "Coverage",
    "build_completion_forest",
    "gold_compiler_decisions",
    "semantic_component_edges",
    "gold_compiler_decision_positions",
]
