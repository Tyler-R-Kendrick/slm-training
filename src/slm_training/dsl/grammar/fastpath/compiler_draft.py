"""Compiler-drafted semantic completions for constrained TwoTower decode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.force_emit import force_next_token_id
from slm_training.dsl.grammar.fastpath.token_map import (
    allowed_id_set,
    apply_literal_frame,
)

Coverage = Literal["complete", "partial", "none"]


@dataclass(frozen=True)
class CompletionPath:
    """One compiler-valid semantic action plus its maximal forced suffix."""

    token_ids: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class CompletionForest:
    """All known next actions for a prefix and their coverage guarantee."""

    paths: tuple[CompletionPath, ...]
    coverage: Coverage
    terminals: tuple[str, ...] = ()

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(path.token_ids[0] for path in self.paths if path.token_ids)


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


def _token_piece(tokenizer: Any, token_id: int) -> str:
    raw = tokenizer.id_to_token.get(int(token_id), "")
    if raw == "NL":
        return "\n"
    if raw in {"LIT_STR", "LIT_END"}:
        return '"'
    decoded = tokenizer.decode([int(token_id)])
    if decoded or raw.startswith(("<BIND_", "<SYM_", "<STATE_")):
        return decoded
    return raw


def _semantic_kind(tokenizer: Any, token_id: int) -> str:
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
        try:
            return str(kind_of(int(token_id)).value)
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

        if lang_core.bridge_available():
            return lang_core.library_schema()
    except Exception:  # noqa: BLE001
        pass
    return None


@lru_cache(maxsize=2048)
def _generated_ast_is_complete(prefix_text: str) -> bool:
    """Ask the official AST parser whether the current document is complete."""
    try:
        from slm_training.dsl import lang_core

        program = lang_core.parse(prefix_text)
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


def _schema_type_terminals(schema_type: str | None) -> frozenset[str] | None:
    """Map generated JSON-schema value types to grammar start terminals."""
    return {
        "string": frozenset({"STRING"}),
        "number": frozenset({"NUMBER"}),
        "integer": frozenset({"NUMBER"}),
        "boolean": frozenset({"BOOL"}),
        "null": frozenset({"NULL"}),
        "array": frozenset({"LSQB"}),
        "object": frozenset({"LBRACE"}),
    }.get(schema_type)


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
    """
    engine = getattr(state, "engine", None) if state is not None else None
    if not isinstance(engine, OpenUIIncrementalEngine):
        engine = OpenUIIncrementalEngine()
    if state is not None:
        prefix_text = state.sync_ids(tokenizer, prefix_ids)
    else:
        prefix_text = tokenizer.decode(prefix_ids)
    if not engine.set_prefix(prefix_text) and prefix_text.strip():
        return CompletionForest((), "none")

    terminals = engine.next_terminals()
    candidates = allowed_id_set(tokenizer, terminals) or set()
    if prefix_ids and tokenizer.id_to_token.get(int(prefix_ids[-1])) == "NL":
        newline_id = tokenizer.token_to_id.get("NL")
        if newline_id is not None:
            candidates.discard(int(newline_id))
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
    needs_schema = bool(terminals & {"COMPONENT", "STRING"}) or _active_call(engine) is not None
    schema = _official_schema() if needs_schema else None
    if schema is not None and "COMPONENT" in terminals:
        component_names = set(schema.get("properties") or {})
        candidates = {
            token_id
            for token_id in candidates
            if _semantic_kind(tokenizer, token_id) != "component"
            or _token_piece(tokenizer, token_id) in component_names
        }
    enum_sequences = (
        _schema_enum_sequences(tokenizer, engine, schema) if schema else None
    )
    if enum_sequences is not None:
        candidates = {sequence[0] for sequence in enum_sequences if sequence}
    schema_type = _schema_slot_type(engine, schema) if schema else None
    schema_slot = _schema_slot_name(engine, schema) if schema else None
    type_terminals = _schema_type_terminals(schema_type)
    arity = _schema_call_arity(engine, schema) if schema else None
    current_started = arity[3] if arity is not None else False
    if type_terminals is not None and enum_sequences is None and not current_started:
        typed_ids = allowed_id_set(tokenizer, type_terminals) or set()
        candidates &= typed_ids
        if schema_type == "string" and slot_contract:
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

    if schema_type == "array" and schema_slot == "children" and current_started:
        node_terminals = frozenset({"NAME", "COMPONENT", "COMMA", "RSQB", "RPAR"})
        node_ids = allowed_id_set(tokenizer, node_terminals) or set()
        candidates &= node_ids

    if arity is not None:
        minimum, maximum, arg_count, current_started = arity
        separator_ids = allowed_id_set(tokenizer, frozenset({"COMMA", "RPAR"})) or set()
        if current_started and "RPAR" in terminals:
            candidates &= separator_ids
        if arg_count < minimum:
            rpar_ids = allowed_id_set(tokenizer, frozenset({"RPAR"})) or set()
            candidates -= rpar_ids
        if arg_count >= maximum:
            comma_ids = allowed_id_set(tokenizer, frozenset({"COMMA"})) or set()
            candidates -= comma_ids

    # Apply tokenizer framing after parser/schema filtering. LIT_STR renders as
    # a quote, so the surface parser sees an empty completed string while the
    # lexer-native token stream still requires BYTE* + LIT_END.
    candidates = apply_literal_frame(tokenizer, prefix_ids, candidates) or set()

    inventory_complete = not (needs_schema and schema is None)
    kind_of = getattr(tokenizer, "kind_of", None)
    if callable(kind_of):
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
                inventory_complete = False
        except Exception:  # noqa: BLE001
            inventory_complete = False

    if _at_declaration_value(tokenizer, prefix_ids):
        candidates = {
            token_id
            for token_id in candidates
            if _semantic_kind(tokenizer, token_id) == "component"
        }

    specials = {
        int(tokenizer.pad_id),
        int(tokenizer.mask_id),
        int(tokenizer.bos_id),
        int(tokenizer.unk_id),
    }
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
            continue
        branch_text = prefix_text
        admitted = True
        for token_id in sequence:
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
            continue
        drafted = [int(token_id) for token_id in sequence]
        while len(drafted) < max_path_tokens:
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

    if not paths:
        coverage: Coverage = "partial" if terminals else "none"
    elif inventory_complete and _known_terminal_coverage(tokenizer, terminals):
        coverage = "complete"
    else:
        coverage = "partial"
    return CompletionForest(
        tuple(paths), coverage, tuple(sorted(str(term) for term in terminals))
    )


def gold_compiler_decisions(
    tokenizer: Any,
    token_ids: list[int] | tuple[int, ...],
    *,
    slot_contract: list[str] | None = None,
    max_path_tokens: int = 8,
) -> tuple[CompilerDecision, ...]:
    """Replay a gold stream and classify every Lark-derived branch decision."""
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
    "active_parent_component_ids",
    "CompletionForest",
    "CompletionPath",
    "CompilerDecision",
    "Coverage",
    "build_completion_forest",
    "gold_compiler_decisions",
    "semantic_component_edges",
    "gold_compiler_decision_positions",
]
