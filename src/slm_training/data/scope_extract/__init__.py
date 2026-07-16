"""Grammar-generic AST scope extraction for scope-graded training data.

Walks a position-preserving parse tree obtained from the active
:class:`~slm_training.dsl.grammar.backends.types.GrammarBackend` and slices the
source text into the four lexical scopes used by scope-graded data families:

* ``document`` — the whole program;
* ``statement`` — each top-level binding;
* ``expression`` — every rule node nested inside a statement's RHS;
* ``lexical`` — every named terminal token.

Scopes are derived purely from grammar rule/terminal names and token
positions — nothing here is OpenUI-specific. Backends expose the capability
via ``parse_tree`` (see ``LarkFileBackend``); backends without it (e.g. a pure
lang-core bridge) resolve through their ``info.grammar_path``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from lark import Lark, Token, Tree

from slm_training.dsl.lang_core import ParseError

# Typed conversion for the conventional literal terminals shared by the
# in-repo grammars (mirrors ``_GenericTransformer``'s terminal callbacks).
TYPED_TERMINALS: dict[str, Callable[[str], Any]] = {
    "STRING": lambda raw: json.loads(raw) if raw.startswith('"') else ast.literal_eval(raw),
    "NUMBER": lambda raw: float(raw) if any(ch in raw for ch in ".eE") else int(raw),
    "BOOL": lambda raw: raw == "true",
    "NULL": lambda raw: None,
}

# Friendly constructor names for the typed-node rendering; anything absent
# falls back to ``terminal.title()``.
DEFAULT_DISPLAY_NAMES: dict[str, str] = {
    "BOOL": "Boolean",
    "NUMBER": "Number",
    "STRING": "String",
    "NULL": "Null",
}

# terminal name -> validate_output lexical category (only well-known ones).
TERMINAL_CATEGORIES: dict[str, str] = {
    "BOOL": "boolean",
    "NUMBER": "number",
    "STRING": "string",
}

SCOPES = ("document", "statement", "expression", "lexical")


@dataclass(frozen=True)
class ScopeSlice:
    """One exact source slice at a single AST-derived lexical scope."""

    scope: str  # document | statement | expression | lexical
    text: str  # exact slice: source[span[0]:span[1]] == text
    span: tuple[int, int]
    category: str  # grammar rule name (statement/expression) or terminal name
    ast_path: tuple[int, ...] = ()
    statement_anchor: str = ""  # LHS binder of the enclosing statement
    typed: bool = False  # True when ``typed_value`` came from TYPED_TERMINALS
    typed_value: Any = field(default=None, compare=False)

    def to_meta(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "span": list(self.span),
            "category": self.category,
            "ast_path": list(self.ast_path),
            "statement_anchor": self.statement_anchor,
        }


def typed_terminal_value(terminal: str, text: str) -> tuple[bool, Any]:
    """Return ``(is_typed, value)`` for a terminal token, mirroring the
    generic Lark transformer's literal typing (BOOL -> bool, NUMBER ->
    int/float, STRING -> str, NULL -> None)."""
    handler = TYPED_TERMINALS.get(terminal)
    if handler is None:
        return False, None
    try:
        return True, handler(text)
    except (ValueError, SyntaxError):
        return False, None


def typed_render(
    terminal: str,
    text: str,
    *,
    display_names: dict[str, str] | None = None,
) -> str | None:
    """Render a typed token as ``DisplayName(json_value)``, e.g.
    ``true`` -> ``Boolean(true)``, ``42`` -> ``Number(42)``.

    Returns None for terminals without a typed conversion.
    """
    is_typed, value = typed_terminal_value(terminal, text)
    if not is_typed:
        return None
    names = {**DEFAULT_DISPLAY_NAMES, **(display_names or {})}
    display = names.get(terminal, terminal.title())
    return f"{display}({json.dumps(value)})"


@lru_cache(maxsize=8)
def _grammar_path_parser(path: str, start: str) -> Lark:
    return Lark(
        Path(path).read_text(encoding="utf-8"),
        start=start,
        parser="lalr",
        maybe_placeholders=False,
        propagate_positions=True,
    )


def _position_tree(source: str, dsl: str) -> Tree:
    """Position-preserving parse tree from the backend for ``dsl``.

    Prefers the backend's ``parse_tree`` capability; falls back to building a
    parser from ``info.grammar_path`` (hybrid backends delegate here). Raises
    ``ParseError`` for invalid input, ``TypeError`` when the backend has no
    grammar file to derive positions from.
    """
    from slm_training.dsl.grammar.backends import get_backend

    backend = get_backend(dsl)
    parse_tree = getattr(backend, "parse_tree", None)
    if callable(parse_tree):
        return parse_tree(source)
    grammar_path = backend.info.grammar_path
    if grammar_path is None or not Path(grammar_path).is_file():
        raise TypeError(
            f"backend {backend.info.id!r} exposes neither parse_tree nor a grammar file"
        )
    text = source if source.endswith("\n") else source + "\n"
    try:
        return _grammar_path_parser(str(grammar_path), "start").parse(text)
    except Exception as exc:  # lark UnexpectedInput and friends
        raise ParseError(str(exc)) from exc


def _span(node: Tree | Token, limit: int) -> tuple[int, int] | None:
    if isinstance(node, Token):
        start, end = node.start_pos, node.end_pos
    else:
        meta = node.meta
        if getattr(meta, "empty", False):
            return None
        start, end = meta.start_pos, meta.end_pos
    if start is None or end is None:
        return None
    return (max(0, int(start)), min(limit, int(end)))


def _unwrap(node: Tree) -> Tree:
    """Descend through single-child wrapper rules (e.g. ``statement`` ->
    ``value_statement``) to the node that carries the real children."""
    while (
        isinstance(node, Tree)
        and len(node.children) == 1
        and isinstance(node.children[0], Tree)
    ):
        node = node.children[0]
    return node


def _statement_anchor(node: Tree) -> str:
    for child in node.children:
        if isinstance(child, Token):
            return str(child)
    return ""


def extract_scope_slices(
    source: str,
    *,
    dsl: str = "openui",
    scopes: tuple[str, ...] = SCOPES,
) -> list[ScopeSlice]:
    """Slice ``source`` into AST-derived scopes with exact character spans.

    Expression candidates are raw grammar rule nodes — callers that need
    parseable fragments should filter through the DSL's fragment validator
    (``dsl.parser.validate_output``). Slices are deduped by (scope, span) and
    returned in source order.
    """
    tree = _position_tree(source, dsl)
    limit = len(source)
    slices: list[ScopeSlice] = []
    seen: set[tuple[str, tuple[int, int]]] = set()

    def emit(slice_: ScopeSlice) -> None:
        key = (slice_.scope, slice_.span)
        if slice_.text.strip() and key not in seen:
            seen.add(key)
            slices.append(slice_)

    if "document" in scopes:
        emit(
            ScopeSlice(
                scope="document",
                text=source,
                span=(0, limit),
                category="document",
            )
        )

    def walk_tokens(
        node: Tree | Token,
        path: tuple[int, ...],
        anchor: str,
        *,
        inside_statement: bool,
    ) -> None:
        if isinstance(node, Token):
            if node.type.startswith("_") or "lexical" not in scopes:
                return
            span = _span(node, limit)
            if span is None:
                return
            is_typed, value = typed_terminal_value(node.type, str(node))
            emit(
                ScopeSlice(
                    scope="lexical",
                    text=source[span[0] : span[1]],
                    span=span,
                    category=node.type,
                    ast_path=path,
                    statement_anchor=anchor,
                    typed=is_typed,
                    typed_value=value,
                )
            )
            return
        span = _span(node, limit)
        if inside_statement and span is not None and "expression" in scopes:
            emit(
                ScopeSlice(
                    scope="expression",
                    text=source[span[0] : span[1]],
                    span=span,
                    category=str(node.data),
                    ast_path=path,
                    statement_anchor=anchor,
                )
            )
        for index, child in enumerate(node.children):
            walk_tokens(child, (*path, index), anchor, inside_statement=True)

    for index, top in enumerate(tree.children):
        if not isinstance(top, Tree):
            continue
        stmt = _unwrap(top)
        span = _span(stmt, limit)
        anchor = _statement_anchor(stmt)
        if span is not None and "statement" in scopes:
            emit(
                ScopeSlice(
                    scope="statement",
                    text=source[span[0] : span[1]],
                    span=span,
                    category=str(stmt.data),
                    ast_path=(index,),
                    statement_anchor=anchor,
                )
            )
        # Children of the statement node are RHS expression territory; the
        # statement node itself is never an expression scope.
        for child_index, child in enumerate(stmt.children):
            walk_tokens(
                child,
                (index, child_index),
                anchor,
                inside_statement=isinstance(child, Tree),
            )

    slices.sort(key=lambda item: (item.span, SCOPES.index(item.scope)))
    return slices


__all__ = [
    "DEFAULT_DISPLAY_NAMES",
    "SCOPES",
    "TERMINAL_CATEGORIES",
    "TYPED_TERMINALS",
    "ScopeSlice",
    "extract_scope_slices",
    "typed_render",
    "typed_terminal_value",
]
