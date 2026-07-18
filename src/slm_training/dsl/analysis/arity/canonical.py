"""Canonical arith-sketch AST: typed template symbols + de Bruijn refs (CAP0-02).

This module owns the *canonical* representation an arity certificate is built on.
It reuses the frozen ``arith-sketch`` grammar backend for parsing and type
rejection (never a private parser) and the ``production_codec`` action sigils for
naming, then normalises the parse into a name-erased, alpha-invariant tree:

* numeric literals collapse to a typed template symbol (``("lit", "N")``) —
  concrete values are irrelevant to arity;
* identifiers collapse to relative **de Bruijn** references
  (``("ref", delta)``) resolved to the nearest preceding binder, so binder
  renaming and shadowing are handled correctly and alpha-equivalent programs map
  to one representative;
* binary operators keep their symbol (``("op", "+", left, right)``).

A :class:`CanonicalProgram` is the ordered tuple of statement right-hand sides
(the last binder is ``root``). Its :meth:`CanonicalProgram.fingerprint` is a
stable sha256 of the canonical JSON — one hash per alpha-equivalence class.

Honesty scope: the counts a downstream report derives from these canonical forms
are certificates *for the committed bounded fixture only*. They do not reproduce
the external CAP0-01 source-reported estimates (see
``docs/design/calculated-arity-adaptive-precision.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from lark import Token, Tree

from slm_training.dsl.grammar.backends import get_backend
from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.production_codec import (
    LIT_PREFIX,
    OP_PREFIX,
    REL_REF_PREFIX,
    STMT,
)

# Typed template symbol for a numeric literal (value erased for arity).
NUMBER_CLASS = "N"

# Arith-sketch grammar rule name -> operator symbol (the full G4 operator set).
_OP_RULE_TO_SYMBOL: dict[str, str] = {
    "add_expr": "+",
    "sub_expr": "-",
    "mul_expr": "*",
    "div_expr": "/",
}
GRAMMAR_OPERATORS: tuple[str, ...] = ("+", "-", "*", "/")

_ARITH_BACKEND_ID = "arith-sketch"

# A canonical expression node is one of:
#   ("lit", class)          - typed template symbol, 1 node
#   ("ref", delta)          - de Bruijn ref to an earlier binder, 1 node
#   ("free", name)          - unresolved free identifier, 1 node (never in fixture)
#   ("op", symbol, l, r)    - binary operator, 1 + nodes(l) + nodes(r) nodes
Expr = tuple[Any, ...]


def lit(cls: str = NUMBER_CLASS) -> Expr:
    """A typed literal template symbol."""
    return ("lit", cls)


def ref(delta: int) -> Expr:
    """A de Bruijn reference to the binder ``delta`` statements earlier."""
    if delta < 1:
        raise ValueError(f"de Bruijn ref delta must be >= 1, got {delta}")
    return ("ref", int(delta))


def op(symbol: str, left: Expr, right: Expr) -> Expr:
    """A binary operator node."""
    return ("op", symbol, left, right)


def expr_nodes(expr: Expr) -> int:
    """Number of AST nodes in ``expr``."""
    if expr[0] in ("lit", "ref", "free"):
        return 1
    return 1 + expr_nodes(expr[2]) + expr_nodes(expr[3])


def expr_depth(expr: Expr) -> int:
    """Nesting depth of ``expr`` (an atom has depth 1)."""
    if expr[0] in ("lit", "ref", "free"):
        return 1
    return 1 + max(expr_depth(expr[2]), expr_depth(expr[3]))


def ref_deltas(expr: Expr) -> list[int]:
    """All de Bruijn ref deltas occurring in ``expr`` (preorder)."""
    if expr[0] == "ref":
        return [int(expr[1])]
    if expr[0] == "op":
        return ref_deltas(expr[2]) + ref_deltas(expr[3])
    return []


def _as_lists(value: Any) -> Any:
    """Recursively turn tuples into JSON-safe lists."""
    if isinstance(value, tuple):
        return [_as_lists(item) for item in value]
    return value


@dataclass(frozen=True)
class CanonicalProgram:
    """A name-erased straight-line program: statement RHS tuple (last is root)."""

    statements: tuple[Expr, ...]

    @property
    def num_statements(self) -> int:
        return len(self.statements)

    @property
    def node_count(self) -> int:
        """Total AST nodes across every statement RHS."""
        return sum(expr_nodes(stmt) for stmt in self.statements)

    @property
    def depth(self) -> int:
        """Maximum RHS depth across statements (0 for the empty program)."""
        return max((expr_depth(stmt) for stmt in self.statements), default=0)

    def to_json(self) -> list[Any]:
        """JSON-safe nested-list view of the canonical program."""
        return [_as_lists(stmt) for stmt in self.statements]

    def canonical_key(self) -> str:
        """Deterministic canonical JSON string (one per alpha-equivalence class)."""
        return json.dumps(self.to_json(), separators=(",", ":"), sort_keys=True)

    def fingerprint(self) -> str:
        """Stable sha256 of the canonical form."""
        return hashlib.sha256(self.canonical_key().encode("utf-8")).hexdigest()


def binder_names(count: int) -> list[str]:
    """Deterministic binder names for ``count`` statements; the last is ``root``."""
    if count < 1:
        return []
    return [f"v{i}" for i in range(count - 1)] + ["root"]


def materialize(program: CanonicalProgram) -> str:
    """Render a canonical program to concrete arith-sketch source.

    Binders are named ``v0, v1, …, root`` (all distinct, so backend
    last-definition-wins resolution coincides with nearest-preceding de Bruijn
    intent). Literals become distinct increasing integers; every operator is
    fully parenthesised so the reparse recovers the exact tree.
    """
    names = binder_names(program.num_statements)
    literal_counter = [0]

    def emit(expr: Expr, stmt_index: int) -> str:
        kind = expr[0]
        if kind == "lit":
            literal_counter[0] += 1
            return str(literal_counter[0])
        if kind == "ref":
            target = stmt_index - int(expr[1])
            if target < 0 or target >= stmt_index:
                raise ValueError(
                    f"ref delta {expr[1]} out of scope at statement {stmt_index}"
                )
            return names[target]
        if kind == "free":
            return str(expr[1])
        if kind == "op":
            left = emit(expr[2], stmt_index)
            right = emit(expr[3], stmt_index)
            return f"({left} {expr[1]} {right})"
        raise ValueError(f"unknown canonical node: {expr!r}")

    lines = [
        f"{names[i]} = {emit(stmt, i)}" for i, stmt in enumerate(program.statements)
    ]
    return "\n".join(lines)


def _canon_expr(node: Any, defined: list[str]) -> Expr:
    """Canonicalise one raw Lark expression node against binders in scope."""
    if isinstance(node, Token):
        if node.type == "NUMBER":
            return lit(NUMBER_CLASS)
        raise ParseError(f"unexpected token in arith expr: {node.type}")
    data = str(node.data)
    if data == "ref":
        name = str(node.children[0])
        for j in range(len(defined) - 1, -1, -1):
            if defined[j] == name:
                return ("ref", len(defined) - j)
        return ("free", name)
    if data in _OP_RULE_TO_SYMBOL:
        left = _canon_expr(node.children[0], defined)
        right = _canon_expr(node.children[1], defined)
        return op(_OP_RULE_TO_SYMBOL[data], left, right)
    raise ParseError(f"unexpected arith rule: {data}")


def program_from_source(source: str) -> CanonicalProgram:
    """Parse concrete arith-sketch source into its canonical program.

    Names are erased to de Bruijn refs against the nearest *preceding* binder,
    so binder renaming yields an identical canonical form (alpha-equivalence)
    and later re-definitions shadow earlier ones correctly.
    """
    tree = get_backend(_ARITH_BACKEND_ID).parse_tree(source)
    defined: list[str] = []
    statements: list[Expr] = []
    for child in tree.children:
        if not (isinstance(child, Tree) and str(child.data) == "statement"):
            continue
        name = str(child.children[0])
        statements.append(_canon_expr(child.children[1], defined))
        defined.append(name)
    return CanonicalProgram(tuple(statements))


def is_type_valid(source: str) -> bool:
    """Whether the arith-sketch backend accepts ``source`` (type/parse gate).

    Delegates to ``backend.validate`` — the authoritative rejection oracle. A
    bare-atom ``root`` (or one that resolves to a bare atom) and any syntax
    error are rejected; a compound ``root`` expression is accepted.
    """
    try:
        get_backend(_ARITH_BACKEND_ID).validate(source)
        return True
    except ParseError:
        return False


def assert_type_valid(source: str) -> None:
    """Raise :class:`ParseError` if the backend rejects ``source``."""
    get_backend(_ARITH_BACKEND_ID).validate(source)


def program_actions(program: CanonicalProgram) -> tuple[str, ...]:
    """Preorder action linearisation over the reused codec sigils.

    Each statement opens with the ``=`` marker (``production_codec.STMT``); an
    operator emits ``o:<sym>`` then its children, a literal emits ``#<class>``,
    a ref emits ``~<delta>``. Fixed node arities make this preorder an
    unambiguous prefix encoding of the program.
    """
    out: list[str] = []
    for stmt in program.statements:
        out.append(STMT)
        _emit_actions(stmt, out)
    return tuple(out)


def _emit_actions(expr: Expr, out: list[str]) -> None:
    kind = expr[0]
    if kind == "lit":
        out.append(f"{LIT_PREFIX}{expr[1]}")
    elif kind == "ref":
        out.append(f"{REL_REF_PREFIX}{expr[1]}")
    elif kind == "free":
        out.append(f"{LIT_PREFIX}free:{expr[1]}")
    elif kind == "op":
        out.append(f"{OP_PREFIX}{expr[1]}")
        _emit_actions(expr[2], out)
        _emit_actions(expr[3], out)
    else:
        raise ValueError(f"unknown canonical node: {expr!r}")
