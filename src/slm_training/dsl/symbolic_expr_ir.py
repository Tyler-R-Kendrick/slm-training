"""SRP-002 (SLM-453): typed symbolic-expression IR, canonical codec, and
symbol-only contract for the symbolic-regression pack.

Supersedes the untyped dict-shaped expression tree
(``symbolic_regression_pack.py``'s ``_collect_operator_and_ref_names``/
``evaluate_expression``) with a real typed AST: three frozen node types
(:class:`VarRef`, :class:`CoefficientHole`, :class:`OpApply`) plus a
deterministic structural codec (:func:`serialize_expr`/:func:`parse_expr`).
The Lark surface grammar registered by SRP-001
(``dsl/grammar/backends/symbolic_regression.py``) stays as an input surface
syntax; this module owns the canonical *structural* form.

Symbol-only, by construction: variables are referenced by an opaque index
into the problem's ``variables`` tuple, never by name, and coefficients are
referenced by an opaque index only -- no free numeric literal or string ever
appears in the structural codec's output. This satisfies the same
"symbol-only" policy ``dsl/language_contract.py``'s ``OUTPUT_CONTRACT_VERSION``
names for OpenUI, expressed natively for this pack's numeric-free shape --
it does not call OpenUI's ``assert_symbol_only_output`` (that walker is
element/call/array/object/literal-shaped, not this pack's), and it does not
introduce a second ``OUTPUT_CONTRACT_VERSION``.

Because a variable/coefficient is only ever an opaque index, the codec is
alpha-renaming safe by construction: renaming the underlying problem's
variable names never changes a single byte of the serialized structural
form, since names never enter the IR at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from slm_training.dsl.symbolic_regression_pack import ALLOWED_OPERATORS

if TYPE_CHECKING:
    from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1

BINARY_OPERATORS = frozenset({"+", "-", "*", "/"})
UNARY_OPERATORS = frozenset({"neg", "sin", "cos", "exp", "log", "sqrt", "abs"})
_OPERATOR_ARITY: dict[str, int] = {
    **{op: 2 for op in BINARY_OPERATORS},
    **{op: 1 for op in UNARY_OPERATORS},
}
assert frozenset(_OPERATOR_ARITY) == ALLOWED_OPERATORS  # kept in lockstep by construction

DEFAULT_MAX_DEPTH = 32
DEFAULT_MAX_NODES = 256


@dataclass(frozen=True)
class VarRef:
    """A reference to input variable ``index`` -- opaque, never a name."""

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"variable index must be non-negative, got {self.index}")


@dataclass(frozen=True)
class CoefficientHole:
    """An opaque free-coefficient slot -- structural identity only.

    Carries no numeric value: the actual coefficient is fit after structure
    synthesis (``coefficient_hole_policy`` on the problem contract), never
    emitted through this codec.
    """

    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"coefficient index must be non-negative, got {self.index}")


@dataclass(frozen=True)
class OpApply:
    """One operator application; arity is checked against the fixed table."""

    operator: str
    args: tuple["SymbolicExprNode", ...]

    def __post_init__(self) -> None:
        expected = _OPERATOR_ARITY.get(self.operator)
        if expected is None:
            raise ValueError(f"unknown operator {self.operator!r}")
        if len(self.args) != expected:
            raise ValueError(
                f"operator {self.operator!r} requires {expected} argument(s), "
                f"got {len(self.args)}"
            )


SymbolicExprNode = Union[VarRef, CoefficientHole, OpApply]


def serialize_expr(node: SymbolicExprNode) -> str:
    """Deterministic canonical S-expression form. Every constructible AST
    round-trips through :func:`parse_expr`."""
    if isinstance(node, VarRef):
        return f"v{node.index}"
    if isinstance(node, CoefficientHole):
        return f"c{node.index}"
    if isinstance(node, OpApply):
        return f"({node.operator} {' '.join(serialize_expr(a) for a in node.args)})"
    raise TypeError(f"unknown symbolic expression node type {type(node).__name__}")


_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
_VAR_TOKEN_RE = re.compile(r"^v(0|[1-9][0-9]*)$")
_COEF_TOKEN_RE = re.compile(r"^c(0|[1-9][0-9]*)$")


def parse_expr(
    text: str,
    *,
    num_variables: int,
    allowed_operators: frozenset[str] = ALLOWED_OPERATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> SymbolicExprNode:
    """Parse the canonical structural form, failing closed on anything that
    is not exactly a well-formed, in-budget, problem-authorized expression.

    The token grammar admits only ``(``, ``)``, operator names, and
    ``v<index>``/``c<index>`` symbol references -- a free numeric literal,
    string, or arbitrary identifier can never lex as a valid token here, so
    unbounded/free-form emission is impossible through this codec, not just
    discouraged by convention.
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        raise ValueError("empty expression")
    pos = 0
    node_count = 0

    def _next() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of expression")
        token = tokens[pos]
        pos += 1
        return token

    def _parse(depth: int) -> SymbolicExprNode:
        nonlocal node_count
        if depth > max_depth:
            raise ValueError(f"expression exceeds max_depth={max_depth}")
        node_count += 1
        if node_count > max_nodes:
            raise ValueError(f"expression exceeds max_nodes={max_nodes}")

        token = _next()
        if token == "(":
            op = _next()
            if op == ")":
                raise ValueError("missing operator after '('")
            if op not in allowed_operators:
                raise ValueError(f"operator {op!r} is not authorized for this problem")
            args: list[SymbolicExprNode] = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("unterminated '(' -- missing ')'")
                if tokens[pos] == ")":
                    break
                args.append(_parse(depth + 1))
            _next()  # consume ')'
            return OpApply(operator=op, args=tuple(args))
        if token == ")":
            raise ValueError("unexpected ')'")

        var_match = _VAR_TOKEN_RE.match(token)
        if var_match:
            index = int(var_match.group(1))
            if not 0 <= index < num_variables:
                raise ValueError(
                    f"variable index {index} out of range [0, {num_variables})"
                )
            return VarRef(index=index)

        coef_match = _COEF_TOKEN_RE.match(token)
        if coef_match:
            return CoefficientHole(index=int(coef_match.group(1)))

        raise ValueError(f"malformed token {token!r}")

    result = _parse(0)
    if pos != len(tokens):
        raise ValueError("trailing tokens after a complete expression")
    return result


def collect_symbols(node: SymbolicExprNode) -> tuple[frozenset[int], frozenset[int]]:
    """The variable-index and coefficient-index symbol table referenced by
    ``node`` -- the derived, read-only view PCT-agnostic callers need."""
    variables: set[int] = set()
    coefficients: set[int] = set()

    def walk(current: SymbolicExprNode) -> None:
        if isinstance(current, VarRef):
            variables.add(current.index)
        elif isinstance(current, CoefficientHole):
            coefficients.add(current.index)
        elif isinstance(current, OpApply):
            for arg in current.args:
                walk(arg)

    walk(node)
    return frozenset(variables), frozenset(coefficients)


def expr_depth_and_node_count(node: SymbolicExprNode) -> tuple[int, int]:
    """``(depth, node_count)`` for an already-constructed tree -- usable on
    trees built directly with the dataclasses, not just parsed ones."""

    def walk(current: SymbolicExprNode) -> tuple[int, int]:
        if isinstance(current, (VarRef, CoefficientHole)):
            return 1, 1
        depths_and_counts = [walk(arg) for arg in current.args]
        depth = 1 + max((d for d, _ in depths_and_counts), default=0)
        count = 1 + sum(c for _, c in depths_and_counts)
        return depth, count

    return walk(node)


def validate_expr_budget(
    node: SymbolicExprNode,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> None:
    """Fail closed on a directly-constructed tree that exceeds budget --
    the same guarantee :func:`parse_expr` enforces eagerly while parsing."""
    depth, count = expr_depth_and_node_count(node)
    if depth > max_depth:
        raise ValueError(f"expression exceeds max_depth={max_depth}")
    if count > max_nodes:
        raise ValueError(f"expression exceeds max_nodes={max_nodes}")


def parse_problem_expr(
    problem: "SymbolicRegressionProblemV1",
    text: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> SymbolicExprNode:
    """Parse ``text`` under ``problem``'s own variable count and operator
    allowlist -- the intended entry point: the authorized operator set
    comes from the active problem contract, never a global default."""
    return parse_expr(
        text,
        num_variables=len(problem.variables),
        allowed_operators=problem.allowed_operators,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


__all__ = [
    "BINARY_OPERATORS",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "UNARY_OPERATORS",
    "CoefficientHole",
    "OpApply",
    "SymbolicExprNode",
    "VarRef",
    "collect_symbols",
    "expr_depth_and_node_count",
    "parse_expr",
    "parse_problem_expr",
    "serialize_expr",
    "validate_expr_budget",
]
