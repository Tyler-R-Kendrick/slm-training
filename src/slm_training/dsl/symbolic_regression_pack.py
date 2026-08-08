"""SRP-001 (SLM-441): symbolic regression as an explicitly-selected DslPack.

Opt-in only: ``symbolic_regression`` is never aliased to the default pack —
``dsl/pack.py``'s ``_ALIASES`` and ``_active_dsl()`` both resolve to
``"openui"`` unless a caller explicitly passes ``dsl="symbolic_regression"``
or sets ``SLM_GRAMMAR_DSL=symbolic_regression``. Registering this pack does
not change that default.

This module owns pack registration and the versioned problem contract.
SRP-002 (SLM-453) owns the typed symbolic-expression IR/grammar/codec that
supersedes the placeholder grammar wired in here, and SRP-003 (SLM-461) owns
the hardened, vectorized NumPy evaluator with numeric-domain policies that
supersedes :func:`evaluate_expression` below. No Julia/PySR/SymPy/SciPy/
scikit-learn dependency: parsing is Lark (an existing core dependency) and
evaluation walks a fixed, hardcoded operator dispatch table — ``eval``/
``exec`` are never called.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from slm_training.dsl.grammar.backends import get_backend
from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.pack import DslPack, PlaceholderPolicy

SYMBOLIC_REGRESSION_PROBLEM_SCHEMA = "symbolic_regression_problem/v1"
SYMBOLIC_REGRESSION_PACK_ID = "symbolic_regression"

# Binary/unary operators plus the allow-listed unary function names a
# symbolic-regression expression may use. Fixed and non-configurable — never
# derived from user input beyond a dict-key membership check.
ALLOWED_OPERATORS = frozenset(
    {"+", "-", "*", "/", "neg", "sin", "cos", "exp", "log", "sqrt", "abs"}
)
NUMERIC_DOMAIN_POLICIES = frozenset({"reject_nonfinite", "allow_nonfinite"})
COEFFICIENT_HOLE_POLICIES = frozenset(
    {"fixed_constants_only", "fit_free_coefficients"}
)

_VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NEVER_MATCH_RE = re.compile(r"(?!)")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


@dataclass(frozen=True)
class SymbolicRegressionProblemV1:
    """Versioned, hashable symbolic-regression problem/request contract.

    Validates on construction — malformed operators, data shapes, or numeric
    policies raise ``ValueError`` immediately rather than failing later deep
    inside a search or evaluator.
    """

    variables: tuple[str, ...]
    table: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]
    allowed_operators: frozenset[str]
    coefficient_hole_policy: str
    complexity_budget: int
    numeric_domain_policy: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    extrapolation_indices: tuple[int, ...] = ()
    seed: int = 0
    dimensional_constraints: Mapping[str, str] | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    pack_id: str = SYMBOLIC_REGRESSION_PACK_ID
    schema: str = SYMBOLIC_REGRESSION_PROBLEM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SYMBOLIC_REGRESSION_PROBLEM_SCHEMA:
            raise ValueError(f"unsupported problem schema {self.schema!r}")
        if self.pack_id != SYMBOLIC_REGRESSION_PACK_ID:
            raise ValueError(f"unsupported pack_id {self.pack_id!r}")
        if not self.variables:
            raise ValueError("at least one input variable is required")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("variable names must be unique")
        for name in self.variables:
            if not _VARIABLE_NAME_RE.match(name):
                raise ValueError(f"invalid variable name {name!r}")
        if not self.table:
            raise ValueError("table must contain at least one row")
        width = len(self.variables)
        for row in self.table:
            if len(row) != width:
                raise ValueError(
                    f"table row width {len(row)} does not match "
                    f"{width} declared variables"
                )
        if len(self.targets) != len(self.table):
            raise ValueError("targets must have one value per table row")
        if not self.allowed_operators:
            raise ValueError("allowed_operators must be non-empty")
        unsupported = self.allowed_operators - ALLOWED_OPERATORS
        if unsupported:
            raise ValueError(f"unsupported operator(s) {sorted(unsupported)}")
        if self.coefficient_hole_policy not in COEFFICIENT_HOLE_POLICIES:
            raise ValueError(
                "unsupported coefficient_hole_policy "
                f"{self.coefficient_hole_policy!r}"
            )
        if self.complexity_budget <= 0:
            raise ValueError("complexity_budget must be positive")
        if self.numeric_domain_policy not in NUMERIC_DOMAIN_POLICIES:
            raise ValueError(
                f"unsupported numeric_domain_policy {self.numeric_domain_policy!r}"
            )
        if self.dimensional_constraints is not None:
            unknown = set(self.dimensional_constraints) - set(self.variables)
            if unknown:
                raise ValueError(
                    "dimensional_constraints reference unknown variable(s) "
                    f"{sorted(unknown)}"
                )
        row_count = len(self.table)
        splits = {
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "extrapolation_indices": self.extrapolation_indices,
        }
        seen: set[int] = set()
        for split_name, indices in splits.items():
            for index in indices:
                if not 0 <= index < row_count:
                    raise ValueError(
                        f"{split_name} contains out-of-range row index {index}"
                    )
                if index in seen:
                    raise ValueError(f"row index {index} appears in more than one split")
                seen.add(index)
        if not self.train_indices:
            raise ValueError("train_indices must be non-empty")

    @property
    def fingerprint(self) -> str:
        """Deterministic sha256 identity over every contract field.

        Two problems with the same content — including ``pack_id`` — always
        hash identically, so this participates in checkpoint/artifact/
        evidence provenance as the problem's stable identity.
        """
        payload = {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "variables": list(self.variables),
            "table": [list(row) for row in self.table],
            "targets": list(self.targets),
            "allowed_operators": sorted(self.allowed_operators),
            "coefficient_hole_policy": self.coefficient_hole_policy,
            "complexity_budget": self.complexity_budget,
            "numeric_domain_policy": self.numeric_domain_policy,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "extrapolation_indices": list(self.extrapolation_indices),
            "seed": self.seed,
            "dimensional_constraints": (
                dict(self.dimensional_constraints)
                if self.dimensional_constraints
                else None
            ),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sr_slot_contract(source: str, *, declared: Any = None) -> tuple[str, ...]:
    # Symbolic-regression sources carry no OpenUI-style user-content
    # placeholders; only explicitly declared slots pass through.
    return tuple(declared or ())


_PLACEHOLDER_POLICY = PlaceholderPolicy(
    placeholder_re=_NEVER_MATCH_RE,
    content_props=frozenset(),
    slot_contract=_sr_slot_contract,
    is_placeholder=lambda value: False,  # noqa: ARG005
    extract=lambda source: [],  # noqa: ARG005
)


def _sr_resolved_root(source: str) -> Any:
    """Parsed+ref-resolved ``root`` binding; raises ``ParseError`` if absent.

    Uses ``resolved_data`` (not ``parse``/``validate``) because the generic
    transformer's ``Program.root`` collapses to ``None`` for a non-dict root
    (e.g. a bare numeric or variable-ref expression) — the same reason
    ``dsl/packs/arith_sketch.py::evaluate_answer`` reads bindings directly.
    """
    backend = get_backend("symbolic-regression")
    data = backend.resolved_data(source)
    if "root" not in (data.get("bindings") or {}):
        raise ParseError("symbolic_regression: program must bind `root`")
    return data.get("root")


def _collect_operator_and_ref_names(node: Any) -> tuple[frozenset[str], frozenset[str]]:
    """Every operator/function name and every free-variable ref name used."""

    operators: set[str] = set()
    refs: set[str] = set()

    def walk(current: Any) -> None:
        if isinstance(current, dict):
            kind = current.get("k")
            if kind == "BinOp":
                operators.add(str(current.get("op")))
                walk(current.get("left"))
                walk(current.get("right"))
            elif kind == "UnaryOp":
                operators.add("neg" if current.get("op") == "-" else str(current.get("op")))
                walk(current.get("operand"))
            elif current.get("type") == "call":
                operators.add(str(current.get("name")))
                for arg in current.get("args") or []:
                    walk(arg)
            elif current.get("type") == "ref":
                refs.add(str(current.get("name")))
        elif isinstance(current, list):
            for item in current:
                walk(item)

    walk(node)
    return frozenset(operators), frozenset(refs)


def _sr_canonicalize(source: str) -> str:
    _sr_resolved_root(source)  # raises ParseError on invalid/missing root
    backend = get_backend("symbolic-regression")
    program = backend.parse(source)
    return backend.serialize(program)


def _sr_oracle(record: Any, context: Any = None) -> Program:  # noqa: ARG001
    """Validity verdict: parses, binds ``root``, and only uses allow-listed
    operators/functions.

    Never checks numeric behavior — SRP-003 (SLM-461) owns numeric-domain-
    policy evaluation, so this oracle's ``reward_label`` stays
    ``"well_formed_not_behavioral"``.
    """
    source = record if isinstance(record, str) else getattr(record, "openui", "")
    root = _sr_resolved_root(source)
    operators, _refs = _collect_operator_and_ref_names(root)
    unsupported = operators - ALLOWED_OPERATORS
    if unsupported:
        raise ParseError(
            f"symbolic_regression: unsupported operator(s) {sorted(unsupported)}"
        )
    backend = get_backend("symbolic-regression")
    return backend.parse(source)


# ---------------------------------------------------------------------------
# Minimal reference evaluator — proves the parse/evaluate round trip only.
# SRP-003 (SLM-461) supersedes this with a vectorized NumPy evaluator that
# honors SymbolicRegressionProblemV1.numeric_domain_policy.
# ---------------------------------------------------------------------------

_ALLOWED_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}


def evaluate_expression(node: Any, bindings: Mapping[str, float]) -> float:
    """Deterministically evaluate a parsed expression tree over ``bindings``.

    Raises ``ValueError`` on anything invalid: undefined variable, division
    by zero, an operator/function outside :data:`ALLOWED_OPERATORS`, or a
    non-finite result. Every operator and function is a fixed, hardcoded
    dispatch-table lookup — ``eval``/``exec`` are never called, so this can
    never run arbitrary Python.
    """
    if isinstance(node, bool):
        raise ValueError("boolean has no numeric value")
    if isinstance(node, (int, float)):
        value = float(node)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite literal")
        return value
    if isinstance(node, dict):
        kind = node.get("k")
        if kind == "BinOp":
            left = evaluate_expression(node.get("left"), bindings)
            right = evaluate_expression(node.get("right"), bindings)
            op = node.get("op")
            if op == "+":
                result = left + right
            elif op == "-":
                result = left - right
            elif op == "*":
                result = left * right
            elif op == "/":
                if right == 0:
                    raise ValueError("division by zero")
                result = left / right
            else:
                raise ValueError(f"unsupported operator {op!r}")
        elif kind == "UnaryOp" and node.get("op") == "-":
            result = -evaluate_expression(node.get("operand"), bindings)
        elif node.get("type") == "call":
            name = str(node.get("name"))
            func = _ALLOWED_FUNCTIONS.get(name)
            if func is None:
                raise ValueError(f"unsupported function {name!r}")
            args = node.get("args") or []
            if len(args) != 1:
                raise ValueError(f"{name} takes exactly one argument")
            arg = evaluate_expression(args[0], bindings)
            try:
                result = func(arg)
            except ValueError as exc:
                raise ValueError(f"{name}({arg!r}) is outside its domain") from exc
        elif node.get("type") == "ref":
            name = str(node.get("name"))
            if name not in bindings:
                raise ValueError(f"undefined variable {name!r}")
            result = float(bindings[name])
        else:
            raise ValueError(f"unsupported expression node {node!r}")
        if result != result or result in (float("inf"), float("-inf")):
            raise ValueError("non-finite result")
        return result
    raise ValueError(f"non-numeric node {type(node).__name__}")


def evaluate_source(source: str, bindings: Mapping[str, float]) -> float:
    """Parse ``source``'s ``root`` binding and evaluate it over ``bindings``."""
    root = _sr_resolved_root(source)
    return evaluate_expression(root, bindings)


def build_symbolic_regression_pack() -> DslPack:
    return DslPack(
        pack_id=SYMBOLIC_REGRESSION_PACK_ID,
        backend=get_backend("symbolic-regression"),
        placeholder_policy=_PLACEHOLDER_POLICY,
        reward_label="well_formed_not_behavioral",
        canonicalize=_sr_canonicalize,
        oracle=_sr_oracle,
    )


__all__ = [
    "ALLOWED_OPERATORS",
    "COEFFICIENT_HOLE_POLICIES",
    "NUMERIC_DOMAIN_POLICIES",
    "SYMBOLIC_REGRESSION_PACK_ID",
    "SYMBOLIC_REGRESSION_PROBLEM_SCHEMA",
    "SymbolicRegressionProblemV1",
    "build_symbolic_regression_pack",
    "evaluate_expression",
    "evaluate_source",
]
