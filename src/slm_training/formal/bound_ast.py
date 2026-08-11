"""Safe symbolic resource-bound AST and exact evaluator (EVID-04 / SLM-525).

Serialisable expression language for formal resource bounds. Never evaluates
Python/Lean source strings, never uses ``eval``/``exec``, and never silently
falls back to floating-point arithmetic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.harness_core.lineage.records import canonical_json, content_sha

BOUND_AST_SCHEMA = "bound_ast/v1"
BOUND_AST_REGISTRY_SCHEMA = "bound_ast_registry/v1"
BOUND_AST_PARITY_SCHEMA = "bound_ast_parity_fixtures/v1"

_ID = r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
_BOUND_ID = r"^bound\.[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_CODELIKE = re.compile(
    r"(?is)\b(eval|exec|compile|__import__|globals|locals|getattr|setattr|"
    r"bytearray|memoryview)\b|__\w+__|\blambda\s|\bimport\s+\w|\bdef\s+\w|"
    r"\bclass\s+\w|`|\$\(|<\?"
)

U64_MAX = (1 << 64) - 1

VarDomain = Literal["nat", "int", "nonneg_rat", "pos_rat", "rat"]
VarKind = Literal["scalar", "finite_sequence"]
OverflowStatus = Literal["fits_u64", "overflow_u64", "not_applicable"]

# Stable ids (must match registry JSON).
BOUND_FINITE_SEARCH_PREFIX_TREE = "bound.finite_search.prefix_tree.v1"
BOUND_FINITE_SEARCH_COARSE = "bound.finite_search.coarse.v1"
BOUND_CLOSURE_LIVE_UPPER = "bound.closure.live_upper.v1"
BOUND_CLOSURE_STRICT_REMOVALS = "bound.closure.strict_removals.v1"
BOUND_PLACEHOLDER_PENDING = "bound.placeholder.pending_evid04.v1"

REGISTERED_BOUND_AST_IDS: frozenset[str] = frozenset(
    {
        BOUND_FINITE_SEARCH_PREFIX_TREE,
        BOUND_FINITE_SEARCH_COARSE,
        BOUND_CLOSURE_LIVE_UPPER,
        BOUND_CLOSURE_STRICT_REMOVALS,
        BOUND_PLACEHOLDER_PENDING,
    }
)


class BoundAstError(ValueError):
    """Malformed, unsafe, or unevaluable bound AST payload."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _reject_codelike(label: str, text: str) -> None:
    if _CODELIKE.search(text):
        raise BoundAstError(
            f"{label} looks code-like; bound AST rejects executable payloads"
        )


class ConstInt(StrictModel):
    op: Literal["const_int"] = "const_int"
    value: int


class ConstRat(StrictModel):
    op: Literal["const_rat"] = "const_rat"
    numerator: int
    denominator: int = Field(gt=0)


class Var(StrictModel):
    op: Literal["var"] = "var"
    name: str = Field(pattern=_ID)


class Add(StrictModel):
    op: Literal["add"] = "add"
    args: tuple[BoundExpr, ...] = Field(min_length=2)


class Mul(StrictModel):
    op: Literal["mul"] = "mul"
    args: tuple[BoundExpr, ...] = Field(min_length=2)


class Max(StrictModel):
    op: Literal["max"] = "max"
    args: tuple[BoundExpr, ...] = Field(min_length=2)


class Min(StrictModel):
    op: Literal["min"] = "min"
    args: tuple[BoundExpr, ...] = Field(min_length=2)


class Div(StrictModel):
    op: Literal["div"] = "div"
    numerator: BoundExpr
    denominator: BoundExpr


class Floor(StrictModel):
    """Floor toward −∞ (explicit semantics; exact rational)."""

    op: Literal["floor"] = "floor"
    arg: BoundExpr


class Ceil(StrictModel):
    """Ceiling toward +∞ (explicit semantics; exact rational)."""

    op: Literal["ceil"] = "ceil"
    arg: BoundExpr


class SumOver(StrictModel):
    """Sum ``body`` over each element of a declared finite sequence."""

    op: Literal["sum_over"] = "sum_over"
    index: str = Field(pattern=_ID)
    sequence: str = Field(pattern=_ID)
    body: BoundExpr


class ProdOver(StrictModel):
    """Product ``body`` over each element; empty product is 1."""

    op: Literal["prod_over"] = "prod_over"
    index: str = Field(pattern=_ID)
    sequence: str = Field(pattern=_ID)
    body: BoundExpr


class PrefixCumprodSum(StrictModel):
    """Justified KERN-03 op: ``Σ_k ∏_{i≤k} d_i`` over a nat sequence."""

    op: Literal["prefix_cumprod_sum"] = "prefix_cumprod_sum"
    sequence: str = Field(pattern=_ID)


class Len(StrictModel):
    op: Literal["len"] = "len"
    sequence: str = Field(pattern=_ID)


BoundExpr = Annotated[
    Union[
        ConstInt,
        ConstRat,
        Var,
        Add,
        Mul,
        Max,
        Min,
        Div,
        Floor,
        Ceil,
        SumOver,
        ProdOver,
        PrefixCumprodSum,
        Len,
    ],
    Field(discriminator="op"),
]


class Le(StrictModel):
    op: Literal["le"] = "le"
    left: BoundExpr
    right: BoundExpr


class Lt(StrictModel):
    op: Literal["lt"] = "lt"
    left: BoundExpr
    right: BoundExpr


class Eq(StrictModel):
    op: Literal["eq"] = "eq"
    left: BoundExpr
    right: BoundExpr


BoundPred = Annotated[Union[Le, Lt, Eq], Field(discriminator="op")]

for _cls in (Add, Mul, Max, Min, Div, Floor, Ceil, SumOver, ProdOver, Le, Lt, Eq):
    _cls.model_rebuild()


class BoundVarDeclV1(StrictModel):
    name: str = Field(pattern=_ID)
    kind: VarKind
    domain: VarDomain
    unit: str = Field(min_length=1, max_length=64)
    meaning: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _no_code(self) -> BoundVarDeclV1:
        _reject_codelike(f"variable {self.name!r} unit", self.unit)
        _reject_codelike(f"variable {self.name!r} meaning", self.meaning)
        return self


class BoundAstDocumentV1(StrictModel):
    schema_version: Literal["bound_ast/v1"] = "bound_ast/v1"
    bound_ast_id: str = Field(pattern=_BOUND_ID)
    meaning: str = Field(min_length=8, max_length=1024)
    variables: tuple[BoundVarDeclV1, ...] = ()
    expression: BoundExpr
    predicate: BoundPred | None = None
    evaluable: bool = True
    wall_clock_claim: Literal[False] = False
    lean_module: str | None = None
    lean_def: str | None = None

    @model_validator(mode="after")
    def _validate_doc(self) -> BoundAstDocumentV1:
        _reject_codelike("bound meaning", self.meaning)
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise BoundAstError("duplicate variable declarations")
        decl = {v.name: v for v in self.variables}
        _check_expr_vars(self.expression, decl, binders=())
        if self.predicate is not None:
            _check_expr_vars(self.predicate.left, decl, binders=())
            _check_expr_vars(self.predicate.right, decl, binders=())
        return self

    def to_canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def digest(self) -> str:
        return content_sha(self.to_canonical_dict())


def _check_expr_vars(
    expr: BoundExpr,
    decl: Mapping[str, BoundVarDeclV1],
    *,
    binders: tuple[str, ...],
) -> None:
    if isinstance(expr, (ConstInt, ConstRat)):
        return
    if isinstance(expr, Var):
        if expr.name in binders:
            return
        if expr.name not in decl:
            raise BoundAstError(f"unknown variable {expr.name!r}")
        if decl[expr.name].kind != "scalar":
            raise BoundAstError(
                f"variable {expr.name!r} is a finite_sequence; use sum/prod/len/prefix"
            )
        return
    if isinstance(expr, (Add, Mul, Max, Min)):
        for arg in expr.args:
            _check_expr_vars(arg, decl, binders=binders)
        return
    if isinstance(expr, Div):
        _check_expr_vars(expr.numerator, decl, binders=binders)
        _check_expr_vars(expr.denominator, decl, binders=binders)
        return
    if isinstance(expr, (Floor, Ceil)):
        _check_expr_vars(expr.arg, decl, binders=binders)
        return
    if isinstance(expr, (SumOver, ProdOver)):
        if expr.index in binders:
            raise BoundAstError(f"shadowed binder {expr.index!r} (cycle/shadow)")
        if expr.index in decl:
            raise BoundAstError(
                f"binder {expr.index!r} collides with declared variable"
            )
        seq = decl.get(expr.sequence)
        if seq is None or seq.kind != "finite_sequence":
            raise BoundAstError(
                f"sum/prod sequence {expr.sequence!r} must be a declared finite_sequence"
            )
        _check_expr_vars(expr.body, decl, binders=(*binders, expr.index))
        return
    if isinstance(expr, (PrefixCumprodSum, Len)):
        seq = decl.get(expr.sequence)
        if seq is None or seq.kind != "finite_sequence":
            raise BoundAstError(
                f"{expr.op} sequence {expr.sequence!r} must be a declared finite_sequence"
            )
        return
    raise BoundAstError(f"unsupported operator {type(expr).__name__}")


class EvalResult(StrictModel):
    value: str
    as_fraction_numerator: int
    as_fraction_denominator: int
    is_integer: bool
    u64_overflow: OverflowStatus

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.as_fraction_numerator, self.as_fraction_denominator)


def _to_result(value: Fraction) -> EvalResult:
    value = Fraction(value)
    if value.denominator == 1 and value.numerator >= 0:
        overflow: OverflowStatus = (
            "overflow_u64" if value.numerator > U64_MAX else "fits_u64"
        )
    else:
        overflow = "not_applicable"
    spelling = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    return EvalResult(
        value=spelling,
        as_fraction_numerator=value.numerator,
        as_fraction_denominator=value.denominator,
        is_integer=value.denominator == 1,
        u64_overflow=overflow,
    )


def _require_domain(name: str, value: Fraction, domain: VarDomain) -> None:
    if domain == "nat" and (value.denominator != 1 or value.numerator < 0):
        raise BoundAstError(f"{name} domain nat requires non-negative integer")
    if domain == "int" and value.denominator != 1:
        raise BoundAstError(f"{name} domain int requires integer")
    if domain == "nonneg_rat" and value < 0:
        raise BoundAstError(f"{name} domain nonneg_rat requires ≥ 0")
    if domain == "pos_rat" and value <= 0:
        raise BoundAstError(f"{name} domain pos_rat requires > 0")


def _parse_env_value(raw: Any) -> Fraction | tuple[Fraction, ...]:
    if isinstance(raw, bool):
        raise BoundAstError("boolean values are not bound scalars")
    if isinstance(raw, int):
        return Fraction(raw)
    if isinstance(raw, Fraction):
        return Fraction(raw)
    if isinstance(raw, str):
        _reject_codelike("env value", raw)
        return Fraction(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: list[Fraction] = []
        for item in raw:
            if isinstance(item, bool) or not isinstance(item, (int, str, Fraction)):
                raise BoundAstError("finite_sequence elements must be int/rational")
            if isinstance(item, str):
                _reject_codelike("sequence element", item)
            out.append(Fraction(item))
        return tuple(out)
    raise BoundAstError(f"unsupported env value type {type(raw).__name__}")


def _floor_fraction(val: Fraction) -> Fraction:
    n, d = val.numerator, val.denominator
    if d == 1:
        return Fraction(n)
    q, _r = divmod(n, d)
    return Fraction(q)


def _ceil_fraction(val: Fraction) -> Fraction:
    n, d = val.numerator, val.denominator
    if d == 1:
        return Fraction(n)
    q, r = divmod(n, d)
    return Fraction(q if r == 0 else q + 1)


def _seq(env: Mapping[str, Any], name: str) -> tuple[Fraction, ...]:
    parsed = _parse_env_value(env[name])
    if not isinstance(parsed, tuple):
        raise BoundAstError(f"{name!r} is not a finite_sequence in env")
    return parsed


def _eval_expr(
    expr: BoundExpr,
    env: Mapping[str, Any],
    binders: dict[str, Fraction],
) -> Fraction:
    if isinstance(expr, ConstInt):
        return Fraction(expr.value)
    if isinstance(expr, ConstRat):
        return Fraction(expr.numerator, expr.denominator)
    if isinstance(expr, Var):
        if expr.name in binders:
            return binders[expr.name]
        raw = _parse_env_value(env[expr.name])
        if not isinstance(raw, Fraction):
            raise BoundAstError(f"{expr.name!r} is not scalar at evaluation")
        return raw
    if isinstance(expr, Add):
        total = Fraction(0)
        for arg in expr.args:
            total += _eval_expr(arg, env, binders)
        return total
    if isinstance(expr, Mul):
        total = Fraction(1)
        for arg in expr.args:
            total *= _eval_expr(arg, env, binders)
        return total
    if isinstance(expr, Max):
        return max(_eval_expr(a, env, binders) for a in expr.args)
    if isinstance(expr, Min):
        return min(_eval_expr(a, env, binders) for a in expr.args)
    if isinstance(expr, Div):
        den = _eval_expr(expr.denominator, env, binders)
        if den == 0:
            raise BoundAstError("division by zero")
        return _eval_expr(expr.numerator, env, binders) / den
    if isinstance(expr, Floor):
        return _floor_fraction(_eval_expr(expr.arg, env, binders))
    if isinstance(expr, Ceil):
        return _ceil_fraction(_eval_expr(expr.arg, env, binders))
    if isinstance(expr, SumOver):
        if expr.index in binders:
            raise BoundAstError(f"binder cycle/shadow on {expr.index!r}")
        total = Fraction(0)
        for elt in _seq(env, expr.sequence):
            child = dict(binders)
            child[expr.index] = elt
            total += _eval_expr(expr.body, env, child)
        return total
    if isinstance(expr, ProdOver):
        if expr.index in binders:
            raise BoundAstError(f"binder cycle/shadow on {expr.index!r}")
        total = Fraction(1)
        for elt in _seq(env, expr.sequence):
            child = dict(binders)
            child[expr.index] = elt
            total *= _eval_expr(expr.body, env, child)
        return total
    if isinstance(expr, PrefixCumprodSum):
        total = Fraction(0)
        pref = Fraction(1)
        for elt in _seq(env, expr.sequence):
            if elt.denominator != 1 or elt.numerator < 0:
                raise BoundAstError("prefix_cumprod_sum requires nat sequence")
            pref *= elt
            total += pref
        return total
    if isinstance(expr, Len):
        return Fraction(len(_seq(env, expr.sequence)))
    raise BoundAstError(f"unsupported operator {type(expr).__name__}")


def evaluate(
    expression: BoundExpr,
    *,
    variables: Sequence[BoundVarDeclV1],
    env: Mapping[str, Any],
) -> EvalResult:
    """Deterministic exact ``Fraction`` evaluator (fail closed)."""

    decl = {v.name: v for v in variables}
    for name, decl_v in decl.items():
        if name not in env:
            raise BoundAstError(f"missing env binding for {name!r}")
        parsed = _parse_env_value(env[name])
        if decl_v.kind == "scalar":
            if not isinstance(parsed, Fraction):
                raise BoundAstError(f"{name!r} expects a scalar")
            _require_domain(name, parsed, decl_v.domain)
        else:
            if not isinstance(parsed, tuple):
                raise BoundAstError(f"{name!r} expects a finite_sequence")
            for i, elt in enumerate(parsed):
                _require_domain(f"{name}[{i}]", elt, decl_v.domain)
    return _to_result(_eval_expr(expression, env, {}))


def evaluate_predicate(
    predicate: BoundPred,
    *,
    variables: Sequence[BoundVarDeclV1],
    env: Mapping[str, Any],
) -> bool:
    left = evaluate(predicate.left, variables=variables, env=env).fraction
    right = evaluate(predicate.right, variables=variables, env=env).fraction
    if isinstance(predicate, Le):
        return left <= right
    if isinstance(predicate, Lt):
        return left < right
    if isinstance(predicate, Eq):
        return left == right
    raise BoundAstError(f"unsupported predicate {type(predicate).__name__}")


def pretty(expr: BoundExpr | BoundPred) -> str:
    if isinstance(expr, ConstInt):
        return str(expr.value)
    if isinstance(expr, ConstRat):
        return f"{expr.numerator}/{expr.denominator}"
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Add):
        return "(" + " + ".join(pretty(a) for a in expr.args) + ")"
    if isinstance(expr, Mul):
        return "(" + " * ".join(pretty(a) for a in expr.args) + ")"
    if isinstance(expr, Max):
        return "max(" + ", ".join(pretty(a) for a in expr.args) + ")"
    if isinstance(expr, Min):
        return "min(" + ", ".join(pretty(a) for a in expr.args) + ")"
    if isinstance(expr, Div):
        return f"({pretty(expr.numerator)} / {pretty(expr.denominator)})"
    if isinstance(expr, Floor):
        return f"floor({pretty(expr.arg)})"
    if isinstance(expr, Ceil):
        return f"ceil({pretty(expr.arg)})"
    if isinstance(expr, SumOver):
        return f"Σ_{{{expr.index}∈{expr.sequence}}}({pretty(expr.body)})"
    if isinstance(expr, ProdOver):
        return f"Π_{{{expr.index}∈{expr.sequence}}}({pretty(expr.body)})"
    if isinstance(expr, PrefixCumprodSum):
        return f"prefix_cumprod_sum({expr.sequence})"
    if isinstance(expr, Len):
        return f"|{expr.sequence}|"
    if isinstance(expr, Le):
        return f"{pretty(expr.left)} ≤ {pretty(expr.right)}"
    if isinstance(expr, Lt):
        return f"{pretty(expr.left)} < {pretty(expr.right)}"
    if isinstance(expr, Eq):
        return f"{pretty(expr.left)} = {pretty(expr.right)}"
    raise BoundAstError(f"unsupported pretty node {type(expr).__name__}")


def _registry_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "formal"
        / "bound_ast_registry.v1.json"
    )


def _parity_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "formal"
        / "bound_ast_parity_fixtures.v1.json"
    )


@lru_cache(maxsize=1)
def load_bound_ast_registry() -> dict[str, BoundAstDocumentV1]:
    raw = json.loads(_registry_path().read_text(encoding="utf-8"))
    if raw.get("schema") != BOUND_AST_REGISTRY_SCHEMA:
        raise BoundAstError(f"unexpected registry schema {raw.get('schema')!r}")
    entries: dict[str, BoundAstDocumentV1] = {}
    for item in raw["entries"]:
        payload = dict(item)
        expected = payload.pop("digest_sha256", None)
        doc = BoundAstDocumentV1.model_validate(payload)
        if doc.bound_ast_id in entries:
            raise BoundAstError(f"duplicate registry id {doc.bound_ast_id!r}")
        if expected is not None and expected != doc.digest():
            raise BoundAstError(
                f"registry digest mismatch for {doc.bound_ast_id!r}: "
                f"file={expected} computed={doc.digest()}"
            )
        entries[doc.bound_ast_id] = doc
    file_ids = frozenset(entries)
    if file_ids != REGISTERED_BOUND_AST_IDS:
        raise BoundAstError(
            "registry ids drifted from REGISTERED_BOUND_AST_IDS: "
            f"file={sorted(file_ids)} const={sorted(REGISTERED_BOUND_AST_IDS)}"
        )
    return entries


def registered_bound_ast_ids() -> frozenset[str]:
    return frozenset(load_bound_ast_registry())


def resolve_bound_ast(bound_ast_id: str) -> BoundAstDocumentV1:
    try:
        return load_bound_ast_registry()[bound_ast_id]
    except KeyError as exc:
        raise BoundAstError(
            f"unknown bound_ast_id {bound_ast_id!r}; not in EVID-04 registry"
        ) from exc


def assert_registered_bound_ast_id(bound_ast_id: str) -> None:
    if bound_ast_id not in REGISTERED_BOUND_AST_IDS:
        # Still consult registry so digest/load errors surface.
        load_bound_ast_registry()
        raise BoundAstError(
            f"bound_ast_id {bound_ast_id!r} is not a registered EVID-04 bound AST id"
        )
    resolve_bound_ast(bound_ast_id)


def evaluate_bound(bound_ast_id: str, env: Mapping[str, Any]) -> EvalResult:
    doc = resolve_bound_ast(bound_ast_id)
    if not doc.evaluable:
        raise BoundAstError(
            f"bound_ast_id {bound_ast_id!r} is registered but not evaluable"
        )
    return evaluate(doc.expression, variables=doc.variables, env=env)


def round_trip(doc: BoundAstDocumentV1) -> BoundAstDocumentV1:
    payload = json.loads(canonical_json(doc.to_canonical_dict()))
    again = BoundAstDocumentV1.model_validate(payload)
    if again.digest() != doc.digest():
        raise BoundAstError("canonical round trip changed digest")
    return again


def load_parity_fixtures() -> dict[str, Any]:
    raw = json.loads(_parity_path().read_text(encoding="utf-8"))
    if raw.get("schema") != BOUND_AST_PARITY_SCHEMA:
        raise BoundAstError(f"unexpected parity schema {raw.get('schema')!r}")
    return raw


__all__ = [
    "BOUND_AST_PARITY_SCHEMA",
    "BOUND_AST_REGISTRY_SCHEMA",
    "BOUND_AST_SCHEMA",
    "BOUND_CLOSURE_LIVE_UPPER",
    "BOUND_CLOSURE_STRICT_REMOVALS",
    "BOUND_FINITE_SEARCH_COARSE",
    "BOUND_FINITE_SEARCH_PREFIX_TREE",
    "BOUND_PLACEHOLDER_PENDING",
    "REGISTERED_BOUND_AST_IDS",
    "Add",
    "BoundAstDocumentV1",
    "BoundAstError",
    "BoundExpr",
    "BoundPred",
    "BoundVarDeclV1",
    "Ceil",
    "ConstInt",
    "ConstRat",
    "Div",
    "Eq",
    "EvalResult",
    "Floor",
    "Le",
    "Len",
    "Lt",
    "Max",
    "Min",
    "Mul",
    "PrefixCumprodSum",
    "ProdOver",
    "SumOver",
    "U64_MAX",
    "Var",
    "assert_registered_bound_ast_id",
    "evaluate",
    "evaluate_bound",
    "evaluate_predicate",
    "load_bound_ast_registry",
    "load_parity_fixtures",
    "pretty",
    "registered_bound_ast_ids",
    "resolve_bound_ast",
    "round_trip",
]
