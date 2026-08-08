"""SRP-005 (SLM-462): pack-owned canonicalization and a proof-scoped,
tamper-evident rewrite-certificate schema for the typed symbolic-expression
IR (``symbolic_expr_ir.py``).

Collapses only demonstrably, exactly equivalent expressions -- there is no
cross-pack global algebra here, just a small, fixed, auditable set of
:data:`REWRITE_RULES`, each declared once with a stable ``rule_id`` and
``rule_version``. Every successful rewrite produces a :class:`RewriteCertificateV1`
recording which rule fired, its version, the input/output canonical hashes,
and a tamper-evident digest -- the certificate schema is deliberately
modeled on ``evals/semantic_failure.py``'s ``VerifierWitnessV1`` (schema
version + typed evidence + trailing self-digest + fail-closed
``from_dict``), per this issue's pre-registered ownership_map guidance that
a rewrite certificate must follow the VCE-001 witness pattern rather than
invent a parallel proof format.

Honest scope note on the ``verifier_gate_stack`` extension point: the
ownership map's downstream_extension_map row for SRP-005 predicted this
would also layer under ``data/verify/stack.py``'s G0-G12 gate numbering.
That gate stack scores LLM decode/verifier evidence; a purely structural
algebraic rewrite over a numeric-DSL AST has no gate/pass-fail outcome to
compose with, so this module does not import or extend ``Gate``/
``GateResult`` -- doing so would be a fabricated integration, not a real
one. What *is* reused, per the row's actual constraint, is the tamper-
evident certificate shape itself. This divergence is recorded in
``ownership_map.json``'s notes for this subsystem, following the same
documented-divergence precedent SGS-008 set.

Every rule below is an exact IEEE-754 float64 identity over the full
real/non-finite domain (commutativity of ``+``/``*``, double negation,
``abs`` idempotence) -- none require a numeric side condition, so none can
silently fold a not-yet-fit coefficient or change declared numerical
semantics. A rule's *structural* precondition (the shape it pattern-matches
against) is enforced by :func:`apply_rewrite_rule`, which raises
:class:`RewriteSideConditionError` rather than silently no-op'ing when a
rule does not apply -- a failed rewrite attempt is never disguised as a
successful, certified one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from slm_training.dsl.symbolic_expr_ir import (
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    serialize_expr,
    validate_expr_budget,
)

CERTIFICATE_SCHEMA_VERSION = "v1"
CHECKER_IDENTITY = "symbolic_expr_canonicalize/v1"
SEMANTIC_SCOPE = "symbolic_regression_pack"

_RULE_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")
_MAX_CANONICALIZE_ITERATIONS = 64


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _expr_hash(node: SymbolicExprNode) -> str:
    return hashlib.sha256(serialize_expr(node).encode("utf-8")).hexdigest()


class RewriteSideConditionError(ValueError):
    """A rewrite rule's structural precondition failed for the given node.

    The input is never rewritten and no certificate is produced -- a
    non-applicable rule is a recorded failure, never a silent no-op
    disguised as success.
    """

    def __init__(self, rule_id: str, reason: str) -> None:
        self.rule_id = rule_id
        self.reason = reason
        super().__init__(f"rewrite rule {rule_id!r} precondition failed: {reason}")


@dataclass(frozen=True)
class RewriteRuleV1:
    """A single pack-owned, versioned equivalence rewrite rule declaration."""

    rule_id: str
    rule_version: str
    operator: str
    commutes: bool
    idempotent: bool
    description: str

    def __post_init__(self) -> None:
        if not _RULE_VERSION_RE.match(self.rule_version):
            raise ValueError(
                f"rule_version must match {_RULE_VERSION_RE.pattern!r}, got {self.rule_version!r}"
            )


def _try_commutative_operand_sort(
    node: SymbolicExprNode, rule: RewriteRuleV1
) -> SymbolicExprNode | None:
    if not isinstance(node, OpApply) or node.operator != rule.operator:
        return None
    left, right = node.args
    if serialize_expr(left) <= serialize_expr(right):
        return None  # already in canonical order -- not a rewrite
    return OpApply(operator=node.operator, args=(right, left))


def _try_double_negation_elimination(
    node: SymbolicExprNode, rule: RewriteRuleV1
) -> SymbolicExprNode | None:
    if not isinstance(node, OpApply) or node.operator != "neg":
        return None
    inner = node.args[0]
    if not (isinstance(inner, OpApply) and inner.operator == "neg"):
        return None
    return inner.args[0]


def _try_double_abs_elimination(
    node: SymbolicExprNode, rule: RewriteRuleV1
) -> SymbolicExprNode | None:
    if not isinstance(node, OpApply) or node.operator != "abs":
        return None
    inner = node.args[0]
    if not (isinstance(inner, OpApply) and inner.operator == "abs"):
        return None
    return inner  # abs(abs(x)) -> abs(x)


ADD_COMMUTATIVE_SORT = RewriteRuleV1(
    rule_id="commutative_operand_sort_add",
    rule_version="v1",
    operator="+",
    commutes=True,
    idempotent=False,
    description=(
        "Deterministically order the two operands of a binary '+' by their canonical "
        "serialized form. Exact under IEEE-754 (a+b == b+a bit-for-bit), so this never "
        "changes numerical semantics."
    ),
)
MUL_COMMUTATIVE_SORT = RewriteRuleV1(
    rule_id="commutative_operand_sort_mul",
    rule_version="v1",
    operator="*",
    commutes=True,
    idempotent=False,
    description=(
        "Deterministically order the two operands of a binary '*' by their canonical "
        "serialized form. Exact under IEEE-754 (a*b == b*a bit-for-bit), so this never "
        "changes numerical semantics."
    ),
)
DOUBLE_NEGATION_ELIMINATION = RewriteRuleV1(
    rule_id="double_negation_elimination",
    rule_version="v1",
    operator="neg",
    commutes=False,
    idempotent=True,
    description=(
        "neg(neg(x)) -> x. Exact under IEEE-754 including signed zero (-(-x) == x for "
        "every finite/non-finite float64 x)."
    ),
)
DOUBLE_ABS_ELIMINATION = RewriteRuleV1(
    rule_id="double_abs_elimination",
    rule_version="v1",
    operator="abs",
    commutes=False,
    idempotent=True,
    description="abs(abs(x)) -> abs(x). abs is idempotent by definition; exact for every finite/non-finite float64 x.",
)

REWRITE_RULES: tuple[RewriteRuleV1, ...] = (
    ADD_COMMUTATIVE_SORT,
    MUL_COMMUTATIVE_SORT,
    DOUBLE_NEGATION_ELIMINATION,
    DOUBLE_ABS_ELIMINATION,
)

_RuleMatcher = Callable[[SymbolicExprNode, RewriteRuleV1], "SymbolicExprNode | None"]
_RULE_MATCHERS: dict[str, _RuleMatcher] = {
    ADD_COMMUTATIVE_SORT.rule_id: _try_commutative_operand_sort,
    MUL_COMMUTATIVE_SORT.rule_id: _try_commutative_operand_sort,
    DOUBLE_NEGATION_ELIMINATION.rule_id: _try_double_negation_elimination,
    DOUBLE_ABS_ELIMINATION.rule_id: _try_double_abs_elimination,
}
assert set(_RULE_MATCHERS) == {rule.rule_id for rule in REWRITE_RULES}  # kept in lockstep by construction


def _certificate_digest(cert: "RewriteCertificateV1") -> str:
    payload = asdict(cert)
    payload.pop("certificate_digest", None)
    return _digest(payload)


@dataclass(frozen=True)
class RewriteCertificateV1:
    """Tamper-evident, replayable evidence for one successful rewrite."""

    schema_version: str
    rule_id: str
    rule_version: str
    input_canonical_hash: str
    output_canonical_hash: str
    semantic_scope: str
    checker_identity: str
    preconditions: tuple[str, ...]
    certificate_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewriteCertificateV1":
        if payload.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported rewrite certificate schema: {payload.get('schema_version')!r}")
        cert = cls(**{**payload, "preconditions": tuple(payload["preconditions"])})
        if _certificate_digest(cert) != cert.certificate_digest:
            raise ValueError("rewrite certificate digest mismatch (tampered or corrupted payload)")
        return cert


def certificate_integrity_ok(cert: RewriteCertificateV1) -> bool:
    """Recompute the certificate digest and compare; used by replay/tamper checks."""
    return _certificate_digest(cert) == cert.certificate_digest


def build_rewrite_certificate(
    rule: RewriteRuleV1,
    input_node: SymbolicExprNode,
    output_node: SymbolicExprNode,
    *,
    preconditions: tuple[str, ...],
) -> RewriteCertificateV1:
    cert = RewriteCertificateV1(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        input_canonical_hash=_expr_hash(input_node),
        output_canonical_hash=_expr_hash(output_node),
        semantic_scope=SEMANTIC_SCOPE,
        checker_identity=CHECKER_IDENTITY,
        preconditions=preconditions,
        certificate_digest="",
    )
    return dataclasses.replace(cert, certificate_digest=_certificate_digest(cert))


def apply_rewrite_rule(
    rule: RewriteRuleV1, node: SymbolicExprNode
) -> tuple[SymbolicExprNode, RewriteCertificateV1]:
    """Apply exactly one declared rewrite rule to ``node``.

    Raises :class:`RewriteSideConditionError` if ``node`` does not
    structurally match ``rule``'s precondition -- callers must not treat a
    non-applicable rule as a no-op success.
    """
    if rule not in REWRITE_RULES:
        raise ValueError(f"unknown rewrite rule: {rule.rule_id!r}")
    matcher = _RULE_MATCHERS[rule.rule_id]
    result = matcher(node, rule)
    if result is None:
        raise RewriteSideConditionError(
            rule.rule_id, f"node does not match rule {rule.rule_id!r}'s applicability precondition"
        )
    precondition = f"operator == {rule.operator!r}"
    cert = build_rewrite_certificate(rule, node, result, preconditions=(precondition,))
    return result, cert


def _rewrite_node_once(
    node: SymbolicExprNode,
) -> tuple[SymbolicExprNode, RewriteCertificateV1 | None]:
    for rule in REWRITE_RULES:
        try:
            new_node, cert = apply_rewrite_rule(rule, node)
        except RewriteSideConditionError:
            continue
        return new_node, cert
    return node, None


def _canonicalize_bottom_up(
    node: SymbolicExprNode, certificates: list[RewriteCertificateV1]
) -> SymbolicExprNode:
    if isinstance(node, (VarRef, CoefficientHole)):
        return node
    if isinstance(node, OpApply):
        new_args = tuple(_canonicalize_bottom_up(arg, certificates) for arg in node.args)
        current: SymbolicExprNode = (
            OpApply(operator=node.operator, args=new_args) if new_args != node.args else node
        )
        for _ in range(_MAX_CANONICALIZE_ITERATIONS):
            rewritten, cert = _rewrite_node_once(current)
            if cert is None:
                return current
            certificates.append(cert)
            current = rewritten
        raise ValueError(
            f"canonicalization did not converge within {_MAX_CANONICALIZE_ITERATIONS} iterations"
        )
    raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")


def canonicalize_with_certificates(
    node: SymbolicExprNode,
) -> tuple[SymbolicExprNode, tuple[RewriteCertificateV1, ...]]:
    """Canonicalize ``node`` bottom-up to a fixed point under :data:`REWRITE_RULES`.

    Returns the canonical node plus an ordered, replayable certificate
    trail -- one certificate per successful rewrite application, each
    independently tamper-evident and re-verifiable.
    """
    if not isinstance(node, (VarRef, CoefficientHole, OpApply)):
        raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")
    validate_expr_budget(node)
    certificates: list[RewriteCertificateV1] = []
    canonical = _canonicalize_bottom_up(node, certificates)
    return canonical, tuple(certificates)


def canonicalize_expr(node: SymbolicExprNode) -> SymbolicExprNode:
    """Canonicalize ``node``, discarding the certificate trail.

    Deterministic and idempotent:
    ``canonicalize_expr(canonicalize_expr(x)) == canonicalize_expr(x)``.
    """
    canonical, _ = canonicalize_with_certificates(node)
    return canonical


def is_canonical(node: SymbolicExprNode) -> bool:
    """Whether ``node`` is already a fixed point of :func:`canonicalize_expr`."""
    return canonicalize_expr(node) == node


__all__ = [
    "ADD_COMMUTATIVE_SORT",
    "CERTIFICATE_SCHEMA_VERSION",
    "CHECKER_IDENTITY",
    "DOUBLE_ABS_ELIMINATION",
    "DOUBLE_NEGATION_ELIMINATION",
    "MUL_COMMUTATIVE_SORT",
    "REWRITE_RULES",
    "SEMANTIC_SCOPE",
    "RewriteCertificateV1",
    "RewriteRuleV1",
    "RewriteSideConditionError",
    "apply_rewrite_rule",
    "build_rewrite_certificate",
    "canonicalize_expr",
    "canonicalize_with_certificates",
    "certificate_integrity_ok",
    "is_canonical",
]
