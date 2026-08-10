"""RSP-004 (SLM-479): proof-scoped e-graph/equivalence-region experiment
(EXP-SR-8) for the symbolic-regression pack.

An isolated experiment module, not a pack-owned compiler-hard subsystem:
nothing in ``dsl/symbolic_regression_pack.py`` imports this file, and it is
never wired into the default canonicalization path -- per SLM-479's own
acceptance criterion, "Result remains default-off regardless of fixture
win."

Seeds rewrite rules exclusively from SRP-005's certified
``symbolic_expr_canonicalize.REWRITE_RULES``: no new rewrite semantics are
introduced here. The e-graph's union-find congruence closure lets an
equivalence class be reached by more than one certified rewrite path before
extracting a representative, which a single deterministic canonicalization
pass cannot represent; every union this module performs still carries the
same :class:`RewriteCertificateV1` SRP-005 already produces, and
:func:`verify_rewrite_numerically` additionally *re-verifies* each
application numerically (evaluating both sides over sampled points and
requiring identical predictions, NaN-aware) before counting it toward
``egraph_certified_equivalence_rate`` -- the pre-registered EXP-SR-8 primary
metric (``exp_sr_catalogue.py``, kill_threshold=1.0, kill_operator="lt": any
uncertified or numerically-unverified union kills the candidate arm
outright).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from slm_training.dsl.symbolic_expr_canonicalize import (
    REWRITE_RULES,
    RewriteCertificateV1,
    RewriteSideConditionError,
    apply_rewrite_rule,
    certificate_integrity_ok,
)
from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized
from slm_training.dsl.symbolic_expr_ir import (
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    serialize_expr,
)

EGRAPH_SCHEMA_VERSION = "v1"
DEFAULT_MAX_SATURATION_ROUNDS = 16


@dataclass(frozen=True)
class ENodeV1:
    """One e-node: an operator applied to child e-class ids, or a leaf.

    Structurally hashable so that two e-nodes built from the same
    operator/children (post congruence) collapse to one hashcons entry --
    the core e-graph invariant.
    """

    kind: str  # "var" | "coef" | operator name
    leaf_index: int | None  # set for "var"/"coef", None for an operator node
    children: tuple[int, ...]  # child e-class ids ("" for a leaf)


class EGraph:
    """A minimal union-find e-graph over :class:`SymbolicExprNode`, seeded
    and rewritten exclusively through SRP-005's certified rule set."""

    def __init__(self) -> None:
        self.schema_version = EGRAPH_SCHEMA_VERSION
        self._parent: list[int] = []
        self._hashcons: dict[ENodeV1, int] = {}
        self._members: list[set[ENodeV1]] = []
        self._certificates: list[RewriteCertificateV1] = []

    def _new_class(self, node: ENodeV1) -> int:
        eclass_id = len(self._parent)
        self._parent.append(eclass_id)
        self._members.append({node})
        self._hashcons[node] = eclass_id
        return eclass_id

    def find(self, eclass_id: int) -> int:
        root = eclass_id
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[eclass_id] != root:
            self._parent[eclass_id], eclass_id = root, self._parent[eclass_id]
        return root

    def add(self, node: SymbolicExprNode) -> int:
        """Recursively add ``node``, returning its e-class id (hashconsed:
        an already-present structurally-identical e-node is never
        duplicated)."""
        if isinstance(node, VarRef):
            enode = ENodeV1(kind="var", leaf_index=node.index, children=())
        elif isinstance(node, CoefficientHole):
            enode = ENodeV1(kind="coef", leaf_index=node.index, children=())
        elif isinstance(node, OpApply):
            child_ids = tuple(self.find(self.add(arg)) for arg in node.args)
            enode = ENodeV1(kind=node.operator, leaf_index=None, children=child_ids)
        else:
            raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")
        existing = self._hashcons.get(enode)
        if existing is not None:
            return self.find(existing)
        return self._new_class(enode)

    def union(self, a: int, b: int, certificate: RewriteCertificateV1) -> int:
        """Merge the e-classes of ``a``/``b``, recording ``certificate`` as
        the evidence for why they are equivalent. Always records the
        certificate, even if ``a``/``b`` were already in the same class
        (re-deriving an equivalence is not an error)."""
        self._certificates.append(certificate)
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return root_a
        # Deterministic tie-break: the lower id always survives as the new
        # root, so repeated runs over the same input produce the same shape.
        keep, absorb = (root_a, root_b) if root_a < root_b else (root_b, root_a)
        self._parent[absorb] = keep
        self._members[keep] |= self._members[absorb]
        self._members[absorb] = set()
        return keep

    def all_class_ids(self) -> tuple[int, ...]:
        """Every e-class id ever minted (roots and absorbed ids alike)."""
        return tuple(range(len(self._parent)))

    def n_enodes(self) -> int:
        return len(self._hashcons)

    def n_eclasses(self) -> int:
        return len({self.find(i) for i in self.all_class_ids()})

    @property
    def certificates(self) -> tuple[RewriteCertificateV1, ...]:
        return tuple(self._certificates)

    def safe_representatives(self) -> dict[int, tuple[SymbolicExprNode, int]]:
        """Bottom-up fixed-point: for every e-class, its lowest-cost (fewest
        AST nodes; ties broken by the canonical S-expression codec's lexical
        order) member that is reconstructible *without cycles*.

        A rule like double-negation elimination unions a class with one of
        its own descendants' classes (``neg(neg(x))`` merges with ``x``),
        which leaves some e-nodes only reconstructible by re-entering their
        own class -- a naive recursive walk over an arbitrary member can
        recurse forever. This computes strictly bottom-up: an operator
        e-node only contributes once every child class already has a known
        acyclic representative, so a cyclic member is simply never selected
        once any acyclic alternative exists (every class reachable from a
        real AST always has one -- e.g. the leaf that started it)."""
        best: dict[int, tuple[SymbolicExprNode, int]] = {}
        class_ids = sorted({self.find(i) for i in self.all_class_ids()})
        changed = True
        while changed:
            changed = False
            for root in class_ids:
                for enode in self._members[root]:
                    if enode.kind == "var":
                        candidate: SymbolicExprNode = VarRef(index=enode.leaf_index)  # type: ignore[arg-type]
                        cost = 1
                    elif enode.kind == "coef":
                        candidate = CoefficientHole(index=enode.leaf_index)  # type: ignore[arg-type]
                        cost = 1
                    else:
                        child_roots = [self.find(child) for child in enode.children]
                        if any(child_root not in best for child_root in child_roots):
                            continue  # a child has no acyclic representative yet
                        child_terms = [best[child_root] for child_root in child_roots]
                        candidate = OpApply(operator=enode.kind, args=tuple(term for term, _ in child_terms))
                        cost = 1 + sum(term_cost for _, term_cost in child_terms)
                    current = best.get(root)
                    key = (cost, serialize_expr(candidate))
                    if current is None or key < (current[1], serialize_expr(current[0])):
                        best[root] = (candidate, cost)
                        changed = True
        return best

    def extract(self, eclass_id: int) -> SymbolicExprNode:
        """The minimal acyclic member of ``eclass_id`` -- deterministic
        regardless of which member was discovered first."""
        best = self.safe_representatives()
        return best[self.find(eclass_id)][0]


@dataclass(frozen=True)
class RewriteApplicationV1:
    """One certified rewrite actually applied during saturation, with both
    sides retained so the application can be independently re-verified
    numerically afterward (a certificate's hashes alone cannot be inverted
    back into an expression)."""

    before: SymbolicExprNode
    after: SymbolicExprNode
    certificate: RewriteCertificateV1


def _reconstruct_member_one_level(
    egraph: EGraph, enode: ENodeV1, safe: dict[int, tuple[SymbolicExprNode, int]]
) -> SymbolicExprNode | None:
    """Reconstruct ``enode`` using each child's already-known acyclic
    representative -- never recurses into a child's own member set, so this
    can never re-enter a cycle. Returns ``None`` if a child has no acyclic
    representative yet (skip this member this round; it will be retried
    once that child resolves)."""
    if enode.kind == "var":
        return VarRef(index=enode.leaf_index)  # type: ignore[arg-type]
    if enode.kind == "coef":
        return CoefficientHole(index=enode.leaf_index)  # type: ignore[arg-type]
    children: list[SymbolicExprNode] = []
    for child in enode.children:
        entry = safe.get(egraph.find(child))
        if entry is None:
            return None
        children.append(entry[0])
    return OpApply(operator=enode.kind, args=tuple(children))


def saturate(
    egraph: EGraph, *, max_rounds: int = DEFAULT_MAX_SATURATION_ROUNDS
) -> tuple[int, tuple[RewriteApplicationV1, ...]]:
    """Apply SRP-005's certified :data:`REWRITE_RULES` to every e-class
    currently in ``egraph`` until a fixed point (no new union in a round) or
    ``max_rounds`` is reached. Returns ``(rounds_run, applications)``.

    Every stored member is tested every round, but reconstructed via each
    child's current acyclic representative (:meth:`EGraph.safe_representatives`)
    rather than a naive recursive walk, so a self-referential e-class (e.g.
    from double-negation elimination) can never cause unbounded recursion.

    Every successful rewrite application produces the certificate SRP-005
    already builds (:func:`apply_rewrite_rule`) and is recorded via
    :meth:`EGraph.union` -- no rewrite here bypasses that certificate path.
    """
    applications: list[RewriteApplicationV1] = []
    round_number = 0
    for round_number in range(1, max_rounds + 1):
        changed = False
        safe = egraph.safe_representatives()
        for eclass_id in sorted({egraph.find(i) for i in egraph.all_class_ids()}):
            for enode in list(egraph._members[egraph.find(eclass_id)]):
                enode_form = _reconstruct_member_one_level(egraph, enode, safe)
                if enode_form is None:
                    continue
                for rule in REWRITE_RULES:
                    try:
                        rewritten, certificate = apply_rewrite_rule(rule, enode_form)
                    except RewriteSideConditionError:
                        continue
                    new_class = egraph.add(rewritten)
                    before = egraph.find(eclass_id)
                    egraph.union(eclass_id, new_class, certificate)
                    applications.append(RewriteApplicationV1(before=enode_form, after=rewritten, certificate=certificate))
                    if egraph.find(eclass_id) != before:
                        changed = True
        if not changed:
            break
    return round_number, tuple(applications)


@dataclass(frozen=True)
class CertificateVerificationV1:
    """Independent runtime re-verification of one rewrite application:
    numerically re-evaluates both sides rather than trusting rule
    provenance -- the EXP-SR-8 falsifier's explicit requirement."""

    rule_id: str
    is_certified: bool  # digest/schema integrity of the certificate itself
    is_numerically_equivalent: bool  # bit-identical predictions, both sides
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "is_certified": self.is_certified,
            "is_numerically_equivalent": self.is_numerically_equivalent,
            "passed": self.passed,
        }


def verify_rewrite_numerically(
    before: SymbolicExprNode,
    after: SymbolicExprNode,
    certificate: RewriteCertificateV1,
    *,
    variables: np.ndarray,
    coefficients: np.ndarray | None = None,
    domain_policy: str = "allow_nonfinite",
) -> CertificateVerificationV1:
    """Independently re-verify one certified rewrite: its own tamper digest
    must be intact, and re-evaluating ``before``/``after`` over ``variables``
    must produce identical predictions row for row (NaN-aware: a matching
    domain violation on both sides is equivalence, not a mismatch)."""
    is_certified = certificate_integrity_ok(certificate)
    before_result = evaluate_vectorized(
        before, variables=variables, coefficients=coefficients, domain_policy=domain_policy
    )
    after_result = evaluate_vectorized(
        after, variables=variables, coefficients=coefficients, domain_policy=domain_policy
    )
    is_numerically_equivalent = bool(
        np.array_equal(
            np.array(before_result.predictions), np.array(after_result.predictions), equal_nan=True
        )
    )
    return CertificateVerificationV1(
        rule_id=certificate.rule_id,
        is_certified=is_certified,
        is_numerically_equivalent=is_numerically_equivalent,
        passed=is_certified and is_numerically_equivalent,
    )


__all__ = [
    "DEFAULT_MAX_SATURATION_ROUNDS",
    "EGRAPH_SCHEMA_VERSION",
    "CertificateVerificationV1",
    "EGraph",
    "ENodeV1",
    "RewriteApplicationV1",
    "saturate",
    "verify_rewrite_numerically",
]
