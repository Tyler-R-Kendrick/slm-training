"""Tests for BEq-analogue semantic-fidelity predicates and rates."""

from __future__ import annotations

from slm_training.dsl.solver.state import (
    DomainValue,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import SupportCertificate, SupportQuery
from slm_training.evals.semantic_fidelity import (
    AST_BEQ_RATE,
    CANONICAL_BEQ_RATE,
    ast_beq,
    canonical_beq,
    certificate_equivalence_rate,
    certificate_equivalent,
    fidelity_rates_from_pairs,
    score_row,
)

_SIMPLE = 'root = Stack([t], "column")\nt = TextContent(":x")'
_REORDER_STYLE = 'root = Stack([t],"column")\nt = TextContent(":x")'


def test_ast_beq_true_for_style_normalized_match() -> None:
    assert ast_beq(_SIMPLE, _REORDER_STYLE) is True


def test_ast_beq_false_for_different_structure() -> None:
    other = 'root = Card([t])\nt = TextContent(":x")'
    assert ast_beq(_SIMPLE, other) is False


def test_canonical_beq_true_for_same_layout() -> None:
    assert canonical_beq(_SIMPLE, _SIMPLE) is True


def test_canonical_beq_false_for_unparseable() -> None:
    assert canonical_beq("not openui {{{", _SIMPLE) is False


def test_fidelity_rates_from_pairs() -> None:
    rates = fidelity_rates_from_pairs(
        [
            (_SIMPLE, _SIMPLE),
            ('root = Card([t])\nt = TextContent(":x")', _SIMPLE),
        ]
    )
    assert rates[AST_BEQ_RATE] == 0.5
    assert rates[CANONICAL_BEQ_RATE] == 0.5
    assert rates["n_pairs"] == 2.0


def test_score_row_flags() -> None:
    row = score_row(_SIMPLE, _SIMPLE)
    assert row["ast_beq"] is True
    assert row["canonical_beq"] is True


def _cert(digest_n: int, *, exhausted: bool = True) -> SupportCertificate:
    query = SupportQuery(
        state_fingerprint=f"{digest_n:064x}",
        hole_id=HoleId(namespace="t", path=("h",), kind="hole"),
        candidate=DomainValue.create("tok", {"i": digest_n}),
    )
    return SupportCertificate(
        schema_version=1,
        query=query,
        verdict=SupportVerdict.UNSUPPORTED if exhausted else SupportVerdict.SUPPORTED,
        problem_id="p",
        pack_id="pack",
        constraint_version="v1",
        bounds=SolverBounds(
            max_tokens=10,
            max_nodes=10,
            max_depth=4,
            max_backtracks=10,
            max_verifier_calls=10,
        ),
        search_order="canonical-domain-value-v1",
        explored_state_fingerprints=(f"{digest_n:064x}",),
        coverage_observations=("complete",),
        verifier_profile="fixture",
        exhausted=exhausted,
        witness_digest=None if exhausted else f"{digest_n + 1:064x}",
        witness_source=None if exhausted else "fixture",
    )


def test_certificate_equivalent_on_digest() -> None:
    a = _cert(1)
    b = SupportCertificate.from_dict(a.to_dict())
    assert certificate_equivalent(a, b) is True
    assert certificate_equivalent(a, _cert(2)) is False


def test_certificate_equivalence_rate() -> None:
    a = _cert(3)
    rate = certificate_equivalence_rate([(a, a), (a, _cert(9))])
    assert rate == 0.5
