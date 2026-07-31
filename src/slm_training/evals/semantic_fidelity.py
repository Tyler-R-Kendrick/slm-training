"""Semantic-fidelity BEq analogues for AST/grammar pairs and certificates.

These checks are stricter than soft structural similarity: they are Boolean
equality predicates (Lean ``BEq`` analogues) averaged into rates for ship
gates. Syntax parse alone never implies semantic fidelity.

Predicates
----------
``ast_beq``
    Structure-normalized exact match after validate/serialize (style-stripped).
    Near parity to equality on the grammar-facing AST surface.
``canonical_beq``
    :func:`canonical_equal` — full D2 canonical form equality.
``certificate_equivalent``
    Content digests of two support certificates (or formal objects) agree.

Rates are means of 0/1 over measured rows. Undefined (empty) denominators
return ``None`` so missing evidence fails closed at the gate, never as a
vacuous 1.0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from slm_training.dsl.canonicalize import canonical_equal

SEMANTIC_FIDELITY_SCHEMA = "semantic_fidelity/v1"

# Ship-gated rate fields (also projected in ship_gates._slim_suite).
AST_BEQ_RATE = "ast_beq_rate"
CANONICAL_BEQ_RATE = "canonical_beq_rate"
CERTIFICATE_EQUIVALENCE_RATE = "certificate_equivalence_rate"

GATED_FIDELITY_METRICS: tuple[str, ...] = (
    AST_BEQ_RATE,
    CANONICAL_BEQ_RATE,
)


def ast_beq(pred: str, gold: str) -> bool:
    """Boolean equality on structure-normalized OpenUI programs.

    Mirrors :func:`slm_training.harnesses.model_build.eval_runner._tree_match`
    but as a pure Bool (Lean BEq style) for fidelity gates.
    """

    from slm_training.data.structure import strip_style_literals
    from slm_training.dsl.parser import ParseError, validate

    pred_s = strip_style_literals(pred).strip()
    gold_s = strip_style_literals(gold).strip()
    if pred_s == gold_s:
        return True
    try:
        pred_p = validate(pred_s)
    except ParseError:
        return False
    try:
        gold_p = validate(gold_s)
    except ParseError:
        # Gold-side parse failure is harness breakage; treat as non-equal here
        # and let the caller record errors separately when needed.
        return False
    if pred_p.serialized and gold_p.serialized:
        ps = strip_style_literals(pred_p.serialized).strip()
        gs = strip_style_literals(gold_p.serialized).strip()
        return ps == gs
    return False


def canonical_beq(pred: str, gold: str, *, dsl: str | None = None) -> bool:
    """Boolean equality on D2-canonical forms (full semantic layout BEq)."""

    return canonical_equal(pred, gold, dsl=dsl)


def certificate_equivalent(
    left: Mapping[str, Any] | Any,
    right: Mapping[str, Any] | Any,
) -> bool:
    """True when two certificates / formal objects have the same content digest.

    Accepts:
    * objects with a ``digest`` property (SupportCertificate);
    * formal objects with ``content_digest``;
    * raw dicts with ``certificate_digest`` / ``content_digest`` / ``digest``.
    """

    def _digest(value: Any) -> str | None:
        if value is None:
            return None
        if hasattr(value, "digest") and not isinstance(value, Mapping):
            try:
                return str(value.digest)
            except Exception:  # noqa: BLE001
                return None
        if isinstance(value, Mapping):
            for key in ("content_digest", "certificate_digest", "digest"):
                raw = value.get(key)
                if isinstance(raw, str) and len(raw) == 64:
                    return raw
            # Nested certificate payload from formal_object export.
            cert = value.get("certificate")
            if isinstance(cert, Mapping):
                try:
                    from slm_training.dsl.solver.support import SupportCertificate

                    return SupportCertificate.from_dict(dict(cert)).digest
                except Exception:  # noqa: BLE001
                    return None
        return None

    a = _digest(left)
    b = _digest(right)
    if a is None or b is None:
        return False
    return a == b


def _mean_bools(flags: Sequence[bool]) -> float | None:
    if not flags:
        return None
    return sum(1.0 if flag else 0.0 for flag in flags) / len(flags)


def fidelity_rates_from_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    dsl: str | None = None,
) -> dict[str, float | None]:
    """Compute AST / canonical BEq rates over (pred, gold) pairs."""

    ast_flags: list[bool] = []
    can_flags: list[bool] = []
    for pred, gold in pairs:
        ast_flags.append(ast_beq(pred, gold))
        can_flags.append(canonical_beq(pred, gold, dsl=dsl))
    return {
        AST_BEQ_RATE: _mean_bools(ast_flags),
        CANONICAL_BEQ_RATE: _mean_bools(can_flags),
        "n_pairs": float(len(ast_flags)),
    }


def certificate_equivalence_rate(
    pairs: Iterable[tuple[Any, Any]],
) -> float | None:
    """Mean certificate_equivalent over pairs; ``None`` when no pairs."""

    flags = [certificate_equivalent(a, b) for a, b in pairs]
    return _mean_bools(flags)


def score_row(
    pred: str,
    gold: str,
    *,
    dsl: str | None = None,
    pred_certificate: Any | None = None,
    gold_certificate: Any | None = None,
) -> dict[str, Any]:
    """Per-row semantic-fidelity evidence for scoreboard details."""

    row: dict[str, Any] = {
        "ast_beq": ast_beq(pred, gold),
        "canonical_beq": canonical_beq(pred, gold, dsl=dsl),
    }
    if pred_certificate is not None or gold_certificate is not None:
        row["certificate_equivalent"] = certificate_equivalent(
            pred_certificate, gold_certificate
        )
    return row


__all__ = [
    "AST_BEQ_RATE",
    "CANONICAL_BEQ_RATE",
    "CERTIFICATE_EQUIVALENCE_RATE",
    "GATED_FIDELITY_METRICS",
    "SEMANTIC_FIDELITY_SCHEMA",
    "ast_beq",
    "canonical_beq",
    "certificate_equivalence_rate",
    "certificate_equivalent",
    "fidelity_rates_from_pairs",
    "score_row",
]
