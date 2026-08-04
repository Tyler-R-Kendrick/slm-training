"""Residual multi-hole support predicate (claim family B / stack step 4).

Implements the Mündler-style *decision problem* — multi-hole completability as
Boolean support under residual forest constraints — **not** a semiring neural
circuit.

Two independent over-approximation axes are labeled honestly:

- **Γ axis**: when Γ is applied only as external leaf filters, the static CFG
  residual may be a superset of Γ-tight legality (``honest_overapprox``).
- **Suffix axis**: the underlying probe (``admit_fill``) validates only the
  contiguous committed span *left of the first hole*. Committed tokens after a
  hole are never checked, so for any canvas that still contains holes the
  result is a left-prefix over-approximation (``left_prefix_overapprox``):
  left-prefix-completable does NOT imply the joint fixed canvas is completable
  (counterexample: grammar ``S -> a b`` with canvas ``a [HOLE] c`` — the
  prefix ``a`` is admissible but no fill preserves the fixed suffix ``c``).
  ``authority="exact"`` is therefore reserved for hole-free canvases with
  Γ-internal filtering, where the full completion check actually runs.

Never softens legality: reject is reject; recovery rewrite of the prefix is not
offered. See ``docs/design/adr-constrained-diffusion-topology-split.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.maskgit_constrain import admit_fill

__all__ = [
    "ResidualSupportResult",
    "joint_multi_hole_support",
    "rank_inside_legal_residual",
]


@dataclass(frozen=True)
class ResidualSupportResult:
    """Boolean residual support with an honest authority label.

    ``authority`` is the OVERALL claim strength, weakest axis wins:
    ``left_prefix_overapprox`` (canvas has holes — suffix never checked) ⊂
    ``honest_overapprox`` (hole-free but Γ external) ⊂ ``exact``.
    """

    admitted: bool
    authority: str  # "exact" | "honest_overapprox" | "left_prefix_overapprox"
    gamma_leaf_filters: bool
    reason: str
    soft_legality: bool = False  # always False for production paths

    def __post_init__(self) -> None:
        if self.soft_legality:
            raise ValueError("soft legality is forbidden under I6")
        if self.authority not in {
            "exact",
            "honest_overapprox",
            "left_prefix_overapprox",
        }:
            raise ValueError(f"unknown residual authority: {self.authority}")


def joint_multi_hole_support(
    engine: OpenUIIncrementalEngine,
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    mask_id: int | None = None,
    gamma_leaf_filters: bool = True,
) -> ResidualSupportResult:
    """Boolean multi-hole completability on the residual left-span.

    Uses the existing MaskGIT hole-admissibility probe (CFG residual /
    InteractiveParser prefix). Two honesty caveats:

    - When ``gamma_leaf_filters`` is True the static residual is an honest
      over-approximation of the fully Γ-tight domain — callers must not treat
      True as a certified Γ-exact ship claim without a separate leaf-filter
      pass.
    - When the canvas still contains holes, ONLY the committed span left of
      the first hole is validated; committed tokens after a hole are ignored
      by the probe, so ``admitted=True`` never certifies that the joint fixed
      canvas is completable. Such results carry
      ``authority="left_prefix_overapprox"`` and can never be ``"exact"``.
    """

    ids = [int(t) for t in token_ids]
    admitted = bool(admit_fill(engine, tokenizer, ids, mask_id=mask_id))
    effective_mask = (
        int(mask_id)
        if mask_id is not None
        else int(getattr(tokenizer, "mask_id", -1))
    )
    has_holes = effective_mask in ids
    if has_holes:
        authority = "left_prefix_overapprox"
        reason = (
            "left_prefix_admissible" if admitted else "left_prefix_inadmissible"
        )
    else:
        authority = "honest_overapprox" if gamma_leaf_filters else "exact"
        reason = (
            "residual_prefix_admissible"
            if admitted
            else "residual_prefix_inadmissible"
        )
    return ResidualSupportResult(
        admitted=admitted,
        authority=authority,
        gamma_leaf_filters=bool(gamma_leaf_filters),
        reason=reason,
        soft_legality=False,
    )


def rank_inside_legal_residual(
    scores: Sequence[float],
    legal_mask: Sequence[bool],
) -> tuple[int | None, tuple[float, ...]]:
    """Reweight / argmax only inside the legal residual (stack steps 3–5).

    Illegal indices receive ``-inf`` and can never win. Empty legal set returns
    ``(None, masked_scores)`` — never falls back to full-vocabulary ranking.
    """

    if len(scores) != len(legal_mask):
        raise ValueError("scores and legal_mask length mismatch")
    masked: list[float] = []
    best_i: int | None = None
    best_s = float("-inf")
    for i, (score, legal) in enumerate(zip(scores, legal_mask, strict=True)):
        if not legal:
            masked.append(float("-inf"))
            continue
        value = float(score)
        masked.append(value)
        if value > best_s:
            best_s = value
            best_i = i
    return best_i, tuple(masked)
