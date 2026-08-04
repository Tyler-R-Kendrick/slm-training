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
    "multi_region_support",
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
            "exact_multi_region",
            "honest_overapprox",
            "left_prefix_overapprox",
            "unknown",
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


def multi_region_support(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    mask_id: int | None = None,
    node_budget: int = 5000,
) -> ResidualSupportResult:
    """Exact multi-region completability under decode semantics (bounded).

    Decides whether SOME assignment of one token per hole makes the ENTIRE
    fixed canvas (committed suffixes included) a valid incomplete prefix —
    the question ``admit_fill`` over-approximates by checking only the span
    left of the first hole. Exactness is relative to the system's own
    decode-then-parse semantics: each position's token contributes its
    surface piece (empty-piece tokens act as epsilon fills, which is why a
    comment byte can legalize a same-line suffix but not one behind a
    newline).

    Memoized DFS keyed on ``(position, parser_state_key)``; hole branching is
    restricted to currently-lexable candidates plus one epsilon fill. Budget
    exhaustion returns ``authority="unknown"`` with ``admitted=False`` —
    fail-closed for callers gating commits, and never conflated with a
    proven rejection (``reason`` distinguishes them).
    """
    from slm_training.dsl.grammar.fastpath.token_map import (
        allowed_id_set,
        token_surface_piece,
    )

    ids = [int(t) for t in token_ids]
    effective_mask = (
        int(mask_id)
        if mask_id is not None
        else int(getattr(tokenizer, "mask_id", -1))
    )
    base = OpenUIIncrementalEngine()
    if not base.set_prefix(""):
        raise RuntimeError("empty prefix must sync")

    budget = {"nodes": int(node_budget), "exhausted": False}
    memo: dict[tuple[int, tuple], bool] = {}

    def search(engine: OpenUIIncrementalEngine, pos: int) -> bool:
        if pos == len(ids):
            return True
        key = (pos, engine.parser_state_key())
        hit = memo.get(key)
        if hit is not None:
            return hit
        if budget["nodes"] <= 0:
            budget["exhausted"] = True
            return False
        budget["nodes"] -= 1
        token = ids[pos]
        if token != effective_mask:
            piece = token_surface_piece(tokenizer, token)
            fork = engine.copy()
            ok = fork.advance_checked(piece) if piece else True
            result = bool(ok) and search(fork if piece else engine, pos + 1)
        else:
            # Epsilon fill first (an empty-piece token in the hole).
            result = search(engine, pos + 1)
            if not result:
                terminals = engine.next_terminals()
                candidates = (
                    allowed_id_set(tokenizer, terminals, use_cache=True) or set()
                )
                for fill in sorted(candidates):
                    piece = token_surface_piece(tokenizer, int(fill))
                    if not piece:
                        continue  # epsilon already tried
                    fork = engine.copy()
                    if fork.advance_checked(piece) and search(fork, pos + 1):
                        result = True
                        break
        memo[key] = result
        return result

    admitted = search(base, 0)
    if budget["exhausted"] and not admitted:
        return ResidualSupportResult(
            admitted=False,
            authority="unknown",
            gamma_leaf_filters=True,
            reason="multi_region_budget_exhausted",
        )
    return ResidualSupportResult(
        admitted=bool(admitted),
        authority="exact_multi_region",
        gamma_leaf_filters=True,
        reason=(
            "multi_region_supported" if admitted else "multi_region_unsupported"
        ),
    )
