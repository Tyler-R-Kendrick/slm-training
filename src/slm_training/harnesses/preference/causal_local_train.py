"""Causal exact-state preference objectives over DecisionEventV2 action tables.

LDI1-02. This is the *causal* counterpart to the TwoTower ``local_decision_loss``
in :mod:`slm_training.harnesses.preference.local_train`, which is kept intact as
the historical whole-program preference surrogate. The causal path differs in two
ways that matter:

* it consumes an admitted :class:`~slm_training.harnesses.preference.decision_events_v2.ObjectiveView`
  (the materialized good/bad/ambiguous/unobserved action partitions of one exact
  ``DecisionStateV2``) rather than a V1 event's frozen ``good_token_ids`` /
  ``bad_token_ids``; and
* it adds the OpenUI-native ``legal_set_mass`` objective, which normalizes the
  probability simplex over ``legal_action_ids`` only, so preference pressure is
  measured inside the legal space the compiler already guarantees — not over the
  full vocabulary.

Only the objective/loss math lives here. Model forwarding, reference-logit parity
(adapter-disable), balancing, and checkpoint selection are layered on top by the
training loop and are intentionally out of this module so the objectives stay
pure and unit-testable on toy logits with no model dependency.

The clipped-FTPO weighting and the reference-locality tether mirror the exact
formulation independently reproduced in ``local_train.local_decision_loss`` (the
Antislop/FTPO lineage, https://arxiv.org/abs/2510.15061). They are re-expressed
here rather than imported so a refactor of the V1 TwoTower path can never silently
change causal training behavior.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from slm_training.harnesses.preference.decision_events_v2 import ObjectiveView

__all__ = ["CausalLocalObjective", "causal_decision_loss"]

CausalLocalObjective = Literal[
    "unlikelihood", "ftpo_single", "ftpo_set", "legal_set_mass"
]
_SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "unlikelihood",
    "ftpo_single",
    "ftpo_set",
    "legal_set_mass",
)


def _index_tensor(values: Sequence[int], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.long, device=like.device)


def _view_weights(view: ObjectiveView) -> dict[int, float]:
    """Per-good-action weights from the materialized view (default 1.0)."""
    return {int(action): float(weight) for action, weight in view.weights}


def causal_decision_loss(
    logits: torch.Tensor,
    view: ObjectiveView,
    *,
    legal_action_ids: Sequence[int],
    objective: CausalLocalObjective,
    epsilon: float = 2.0,
    tau: float = 1.0,
    evidence_confidence: float = 1.0,
    reference_logits: torch.Tensor | None = None,
    non_target_tether: float = 0.0,
    target_tether: float = 0.0,
    target_grace: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return one causal decision loss and detached decision/locality telemetry.

    ``logits`` are the full-vocabulary next-token logits at the exact supervised
    prefix position (1-D). ``view`` is a *trainable* materialized objective view;
    ``legal_action_ids`` is the state's legal set, used both to validate the view
    and to normalize the ``legal_set_mass`` objective and legal-space metrics.

    Raises before touching any gradient when the view is a non-trainable
    (constraint-shadow) view, when good/bad partitions fall outside the legal set,
    or when an objective's shape contract is violated. Legality is not a semantic
    label (the E284 lesson), so a legality-only view is refused here rather than
    silently trained.
    """
    if logits.ndim != 1:
        raise ValueError("decision logits must be one-dimensional")
    if not all(
        math.isfinite(value)
        for value in (
            epsilon,
            tau,
            target_grace,
            evidence_confidence,
            non_target_tether,
            target_tether,
        )
    ):
        raise ValueError("loss hyperparameters must be finite (NaN bypasses comparisons)")
    if tau <= 0 or epsilon <= 0 or target_grace < 0:
        raise ValueError("epsilon/tau must be positive and target_grace non-negative")
    if evidence_confidence < 0 or non_target_tether < 0 or target_tether < 0:
        raise ValueError("confidence and tether weights must be non-negative")
    if not view.trainable:
        raise ValueError(
            "semantic causal training refuses a non-trainable objective view "
            "(constraint-shadow / legality-only evidence is not a preference label)"
        )

    legal = tuple(int(a) for a in legal_action_ids)
    if not legal:
        raise ValueError("legal_action_ids must be non-empty")
    legal_set = set(legal)
    if len(legal_set) != len(legal):
        raise ValueError("legal_action_ids must be unique")
    good = tuple(int(a) for a in view.good_action_ids)
    bad = tuple(int(a) for a in view.bad_action_ids)
    if not legal_set.issuperset(good) or not legal_set.issuperset(bad):
        raise ValueError("good/bad action ids must be inside the legal set")
    if objective not in _SUPPORTED_OBJECTIVES:
        raise ValueError(f"unknown causal objective {objective!r}")
    if not good:
        raise ValueError("a trainable objective requires at least one good action")
    if objective in ("unlikelihood", "ftpo_single", "ftpo_set") and not bad:
        raise ValueError(f"{objective} requires at least one bad action")
    if objective == "ftpo_single" and (len(good) != 1 or len(bad) != 1):
        raise ValueError("ftpo_single requires exactly one good and one bad action")

    good_ids = _index_tensor(good, logits)
    bad_ids = _index_tensor(bad, logits) if bad else _index_tensor((), logits)
    legal_ids = _index_tensor(legal, logits)
    good_logits = logits.index_select(0, good_ids)
    weight_map = _view_weights(view)
    good_weights = logits.new_tensor([weight_map.get(a, 1.0) for a in good])

    # Full-vocabulary and legal-space probability views (metrics report both).
    full_probs = F.softmax(logits, dim=-1)
    legal_probs = F.softmax(logits.index_select(0, legal_ids), dim=-1)
    legal_pos = {action: position for position, action in enumerate(legal)}
    good_legal_pos = _index_tensor([legal_pos[a] for a in good], logits)
    bad_legal_pos = (
        _index_tensor([legal_pos[a] for a in bad], logits)
        if bad
        else _index_tensor((), logits)
    )
    good_legal_mass = legal_probs.index_select(0, good_legal_pos).sum()
    bad_legal_mass = (
        legal_probs.index_select(0, bad_legal_pos).sum()
        if bad
        else legal_probs.new_zeros(())
    )

    if objective == "unlikelihood":
        # Negative control: drive the full-vocabulary bad mass toward zero.
        bad_mass = full_probs.index_select(0, bad_ids).sum() if bad else full_probs.new_zeros(())
        pref_loss = -torch.log1p(-bad_mass.clamp(max=1.0 - 1e-7))
        active_weight = logits.new_tensor(1.0)
        deltas = good_logits[:, None] - (
            logits.index_select(0, bad_ids)[None, :] if bad else good_logits.new_zeros((1, 1))
        )
    elif objective == "legal_set_mass":
        # Move probability mass from the bad set toward the good set *inside the
        # legal simplex*: reward good legal mass, penalize bad legal mass.
        pref_loss = -torch.log(good_legal_mass.clamp(min=1e-7))
        if bad:
            pref_loss = pref_loss - torch.log1p(-bad_legal_mass.clamp(max=1.0 - 1e-7))
        pref_loss = pref_loss * float(evidence_confidence)
        active_weight = good_legal_mass.detach()
        bad_logits = logits.index_select(0, bad_ids) if bad else good_logits.new_zeros((0,))
        deltas = good_logits[:, None] - bad_logits[None, :] if bad else good_logits[:, None]
    else:  # ftpo_single / ftpo_set — clipped FTPO on good x bad margins.
        bad_logits = logits.index_select(0, bad_ids)
        deltas = good_logits[:, None] - bad_logits[None, :]
        clip = ((epsilon - deltas) / epsilon).clamp(0.0, 1.0)
        weighted = clip * good_weights[:, None]
        pref_loss = (
            weighted * F.softplus((epsilon - deltas) / tau)
        ).mean() * float(evidence_confidence)
        active_weight = clip.mean()

    # Reference-locality tether — protect vocabulary geometry outside the decision.
    target_ids = torch.unique(
        torch.cat((good_ids, bad_ids)) if bad else good_ids
    )
    non_target_mse = logits.new_zeros(())
    target_excess_mse = logits.new_zeros(())
    if non_target_tether > 0 or target_tether > 0:
        if reference_logits is None or reference_logits.shape != logits.shape:
            raise ValueError("matching reference logits are required for tethering")
        diff = logits - reference_logits.detach()
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask[target_ids] = True
        if non_target_tether > 0 and (~target_mask).any():
            non_target_mse = diff[~target_mask].pow(2).mean()
        if target_tether > 0:
            excess = (diff[target_mask].abs() - target_grace).clamp(min=0.0)
            target_excess_mse = excess.pow(2).mean()

    loss = (
        pref_loss
        + non_target_tether * non_target_mse
        + target_tether * target_excess_mse
    )

    metrics = {
        "loss": float(loss.detach()),
        "preference_loss": float(pref_loss.detach()),
        "objective": objective,
        "chosen_win": float((deltas > 0).float().mean().detach()) if deltas.numel() else 0.0,
        "margin_win": float((deltas >= epsilon).float().mean().detach()) if deltas.numel() else 0.0,
        "mean_margin": float(deltas.mean().detach()) if deltas.numel() else 0.0,
        "active_weight": float(active_weight.detach()),
        "good_probability_mass": float(full_probs.index_select(0, good_ids).sum().detach()),
        "bad_probability_mass": float(
            full_probs.index_select(0, bad_ids).sum().detach()
        )
        if bad
        else 0.0,
        "good_legal_mass": float(good_legal_mass.detach()),
        "bad_legal_mass": float(bad_legal_mass.detach()),
        "non_target_logit_mse": float(non_target_mse.detach()),
        "target_excess_logit_mse": float(target_excess_mse.detach()),
    }
    return loss, metrics
