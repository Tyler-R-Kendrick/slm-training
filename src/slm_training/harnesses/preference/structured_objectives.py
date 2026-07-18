"""LDI3-01 structured local-preference objectives (SLM-128).

An architecture-neutral objective library shared by the causal and TwoTower local
trainers. It adds three OpenUI-native structured objectives *adapted* (not
reproduced) from recent token-level preference work:

* ``legal_set_ftpo`` — Legal-Set FTPO with a ``pairwise`` set-margin variant and a
  ``mass`` legal-probability-mass-margin variant (the primary constrained-space
  set objective).
* ``tab_barrier`` — a TAB-PO-inspired confidence-gated anchor that lifts
  under-confident, verified-critical good actions, metered alongside a
  likelihood-erosion signal.
* ``tbpo_inspired`` — a TokenRatio/TBPO-inspired state-normalized action-ratio
  control comparing good/bad log-ratios against a reference at the *same* state.

Every objective consumes a materialized :class:`ObjectiveView` plus the exact
legal set, operates on 1-D decision logits, and returns ``(loss, metrics)`` with
the loss differentiable and the metrics detached. Ambiguous and unobserved legal
actions are kept in legal normalization and reported separately (never mislabeled
as good/bad targets), the barrier honors a per-action semantic role weight, and any
objective optionally composes with the target / non-target MSE locality tethers,
each metered independently. Names carry ``_inspired`` where they adapt rather than
reproduce a paper. No model update or quality claim is made here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from slm_training.harnesses.preference.decision_events_v2 import ObjectiveView

__all__ = [
    "StructuredObjectiveConfig",
    "StructuredObjectiveError",
    "structured_decision_loss",
    "SUPPORTED_STRUCTURED_OBJECTIVES",
]

SUPPORTED_STRUCTURED_OBJECTIVES: tuple[str, ...] = (
    "legal_set_ftpo",
    "tab_barrier",
    "tbpo_inspired",
)
_FTPO_VARIANTS: tuple[str, ...] = ("pairwise", "mass")


class StructuredObjectiveError(ValueError):
    """Raised for an invalid config or a violated objective shape/support contract."""


@dataclass(frozen=True)
class StructuredObjectiveConfig:
    """Typed, versioned, fail-closed config. Its fingerprint is part of run and
    adapter/checkpoint identity."""

    name: Literal["legal_set_ftpo", "tab_barrier", "tbpo_inspired"]
    variant: str = "pairwise"  # legal_set_ftpo: "pairwise" | "mass"
    epsilon: float = 2.0
    tau: float = 1.0
    temperature: float = 1.0
    margin: float = 0.0
    normalize_per_state: bool = True
    evidence_clip: float = 10.0
    barrier_p: float = 0.1
    barrier_strength: float = 1.0
    critical_roles: tuple[str, ...] = ()
    default_role_weight: float = 1.0
    non_target_tether: float = 0.0
    target_tether: float = 0.0
    target_grace: float = 1.0
    state_baseline: Literal["none", "advantage"] = "none"
    numeric_eps: float = 1e-8
    version: int = 1

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_STRUCTURED_OBJECTIVES:
            raise StructuredObjectiveError(f"unknown structured objective {self.name!r}")
        if self.name == "legal_set_ftpo" and self.variant not in _FTPO_VARIANTS:
            raise StructuredObjectiveError(
                f"legal_set_ftpo variant must be one of {_FTPO_VARIANTS}, got {self.variant!r}"
            )
        for key in ("epsilon", "tau", "temperature", "numeric_eps"):
            if getattr(self, key) <= 0:
                raise StructuredObjectiveError(f"{key} must be positive")
        for key in (
            "barrier_strength",
            "default_role_weight",
            "evidence_clip",
            "non_target_tether",
            "target_tether",
            "target_grace",
        ):
            if getattr(self, key) < 0:
                raise StructuredObjectiveError(f"{key} must be non-negative")
        if not 0.0 < self.barrier_p < 1.0:
            raise StructuredObjectiveError("barrier_p must be in (0, 1)")
        if not all(
            math.isfinite(float(getattr(self, k)))
            for k in ("epsilon", "tau", "temperature", "margin", "numeric_eps")
        ):
            raise StructuredObjectiveError("numeric hyperparameters must be finite")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> StructuredObjectiveConfig:
        """Build from a mapping, failing closed on unknown fields."""
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise StructuredObjectiveError(f"unknown config fields (fail closed): {sorted(unknown)}")
        payload = dict(data)
        if "critical_roles" in payload:
            payload["critical_roles"] = tuple(payload["critical_roles"])
        return cls(**payload)


@dataclass(frozen=True)
class _Prepared:
    good: tuple[int, ...]
    bad: tuple[int, ...]
    legal: tuple[int, ...]
    ambiguous: tuple[int, ...]
    unobserved: tuple[int, ...]
    good_logits: torch.Tensor
    bad_logits: torch.Tensor
    good_weights: torch.Tensor
    legal_probs: torch.Tensor
    good_legal_mass: torch.Tensor
    bad_legal_mass: torch.Tensor
    ambiguous_legal_mass: torch.Tensor
    unobserved_legal_mass: torch.Tensor
    good_legal_prob: torch.Tensor  # per-good legal-space probability


def _prepare(
    logits: torch.Tensor,
    view: ObjectiveView,
    legal_action_ids: Sequence[int],
    config: StructuredObjectiveConfig,
) -> _Prepared:
    if logits.ndim != 1:
        raise StructuredObjectiveError("decision logits must be one-dimensional")
    if not view.trainable:
        raise StructuredObjectiveError(
            "structured objectives refuse a non-trainable (constraint-shadow) view"
        )
    legal = tuple(int(a) for a in legal_action_ids)
    if not legal:
        raise StructuredObjectiveError("legal_action_ids must be non-empty")
    legal_set = set(legal)
    if len(legal_set) != len(legal):
        raise StructuredObjectiveError("legal_action_ids must be unique")
    good = tuple(int(a) for a in view.good_action_ids)
    bad = tuple(int(a) for a in view.bad_action_ids)
    ambiguous = tuple(int(a) for a in view.ambiguous_action_ids)
    unobserved = tuple(int(a) for a in view.unobserved_action_ids)
    if not good:
        raise StructuredObjectiveError("a trainable objective requires at least one good action")
    for label, ids in (
        ("good", good), ("bad", bad), ("ambiguous", ambiguous), ("unobserved", unobserved)
    ):
        if not legal_set.issuperset(ids):
            raise StructuredObjectiveError(f"{label} action ids must be inside the legal set")
    # Ambiguous/unobserved actions stay part of legal normalization but must never be
    # mislabeled as a good/bad target: the four partitions are disjoint.
    pooled = good + bad + ambiguous + unobserved
    if len(pooled) != len(set(pooled)):
        raise StructuredObjectiveError(
            "good/bad/ambiguous/unobserved action sets must be disjoint"
        )

    def idx(values: Sequence[int]) -> torch.Tensor:
        return torch.tensor(tuple(values), dtype=torch.long, device=logits.device)

    weight_map = {int(a): float(w) for a, w in view.weights}
    clip = config.evidence_clip if config.evidence_clip > 0 else float("inf")
    good_weights = logits.new_tensor(
        [min(weight_map.get(a, 1.0), clip) for a in good]
    )
    good_logits = logits.index_select(0, idx(good))
    bad_logits = (
        logits.index_select(0, idx(bad)) if bad else logits.new_zeros(0)
    )
    legal_logits = logits.index_select(0, idx(legal))
    legal_probs = F.softmax(legal_logits / config.temperature, dim=-1)
    legal_pos = {a: p for p, a in enumerate(legal)}

    def _mass(ids: Sequence[int]) -> torch.Tensor:
        if not ids:
            return legal_probs.new_zeros(())
        return legal_probs.index_select(0, idx([legal_pos[a] for a in ids])).sum()

    good_legal_prob = legal_probs.index_select(0, idx([legal_pos[a] for a in good]))
    good_legal_mass = good_legal_prob.sum()
    bad_legal_mass = _mass(bad)
    return _Prepared(
        good=good,
        bad=bad,
        legal=legal,
        ambiguous=ambiguous,
        unobserved=unobserved,
        good_logits=good_logits,
        bad_logits=bad_logits,
        good_weights=good_weights,
        legal_probs=legal_probs,
        good_legal_mass=good_legal_mass,
        bad_legal_mass=bad_legal_mass,
        ambiguous_legal_mass=_mass(ambiguous),
        unobserved_legal_mass=_mass(unobserved),
        good_legal_prob=good_legal_prob,
    )


def _base_metrics(p: _Prepared) -> dict[str, float]:
    return {
        "good_legal_mass": float(p.good_legal_mass.detach()),
        "bad_legal_mass": float(p.bad_legal_mass.detach()),
        "ambiguous_legal_mass": float(p.ambiguous_legal_mass.detach()),
        "unobserved_legal_mass": float(p.unobserved_legal_mass.detach()),
        "legal_entropy": float(
            -(p.legal_probs * (p.legal_probs + 1e-12).log()).sum().detach()
        ),
        "num_good": float(len(p.good)),
        "num_bad": float(len(p.bad)),
        "num_ambiguous": float(len(p.ambiguous)),
        "num_unobserved": float(len(p.unobserved)),
    }


def _legal_set_ftpo(
    p: _Prepared, config: StructuredObjectiveConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    eps, tau = config.epsilon, config.tau
    if config.variant == "mass":
        loss = F.softplus(
            config.margin
            + (p.bad_legal_mass + config.numeric_eps).log()
            - (p.good_legal_mass + config.numeric_eps).log()
        )
        metrics = _base_metrics(p)
        metrics["objective_loss"] = float(loss.detach())
        return loss, metrics
    # pairwise set margin over verified G x B pairs, weighted and per-state-normalized
    if not p.bad:
        raise StructuredObjectiveError("legal_set_ftpo pairwise requires at least one bad action")
    delta = p.good_logits.unsqueeze(1) - p.bad_logits.unsqueeze(0)  # [G, B]
    pair_w = p.good_weights.unsqueeze(1).expand_as(delta)
    hinge_w = torch.clamp((eps - delta) / eps, min=0.0, max=1.0)
    per_pair = pair_w * hinge_w * F.softplus((eps - delta) / tau)
    if config.normalize_per_state:
        loss = per_pair.mean()  # mean over pairs so large sets do not dominate
    else:
        loss = per_pair.sum()
    metrics = _base_metrics(p)
    metrics["objective_loss"] = float(loss.detach())
    metrics["mean_margin"] = float(delta.mean().detach())
    metrics["active_pair_fraction"] = float((hinge_w > 0).float().mean().detach())
    return loss, metrics


def _tab_barrier(
    p: _Prepared,
    config: StructuredObjectiveConfig,
    *,
    critical_good_mask: torch.Tensor | None,
    good_role_weights: torch.Tensor | None = None,
    reference_good_prob: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Base preference term (reuse the pairwise set margin) so the barrier is an
    # additive, separately-metered component — never a replacement.
    base, metrics = _legal_set_ftpo(p, StructuredObjectiveConfig(name="legal_set_ftpo",
        variant="pairwise", epsilon=config.epsilon, tau=config.tau,
        temperature=config.temperature, normalize_per_state=config.normalize_per_state,
        evidence_clip=config.evidence_clip)) if p.bad else (p.good_logits.new_zeros(()), _base_metrics(p))

    role_w = (
        critical_good_mask.to(p.good_legal_prob.dtype)
        if critical_good_mask is not None
        else p.good_legal_prob.new_ones(len(p.good))
    )
    # Semantic role weight per good action: structural tokens can be down-weighted by the
    # caller (default ``default_role_weight``), so the barrier does not over-anchor
    # low-criticality structural punctuation.
    role_scale = (
        good_role_weights.to(p.good_legal_prob.dtype)
        if good_role_weights is not None
        else p.good_legal_prob.new_full((len(p.good),), config.default_role_weight)
    )
    # Anchor only under-confident (legal prob < barrier_p) critical good actions.
    under = (p.good_legal_prob < config.barrier_p).to(p.good_legal_prob.dtype)
    anchor_w = config.barrier_strength * role_w * role_scale * under * p.good_weights
    barrier = (anchor_w * -(p.good_legal_prob + config.numeric_eps).log()).sum()
    loss = base + barrier

    metrics["objective_loss"] = float(loss.detach())
    metrics["barrier_loss"] = float(barrier.detach())
    metrics["barrier_active_fraction"] = float(under.mean().detach())
    metrics["mean_role_weight"] = float(role_scale.mean().detach())
    if reference_good_prob is not None:
        # Likelihood-erosion: verified good actions whose absolute prob dropped.
        eroded = (p.good_legal_prob < reference_good_prob).to(p.good_legal_prob.dtype)
        metrics["erosion_rate"] = float(eroded.mean().detach())
    return loss, metrics


def _tbpo_inspired(
    p: _Prepared,
    config: StructuredObjectiveConfig,
    *,
    reference_legal_probs: torch.Tensor | None,
    state_baseline: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if reference_legal_probs is None:
        raise StructuredObjectiveError("tbpo_inspired requires reference logits at the same state")
    if not p.bad:
        raise StructuredObjectiveError("tbpo_inspired requires at least one bad action")
    if len(p.legal) < 2:
        raise StructuredObjectiveError("tbpo_inspired disabled: inadequate legal-state support")
    legal_pos = {a: i for i, a in enumerate(p.legal)}
    g_pos = torch.tensor([legal_pos[a] for a in p.good], dtype=torch.long, device=p.legal_probs.device)
    b_pos = torch.tensor([legal_pos[a] for a in p.bad], dtype=torch.long, device=p.legal_probs.device)
    eps = config.numeric_eps
    log_ratio = (p.legal_probs + eps).log() - (reference_legal_probs + eps).log()
    if config.state_baseline == "advantage":
        log_ratio = log_ratio - log_ratio.mean() - state_baseline  # advantage-centered
    good_r = log_ratio.index_select(0, g_pos).mean()
    bad_r = log_ratio.index_select(0, b_pos).mean()
    loss = F.softplus(config.margin - (good_r - bad_r))
    metrics = _base_metrics(p)
    metrics["objective_loss"] = float(loss.detach())
    metrics["good_ratio"] = float(good_r.detach())
    metrics["bad_ratio"] = float(bad_r.detach())
    return loss, metrics


def _locality_tether(
    logits: torch.Tensor,
    reference_logits: torch.Tensor | None,
    target_ids: Sequence[int],
    config: StructuredObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Target / non-target MSE locality tethers against the reference, separately metered.

    Holds logits near the reference off-target (``non_target_tether``) and bounds on-target
    drift beyond ``target_grace`` (``target_tether``), so a structured objective composes
    with locality while the barrier and the target tether stay independently metered.
    """
    zero = logits.new_zeros(())
    if config.non_target_tether <= 0 and config.target_tether <= 0:
        return zero, {"non_target_logit_mse": 0.0, "target_excess_logit_mse": 0.0}
    if reference_logits is None or reference_logits.shape != logits.shape:
        raise StructuredObjectiveError("locality tethers require matching reference logits")
    diff = logits - reference_logits.detach()
    target_mask = torch.zeros_like(logits, dtype=torch.bool)
    target_mask[torch.tensor(tuple(target_ids), dtype=torch.long, device=logits.device)] = True
    non_target_mse = zero
    target_excess_mse = zero
    if config.non_target_tether > 0 and (~target_mask).any():
        non_target_mse = diff[~target_mask].pow(2).mean()
    if config.target_tether > 0 and target_mask.any():
        excess = (diff[target_mask].abs() - config.target_grace).clamp(min=0.0)
        target_excess_mse = excess.pow(2).mean()
    loss = config.non_target_tether * non_target_mse + config.target_tether * target_excess_mse
    return loss, {
        "non_target_logit_mse": float(non_target_mse.detach()),
        "target_excess_logit_mse": float(target_excess_mse.detach()),
    }


def structured_decision_loss(
    logits: torch.Tensor,
    view: ObjectiveView,
    *,
    legal_action_ids: Sequence[int],
    config: StructuredObjectiveConfig,
    reference_logits: torch.Tensor | None = None,
    critical_good_mask: torch.Tensor | None = None,
    good_role_weights: torch.Tensor | None = None,
    state_baseline: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dispatch one structured objective. Architecture-neutral: the same call
    serves causal and TwoTower logits. Returns ``(loss, detached metrics)``.

    Every objective optionally composes with the target / non-target MSE locality
    tethers (``config.non_target_tether`` / ``target_tether``), metered separately from the
    objective and any barrier so locality never double-counts the target anchor.
    """
    p = _prepare(logits, view, legal_action_ids, config)
    if config.name == "legal_set_ftpo":
        loss, metrics = _legal_set_ftpo(p, config)
    elif config.name == "tab_barrier":
        ref_good_prob = None
        if reference_logits is not None:
            ref_p = _prepare(reference_logits, view, legal_action_ids, config)
            ref_good_prob = ref_p.good_legal_prob.detach()
        loss, metrics = _tab_barrier(
            p,
            config,
            critical_good_mask=critical_good_mask,
            good_role_weights=good_role_weights,
            reference_good_prob=ref_good_prob,
        )
    else:  # tbpo_inspired
        ref_legal = None
        if reference_logits is not None:
            ref_legal = _prepare(reference_logits, view, legal_action_ids, config).legal_probs.detach()
        loss, metrics = _tbpo_inspired(
            p, config, reference_legal_probs=ref_legal, state_baseline=state_baseline
        )
    if config.non_target_tether > 0 or config.target_tether > 0:
        tether_loss, tether_metrics = _locality_tether(
            logits, reference_logits, sorted(set(p.good) | set(p.bad)), config
        )
        loss = loss + tether_loss
        metrics.update(tether_metrics)
    metrics["config_fingerprint_present"] = 1.0
    return loss, metrics
