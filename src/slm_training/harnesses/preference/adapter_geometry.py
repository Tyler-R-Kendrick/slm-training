"""Adapter-subspace objective geometry profiling for TwoTower (LDI2-02 / SLM-125).

Re-asks the E285/E286 "is there a safe common descent direction?" question inside
a small, explicit adapter parameter subspace instead of the full parameter space.
It reuses the merged SLM-123 removable low-rank adapter (frozen parent,
adapter-only gradients) and the SLM-116 legal-set semantics, and differentiates
the protected preference quantities in **grammar-legal probability space**:

* ``loss``      — the local objective loss (minimized);
* ``good_mass`` — legal-space probability mass on good actions (maximized);
* ``bad_mass``  — legal-space probability mass on bad actions (minimized);
* ``margin``    — mean good-minus-bad logit margin (maximized).

Each quantity yields a *descent gradient* (sign-adjusted so that a step along its
negative improves that quantity); the report exposes raw and unit-normalized
gradients, pairwise cosine alignment, and whether a strictly common descent
direction exists. This is a bounded diagnostic: no training, no checkpoint, no
quality claim is produced here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from slm_training.harnesses.preference.local_decisions import DecisionEventV1
from slm_training.harnesses.preference.local_train import _event_logits, local_decision_loss
from slm_training.models.twotower import TwoTowerModel

__all__ = [
    "PROTECTED_QUANTITIES",
    "AdapterGeometryReport",
    "legal_space_quantities",
    "profile_adapter_objective_geometry",
]

PROTECTED_QUANTITIES: tuple[str, ...] = ("loss", "good_mass", "bad_mass", "margin")
# Sign that turns each raw quantity gradient into a *descent* gradient: +1 for
# quantities we minimize (loss, bad_mass), -1 for quantities we maximize.
_DESCENT_SIGN: dict[str, float] = {
    "loss": 1.0,
    "good_mass": -1.0,
    "bad_mass": 1.0,
    "margin": -1.0,
}


def _positions(action_ids: Sequence[int], legal: Sequence[int]) -> list[int]:
    index = {action: position for position, action in enumerate(legal)}
    return [index[a] for a in action_ids]


def legal_space_quantities(
    logits: torch.Tensor,
    event: DecisionEventV1,
    *,
    objective: str = "ftpo_single",
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The four protected quantities as differentiable scalars (legal-space)."""
    legal = tuple(int(a) for a in event.legal_token_ids)
    good = tuple(int(a) for a in event.good_token_ids)
    bad = tuple(int(a) for a in event.bad_token_ids)
    legal_set = set(legal)
    if not legal_set.issuperset(good) or not legal_set.issuperset(bad):
        raise ValueError("good/bad token ids must be inside the event's legal set")

    legal_ids = torch.tensor(legal, dtype=torch.long, device=logits.device)
    legal_probs = F.softmax(logits.index_select(0, legal_ids), dim=-1)
    good_pos = torch.tensor(_positions(good, legal), dtype=torch.long, device=logits.device)
    bad_pos = torch.tensor(_positions(bad, legal), dtype=torch.long, device=logits.device)
    good_mass = legal_probs.index_select(0, good_pos).sum()
    bad_mass = legal_probs.index_select(0, bad_pos).sum()

    good_logits = logits.index_select(0, torch.tensor(good, dtype=torch.long, device=logits.device))
    bad_logits = logits.index_select(0, torch.tensor(bad, dtype=torch.long, device=logits.device))
    margin = good_logits.mean() - bad_logits.mean()

    loss, _ = local_decision_loss(logits, event, objective=objective, epsilon=epsilon, tau=tau)
    return {"loss": loss, "good_mass": good_mass, "bad_mass": bad_mass, "margin": margin}


@dataclass(frozen=True)
class AdapterGeometryReport:
    """Adapter-subspace geometry of the protected preference quantities."""

    parameter_dim: int
    descent_gradients: dict[str, torch.Tensor]
    gradient_norms: dict[str, float]
    cosine_alignment: dict[str, float]
    common_descent: bool
    unit_normalized_norms: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_dim": self.parameter_dim,
            "gradient_norms": self.gradient_norms,
            "cosine_alignment": self.cosine_alignment,
            "common_descent": self.common_descent,
            "unit_normalized_norms": self.unit_normalized_norms,
        }


def _flat_grad(quantity: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(quantity, params, retain_graph=True, allow_unused=True)
    parts = [
        g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(grads, params)
    ]
    return torch.cat(parts)


def profile_adapter_objective_geometry(
    model: TwoTowerModel,
    event: DecisionEventV1,
    *,
    objective: str = "ftpo_single",
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> AdapterGeometryReport:
    """Profile the adapter-subspace descent geometry for one exact decision state.

    Requires an attached adapter (frozen parent). Parent parameters are never
    differentiated — only the adapter tensors — so the diagnostic isolates the
    bounded subspace rather than full-parameter geometry.
    """
    if not model.has_adapter():
        raise ValueError("profiling requires an attached adapter (frozen parent)")
    params = [p for p in model.adapter_parameters() if p.requires_grad]
    if not params:
        raise ValueError("the attached adapter exposes no trainable parameters")

    logits = _event_logits(model, event)
    quantities = legal_space_quantities(
        logits, event, objective=objective, epsilon=epsilon, tau=tau
    )
    descent: dict[str, torch.Tensor] = {}
    for name in PROTECTED_QUANTITIES:
        descent[name] = _DESCENT_SIGN[name] * _flat_grad(quantities[name], params)

    norms = {name: float(vector.norm()) for name, vector in descent.items()}
    units = {
        name: (vector / vector.norm()) if norms[name] > 1e-12 else vector
        for name, vector in descent.items()
    }
    unit_norms = {name: float(vector.norm()) for name, vector in units.items()}

    # Pairwise cosine alignment among descent directions; a strictly common
    # descent direction requires every pair to be non-antagonistic.
    cosine: dict[str, float] = {}
    min_pair = 1.0
    names = list(PROTECTED_QUANTITIES)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if norms[left] <= 1e-12 or norms[right] <= 1e-12:
                value = 0.0
            else:
                value = float(torch.dot(units[left], units[right]))
            cosine[f"{left}|{right}"] = value
            min_pair = min(min_pair, value)

    return AdapterGeometryReport(
        parameter_dim=int(sum(p.numel() for p in params)),
        descent_gradients=descent,
        gradient_norms=norms,
        cosine_alignment=cosine,
        common_descent=min_pair > 0.0,
        unit_normalized_norms=unit_norms,
    )
