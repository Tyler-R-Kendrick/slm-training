"""AbstractPlanConnector: a default-off, gated-additive logit bias that
conditions MaskGIT/tree denoising on the discrete abstract plan (AP-024 /
SLM-316).

Consumes an :class:`~slm_training.models.abstract_plan_head.AbstractPlanTrace`
(AP-023 / SLM-315) and, when enabled, biases every round's vocabulary logits
by a learned projection of the plan's own (codebook-local) predicted slot
sequence. This mirrors the repository's existing conditioning idiom --
``DenoiserTower._runtime_symbol_features``
and ``TwoTowerModel._component_inventory_bias``/``_component_plan_bias`` -- a
post-hoc additive term on logits, not a change to the transformer stack
itself, so:

* it composes with the existing per-round MaskGIT/tree denoising loop with
  **zero changes to that loop's scheduling logic** -- the bias is applied
  once inside :meth:`DenoiserTower.project`, which every round already calls;
* "off" costs zero parameters and zero forward compute (the connector is not
  constructed at all, matching ``AbstractPlanHead``'s own convention);
* per-round, per-position "did this change the argmax token" instrumentation
  is nearly free (two argmax calls on already-computed logits), matching the
  existing ``*_applications``/``*_choice_changes`` counter idiom -- this is
  explicitly a correlational signal, not a causal claim (see
  ``CausalLatentUseFalsificationSpecV1`` in ``models/causal_trace.py``).

One connector, one code path, seven arms (:class:`PlanConnectorArm`) select
what vector is fed to that same path: ``oracle``/``learned`` (the actual
predicted plan, gradient-connected), ``detached`` (same content, gradient
blocked), ``empty``/``random``/``shuffled`` (content-null controls at matched
architecture), and ``disabled`` (the connector does not exist). Every arm
resolves through :func:`resolve_plan_vector` so control arms cannot silently
diverge in scheduling, seeding, or prompt handling from the ``learned`` arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from slm_training.models.abstract_plan_head import AbstractPlanTrace

__all__ = [
    "AbstractPlanConnector",
    "PlanConnectorArm",
    "PlanConnectorTrace",
    "resolve_plan_vector",
]


class PlanConnectorArm(str, Enum):
    """Named conditioning arms sharing one connector code path."""

    DISABLED = "disabled"
    LEARNED = "learned"
    ORACLE = "oracle"
    DETACHED = "detached"
    EMPTY = "empty"
    RANDOM = "random"
    SHUFFLED = "shuffled"


# Arms whose vector is the trace's own content, undisturbed.
_PASSTHROUGH = (PlanConnectorArm.LEARNED, PlanConnectorArm.ORACLE)


def resolve_plan_vector(
    vector: torch.Tensor,
    *,
    arm: PlanConnectorArm,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Resolve the connector's input vector for one arm from the same call site.

    ``vector`` supplies the shape/device/dtype template for every arm
    (including the content-null ones), so every arm pays the identical
    plan-head forward cost and only the connector's *content* differs.
    """
    if arm is PlanConnectorArm.DISABLED:
        raise ValueError(
            "resolve_plan_vector called with arm=disabled; the connector "
            "must not be constructed or invoked at all in that case"
        )
    if generator is not None and torch.device(generator.device) != vector.device:
        raise ValueError(
            f"generator device {generator.device!r} does not match plan "
            f"vector device {vector.device!r}; construct the generator on "
            "the same device"
        )
    if arm in _PASSTHROUGH:
        return vector
    if arm is PlanConnectorArm.DETACHED:
        return vector.detach()
    if arm is PlanConnectorArm.EMPTY:
        return torch.zeros_like(vector)
    if arm is PlanConnectorArm.RANDOM:
        return torch.randn(
            vector.shape, generator=generator, device=vector.device, dtype=vector.dtype
        )
    # SHUFFLED: permute the batch dimension so each example conditions on a
    # different example's plan content, breaking the example<->plan
    # correspondence while preserving the marginal content distribution.
    batch_size = vector.size(0)
    if batch_size < 2:
        return vector.clone()
    perm = torch.randperm(batch_size, generator=generator, device=vector.device)
    return vector.index_select(0, perm)


@dataclass(frozen=True)
class PlanConnectorTrace:
    """One :meth:`DenoiserTower.project` call's worth of connector evidence.

    ``choice_changed_count`` / ``position_count`` are a correlational
    "did the argmax flip" readout, not a causal-use claim -- see
    ``CausalLatentUseFalsificationSpecV1`` in ``models/causal_trace.py``.
    """

    arm: str
    gate_value: float
    bias_norm: float
    choice_changed_count: int
    position_count: int
    applications: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "gate_value": self.gate_value,
            "bias_norm": self.bias_norm,
            "choice_changed_count": self.choice_changed_count,
            "position_count": self.position_count,
            "applications": self.applications,
        }


class AbstractPlanConnector(nn.Module):
    """Gated-additive vocabulary-logit bias from a pooled plan vector.

    Owns its own small ``plan_slot_count``-row embedding table for the
    plan's *codebook-local* slot indices (``AbstractPlanTrace.plan_tokens``,
    values in ``[0, plan.slot_count)``) rather than reusing the host model's
    real token embeddings: ``AbstractPlanV1``'s *absolute* reserved
    vocabulary ids (``.token_ids()``) are reserved in the causal-LM track's
    tokenizer namespace (AP-016/017), not in an arbitrary TwoTower
    tokenizer's, so indexing the host embedding table by them is unsafe in
    general. A dedicated table sized exactly to the codebook is always valid
    and adds only ``plan_slot_count * d_model`` parameters.

    The gate is a single learnable scalar initialized at ``0.0``
    (``tanh(0) == 0``), so an enabled-but-freshly-initialized connector
    starts as a numerical no-op -- the "off" *guarantee* still comes from
    never constructing/invoking the connector at all, but this keeps a
    warm-started model's early training identical in effect to disabled.
    """

    def __init__(self, d_model: int, plan_slot_count: int, vocab_size: int) -> None:
        super().__init__()
        self.plan_embed = nn.Embedding(plan_slot_count, d_model)
        self.gate = nn.Parameter(torch.zeros(1))
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def plan_vector_from_trace(self, trace: AbstractPlanTrace) -> torch.Tensor:
        """Pool this connector's own embeddings of the trace's codebook-local slot ids."""
        embeds = self.plan_embed(trace.plan_tokens)  # [batch, rounds, d_model]
        return embeds.mean(dim=1)  # [batch, d_model]

    def bias_for_vocab(
        self,
        plan_vector: torch.Tensor,
        *,
        candidate_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Additive logit bias, gathered to ``candidate_ids`` when given."""
        full = torch.tanh(self.gate) * self.proj(plan_vector)  # [batch, vocab]
        if candidate_ids is None:
            return full
        return full.index_select(1, candidate_ids)
