"""AbstractPlanHead: a lightweight, default-off discrete plan head over
TwoTower context-tower states (AP-023 / SLM-315).

Predicts a short sequence of :class:`~slm_training.dsl.abstract_plan.AbstractPlanV1`
codebook indices from the pooled context-tower hidden state. This is a pure
side channel: nothing in :mod:`slm_training.models.twotower` calls it from
``training_loss``/``forward`` (the denoiser's inputs are untouched), so no
plan signal reaches the decoder unless a caller explicitly invokes
``TwoTowerModel.abstract_plan_trace(...)``, matching AP-016's existing
default-off convention (``abstract_plan_slots=0``) for the reserved codebook
itself. Wiring an actual connector into the decoder is AP-027's job, not this
issue's -- see ``TwoTowerConfig.semantic_connector``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.dsl.abstract_plan import AbstractPlanV1


class AbstractPlanMode(str, enum.Enum):
    """Named plan-generation modes. ``DISABLED`` costs zero parameters and
    compute -- callers must not construct :class:`AbstractPlanHead` at all
    for it, mirroring every other optional aux head in ``twotower.py``
    (e.g. ``binder_arity_head``, gated ``None`` unless its loss weight > 0).
    """

    DISABLED = "disabled"
    TEACHER_FORCED = "teacher_forced"
    SAMPLED = "sampled"
    ORACLE = "oracle"
    RANDOM = "random"
    SHUFFLED = "shuffled"


# Modes that require a caller-supplied gold/teacher plan.
_REQUIRES_TARGET = (AbstractPlanMode.TEACHER_FORCED, AbstractPlanMode.ORACLE)
# Modes that run the learned projection (and therefore report logits/entropy).
_RUNS_PROJECTION = (
    AbstractPlanMode.TEACHER_FORCED,
    AbstractPlanMode.SAMPLED,
    AbstractPlanMode.SHUFFLED,
)


@dataclass
class AbstractPlanTrace:
    """One forward's worth of plan prediction.

    Distinct from :class:`AbstractPlanV1`, which only fixes the static
    shape/token contract (codebook size, delimiters, version) -- this carries
    the actual per-example prediction and its full provenance.
    """

    plan: AbstractPlanV1
    mode: str
    plan_tokens: torch.Tensor  # [batch, plan.rounds] long, in [0, plan.slot_count)
    hidden_states: torch.Tensor  # [batch, d_model] pooled context vector consumed
    logits: torch.Tensor | None  # [batch, plan.rounds, plan.slot_count] or None
    entropy: torch.Tensor | None  # [batch, plan.rounds] nats, or None
    provenance: dict[str, object] = field(default_factory=dict)

    def token_ids(self) -> torch.Tensor:
        """Map codebook-local indices to absolute reserved vocabulary ids."""
        slot_ids = torch.tensor(
            self.plan.slot_token_ids,
            dtype=torch.long,
            device=self.plan_tokens.device,
        )
        return slot_ids[self.plan_tokens]


class AbstractPlanHead(nn.Module):
    """Predicts an :class:`AbstractPlanTrace` from pooled context states."""

    def __init__(self, d_model: int, plan: AbstractPlanV1) -> None:
        super().__init__()
        self.plan = plan
        self.proj = nn.Linear(d_model, plan.rounds * plan.slot_count)

    def forward(
        self,
        context: torch.Tensor,
        context_pad_mask: torch.Tensor | None,
        *,
        mode: str | AbstractPlanMode,
        target_plan_ids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> AbstractPlanTrace:
        mode = AbstractPlanMode(mode)
        if mode is AbstractPlanMode.DISABLED:
            raise ValueError(
                "AbstractPlanHead.forward called with mode='disabled'; "
                "construct the head only when a non-disabled mode is configured"
            )
        if mode in _REQUIRES_TARGET and target_plan_ids is None:
            raise ValueError(f"mode={mode.value!r} requires target_plan_ids")

        batch_size = context.size(0)
        device = context.device
        hidden_states = _masked_mean_pool(context, context_pad_mask)

        logits: torch.Tensor | None = None
        entropy: torch.Tensor | None = None
        if mode in _RUNS_PROJECTION:
            logits = self.proj(hidden_states).view(
                batch_size, self.plan.rounds, self.plan.slot_count
            )
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        if mode is AbstractPlanMode.ORACLE:
            plan_tokens = target_plan_ids
        elif mode is AbstractPlanMode.TEACHER_FORCED:
            plan_tokens = target_plan_ids
        elif mode is AbstractPlanMode.RANDOM:
            plan_tokens = torch.randint(
                0,
                self.plan.slot_count,
                (batch_size, self.plan.rounds),
                device=device,
                generator=generator,
            )
        elif mode is AbstractPlanMode.SAMPLED:
            plan_tokens = _sample_categorical(probs, generator)
        else:  # AbstractPlanMode.SHUFFLED
            sampled = _sample_categorical(probs, generator)
            plan_tokens = _shuffle_rounds(sampled, generator)

        return AbstractPlanTrace(
            plan=self.plan,
            mode=mode.value,
            plan_tokens=plan_tokens,
            hidden_states=hidden_states,
            logits=logits,
            entropy=entropy,
            provenance={
                "mode": mode.value,
                "compatibility_fingerprint": self.plan.compatibility_fingerprint,
                "used_target_plan_ids": mode in _REQUIRES_TARGET,
            },
        )


def _sample_categorical(
    probs: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    batch_size, rounds, slot_count = probs.shape
    flat = probs.reshape(-1, slot_count)
    drawn = torch.multinomial(flat, num_samples=1, generator=generator)
    return drawn.view(batch_size, rounds)


def _shuffle_rounds(
    plan_tokens: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    batch_size, rounds = plan_tokens.shape
    device = plan_tokens.device
    perm = torch.stack(
        [
            torch.randperm(rounds, device=device, generator=generator)
            for _ in range(batch_size)
        ]
    )
    return torch.gather(plan_tokens, 1, perm)


def _masked_mean_pool(
    context: torch.Tensor, pad_mask: torch.Tensor | None
) -> torch.Tensor:
    """Mean-pool over the sequence dim; ``pad_mask`` True marks a padded key
    (the same polarity ``TwoTowerModel._encode_context`` already returns).
    """
    if pad_mask is None:
        return context.mean(dim=1)
    keep = (~pad_mask.bool()).unsqueeze(-1).to(context.dtype)
    denom = keep.sum(dim=1).clamp_min(1.0)
    return (context * keep).sum(dim=1) / denom
