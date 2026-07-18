"""Causal exact-state local-preference training loop (LDI1-02).

Ties the objective core, balancing, and a PEFT-enabled causal policy into one
bounded training pass that updates only adapter parameters. The policy is abstract
(:class:`CausalPolicy`) so the loop is exercised here against a torch-only mock;
the real Transformers/PEFT policy plugs in behind the same protocol without the
loop changing.

Reference logits for the locality tether come from the *same* policy with the
adapter disabled — never a re-encoded surface and never a separately loaded model
(unless one is explicitly configured upstream). Checkpoint selection is driven by
held-out event loss, and the best adapter state is restored before returning, so a
chosen/margin win alone can never promote a drifting adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch

from slm_training.harnesses.preference.causal_balancing import (
    BalanceStratum,
    CausalTrainingItem,
    balance_items,
)
from slm_training.harnesses.preference.causal_local_train import (
    CausalLocalObjective,
    causal_decision_loss,
)

__all__ = ["CausalPolicy", "train_causal_local", "evaluate_items"]


@runtime_checkable
class CausalPolicy(Protocol):
    """Minimal surface the training loop needs from a causal PEFT policy."""

    def forward_logits(self, prefix_ids: Sequence[int]) -> torch.Tensor:
        """Next-token logits (1-D) at the final position of ``prefix_ids``."""

    def set_adapter_enabled(self, enabled: bool) -> None:
        """Enable/disable the adapter delta (disabled == frozen base reference)."""

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """The adapter parameters — the *only* tensors this loop updates."""


def _prefix_ids(item: CausalTrainingItem) -> tuple[int, ...]:
    ids = item.state.context_ids
    if ids is None:
        raise ValueError(
            "causal training requires stored context_ids (the exact supervised prefix)"
        )
    return tuple(ids)


def evaluate_items(
    policy: CausalPolicy,
    items: Sequence[CausalTrainingItem],
    *,
    objective: CausalLocalObjective,
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> dict[str, float]:
    """Mean preference loss/metrics over ``items`` with the adapter enabled."""
    if not items:
        return {"loss": 0.0, "preference_loss": 0.0, "count": 0.0}
    policy.set_adapter_enabled(True)
    total = 0.0
    good_mass = 0.0
    with torch.no_grad():
        for item in items:
            logits = policy.forward_logits(_prefix_ids(item))
            _, metrics = causal_decision_loss(
                logits,
                item.view,
                legal_action_ids=item.state.legal_action_ids,
                objective=objective,
                epsilon=epsilon,
                tau=tau,
            )
            total += metrics["preference_loss"]
            good_mass += metrics["good_legal_mass"]
    count = len(items)
    return {
        "loss": total / count,
        "preference_loss": total / count,
        "good_legal_mass": good_mass / count,
        "count": float(count),
    }


def _snapshot(params: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _restore(params: Sequence[torch.nn.Parameter], state: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, saved in zip(params, state):
            param.copy_(saved)


def train_causal_local(
    items: Sequence[CausalTrainingItem],
    policy: CausalPolicy,
    *,
    objective: CausalLocalObjective,
    strata: Sequence[BalanceStratum],
    seed: int,
    max_epochs: int = 8,
    lr: float = 0.1,
    patience: int = 2,
    per_stratum: int | None = None,
    held_out: Sequence[CausalTrainingItem] = (),
    epsilon: float = 2.0,
    tau: float = 1.0,
    non_target_tether: float = 0.0,
    target_tether: float = 0.0,
    target_grace: float = 1.0,
) -> dict[str, Any]:
    """Run a bounded adapter-only training pass; restore the best held-out state.

    Returns a summary with the balancing report, pre/post held-out metrics,
    trainable-parameter count, and the stopping reason. No quality claim is made —
    the caller decides whether the pre/post delta justifies keeping the adapter.
    """
    balanced, balancing = balance_items(
        items, strata=strata, seed=seed, per_stratum=per_stratum
    )
    if not balanced:
        raise ValueError("no trainable items after balancing")
    tether_on = non_target_tether > 0 or target_tether > 0

    params = policy.trainable_parameters()
    if not params:
        raise ValueError("policy exposes no trainable adapter parameters")
    optimizer = torch.optim.AdamW(params, lr=lr)

    selection = list(held_out) if held_out else list(balanced)
    pre = evaluate_items(policy, selection, objective=objective, epsilon=epsilon, tau=tau)
    best = pre["loss"]
    best_state = _snapshot(params)
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    completed_epochs = 0

    for epoch in range(max_epochs):
        completed_epochs = epoch + 1
        for item in balanced:
            optimizer.zero_grad()
            policy.set_adapter_enabled(True)
            logits = policy.forward_logits(_prefix_ids(item))
            reference = None
            if tether_on:
                policy.set_adapter_enabled(False)
                with torch.no_grad():
                    reference = policy.forward_logits(_prefix_ids(item)).detach()
                policy.set_adapter_enabled(True)
            loss, _ = causal_decision_loss(
                logits,
                item.view,
                legal_action_ids=item.state.legal_action_ids,
                objective=objective,
                epsilon=epsilon,
                tau=tau,
                reference_logits=reference,
                non_target_tether=non_target_tether,
                target_tether=target_tether,
                target_grace=target_grace,
            )
            loss.backward()
            optimizer.step()

        current = evaluate_items(
            policy, selection, objective=objective, epsilon=epsilon, tau=tau
        )
        if current["loss"] < best - 1e-9:
            best = current["loss"]
            best_state = _snapshot(params)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stop_reason = "held_out_no_improvement"
                break

    _restore(params, best_state)
    post = evaluate_items(policy, selection, objective=objective, epsilon=epsilon, tau=tau)
    return {
        "objective": objective,
        "balancing": balancing,
        "pre": pre,
        "post": post,
        "best_held_out_loss": best,
        "epochs": completed_epochs,
        "stop_reason": stop_reason,
        "trainable_parameters": int(sum(p.numel() for p in params)),
        "selection_size": len(selection),
        "claim": "wiring only; no quality claim",
    }
