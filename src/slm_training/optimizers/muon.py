"""Faithful Muon/AdamW hybrid optimizer for the model-build harness.

Muon reference: Keller Jordan, "Muon: An optimizer for hidden layers in neural
networks", 2024. The core update orthogonalizes the momentum buffer with a
Newton–Schulz iteration so that dense 2-D matrix updates have orthogonal rows
(or columns for wide matrices). Embeddings, tied/untied output heads, norms,
biases, scalars, and auxiliary heads stay on AdamW.

This module is a self-contained PyTorch implementation with no external
optimizer dependencies.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim import Optimizer


__all__ = ["MuonHybrid", "build_muon_hybrid", "newton_schulz_orthogonalize"]


def newton_schulz_orthogonalize(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return an orthonormalized version of 2-D matrix ``G``.

    Uses the Newton–Schulz iteration for the polar factor. For a tall matrix
    (rows >= cols) we drive ``X.T @ X -> I``; for a wide matrix we drive
    ``X @ X.T -> I``. The input is normalized by its Frobenius norm so the
    iteration starts in the basin of convergence.
    """
    if G.ndim != 2:
        raise ValueError("newton_schulz_orthogonalize expects a 2-D matrix")
    X = G / (G.norm("fro") + eps)
    rows, cols = X.shape
    if rows >= cols:
        # Tall / square: orthonormalize columns.
        eye = torch.eye(cols, device=X.device, dtype=X.dtype)
        for _ in range(steps):
            X = X @ (1.5 * eye - 0.5 * (X.T @ X))
    else:
        # Wide: orthonormalize rows.
        eye = torch.eye(rows, device=X.device, dtype=X.dtype)
        for _ in range(steps):
            X = (1.5 * eye - 0.5 * (X @ X.T)) @ X
    return X


class MuonHybrid(Optimizer):
    """Muon for eligible 2-D matrices, AdamW for everything else.

    Parameters are split at construction time into two groups:

    * ``muon`` parameters receive the orthogonalized-momentum update;
    * ``adamw`` parameters receive a standard AdamW update.

    The split is controlled by shape and optional name prefixes. By default
    only dense 2-D matrices that are not token embeddings, output heads,
    normalization layers, biases, or auxiliary heads are eligible for Muon.
    """

    def __init__(
        self,
        named_params: Iterable[tuple[str, torch.nn.Parameter]]
        | Iterable[torch.nn.Parameter],
        *,
        lr: float = 3e-4,
        muon_lr: float | None = None,
        adamw_lr: float | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        muon_momentum: float = 0.9,
        muon_nesterov: bool = False,
        muon_ns_steps: int = 5,
        deny_prefixes: tuple[str, ...] = (
            "length_head.",
            "component_inventory_head.",
            "component_plan_head.",
            "slot_component_head.",
            "component_edge_head.",
            "binder_component_plan_head.",
            "binder_topology_head.",
            "binder_arity_head.",
            "root_reference_arity_head.",
            "root_reference_identity_head.",
            "trust_gate.",
            "survival_head.",
        ),
        matrix_min_dim: int = 2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if muon_ns_steps < 1:
            raise ValueError("muon_ns_steps must be positive")

        embedding_suffixes = (
            ".tok.weight",
            ".lm_head.weight",
            ".pos.weight",
            ".emb.weight",
            ".embed.weight",
            ".embedding.weight",
        )
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for item in named_params:
            if isinstance(item, tuple) and len(item) == 2:
                name, p = item
            else:
                name, p = "", item
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if not p.requires_grad:
                continue
            is_eligible_matrix = (
                p.ndim == 2
                and min(p.shape) >= matrix_min_dim
                and not any(str(name).endswith(suffix) for suffix in embedding_suffixes)
                and ".norm." not in name
                and not str(name).endswith("norm.weight")
                and ".bias" not in name
                and not any(str(name).startswith(prefix) for prefix in deny_prefixes)
            )
            if is_eligible_matrix:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        defaults_muon = {
            "lr": muon_lr if muon_lr is not None else lr,
            "momentum": muon_momentum,
            "nesterov": muon_nesterov,
            "ns_steps": muon_ns_steps,
            "weight_decay": 0.0,
            "optimizer": "muon",
        }
        defaults_adamw = {
            "lr": adamw_lr if adamw_lr is not None else lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "optimizer": "adamw",
        }
        param_groups: list[dict[str, Any]] = []
        if muon_params:
            param_groups.append({"params": muon_params, **defaults_muon})
        if adamw_params:
            param_groups.append({"params": adamw_params, **defaults_adamw})
        if not param_groups:
            raise ValueError("MuonHybrid received no trainable parameters")
        super().__init__(param_groups, {"lr": lr})
        self._fingerprint = {
            "optimizer": "muon_hybrid",
            "muon_ns_steps": muon_ns_steps,
            "muon_nesterov": muon_nesterov,
        }

    @property
    def fingerprint(self) -> dict[str, Any]:
        return dict(self._fingerprint)

    def _partition_groups(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        muon_groups = [g for g in self.param_groups if g.get("optimizer") == "muon"]
        adamw_groups = [g for g in self.param_groups if g.get("optimizer") == "adamw"]
        return muon_groups, adamw_groups

    @torch.no_grad()
    def step(self, closure: Any = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups, adamw_groups = self._partition_groups()

        # Muon update: orthogonalized momentum.
        for group in muon_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                state["step"] += 1
                buf.mul_(momentum).add_(grad)
                direction = grad.add(buf, alpha=momentum) if nesterov else buf
                update = newton_schulz_orthogonalize(direction, steps=ns_steps)
                p.add_(update, alpha=-lr)

        # AdamW update for non-matrix parameters.
        beta1, beta2 = None, None
        for group in adamw_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                # AdamW decoupled weight decay.
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def build_muon_hybrid(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float = 3e-4,
    muon_lr: float | None = None,
    adamw_lr: float | None = None,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    muon_momentum: float = 0.9,
    muon_nesterov: bool = False,
    muon_ns_steps: int = 5,
) -> MuonHybrid:
    """Build a Muon/AdamW hybrid optimizer from named model parameters."""
    return MuonHybrid(
        named_parameters,
        lr=lr,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        muon_ns_steps=muon_ns_steps,
    )
