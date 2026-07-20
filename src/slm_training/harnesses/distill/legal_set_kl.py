"""Dense legal-set KL distillation objective (SPV2-03).

This is a fixture-only wiring baseline: a deterministic teacher-trace manifest,
a legal-set-masked KL objective, and a tiny fixture trainer. It does not run a
full TwoTower train, does not download an external teacher, and makes no ship
readiness claim. Real teacher scoring is deferred to the SLM-108 external
scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may be absent in minimal environments
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

__all__ = [
    "LegalSetKLConfig",
    "LegalSetDistillExample",
    "legal_set_kl_loss",
    "legal_set_kl_loss_from_examples",
    "legal_set_teacher_distribution",
    "train_legal_set_kl_fixture",
]

_REDUCTIONS = frozenset({"none", "sum", "mean", "batchmean"})


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("legal_set_kl requires torch")


@dataclass
class LegalSetKLConfig:
    """Hyperparameters for the legal-set KL objective."""

    temperature: float = 1.0
    teacher_is_prob: bool = False
    reduction: str = "batchmean"
    kl_weight: float = 1.0
    epsilon: float = 1e-8


@dataclass
class LegalSetDistillExample:
    """One legal-set distillation decision."""

    state_id: str
    legal_action_ids: tuple[int, ...]
    teacher_logits: torch.Tensor | None = None
    teacher_probs: torch.Tensor | None = None
    student_logits: torch.Tensor | None = None
    accepted_action_ids: tuple[int, ...] = ()
    source: str = "fixture"
    coverage: str = "complete"


def _entropy_bits(probs: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Shannon entropy in bits for a 1-D probability vector."""
    if probs.numel() == 0:
        return float("nan")
    p = probs.clamp(min=epsilon)
    return float((-p * p.log2()).sum().detach())


def legal_set_teacher_distribution(
    logits_or_probs: torch.Tensor,
    legal_action_ids: tuple[int, ...],
    temperature: float = 1.0,
    is_prob: bool = False,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return a teacher probability vector indexed by ``legal_action_ids``.

    ``logits_or_probs`` may be either a full-vocabulary 1-D tensor (the usual
    case) or a tensor already sliced to the legal set. If ``is_prob`` is True
    the input is treated as probabilities and renormalized over the legal set.
    A small ``epsilon`` is mixed in and renormalized to avoid zero probabilities.
    """
    _require_torch()
    if logits_or_probs.ndim != 1:
        raise ValueError("teacher logits/probs must be one-dimensional")
    if not legal_action_ids:
        return torch.empty(0, dtype=torch.float32, device=logits_or_probs.device)

    already_sliced = logits_or_probs.size(0) == len(legal_action_ids)
    if already_sliced:
        values = logits_or_probs
    else:
        legal_ids = torch.tensor(
            legal_action_ids, dtype=torch.long, device=logits_or_probs.device
        )
        values = logits_or_probs.index_select(0, legal_ids)

    if is_prob:
        probs = values.clamp(min=epsilon)
    else:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        values = values / float(temperature)
        values = values - values.max()
        probs = values.exp()

    probs = probs / probs.sum()
    # Mix in epsilon and renormalize for stability.
    probs = probs.clamp(min=epsilon)
    return probs / probs.sum()


def legal_set_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits_or_probs: torch.Tensor,
    legal_action_ids: tuple[int, ...],
    *,
    config: LegalSetKLConfig | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute KL(student || teacher) restricted to ``legal_action_ids``.

    Both ``student_logits`` and ``teacher_logits_or_probs`` are assumed to be
    full-vocabulary 1-D tensors. The loss ignores every action outside the legal
    set. Returns the scalar KL (or weighted scalar) and a metrics dict.
    """
    _require_torch()
    cfg = config if config is not None else LegalSetKLConfig(**kwargs)
    if cfg.reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}")
    if student_logits.ndim != 1:
        raise ValueError("student_logits must be one-dimensional")

    device = student_logits.device
    if not legal_action_ids:
        loss = student_logits.new_zeros(())
        return loss, {
            "kl_div": 0.0,
            "legal_entropy": float("nan"),
            "student_entropy": float("nan"),
            "teacher_entropy": float("nan"),
            "legal_set_size": 0,
        }

    legal_ids = torch.tensor(legal_action_ids, dtype=torch.long, device=device)
    student_legal = student_logits.index_select(0, legal_ids)
    student_log_probs = F.log_softmax(student_legal / cfg.temperature, dim=-1)
    student_probs = student_log_probs.exp()

    teacher_probs = legal_set_teacher_distribution(
        teacher_logits_or_probs,
        legal_action_ids,
        temperature=cfg.temperature,
        is_prob=cfg.teacher_is_prob,
        epsilon=cfg.epsilon,
    )
    teacher_probs = teacher_probs.to(device)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=cfg.reduction,
    )
    loss = kl * cfg.kl_weight

    student_entropy = _entropy_bits(student_probs, epsilon=cfg.epsilon)
    teacher_entropy = _entropy_bits(teacher_probs, epsilon=cfg.epsilon)
    metrics = {
        "kl_div": float(kl.detach()),
        "legal_entropy": (student_entropy + teacher_entropy) / 2.0,
        "student_entropy": student_entropy,
        "teacher_entropy": teacher_entropy,
        "legal_set_size": len(legal_action_ids),
    }
    return loss, metrics


def legal_set_kl_loss_from_examples(
    student_logits_full: torch.Tensor,
    examples: list[LegalSetDistillExample],
    config: LegalSetKLConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Batch over examples (each may have a different legal set).

    ``student_logits_full`` has shape ``(len(examples), vocab_size)``. Per-example
    ``student_logits`` on an example overrides the corresponding row. Returns the
    mean loss and aggregated metrics.
    """
    _require_torch()
    if not examples:
        loss = student_logits_full.new_zeros(())
        return loss, {
            "kl_div": 0.0,
            "legal_entropy": float("nan"),
            "student_entropy": float("nan"),
            "teacher_entropy": float("nan"),
            "legal_set_size": 0,
            "n_examples": 0,
        }

    losses: list[torch.Tensor] = []
    metrics_list: list[dict[str, float]] = []
    for i, example in enumerate(examples):
        student_logits = example.student_logits
        if student_logits is None:
            student_logits = student_logits_full[i]
        teacher_logits_or_probs: torch.Tensor | None = example.teacher_logits
        if teacher_logits_or_probs is None:
            teacher_logits_or_probs = example.teacher_probs
        if teacher_logits_or_probs is None:
            raise ValueError(
                f"example {example.state_id!r} has neither teacher_logits "
                "nor teacher_probs"
            )
        loss_i, metrics_i = legal_set_kl_loss(
            student_logits,
            teacher_logits_or_probs,
            example.legal_action_ids,
            config=config,
        )
        losses.append(loss_i)
        metrics_list.append(metrics_i)

    loss = torch.stack(losses).mean()
    aggregated: dict[str, Any] = {
        "kl_div": sum(m["kl_div"] for m in metrics_list) / len(metrics_list),
        "legal_entropy": sum(
            m["legal_entropy"] for m in metrics_list if math.isfinite(m["legal_entropy"])
        )
        / max(1, sum(1 for m in metrics_list if math.isfinite(m["legal_entropy"]))),
        "student_entropy": sum(
            m["student_entropy"]
            for m in metrics_list
            if math.isfinite(m["student_entropy"])
        )
        / max(
            1,
            sum(1 for m in metrics_list if math.isfinite(m["student_entropy"])),
        ),
        "teacher_entropy": sum(
            m["teacher_entropy"]
            for m in metrics_list
            if math.isfinite(m["teacher_entropy"])
        )
        / max(
            1,
            sum(1 for m in metrics_list if math.isfinite(m["teacher_entropy"])),
        ),
        "legal_set_size": sum(m["legal_set_size"] for m in metrics_list),
        "n_examples": len(examples),
    }
    return loss, aggregated


def train_legal_set_kl_fixture(
    student_net: torch.nn.Module,
    teacher_net: torch.nn.Module,
    examples: list[LegalSetDistillExample],
    *,
    config: LegalSetKLConfig | None = None,
    steps: int = 20,
    lr: float = 0.05,
) -> dict[str, Any]:
    """Tiny fixture trainer that updates ``student_net`` to match the teacher.

    Both networks are called with ``torch.arange(len(examples))`` and are
    expected to return a ``(len(examples), vocab_size)`` logits tensor. The
    teacher network is evaluated under ``torch.no_grad()`` and left in eval
    mode. Returns loss history and final metrics.
    """
    _require_torch()
    cfg = config if config is not None else LegalSetKLConfig()
    student_net.train()
    teacher_net.eval()
    optimizer = torch.optim.Adam(student_net.parameters(), lr=lr)
    indices = torch.arange(len(examples), dtype=torch.long)
    history: list[dict[str, Any]] = []

    for step in range(steps):
        optimizer.zero_grad()
        student_logits_full = student_net(indices)
        with torch.no_grad():
            teacher_logits_full = teacher_net(indices)
        # Attach per-example teacher targets so the loss can slice per legal set.
        for i, example in enumerate(examples):
            example.student_logits = student_logits_full[i]
            example.teacher_logits = teacher_logits_full[i]
        loss, metrics = legal_set_kl_loss_from_examples(
            student_logits_full, examples, cfg
        )
        loss.backward()
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach()), **metrics})

    with torch.no_grad():
        final_student_logits = student_net(indices)
        final_teacher_logits = teacher_net(indices)
    for i, example in enumerate(examples):
        example.student_logits = final_student_logits[i]
        example.teacher_logits = final_teacher_logits[i]
    _, final_metrics = legal_set_kl_loss_from_examples(
        final_student_logits, examples, cfg
    )

    return {
        "steps": steps,
        "lr": lr,
        "n_examples": len(examples),
        "history": history,
        "final_metrics": final_metrics,
    }
