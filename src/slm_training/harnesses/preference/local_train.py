"""Exact-state preference objectives for TwoTower decision events."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from slm_training.harnesses.distill.trace_store import checkpoint_sha
from slm_training.harnesses.preference.local_decisions import (
    DecisionEventV1,
    load_decision_events,
)
from slm_training.models.twotower import TwoTowerModel


LocalObjective = Literal["ce_margin", "unlikelihood", "ftpo_single", "ftpo_set"]


def _indices(values: tuple[int, ...], logits: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=logits.device)


def local_decision_loss(
    logits: torch.Tensor,
    event: DecisionEventV1,
    *,
    objective: LocalObjective,
    epsilon: float = 2.0,
    tau: float = 1.0,
    reference_logits: torch.Tensor | None = None,
    non_target_tether: float = 0.0,
    target_tether: float = 0.0,
    target_grace: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return one event loss and detached decision/locality telemetry."""
    if logits.ndim != 1:
        raise ValueError("decision logits must be one-dimensional")
    if tau <= 0 or epsilon <= 0 or target_grace < 0:
        raise ValueError("epsilon/tau must be positive and target_grace non-negative")
    if non_target_tether < 0 or target_tether < 0:
        raise ValueError("tether weights must be non-negative")
    good_ids = _indices(event.good_token_ids, logits)
    bad_ids = _indices(event.bad_token_ids, logits)
    good = logits.index_select(0, good_ids)
    bad = logits.index_select(0, bad_ids)
    deltas = good[:, None] - bad[None, :]

    if objective == "ce_margin":
        if good.numel() != 1:
            raise ValueError("ce_margin requires exactly one good token")
        candidates = torch.cat((good, bad))
        pref_loss = F.cross_entropy(
            candidates.unsqueeze(0),
            torch.zeros(1, dtype=torch.long, device=logits.device),
        ) + F.relu(logits.new_tensor(epsilon) - deltas.min())
        active_weight = logits.new_tensor(1.0)
    elif objective == "unlikelihood":
        bad_mass = F.softmax(logits, dim=-1).index_select(0, bad_ids).sum()
        pref_loss = -torch.log1p(-bad_mass.clamp(max=1.0 - 1e-7))
        active_weight = logits.new_tensor(1.0)
    else:
        if objective == "ftpo_single" and (good.numel() != 1 or bad.numel() != 1):
            raise ValueError("ftpo_single requires one good and one bad token")
        weights = ((epsilon - deltas) / epsilon).clamp(0.0, 1.0)
        pref_loss = (
            weights * F.softplus((epsilon - deltas) / tau)
        ).mean() * event.evidence_confidence
        active_weight = weights.mean()

    target_ids = torch.unique(torch.cat((good_ids, bad_ids)))
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
    probs = F.softmax(logits, dim=-1)
    metrics = {
        "loss": float(loss.detach()),
        "preference_loss": float(pref_loss.detach()),
        "chosen_win": float((deltas > 0).float().mean().detach()),
        "margin_win": float((deltas >= epsilon).float().mean().detach()),
        "mean_margin": float(deltas.mean().detach()),
        "median_margin": float(deltas.flatten().median().detach()),
        "active_weight": float(active_weight.detach()),
        "good_probability_mass": float(probs.index_select(0, good_ids).sum().detach()),
        "bad_probability_mass": float(probs.index_select(0, bad_ids).sum().detach()),
        "non_target_logit_mse": float(non_target_mse.detach()),
        "target_excess_logit_mse": float(target_excess_mse.detach()),
    }
    return loss, metrics


def event_schedule(
    events: list[DecisionEventV1], *, steps: int, seed: int, balanced: bool
) -> list[DecisionEventV1]:
    if not events:
        raise ValueError("no training decision events")
    if not balanced:
        return [events[index % len(events)] for index in range(steps)]
    groups: dict[
        tuple[str | None, str, tuple[int, ...]], list[DecisionEventV1]
    ] = defaultdict(list)
    for event in events:
        groups[(event.source_suite, event.decision_kind, event.bad_token_ids)].append(
            event
        )
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups, key=str)
    offsets = defaultdict(int)
    schedule: list[DecisionEventV1] = []
    for step in range(steps):
        key = keys[step % len(keys)]
        group = groups[key]
        schedule.append(group[offsets[key] % len(group)])
        offsets[key] += 1
    return schedule


def _event_logits(model: TwoTowerModel, event: DecisionEventV1) -> torch.Tensor:
    ctx, ctx_pad = model._encode_context([event.context_text])
    ids = torch.tensor([event.canvas_ids], dtype=torch.long, device=model.device_name)
    logits = model.denoiser(
        ids, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
    )
    return logits[0, event.position]


def train_local_decisions(
    model: TwoTowerModel,
    events: list[DecisionEventV1],
    *,
    objective: LocalObjective,
    reference_model: TwoTowerModel | None = None,
    steps: int = 50,
    lr: float = 5e-5,
    epsilon: float = 2.0,
    tau: float = 1.0,
    non_target_tether: float = 0.0,
    target_tether: float = 0.0,
    target_grace: float = 1.0,
    balanced: bool = False,
    seed: int = 0,
) -> dict:
    if steps <= 0 or lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    all_train_events = [event for event in events if event.split == "train"]
    if objective == "ce_margin":
        train_events = [
            event for event in all_train_events if len(event.good_token_ids) == 1
        ]
    elif objective == "ftpo_single":
        train_events = [
            event
            for event in all_train_events
            if len(event.good_token_ids) == 1 and len(event.bad_token_ids) == 1
        ]
    else:
        train_events = all_train_events
    if objective == "ftpo_set" and not any(
        len(event.good_token_ids) > 1 or len(event.bad_token_ids) > 1
        for event in train_events
    ):
        raise ValueError("ftpo_set requires a verified set-valued training event")
    if (non_target_tether > 0 or target_tether > 0) and reference_model is None:
        raise ValueError("reference_model is required for tethered training")
    schedule = event_schedule(
        train_events, steps=max(0, int(steps)), seed=int(seed), balanced=balanced
    )
    if reference_model is not None:
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    totals: dict[str, float] = defaultdict(float)
    by_kind: dict[str, int] = defaultdict(int)
    for event in schedule:
        logits = _event_logits(model, event)
        with torch.no_grad():
            reference_logits = (
                _event_logits(reference_model, event)
                if reference_model is not None
                else None
            )
        loss, metrics = local_decision_loss(
            logits,
            event,
            objective=objective,
            epsilon=epsilon,
            tau=tau,
            reference_logits=reference_logits,
            non_target_tether=non_target_tether,
            target_tether=target_tether,
            target_grace=target_grace,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for name, value in metrics.items():
            totals[name] += value
        by_kind[event.decision_kind] += 1
    count = len(schedule)
    return {
        "objective": objective,
        "steps": count,
        "train_events": len(train_events),
        "excluded_train_events": len(all_train_events) - len(train_events),
        "held_out_events": len(events) - len(train_events),
        "balanced": bool(balanced),
        "reference_tethered": bool(non_target_tether > 0 or target_tether > 0),
        "metrics": {name: value / count for name, value in sorted(totals.items())},
        "decision_kind_steps": dict(sorted(by_kind.items())),
    }


def _validate_identity(
    events: list[DecisionEventV1], checkpoint: Path, model: TwoTowerModel
) -> None:
    sha = checkpoint_sha(checkpoint)
    tokenizer_sha = model.artifact_identity()["tokenizer_sha"]
    if any(event.policy_checkpoint_sha != sha for event in events):
        raise ValueError("decision events do not match the policy checkpoint")
    if any(event.tokenizer_sha != tokenizer_sha for event in events):
        raise ValueError("decision events do not match the checkpoint tokenizer")


def train_local_from_paths(
    checkpoint: Path,
    events_path: Path,
    *,
    out_dir: Path,
    objective: LocalObjective,
    reference_checkpoint: Path | None = None,
    steps: int = 50,
    device: str = "cpu",
    lr: float = 5e-5,
    epsilon: float = 2.0,
    tau: float = 1.0,
    non_target_tether: float = 0.0,
    target_tether: float = 0.0,
    target_grace: float = 1.0,
    balanced: bool = False,
    seed: int = 0,
) -> dict:
    events = load_decision_events(events_path)
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    _validate_identity(events, checkpoint, model)
    reference_model = None
    if reference_checkpoint is not None:
        if checkpoint_sha(reference_checkpoint) != checkpoint_sha(checkpoint):
            raise ValueError("reference checkpoint must be the event policy parent")
        reference_model = TwoTowerModel.from_checkpoint(reference_checkpoint, device=device)
    summary = train_local_decisions(
        model,
        events,
        objective=objective,
        reference_model=reference_model,
        steps=steps,
        lr=lr,
        epsilon=epsilon,
        tau=tau,
        non_target_tether=non_target_tether,
        target_tether=target_tether,
        target_grace=target_grace,
        balanced=balanced,
        seed=seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "model.pt"
    model.save(output)
    summary["checkpoint"] = str(output)
    summary["source_checkpoint_sha"] = checkpoint_sha(checkpoint)
    (out_dir / "local_preference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
