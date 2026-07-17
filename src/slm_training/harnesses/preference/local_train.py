"""Exact-state preference objectives for TwoTower decision events."""

from __future__ import annotations

import copy
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

_GUARD_DIRECTIONS = {
    "loss": "min",
    "bad_probability_mass": "min",
    "good_probability_mass": "max",
    "mean_margin": "max",
}


def _guard_dominates(
    candidate: dict[str, float], baseline: dict[str, float]
) -> bool:
    weak = all(
        candidate[key] <= baseline[key]
        if direction == "min"
        else candidate[key] >= baseline[key]
        for key, direction in _GUARD_DIRECTIONS.items()
    )
    strict = any(
        candidate[key] < baseline[key]
        if direction == "min"
        else candidate[key] > baseline[key]
        for key, direction in _GUARD_DIRECTIONS.items()
    )
    return weak and strict


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


def _objective_events(
    events: list[DecisionEventV1], objective: LocalObjective
) -> list[DecisionEventV1]:
    if objective == "ce_margin":
        return [event for event in events if len(event.good_token_ids) == 1]
    if objective == "ftpo_single":
        return [
            event
            for event in events
            if len(event.good_token_ids) == 1 and len(event.bad_token_ids) == 1
        ]
    return events


@torch.inference_mode()
def evaluate_local_decisions(
    model: TwoTowerModel,
    events: list[DecisionEventV1],
    *,
    objective: LocalObjective,
    split: str = "held_out",
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> dict:
    """Measure exact-state recurrence without updating the policy."""
    selected = _objective_events(
        [event for event in events if event.split == split], objective
    )
    was_training = model.training
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    by_kind: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    kind_counts: dict[str, int] = defaultdict(int)
    for event in selected:
        _, metrics = local_decision_loss(
            _event_logits(model, event),
            event,
            objective=objective,
            epsilon=epsilon,
            tau=tau,
        )
        for name, value in metrics.items():
            totals[name] += value
            by_kind[event.decision_kind][name] += value
        kind_counts[event.decision_kind] += 1
    model.train(was_training)
    count = len(selected)
    return {
        "split": split,
        "event_count": count,
        "excluded_events": sum(event.split == split for event in events) - count,
        "metrics": {
            name: value / count for name, value in sorted(totals.items())
        }
        if count
        else {},
        "by_decision_kind": {
            kind: {
                "event_count": kind_counts[kind],
                "metrics": {
                    name: value / kind_counts[kind]
                    for name, value in sorted(metrics.items())
                },
            }
            for kind, metrics in sorted(by_kind.items())
        },
    }


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
    validation_events: list[DecisionEventV1] | None = None,
    validation_baseline: dict | None = None,
    validation_every: int = 0,
    guarded_selection: bool = False,
    guarded_updates: bool = False,
    guard_backtrack_steps: int = 4,
) -> dict:
    if steps <= 0 or lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    if any(event.evidence_kind == "constraint_shadow" for event in events):
        raise ValueError(
            "constraint shadows encode decoder legality, not semantic preferences; "
            "train only on judge-backed counterfactual events"
        )
    all_train_events = [event for event in events if event.split == "train"]
    train_events = _objective_events(all_train_events, objective)
    if objective == "ftpo_set" and not any(
        len(event.good_token_ids) > 1 or len(event.bad_token_ids) > 1
        for event in train_events
    ):
        raise ValueError("ftpo_set requires a verified set-valued training event")
    if (non_target_tether > 0 or target_tether > 0) and reference_model is None:
        raise ValueError("reference_model is required for tethered training")
    if guarded_selection and validation_every <= 0:
        raise ValueError("guarded selection requires positive validation_every")
    validation = validation_events or []
    if (guarded_selection or guarded_updates) and not any(
        event.split == "held_out" for event in validation
    ):
        raise ValueError("guarded training requires held-out decision events")
    if guard_backtrack_steps < 0:
        raise ValueError("guard backtrack steps must be non-negative")
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
    selection: dict | None = None
    best_state = None
    best_metrics: dict[str, float] = {}
    best_step = 0
    if guarded_selection or guarded_updates:
        baseline = validation_baseline or evaluate_local_decisions(
            model,
            validation,
            objective=objective,
            epsilon=epsilon,
            tau=tau,
        )
        best_metrics = dict(baseline.get("metrics") or {})
        if not all(key in best_metrics for key in _GUARD_DIRECTIONS):
            raise ValueError("guarded selection baseline lacks required metrics")
        best_state = copy.deepcopy(model.state_dict())
        selection = {
            "guard": dict(_GUARD_DIRECTIONS),
            "baseline": best_metrics,
            "history": [{"step": 0, "eligible": True, "metrics": best_metrics}],
            "mode": "updates" if guarded_updates else "selection",
        }
    for step, event in enumerate(schedule, start=1):
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
        if guarded_updates:
            model_state = copy.deepcopy(model.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            base_lrs = [group["lr"] for group in optimizer.param_groups]
            trials = []
            accepted = False
            for backtrack in range(guard_backtrack_steps + 1):
                if backtrack:
                    model.load_state_dict(model_state)
                    optimizer.load_state_dict(optimizer_state)
                scale = 0.5**backtrack
                for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                    group["lr"] = base_lr * scale
                optimizer.step()
                report = evaluate_local_decisions(
                    model,
                    validation,
                    objective=objective,
                    epsilon=epsilon,
                    tau=tau,
                )
                candidate = dict(report.get("metrics") or {})
                eligible = _guard_dominates(candidate, best_metrics)
                trials.append(
                    {"scale": scale, "eligible": eligible, "metrics": candidate}
                )
                if eligible:
                    accepted = True
                    best_metrics = candidate
                    best_step = step
                    best_state = copy.deepcopy(model.state_dict())
                    break
            for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base_lr
            if not accepted:
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
            selection["history"].append(
                {
                    "step": step,
                    "eligible": accepted,
                    "accepted_scale": trials[-1]["scale"] if accepted else None,
                    "metrics": best_metrics,
                    "trials": trials,
                }
            )
        else:
            optimizer.step()
        for name, value in metrics.items():
            totals[name] += value
        by_kind[event.decision_kind] += 1
        if guarded_selection and (step % validation_every == 0 or step == len(schedule)):
            report = evaluate_local_decisions(
                model,
                validation,
                objective=objective,
                epsilon=epsilon,
                tau=tau,
            )
            candidate = dict(report.get("metrics") or {})
            eligible = all(
                candidate[key] <= best_metrics[key]
                if direction == "min"
                else candidate[key] >= best_metrics[key]
                for key, direction in _GUARD_DIRECTIONS.items()
            )
            selection["history"].append(
                {"step": step, "eligible": eligible, "metrics": candidate}
            )
            if eligible and candidate["loss"] < best_metrics["loss"]:
                best_metrics = candidate
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    count = len(schedule)
    if guarded_selection:
        model.load_state_dict(best_state)
        selection["selected_step"] = best_step
        selection["restored"] = best_step != count
    elif guarded_updates:
        selection["selected_step"] = best_step
        selection["restored"] = False
        selection["accepted_steps"] = sum(
            bool(item.get("eligible")) for item in selection["history"][1:]
        )
        selection["rejected_steps"] = count - selection["accepted_steps"]
    return {
        "objective": objective,
        "steps": count,
        "train_events": len(train_events),
        "excluded_train_events": len(all_train_events) - len(train_events),
        "held_out_events": len(events) - len(train_events),
        "balanced": bool(balanced),
        "reference_tethered": bool(non_target_tether > 0 or target_tether > 0),
        "guarded_selection": bool(guarded_selection),
        "guarded_updates": bool(guarded_updates),
        "guard_backtrack_steps": int(guard_backtrack_steps) if guarded_updates else 0,
        "validation_every": (
            1 if guarded_updates else int(validation_every) if guarded_selection else 0
        ),
        "validation_selection": selection,
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
    validation_every: int = 0,
    guarded_selection: bool = False,
    guarded_updates: bool = False,
    guard_backtrack_steps: int = 4,
) -> dict:
    events = load_decision_events(events_path)
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    _validate_identity(events, checkpoint, model)
    held_out_before = evaluate_local_decisions(
        model,
        events,
        objective=objective,
        epsilon=epsilon,
        tau=tau,
    )
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
        validation_events=events,
        validation_baseline=held_out_before,
        validation_every=validation_every,
        guarded_selection=guarded_selection,
        guarded_updates=guarded_updates,
        guard_backtrack_steps=guard_backtrack_steps,
    )
    held_out_after = evaluate_local_decisions(
        model,
        events,
        objective=objective,
        epsilon=epsilon,
        tau=tau,
    )
    summary["held_out_before"] = held_out_before
    summary["held_out_after"] = held_out_after
    summary["held_out_delta"] = {
        name: held_out_after["metrics"][name] - value
        for name, value in held_out_before["metrics"].items()
        if name in held_out_after["metrics"]
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "model.pt"
    model.save(output)
    summary["checkpoint"] = str(output)
    summary["source_checkpoint_sha"] = checkpoint_sha(checkpoint)
    (out_dir / "local_preference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
