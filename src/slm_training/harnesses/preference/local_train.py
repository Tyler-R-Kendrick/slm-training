"""Exact-state preference objectives for TwoTower decision events."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
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
GradientCombination = Literal["proposal", "pcgrad", "mgda"]
LocalOptimizer = Literal["adamw", "sgd"]
GradientScaling = Literal["raw", "unit_norm"]

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


def _guard_strata_regressions(candidate: dict, baseline: dict) -> list[dict]:
    regressions = []
    candidate_kinds = candidate.get("by_decision_kind") or {}
    for kind, baseline_report in sorted(
        (baseline.get("by_decision_kind") or {}).items()
    ):
        candidate_report = candidate_kinds.get(kind) or {}
        before = baseline_report.get("metrics") or {}
        after = candidate_report.get("metrics") or {}
        for key, direction in _GUARD_DIRECTIONS.items():
            if key not in before or key not in after:
                regressions.append({"decision_kind": kind, "metric": key, "missing": True})
                continue
            regressed = after[key] > before[key] if direction == "min" else after[key] < before[key]
            if regressed:
                regressions.append(
                    {
                        "decision_kind": kind,
                        "metric": key,
                        "before": before[key],
                        "after": after[key],
                    }
                )
    return regressions


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


def _guard_objective_tensors(
    logits: torch.Tensor,
    event: DecisionEventV1,
    *,
    objective: LocalObjective,
    probability_space: Literal["full_vocab", "legal_tokens"] = "full_vocab",
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return minimization-oriented tensors for every guarded metric."""
    loss, _ = local_decision_loss(
        logits, event, objective=objective, epsilon=epsilon, tau=tau
    )
    good_ids = _indices(event.good_token_ids, logits)
    bad_ids = _indices(event.bad_token_ids, logits)
    if probability_space == "full_vocab":
        probs = F.softmax(logits, dim=-1)
        good_mass = probs.index_select(0, good_ids).sum()
        bad_mass = probs.index_select(0, bad_ids).sum()
    elif probability_space == "legal_tokens":
        legal_ids = _indices(event.legal_token_ids, logits)
        legal_probs = F.softmax(logits.index_select(0, legal_ids), dim=-1)
        legal_index = {
            token_id: index for index, token_id in enumerate(event.legal_token_ids)
        }
        legal_good_ids = _indices(
            tuple(legal_index[token_id] for token_id in event.good_token_ids),
            legal_probs,
        )
        legal_bad_ids = _indices(
            tuple(legal_index[token_id] for token_id in event.bad_token_ids),
            legal_probs,
        )
        good_mass = legal_probs.index_select(0, legal_good_ids).sum()
        bad_mass = legal_probs.index_select(0, legal_bad_ids).sum()
    else:
        raise ValueError(f"unknown probability space: {probability_space}")
    good = logits.index_select(0, good_ids)
    bad = logits.index_select(0, bad_ids)
    return {
        "loss": loss,
        "bad_probability_mass": bad_mass,
        "good_probability_mass": -good_mass,
        "mean_margin": -(good[:, None] - bad[None, :]).mean(),
    }


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


def proposal_schedule(
    events: list[DecisionEventV1],
    *,
    steps: int,
    seed: int,
    balanced: bool,
    block_by_decision_kind: bool,
) -> list[list[DecisionEventV1]]:
    """Build single-event or grammar/AST decision-kind block proposals."""
    if not block_by_decision_kind:
        return [
            [event]
            for event in event_schedule(
                events, steps=steps, seed=seed, balanced=balanced
            )
        ]
    groups: dict[str, list[DecisionEventV1]] = defaultdict(list)
    for event in events:
        groups[event.decision_kind].append(event)
    if not groups:
        raise ValueError("no training decision events")
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    return [groups[keys[step % len(keys)]] for step in range(steps)]


def _project_conflicting_gradients(
    gradients: list[list[torch.Tensor | None]],
) -> tuple[list[torch.Tensor | None], dict[str, float | int]]:
    """Deterministically PCGrad-project task gradients, then average them."""
    if not gradients:
        raise ValueError("at least one task gradient is required")
    width = len(gradients[0])
    if any(len(row) != width for row in gradients):
        raise ValueError("task gradients must share a parameter layout")
    projected = [
        [value.detach().clone() if value is not None else None for value in row]
        for row in gradients
    ]
    conflicts = 0
    projections = 0
    for index, row in enumerate(projected):
        for other_index, other in enumerate(gradients):
            if index == other_index:
                continue
            dot = sum(
                (left * right).sum()
                for left, right in zip(row, other, strict=True)
                if left is not None and right is not None
            )
            norm_sq = sum(
                value.square().sum() for value in other if value is not None
            )
            if float(dot) < 0:
                conflicts += 1
                if float(norm_sq) > 0:
                    scale = dot / norm_sq
                    for parameter_index, other_value in enumerate(other):
                        if other_value is not None:
                            if row[parameter_index] is None:
                                row[parameter_index] = torch.zeros_like(other_value)
                            row[parameter_index].sub_(scale * other_value)
                    projections += 1
    combined: list[torch.Tensor | None] = []
    for parameter_index in range(width):
        values = [
            row[parameter_index]
            for row in projected
            if row[parameter_index] is not None
        ]
        combined.append(torch.stack(values).mean(0) if values else None)
    return combined, {
        "task_count": len(gradients),
        "ordered_pair_count": len(gradients) * (len(gradients) - 1),
        "conflict_count": conflicts,
        "projection_count": projections,
    }


def _minimum_norm_gradient(
    gradients: list[list[torch.Tensor | None]],
    *,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
) -> tuple[list[torch.Tensor | None], dict]:
    """Find the minimum-norm convex task-gradient combination with Frank-Wolfe."""
    if not gradients:
        raise ValueError("at least one task gradient is required")
    width = len(gradients[0])
    if any(len(row) != width for row in gradients):
        raise ValueError("task gradients must share a parameter layout")
    count = len(gradients)
    gram = torch.empty((count, count), dtype=torch.float64)
    for left in range(count):
        for right in range(left, count):
            value = sum(
                (a.detach() * b.detach()).sum().double().cpu()
                for a, b in zip(gradients[left], gradients[right], strict=True)
                if a is not None and b is not None
            )
            gram[left, right] = value
            gram[right, left] = value
    diagonal = torch.diagonal(gram)
    active = torch.nonzero(diagonal > tolerance).flatten()
    weights = torch.zeros(count, dtype=torch.float64)
    if not len(active):
        return [None] * width, {
            "task_count": count,
            "active_task_count": 0,
            "inactive_task_count": count,
            "iterations": 0,
            "duality_gap": 0.0,
            "converged": True,
            "norm_sq": 0.0,
            "min_task_dot": 0.0,
            "min_active_task_dot": 0.0,
            "common_descent": False,
            "weights": weights.tolist(),
        }
    active_gram = gram.index_select(0, active).index_select(1, active)
    active_weights = torch.zeros(len(active), dtype=torch.float64)
    active_weights[int(torch.diagonal(active_gram).argmin())] = 1.0
    converged = False
    iterations = 0
    gap = float("inf")
    for iterations in range(1, max_iterations + 1):
        gram_weights = active_gram @ active_weights
        vertex = int(gram_weights.argmin())
        direction = -active_weights
        direction[vertex] += 1.0
        gap = float(active_weights @ gram_weights - gram_weights[vertex])
        if gap <= tolerance:
            converged = True
            break
        denominator = float(direction @ active_gram @ direction)
        if denominator <= tolerance:
            break
        step = max(0.0, min(1.0, -float(direction @ gram_weights) / denominator))
        active_weights.add_(direction, alpha=step)
    weights[active] = active_weights
    combined: list[torch.Tensor | None] = []
    for parameter_index in range(width):
        values = [
            (float(weights[index]), row[parameter_index])
            for index, row in enumerate(gradients)
            if row[parameter_index] is not None
        ]
        combined.append(
            sum(weight * value.detach() for weight, value in values)
            if values
            else None
        )
    task_dots = gram @ weights
    norm_sq = float(weights @ task_dots)
    min_task_dot = float(task_dots.min())
    min_active_task_dot = float(task_dots.index_select(0, active).min())
    return combined, {
        "task_count": count,
        "active_task_count": len(active),
        "inactive_task_count": count - len(active),
        "iterations": iterations,
        "duality_gap": gap,
        "converged": converged,
        "norm_sq": norm_sq,
        "min_task_dot": min_task_dot,
        "min_active_task_dot": min_active_task_dot,
        "common_descent": (
            min_active_task_dot > tolerance and min_task_dot >= -tolerance
        ),
        "weights": weights.tolist(),
    }


def _gradient_alignment(
    left: list[torch.Tensor | None], right: list[torch.Tensor | None]
) -> dict[str, float]:
    dot = sum(
        (a.detach() * b.detach()).sum().double().cpu()
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    )
    left_norm_sq = sum(
        value.detach().square().sum().double().cpu()
        for value in left
        if value is not None
    )
    right_norm_sq = sum(
        value.detach().square().sum().double().cpu()
        for value in right
        if value is not None
    )
    denominator = float(left_norm_sq.sqrt() * right_norm_sq.sqrt())
    return {
        "dot": float(dot),
        "cosine": float(dot) / denominator if denominator else 0.0,
        "left_norm": float(left_norm_sq.sqrt()),
        "right_norm": float(right_norm_sq.sqrt()),
    }


def _scale_gradient(
    gradient: list[torch.Tensor | None], scaling: GradientScaling
) -> list[torch.Tensor | None]:
    if scaling == "raw":
        return gradient
    if scaling != "unit_norm":
        raise ValueError(f"unknown gradient scaling: {scaling}")
    norm_sq = sum(
        value.detach().square().sum().double().cpu()
        for value in gradient
        if value is not None
    )
    norm = float(norm_sq.sqrt())
    return [
        value.detach() / norm if value is not None and norm else value
        for value in gradient
    ]


def _fresh_adamw_direction(
    gradients: list[torch.Tensor | None],
    parameters: list[torch.nn.Parameter],
    *,
    epsilon: float = 1e-8,
    weight_decay: float = 0.01,
) -> list[torch.Tensor | None]:
    """Return the descent direction applied by a fresh AdamW optimizer."""
    if len(gradients) != len(parameters):
        raise ValueError("gradients and parameters must share a layout")
    return [
        gradient.detach() / (gradient.detach().abs() + epsilon)
        + weight_decay * parameter.detach()
        if gradient is not None
        else None
        for gradient, parameter in zip(gradients, parameters, strict=True)
    ]


def diagnose_decision_gradient_alignment(
    model: TwoTowerModel,
    events: list[DecisionEventV1],
    *,
    objective: LocalObjective,
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> dict:
    """Compare exact train and held-out gradients by grammar/AST decision kind."""
    trainable = list(model.trainable_parameters())
    gradients: dict[str, dict[str, list[torch.Tensor | None]]] = {}
    provenance: dict[str, dict[str, dict]] = {}
    for split in ("train", "held_out"):
        selected = _objective_events(
            [event for event in events if event.split == split], objective
        )
        logits_rows = _event_logits_many(model, selected)
        losses: dict[str, list[torch.Tensor]] = defaultdict(list)
        rows: dict[str, list[DecisionEventV1]] = defaultdict(list)
        for event, logits in zip(selected, logits_rows, strict=True):
            loss, _ = local_decision_loss(
                logits, event, objective=objective, epsilon=epsilon, tau=tau
            )
            losses[event.decision_kind].append(loss)
            rows[event.decision_kind].append(event)
        gradients[split] = {}
        provenance[split] = {}
        kinds = sorted(losses)
        for index, kind in enumerate(kinds):
            gradients[split][kind] = list(
                torch.autograd.grad(
                    torch.stack(losses[kind]).mean(),
                    trainable,
                    retain_graph=index + 1 < len(kinds),
                    allow_unused=True,
                )
            )
            kind_rows = rows[kind]
            provenance[split][kind] = {
                "event_count": len(kind_rows),
                "group_count": len({event.group_id for event in kind_rows}),
                "evidence_kinds": dict(
                    sorted(Counter(event.evidence_kind for event in kind_rows).items())
                ),
                "mean_evidence_confidence": sum(
                    event.evidence_confidence for event in kind_rows
                )
                / len(kind_rows),
            }
    train_kinds = set(gradients["train"])
    held_kinds = set(gradients["held_out"])
    ordered_train_kinds = sorted(train_kinds)
    combined, solver = _minimum_norm_gradient(
        [gradients["train"][kind] for kind in ordered_train_kinds]
    )
    solver["weight_by_decision_kind"] = dict(
        zip(ordered_train_kinds, solver.pop("weights"), strict=True)
    )
    adamw_direction = _fresh_adamw_direction(combined, trainable)
    adam_direction = _fresh_adamw_direction(
        combined, trainable, weight_decay=0.0
    )
    def optimizer_alignment(direction: list[torch.Tensor | None]) -> dict:
        return {
            "train_alignment": {
                kind: _gradient_alignment(gradients["train"][kind], direction)
                for kind in ordered_train_kinds
            },
            "held_out_alignment": {
                kind: _gradient_alignment(gradients["held_out"][kind], direction)
                for kind in sorted(held_kinds)
            },
        }
    return {
        "objective": objective,
        "by_decision_kind": {
            kind: {
                **_gradient_alignment(
                    gradients["train"][kind], gradients["held_out"][kind]
                ),
                "train": provenance["train"][kind],
                "held_out": provenance["held_out"][kind],
            }
            for kind in sorted(train_kinds & held_kinds)
        },
        "train_only_decision_kinds": sorted(train_kinds - held_kinds),
        "held_out_only_decision_kinds": sorted(held_kinds - train_kinds),
        "train_minimum_norm_solver": solver,
        "held_out_to_train_combination": {
            kind: _gradient_alignment(gradients["held_out"][kind], combined)
            for kind in sorted(held_kinds)
        },
        "fresh_adamw": {
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            **optimizer_alignment(adamw_direction),
        },
        "fresh_adam": {
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            **optimizer_alignment(adam_direction),
        },
        "cross_kind_alignment": {
            held_kind: {
                train_kind: _gradient_alignment(
                    gradients["held_out"][held_kind],
                    gradients["train"][train_kind],
                )
                for train_kind in ordered_train_kinds
            }
            for held_kind in sorted(held_kinds)
        },
    }


def diagnose_metric_complete_gradient_feasibility(
    model: TwoTowerModel,
    events: list[DecisionEventV1],
    *,
    objective: LocalObjective,
    probability_space: Literal["full_vocab", "legal_tokens"] = "full_vocab",
    gradient_scaling: GradientScaling = "raw",
    epsilon: float = 2.0,
    tau: float = 1.0,
) -> dict:
    """Test common descent across every decision-kind guard objective."""
    trainable = list(model.trainable_parameters())
    gradients: dict[str, dict[str, list[torch.Tensor | None]]] = {}
    counts: dict[str, Counter[str]] = {}
    for split in ("train", "held_out"):
        selected = _objective_events(
            [event for event in events if event.split == split], objective
        )
        values: dict[str, list[torch.Tensor]] = defaultdict(list)
        counts[split] = Counter()
        for event, logits in zip(
            selected, _event_logits_many(model, selected), strict=True
        ):
            for metric, value in _guard_objective_tensors(
                logits,
                event,
                objective=objective,
                probability_space=probability_space,
                epsilon=epsilon,
                tau=tau,
            ).items():
                values[f"{event.decision_kind}:{metric}"].append(value)
            counts[split][event.decision_kind] += 1
        gradients[split] = {}
        keys = sorted(values)
        for index, key in enumerate(keys):
            gradients[split][key] = list(
                torch.autograd.grad(
                    torch.stack(values[key]).mean(),
                    trainable,
                    retain_graph=index + 1 < len(keys),
                    allow_unused=True,
                )
            )
    train_keys = sorted(gradients["train"])
    scaled_train_gradients = [
        _scale_gradient(gradients["train"][key], gradient_scaling)
        for key in train_keys
    ]
    combined, solver = _minimum_norm_gradient(
        scaled_train_gradients
    )
    solver["weight_by_objective"] = dict(
        zip(train_keys, solver.pop("weights"), strict=True)
    )
    held_alignment = {
        key: _gradient_alignment(gradient, combined)
        for key, gradient in sorted(gradients["held_out"].items())
    }
    train_alignment = {
        key: _gradient_alignment(gradient, combined)
        for key, gradient in sorted(gradients["train"].items())
    }
    return {
        "objective": objective,
        "probability_space": probability_space,
        "gradient_scaling": gradient_scaling,
        "guard_objective_directions": dict(_GUARD_DIRECTIONS),
        "train_event_counts": dict(sorted(counts["train"].items())),
        "held_out_event_counts": dict(sorted(counts["held_out"].items())),
        "train_objective_count": len(train_keys),
        "held_out_objective_count": len(held_alignment),
        "train_minimum_norm_solver": solver,
        "train_alignment": train_alignment,
        "train_regressions": sorted(
            key for key, alignment in train_alignment.items() if alignment["dot"] < 0
        ),
        "held_out_alignment": held_alignment,
        "held_out_regressions": sorted(
            key for key, alignment in held_alignment.items() if alignment["dot"] < 0
        ),
    }


def diagnose_decision_gradient_alignment_from_paths(
    checkpoint: Path,
    events_path: Path,
    *,
    objective: LocalObjective,
    device: str = "cpu",
    metric_complete: bool = False,
    probability_space: Literal["full_vocab", "legal_tokens"] = "full_vocab",
    gradient_scaling: GradientScaling = "raw",
) -> dict:
    events = load_decision_events(events_path)
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    _validate_identity(events, checkpoint, model)
    report = (
        diagnose_metric_complete_gradient_feasibility(
            model,
            events,
            objective=objective,
            probability_space=probability_space,
            gradient_scaling=gradient_scaling,
        )
        if metric_complete
        else diagnose_decision_gradient_alignment(model, events, objective=objective)
    )
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_sha"] = checkpoint_sha(checkpoint)
    report["events_path"] = str(events_path)
    return report


def _event_logits(model: TwoTowerModel, event: DecisionEventV1) -> torch.Tensor:
    ctx, ctx_pad = model._encode_context([event.context_text])
    ids = torch.tensor([event.canvas_ids], dtype=torch.long, device=model.device_name)
    logits = model.denoiser(
        ids, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
    )
    return logits[0, event.position]


def _event_logits_many(
    model: TwoTowerModel, events: list[DecisionEventV1]
) -> list[torch.Tensor]:
    """Evaluate exact states in same-length batches with reusable context keys."""
    outputs: list[torch.Tensor | None] = [None] * len(events)
    groups: dict[int, list[tuple[int, DecisionEventV1]]] = defaultdict(list)
    for index, event in enumerate(events):
        groups[len(event.canvas_ids)].append((index, event))
    for group in groups.values():
        prompts = [event.context_text for _, event in group]
        cache_keys = [
            "local-decision:"
            + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in prompts
        ]
        ctx, ctx_pad = model._encode_context(prompts, cache_keys=cache_keys)
        ids = torch.tensor(
            [event.canvas_ids for _, event in group],
            dtype=torch.long,
            device=model.device_name,
        )
        logits = model.denoiser(
            ids, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
        )
        for row, (index, event) in enumerate(group):
            outputs[index] = logits[row, event.position]
    assert all(output is not None for output in outputs)
    return [output for output in outputs if output is not None]


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
    for event, logits in zip(
        selected, _event_logits_many(model, selected), strict=True
    ):
        _, metrics = local_decision_loss(
            logits,
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
    guard_by_decision_kind: bool = False,
    block_by_decision_kind: bool = False,
    gradient_combination: GradientCombination = "proposal",
    optimizer_name: LocalOptimizer = "adamw",
) -> dict:
    if steps <= 0 or lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    if optimizer_name not in ("adamw", "sgd"):
        raise ValueError("local optimizer must be adamw or sgd")
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
    if guard_by_decision_kind and not (guarded_selection or guarded_updates):
        raise ValueError("decision-kind guard requires guarded training")
    if gradient_combination != "proposal" and not guard_by_decision_kind:
        raise ValueError("multi-objective gradients require the decision-kind guard")
    schedule = proposal_schedule(
        train_events,
        steps=max(0, int(steps)),
        seed=int(seed),
        balanced=balanced,
        block_by_decision_kind=block_by_decision_kind,
    )
    if gradient_combination != "proposal":
        schedule = [train_events] * len(schedule)
    if reference_model is not None:
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
    model.train()
    optimizer_type = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.SGD
    optimizer = optimizer_type(model.trainable_parameters(), lr=lr)
    totals: dict[str, float] = defaultdict(float)
    by_kind: dict[str, int] = defaultdict(int)
    selection: dict | None = None
    best_state = None
    best_metrics: dict[str, float] = {}
    best_report: dict = {}
    best_step = 0
    strata_regression_counts: Counter[str] = Counter()
    gradient_projection_totals: Counter[str] = Counter()
    gradient_solver_history: list[dict] = []
    if guarded_selection or guarded_updates:
        baseline = validation_baseline or evaluate_local_decisions(
            model,
            validation,
            objective=objective,
            epsilon=epsilon,
            tau=tau,
        )
        best_metrics = dict(baseline.get("metrics") or {})
        best_report = baseline
        if not all(key in best_metrics for key in _GUARD_DIRECTIONS):
            raise ValueError("guarded selection baseline lacks required metrics")
        best_state = copy.deepcopy(model.state_dict())
        selection = {
            "guard": dict(_GUARD_DIRECTIONS),
            "baseline": best_metrics,
            "history": [{"step": 0, "eligible": True, "metrics": best_metrics}],
            "mode": "updates" if guarded_updates else "selection",
            "by_decision_kind": bool(guard_by_decision_kind),
        }
    for step, proposal in enumerate(schedule, start=1):
        gradient_certified = True
        logits_rows = _event_logits_many(model, proposal)
        with torch.no_grad():
            reference_rows = (
                _event_logits_many(reference_model, proposal)
                if reference_model is not None
                else None
            )
        proposal_losses = []
        proposal_metrics: dict[str, float] = defaultdict(float)
        for index, (event, logits) in enumerate(
            zip(proposal, logits_rows, strict=True)
        ):
            event_loss, event_metrics = local_decision_loss(
                logits,
                event,
                objective=objective,
                epsilon=epsilon,
                tau=tau,
                reference_logits=(
                    reference_rows[index] if reference_rows is not None else None
                ),
                non_target_tether=non_target_tether,
                target_tether=target_tether,
                target_grace=target_grace,
            )
            proposal_losses.append(event_loss)
            for name, value in event_metrics.items():
                proposal_metrics[name] += value
        loss = torch.stack(proposal_losses).mean()
        metrics = {
            name: value / len(proposal)
            for name, value in proposal_metrics.items()
        }
        optimizer.zero_grad(set_to_none=True)
        if gradient_combination != "proposal":
            trainable = list(model.trainable_parameters())
            losses_by_kind: dict[str, list[torch.Tensor]] = defaultdict(list)
            for event, event_loss in zip(proposal, proposal_losses, strict=True):
                losses_by_kind[event.decision_kind].append(event_loss)
            kind_losses = [
                torch.stack(losses_by_kind[kind]).mean()
                for kind in sorted(losses_by_kind)
            ]
            task_gradients = [
                list(
                    torch.autograd.grad(
                        kind_loss,
                        trainable,
                        retain_graph=index + 1 < len(kind_losses),
                        allow_unused=True,
                    )
                )
                for index, kind_loss in enumerate(kind_losses)
            ]
            if gradient_combination == "pcgrad":
                combined, projection = _project_conflicting_gradients(task_gradients)
            else:
                combined, projection = _minimum_norm_gradient(task_gradients)
            gradient_certified = bool(projection.get("common_descent", True))
            for parameter, gradient in zip(trainable, combined, strict=True):
                parameter.grad = gradient if gradient_certified else None
            gradient_projection_totals.update(
                {
                    name: int(value)
                    for name, value in projection.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            )
            gradient_solver_history.append(projection)
        else:
            loss.backward()
        if not gradient_certified:
            for name, value in metrics.items():
                totals[name] += value
            for kind in sorted({event.decision_kind for event in proposal}):
                by_kind[kind] += 1
            if guarded_updates:
                selection["history"].append(
                    {
                        "step": step,
                        "eligible": False,
                        "accepted_scale": None,
                        "metrics": best_metrics,
                        "trials": [],
                        "rejection_reason": "no_common_descent_certificate",
                    }
                )
            continue
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
                strata_regressions = (
                    _guard_strata_regressions(report, best_report)
                    if guard_by_decision_kind
                    else []
                )
                eligible = (
                    _guard_dominates(candidate, best_metrics)
                    and not strata_regressions
                )
                strata_regression_counts.update(
                    f"{item['decision_kind']}:{item['metric']}"
                    for item in strata_regressions
                )
                trials.append(
                    {
                        "scale": scale,
                        "eligible": eligible,
                        "metrics": candidate,
                        "strata_regression_count": len(strata_regressions),
                        "strata_regression_kinds": sorted(
                            {item["decision_kind"] for item in strata_regressions}
                        ),
                        "strata_regression_metrics": sorted(
                            {item["metric"] for item in strata_regressions}
                        ),
                    }
                )
                if eligible:
                    accepted = True
                    best_metrics = candidate
                    best_report = report
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
        if gradient_combination != "proposal":
            for kind in sorted({event.decision_kind for event in proposal}):
                by_kind[kind] += 1
        else:
            by_kind[proposal[0].decision_kind] += 1
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
        selection["strata_regression_counts"] = dict(
            sorted(strata_regression_counts.items())
        )
    return {
        "objective": objective,
        "steps": count,
        "train_events": len(train_events),
        "excluded_train_events": len(all_train_events) - len(train_events),
        "held_out_events": len(events) - len(train_events),
        "validation_batch_groups": len(
            {
                len(event.canvas_ids)
                for event in validation
                if event.split == "held_out"
            }
        ),
        "balanced": bool(balanced),
        "reference_tethered": bool(non_target_tether > 0 or target_tether > 0),
        "guarded_selection": bool(guarded_selection),
        "guarded_updates": bool(guarded_updates),
        "guard_backtrack_steps": int(guard_backtrack_steps) if guarded_updates else 0,
        "guard_by_decision_kind": bool(guard_by_decision_kind),
        "block_by_decision_kind": bool(block_by_decision_kind),
        "gradient_combination": gradient_combination,
        "optimizer": optimizer_name,
        "gradient_projection": dict(sorted(gradient_projection_totals.items())),
        "gradient_solver_history": gradient_solver_history,
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
    guard_by_decision_kind: bool = False,
    block_by_decision_kind: bool = False,
    gradient_combination: GradientCombination = "proposal",
    optimizer_name: LocalOptimizer = "adamw",
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
        guard_by_decision_kind=guard_by_decision_kind,
        block_by_decision_kind=block_by_decision_kind,
        gradient_combination=gradient_combination,
        optimizer_name=optimizer_name,
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
