"""Trajectory-aligned RL objective (E64 / P3)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.models.twotower import TwoTowerModel, _pad_batch
from slm_training.runtime.telemetry import CycleTelemetry, bind_telemetry, timed


@dataclass
class TrajectoryRLConfig:
    steps: int = 50
    lr: float = 1e-5
    group_size: int = 4
    clip_ratio: float = 0.2
    kl_beta: float = 0.0
    seed: int = 0
    max_grad_norm: float = 0.5
    min_reward_std: float = 1e-3
    require_same_policy_sha: bool = True


def lexicographic_reward(reward: dict[str, Any] | None, labels: dict[str, Any] | None) -> float:
    """Lexicographic gates: validity → slots → structure → style → efficiency."""
    reward = dict(reward or {})
    labels = dict(labels or {})
    if not labels.get("accepted"):
        return 0.0
    grammar = float(reward.get("grammar") or 0.0)
    if grammar <= 0:
        return 0.0
    placeholder = float(reward.get("placeholder") or 0.0)
    layout = float(reward.get("layout") or 0.0)
    composite = float(reward.get("composite") or 0.0)
    nfe = float((reward.get("nfe") or reward.get("efficiency") or 0.0))
    # Rank-like scalar that respects the gate order via large bases.
    return (
        1_000_000.0 * (1.0 if labels.get("accepted") else 0.0)
        + 10_000.0 * grammar
        + 1_000.0 * placeholder
        + 100.0 * layout
        + 10.0 * composite
        - nfe
    )


def _group_traces(
    traces: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        meta = dict(trace.get("meta") or {})
        key = f"{meta.get('record_id')}|{meta.get('prompt')}"
        groups.setdefault(key, []).append(trace)
    return groups


def trajectory_logprob(
    model: TwoTowerModel,
    trace: dict[str, Any],
) -> torch.Tensor | None:
    """Sum log π_θ(a_t | canvas_t) over recorded commits (same support when present)."""
    prompt = str((trace.get("meta") or {}).get("prompt") or "")
    steps = list(trace.get("steps") or [])
    if not prompt or not steps:
        return None
    device = model.device_name
    total = None
    n = 0
    for step in steps:
        canvas = step.get("canvas")
        commits = step.get("commits") or []
        if not canvas or not commits:
            continue
        ctx, ctx_pad = model._encode_context([prompt], cache_keys=None)
        noisy = _pad_batch([list(canvas)], model.tokenizer.pad_id, device=device)
        logits = model.denoiser(
            noisy, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
        )
        # Mask illegal ids when support was recorded (diffusion top-p analogue).
        for commit in commits:
            allowed = commit.get("allowed_id_set")
            pos = int(commit["t"])
            tid = int(commit["id"])
            row = logits[0, pos].float()
            if allowed:
                mask = torch.full_like(row, float("-inf"))
                idxs = [int(x) for x in allowed if 0 <= int(x) < row.numel()]
                if not idxs:
                    continue
                mask[idxs] = row[idxs]
                row = mask
            lp = F.log_softmax(row, dim=-1)[tid]
            total = lp if total is None else total + lp
            n += 1
    if total is None or n == 0:
        return None
    return total / n


def importance_weighted_loss(
    model: TwoTowerModel,
    group: list[dict[str, Any]],
    *,
    clip_ratio: float = 0.2,
) -> torch.Tensor | None:
    if len(group) < 2:
        return None
    rewards = [
        lexicographic_reward(t.get("reward"), t.get("labels")) for t in group
    ]
    mean_r = sum(rewards) / len(rewards)
    advantages = [r - mean_r for r in rewards]
    var = sum(a * a for a in advantages) / len(advantages)
    std = math.sqrt(var)
    if std < 1e-6 or mean_r <= 0:
        return None
    advantages = [a / std for a in advantages]

    losses: list[torch.Tensor] = []
    for trace, adv in zip(group, advantages):
        if abs(adv) < 1e-8:
            continue
        new_lp = trajectory_logprob(model, trace)
        if new_lp is None:
            continue
        # Rollout log-prob from recorded commit lps (behavior policy).
        old_lps: list[float] = []
        for step in trace.get("steps") or []:
            for commit in step.get("commits") or []:
                if "lp" in commit:
                    old_lps.append(float(commit["lp"]))
        if not old_lps:
            # Fall back to on-policy PG without importance weights.
            losses.append(-float(adv) * new_lp)
            continue
        old_lp = sum(old_lps) / len(old_lps)
        ratio = torch.exp(new_lp - float(old_lp))
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        # PPO-style clipped surrogate on trajectory likelihoods.
        unclipped = ratio * float(adv)
        clipped_obj = clipped * float(adv)
        losses.append(-torch.min(unclipped, clipped_obj))
    if not losses:
        return None
    return torch.stack(losses).mean()


def train_trajectory_rl(
    model: TwoTowerModel,
    traces: list[dict[str, Any]],
    *,
    config: TrajectoryRLConfig | None = None,
    out_dir: Path | None = None,
    base_policy_sha: str | None = None,
) -> dict[str, Any]:
    cfg = config or TrajectoryRLConfig()
    out_dir = Path(out_dir) if out_dir else Path("outputs/runs/traj_rl")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Never mix traces from different policy checkpoints.
    if cfg.require_same_policy_sha:
        shas = {
            str((t.get("meta") or {}).get("policy_checkpoint_sha") or "")
            for t in traces
        }
        shas.discard("")
        if base_policy_sha:
            traces = [
                t
                for t in traces
                if str((t.get("meta") or {}).get("policy_checkpoint_sha") or "")
                == base_policy_sha
            ]
        elif len(shas) > 1:
            # Keep the most common sha.
            from collections import Counter

            top = Counter(
                str((t.get("meta") or {}).get("policy_checkpoint_sha") or "")
                for t in traces
            ).most_common(1)[0][0]
            traces = [
                t
                for t in traces
                if str((t.get("meta") or {}).get("policy_checkpoint_sha") or "") == top
            ]

    groups = {
        k: v
        for k, v in _group_traces(traces).items()
        if len(v) >= 2
    }
    if not groups:
        raise ValueError("need ≥2 rollouts per prompt for trajectory RL")

    # All-fail groups → leave for repair pipeline (do not update).
    usable: list[list[dict[str, Any]]] = []
    repair_routed = 0
    for group in groups.values():
        if not any((t.get("labels") or {}).get("accepted") for t in group):
            repair_routed += 1
            continue
        usable.append(group)
    if not usable:
        raise ValueError("all groups failed — route to repair, no RL update")

    tel = CycleTelemetry(
        enabled=True,
        meta={"algo": "e64_trajectory_rl", "group_size": cfg.group_size},
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr)
    history: list[dict[str, Any]] = []
    import random

    rng = random.Random(cfg.seed)

    with bind_telemetry(tel):
        for step in range(cfg.steps):
            group = usable[rng.randrange(len(usable))]
            # Subsample to group_size when larger.
            if len(group) > cfg.group_size:
                group = rng.sample(group, cfg.group_size)
            opt.zero_grad(set_to_none=True)
            with timed("traj_pg"):
                loss = importance_weighted_loss(
                    model, group, clip_ratio=cfg.clip_ratio
                )
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.trainable_parameters()), cfg.max_grad_norm
            )
            opt.step()
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})

    model.save(out_dir / "model.pt")
    summary = {
        "algo": "e64_trajectory_rl",
        "steps": cfg.steps,
        "n_traces": len(traces),
        "n_groups": len(usable),
        "repair_routed_groups": repair_routed,
        "history": history[-50:],
        "telemetry": tel.summary(),
    }
    (out_dir / "traj_rl_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
