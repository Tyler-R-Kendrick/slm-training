"""GRPO-lite online RL on discrete OpenUI rollouts (no value head)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.twotower import TwoTowerModel
from slm_training.harnesses.preference import composite_reward, grammar_score
from slm_training.harnesses.preference.train import _logprob_of_target
from slm_training.runtime.telemetry import CycleTelemetry, bind_telemetry, timed
from slm_training.autoresearch.rl_gate import assert_rl_ready
from slm_training.autoresearch.schemas import RLReadinessReport


@dataclass
class GRPOConfig:
    group_size: int = 4
    steps: int = 50
    lr: float = 1e-5
    kl_beta: float = 0.02
    parse_bonus: float = 0.1
    batch_prompts: int = 2
    seed: int = 0
    min_reward_std: float = 1e-3
    max_grad_norm: float = 0.5


def structure_reward(
    openui: str,
    *,
    gold: ExampleRecord | None = None,
    parse_bonus: float = 0.1,
) -> float:
    """Structure-only reward for RL (never DESIGN.md style lint)."""
    base = float(composite_reward(openui, gold=gold, design_md=None))
    if parse_bonus > 0.0 and grammar_score(openui) > 0.0:
        base = min(1.0, base + parse_bonus * 0.1)
    return base


def grpo_loss_for_group(
    model: TwoTowerModel,
    prompt: str,
    completions: list[str],
    rewards: list[float],
    *,
    design_md: str | None = None,
    ref_model: TwoTowerModel | None = None,
    kl_beta: float = 0.02,
) -> torch.Tensor | None:
    """
    Group-relative policy gradient on teacher-forced log-probs.

    Returns None when the group has no usable reward signal (skip update).
    """
    if len(completions) < 2:
        return None
    mean_r = sum(rewards) / len(rewards)
    advantages = [float(r) - mean_r for r in rewards]
    var = sum(a * a for a in advantages) / len(advantages)
    std = var**0.5
    if std < 1e-6 or mean_r <= 0.0:
        return None
    advantages = [a / std for a in advantages]

    losses: list[torch.Tensor] = []
    for comp, adv in zip(completions, advantages):
        if abs(adv) < 1e-6:
            continue
        lp = _logprob_of_target(model, prompt, comp, design_md)
        losses.append(-float(adv) * lp)
        if ref_model is not None and kl_beta > 0.0:
            with torch.no_grad():
                ref_lp = _logprob_of_target(ref_model, prompt, comp, design_md)
            losses.append(kl_beta * (lp - ref_lp).pow(2))
    if not losses:
        return None
    return torch.stack(losses).mean()


def train_grpo(
    model: TwoTowerModel,
    records: list[ExampleRecord],
    *,
    config: GRPOConfig | None = None,
    ref_model: TwoTowerModel | None = None,
    out_dir: Path | None = None,
    readiness_report: RLReadinessReport | Path | str | None = None,
) -> dict[str, Any]:
    """Online GRPO-lite: rollout K samples / prompt, update denoiser."""
    readiness = assert_rl_ready(readiness_report)
    cfg = config or GRPOConfig()
    if not records:
        raise ValueError("no records for GRPO")
    out_dir = Path(out_dir) if out_dir else Path("outputs/runs/grpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    tel = CycleTelemetry(
        enabled=True,
        meta={"algo": "grpo-lite", "group_size": cfg.group_size, "lr": cfg.lr},
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr)
    history: list[dict[str, Any]] = []
    model.config.grammar_ltr_primary = True
    model.config.best_of_n = 1
    model.config.grammar_ltr_repair = True

    import random

    rng = random.Random(cfg.seed)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_reward = -1.0
    skipped = 0

    with bind_telemetry(tel):
        for step in range(cfg.steps):
            batch = [
                records[rng.randrange(len(records))]
                for _ in range(max(1, cfg.batch_prompts))
            ]
            step_losses: list[float] = []
            step_rewards: list[float] = []
            opt.zero_grad(set_to_none=True)
            did_backward = False
            n_loss_terms = 0
            for record in batch:
                with timed("rollout"):
                    model.eval()
                    comps: list[str] = []
                    for _ in range(max(2, cfg.group_size)):
                        comps.append(
                            model.generate(record.prompt, gold=None, design_md=None)
                        )
                    model.train()
                with timed("reward"):
                    rewards = [
                        structure_reward(
                            c, gold=record, parse_bonus=cfg.parse_bonus
                        )
                        for c in comps
                    ]
                step_rewards.extend(rewards)
                with timed("grpo_loss"):
                    loss = grpo_loss_for_group(
                        model,
                        record.prompt,
                        comps,
                        rewards,
                        design_md=None,
                        ref_model=ref_model,
                        kl_beta=cfg.kl_beta if ref_model is not None else 0.0,
                    )
                if loss is None:
                    skipped += 1
                    continue
                with timed("backward"):
                    loss.backward()
                    did_backward = True
                    n_loss_terms += 1
                step_losses.append(float(loss.detach().cpu()))
            mean_r = sum(step_rewards) / max(1, len(step_rewards))
            if did_backward:
                with timed("optim_step"):
                    if n_loss_terms > 1:
                        for p in model.trainable_parameters():
                            if p.grad is not None:
                                p.grad.div_(float(n_loss_terms))
                    torch.nn.utils.clip_grad_norm_(
                        list(model.trainable_parameters()), cfg.max_grad_norm
                    )
                    opt.step()
            model.eval()
            if mean_r >= best_reward:
                best_reward = mean_r
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            history.append(
                {
                    "step": step + 1,
                    "loss": (
                        sum(step_losses) / max(1, len(step_losses)) if step_losses else None
                    ),
                    "reward_mean": mean_r,
                    "reward_max": max(step_rewards) if step_rewards else 0.0,
                    "updated": did_backward,
                }
            )

    # Restore best-on-reward weights (protect against collapse).
    model.load_state_dict(best_state)
    ckpt = out_dir / "model.pt"
    model.save(ckpt)
    tel_path = tel.write(out_dir / "rl_telemetry.json")
    summary = {
        "algo": "grpo-lite",
        "steps": cfg.steps,
        "group_size": cfg.group_size,
        "lr": cfg.lr,
        "kl_beta": cfg.kl_beta if ref_model is not None else 0.0,
        "reference_free": ref_model is None,
        "skipped_groups": skipped,
        "best_reward_mean": best_reward,
        "last_loss": history[-1]["loss"] if history else None,
        "last_reward_mean": history[-1]["reward_mean"] if history else None,
        "history": history,
        "checkpoint": str(ckpt),
        "telemetry": tel.summary(),
        "telemetry_path": str(tel_path),
        "rl_readiness_report_id": readiness.report_id,
        "note": (
            "On-policy group-relative updates; skips zero-variance / zero-reward "
            "groups; restores best-reward weights at end."
        ),
    }
    (out_dir / "rl_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def train_grpo_from_paths(
    checkpoint: Path,
    train_records: Path,
    *,
    out_dir: Path,
    steps: int = 50,
    group_size: int = 4,
    device: str = "cpu",
    ref_checkpoint: Path | None = None,
    limit: int | None = 64,
    kl_beta: float = 0.02,
    lr: float = 1e-5,
    readiness_report: RLReadinessReport | Path | str | None = None,
) -> dict[str, Any]:
    readiness = assert_rl_ready(readiness_report)
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    records = load_jsonl(train_records)
    if limit is not None:
        records = records[: max(0, int(limit))]
    ref = None
    if ref_checkpoint is not None and Path(ref_checkpoint).exists():
        ref = TwoTowerModel.from_checkpoint(ref_checkpoint, device=device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
    return train_grpo(
        model,
        records,
        config=GRPOConfig(
            steps=steps, group_size=group_size, kl_beta=kl_beta, lr=lr
        ),
        ref_model=ref,
        out_dir=out_dir,
        readiness_report=readiness,
    )
