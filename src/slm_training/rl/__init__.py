"""GRPO-lite online RL on discrete OpenUI rollouts (no value head)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.twotower import TwoTowerModel
from slm_training.preference import composite_reward, grammar_score
from slm_training.preference.train import _logprob_of_target
from slm_training.telemetry import CycleTelemetry, bind_telemetry, timed


@dataclass
class GRPOConfig:
    group_size: int = 4
    steps: int = 50
    lr: float = 1e-4
    kl_beta: float = 0.05
    parse_bonus: float = 0.1
    batch_prompts: int = 2
    seed: int = 0


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
    kl_beta: float = 0.05,
) -> torch.Tensor:
    """
    Group-relative policy gradient on teacher-forced log-probs.

    advantage_i = r_i - mean(r); loss = mean(-adv * logp(y_i)) + optional KL to ref.
    """
    if len(completions) < 2:
        raise ValueError("GRPO group needs ≥2 completions")
    mean_r = sum(rewards) / len(rewards)
    advantages = [float(r) - mean_r for r in rewards]
    # Normalize by std when non-degenerate.
    var = sum((a - 0.0) ** 2 for a in advantages) / len(advantages)
    std = var**0.5
    if std > 1e-6:
        advantages = [a / std for a in advantages]

    losses: list[torch.Tensor] = []
    for comp, adv in zip(completions, advantages):
        lp = _logprob_of_target(model, prompt, comp, design_md)
        losses.append(-float(adv) * lp)
        if ref_model is not None and kl_beta > 0.0:
            with torch.no_grad():
                ref_lp = _logprob_of_target(ref_model, prompt, comp, design_md)
            # Approximate token-mean KL via logp gap.
            losses.append(kl_beta * (lp - ref_lp).pow(2))
    return torch.stack(losses).mean()


def train_grpo(
    model: TwoTowerModel,
    records: list[ExampleRecord],
    *,
    config: GRPOConfig | None = None,
    ref_model: TwoTowerModel | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Online GRPO-lite: rollout K samples / prompt, update denoiser."""
    cfg = config or GRPOConfig()
    if not records:
        raise ValueError("no records for GRPO")
    out_dir = Path(out_dir) if out_dir else Path("outputs/runs/grpo")
    out_dir.mkdir(parents=True, exist_ok=True)

    tel = CycleTelemetry(
        enabled=True,
        meta={"algo": "grpo-lite", "group_size": cfg.group_size},
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr)
    history: list[dict[str, Any]] = []
    # Prefer LTR for fast on-policy rollouts.
    model.config.grammar_ltr_primary = True
    model.config.best_of_n = 1

    import random

    rng = random.Random(cfg.seed)

    with bind_telemetry(tel):
        for step in range(cfg.steps):
            batch = [
                records[rng.randrange(len(records))]
                for _ in range(max(1, cfg.batch_prompts))
            ]
            step_losses: list[float] = []
            step_rewards: list[float] = []
            opt.zero_grad(set_to_none=True)
            for record in batch:
                with timed("rollout"):
                    prev_train = model.training
                    model.eval()
                    comps: list[str] = []
                    for _ in range(max(2, cfg.group_size)):
                        comps.append(
                            model.generate(record.prompt, gold=None, design_md=None)
                        )
                    if prev_train:
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
                with timed("backward"):
                    (loss / len(batch)).backward()
                step_losses.append(float(loss.detach().cpu()))
            with timed("optim_step"):
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
                opt.step()
            row = {
                "step": step + 1,
                "loss": sum(step_losses) / max(1, len(step_losses)),
                "reward_mean": sum(step_rewards) / max(1, len(step_rewards)),
                "reward_max": max(step_rewards) if step_rewards else 0.0,
            }
            history.append(row)

    ckpt = out_dir / "model.pt"
    model.save(ckpt)
    tel_path = tel.write(out_dir / "rl_telemetry.json")
    summary = {
        "algo": "grpo-lite",
        "steps": cfg.steps,
        "group_size": cfg.group_size,
        "kl_beta": cfg.kl_beta if ref_model is not None else 0.0,
        "reference_free": ref_model is None,
        "last_loss": history[-1]["loss"] if history else None,
        "last_reward_mean": history[-1]["reward_mean"] if history else None,
        "history": history,
        "checkpoint": str(ckpt),
        "telemetry": tel.summary(),
        "telemetry_path": str(tel_path),
        "note": (
            "On-policy group-relative updates on teacher-forced log-probs; "
            "structure-only composite_reward; no value head."
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
    kl_beta: float = 0.05,
    lr: float = 1e-4,
) -> dict[str, Any]:
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
    )
