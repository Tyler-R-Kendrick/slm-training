"""Preference / DPO-style training stage for TwoTower denoiser."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from slm_training.dsl.schema import load_jsonl
from slm_training.models.twotower import TwoTowerModel, format_context_text
from slm_training.preference import PreferencePair, load_pairs


def _logprob_of_target(model: TwoTowerModel, prompt: str, target: str, design_md: str | None) -> torch.Tensor:
    """Teacher-forced mean log-prob of target tokens under one-step denoiser."""
    ctx_text = format_context_text(
        prompt,
        design_md if model.config.design_md_in_context else None,
        budget=model.config.design_md_budget,
    )
    ctx, ctx_pad = model._encode_context([ctx_text])
    target_ids = model.tokenizer.encode(target)[: model.config.max_target_len]
    ids = torch.tensor([target_ids], dtype=torch.long, device=model.device_name)
    # Mild noise: mask ~30% for a preference-compatible diffusion surrogate.
    mask = torch.rand_like(ids, dtype=torch.float) < 0.3
    mask[:, 0] = False  # keep BOS
    noisy = ids.clone()
    noisy[mask] = model.tokenizer.mask_id
    logits = model.denoiser(noisy, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad)
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    # Score masked positions (where preference signal is applied).
    if mask.any():
        return token_lp[mask].mean()
    return token_lp.mean()


def dpo_loss(
    model: TwoTowerModel,
    pair: PreferencePair,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    """Simple DPO-style loss on masked-token log-probs (no reference model)."""
    chosen_lp = _logprob_of_target(model, pair.prompt, pair.chosen, pair.design_md)
    rejected_lp = _logprob_of_target(model, pair.prompt, pair.rejected, pair.design_md)
    return -F.logsigmoid(beta * (chosen_lp - rejected_lp))


def train_preference(
    model: TwoTowerModel,
    pairs: list[PreferencePair],
    *,
    steps: int = 50,
    lr: float = 1e-4,
    beta: float = 0.1,
) -> dict:
    if not pairs:
        raise ValueError("no preference pairs")
    model.train()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    history: list[float] = []
    for step in range(steps):
        pair = pairs[step % len(pairs)]
        loss = dpo_loss(model, pair, beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))
    return {
        "steps": steps,
        "last_loss": history[-1] if history else None,
        "mean_loss": sum(history) / max(1, len(history)),
        "n_pairs": len(pairs),
    }


def train_preference_from_paths(
    checkpoint: Path,
    pairs_path: Path,
    *,
    out_dir: Path,
    steps: int = 50,
    device: str = "cpu",
) -> dict:
    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    pairs = load_pairs(pairs_path)
    summary = train_preference(model, pairs, steps=steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"
    model.save(ckpt)
    (out_dir / "preference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary["checkpoint"] = str(ckpt)
    return summary
