"""Preference / DPO-style training stage for TwoTower denoiser."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from slm_training.models.twotower import TwoTowerModel, format_context_text
from slm_training.harnesses.preference import PreferencePair, load_pairs


def _encoder_deterministic(model: TwoTowerModel) -> bool:
    """True when re-encoding an identical prompt is provably a no-op."""
    return (
        not model.training
        or float(getattr(model.config, "dropout", 0.0) or 0.0) == 0.0
    )


def _context_of(
    model: TwoTowerModel, prompt: str, design_md: str | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ctx_text = format_context_text(
        prompt,
        design_md if model.config.design_md_in_context else None,
        budget=model.config.design_md_budget,
    )
    return model._encode_context([ctx_text])


def _logprob_of_target(
    model: TwoTowerModel,
    prompt: str,
    target: str,
    design_md: str | None,
    *,
    context: tuple[torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Teacher-forced mean log-prob of target tokens under one-step denoiser."""
    ctx, ctx_pad = (
        context if context is not None else _context_of(model, prompt, design_md)
    )
    target_ids = model.tokenizer.encode(target)[: model.config.max_target_len]
    ids = torch.tensor([target_ids], dtype=torch.long, device=model.device_name)
    # Mild noise: mask ~30% for a preference-compatible diffusion surrogate.
    mask = torch.rand_like(ids, dtype=torch.float) < 0.3
    mask[:, 0] = False  # keep BOS
    noisy = ids.clone()
    noisy[mask] = model.tokenizer.mask_id
    logits = model.denoiser(
        noisy, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
    )
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
    # Both targets share the pair's prompt context; encode it once unless the
    # encoder is stochastic (train mode with dropout), where sharing would
    # change the sampled masks.
    context = (
        _context_of(model, pair.prompt, pair.design_md)
        if _encoder_deterministic(model)
        else None
    )
    chosen_lp = _logprob_of_target(
        model, pair.prompt, pair.chosen, pair.design_md, context=context
    )
    rejected_lp = _logprob_of_target(
        model, pair.prompt, pair.rejected, pair.design_md, context=context
    )
    return -F.logsigmoid(beta * (chosen_lp - rejected_lp))


def train_preference(
    model: TwoTowerModel,
    pairs: list[PreferencePair],
    *,
    steps: int = 50,
    lr: float = 5e-5,
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
        "reference_free": True,
        "note": (
            "Surrogate preference loss on masked-token log-probs; "
            "no frozen reference model (not textbook DPO)."
        ),
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


# --------------------------------------------------------------------------- #
# SLM-418 (DSH5-10) seventh slice: held-out pairwise-preference measurement.
#
# Standalone exposure of the exact chosen/rejected log-prob margin
# ``dpo_loss`` already trains on, so a caller can measure whether a real
# TwoTowerModel checkpoint already prefers ``chosen`` over ``rejected`` on a
# held-out split -- honest evidence, never a certified benefit claim at
# fixture scale. See docs/design/dsh5-10-replay-preference-rows.md's
# "Seventh slice" for what this closes and what it still does not.
# --------------------------------------------------------------------------- #
def pairwise_preference_margin(model: TwoTowerModel, pair: PreferencePair) -> float:
    """``chosen_logprob - rejected_logprob`` under ``model``'s current weights.

    The same masked teacher-forced log-prob :func:`dpo_loss` trains on,
    exposed standalone for measurement. A positive margin means the model
    already ranks ``chosen`` above ``rejected`` for this pair's prompt.
    """
    context = (
        _context_of(model, pair.prompt, pair.design_md)
        if _encoder_deterministic(model)
        else None
    )
    chosen_lp = _logprob_of_target(
        model, pair.prompt, pair.chosen, pair.design_md, context=context
    )
    rejected_lp = _logprob_of_target(
        model, pair.prompt, pair.rejected, pair.design_md, context=context
    )
    return float((chosen_lp - rejected_lp).detach().cpu())


def held_out_pairwise_accuracy(
    model: TwoTowerModel, pairs: Sequence[PreferencePair]
) -> dict:
    """Pairwise chosen>rejected accuracy over ``pairs`` under ``model``'s weights.

    Puts ``model`` in eval mode (so the shared-context caching branch in
    :func:`pairwise_preference_margin` reuses one encoded context per pair
    instead of re-sampling it) and runs under ``torch.no_grad()`` -- this is
    measurement, not training. This is **not** fully deterministic across
    repeated calls: ``_logprob_of_target`` applies fresh random masking noise
    each time it scores a target (the same "mild noise" ``dpo_loss`` trains
    under), so the exact accuracy/margin can vary run-to-run by a small
    amount unless the caller fixes ``torch.manual_seed`` immediately before
    calling. Raises ``ValueError`` on an empty ``pairs`` rather than silently
    reporting a vacuous 0/0 accuracy.
    """
    if not pairs:
        raise ValueError("held_out_pairwise_accuracy requires at least one pair")
    model.eval()
    correct = 0.0
    margins: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            margin = pairwise_preference_margin(model, pair)
            margins.append(margin)
            if margin > 0:
                correct += 1.0
            elif margin == 0:
                correct += 0.5
    return {
        "pairwise_preference_accuracy": correct / len(pairs),
        "n_pairs": len(pairs),
        "mean_margin": sum(margins) / len(margins),
    }


def evaluate_replay_preference_held_out_benefit(
    *,
    baseline_checkpoint: Path,
    held_out_pairs: Sequence[PreferencePair],
    trained_checkpoint: Path | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Honest held-out pairwise-preference benefit report for two checkpoints.

    ``baseline_checkpoint`` is the pre-preference-training checkpoint (e.g. a
    freshly built ``wf_smoke_v2`` scratch checkpoint); ``trained_checkpoint``
    is the same checkpoint after a real ``slm preference train`` run against
    the replay-preference train-split pairs. When ``trained_checkpoint`` is
    omitted, only the baseline is measured and no benefit verdict is made.

    ``seed`` is applied via ``torch.manual_seed`` immediately before each
    ``held_out_pairwise_accuracy`` call (once for the baseline, once for the
    trained checkpoint, if given) so the same ``seed`` reproduces the same
    masking-noise draw for a fixed set of pairs and model architecture --
    ``held_out_pairwise_accuracy`` itself is not unconditionally deterministic
    (see its own docstring), so this is what makes a reported number here
    reproducible rather than a one-off random draw.

    ``verdict`` uses the same strict-greater-than convention as the sixth
    slice's ``held_out_benefit`` (a tie or regression is honestly
    ``no_benefit_fixture_scale``, never rounded up). This is fixture-scale
    wiring evidence over a few real pairs -- never a certified or powered
    ship-readiness claim.
    """
    baseline_model = TwoTowerModel.from_checkpoint(baseline_checkpoint, device=device)
    torch.manual_seed(seed)
    baseline = held_out_pairwise_accuracy(baseline_model, held_out_pairs)
    if trained_checkpoint is None:
        return {
            "baseline": baseline,
            "trained": None,
            "verdict": "no_trained_checkpoint",
            "notes": (
                "only the baseline checkpoint was measured; pass "
                "trained_checkpoint (the output of a real `slm preference "
                "train` run against the replay-preference train-split "
                "pairs) for a benefit comparison"
            ),
        }
    trained_model = TwoTowerModel.from_checkpoint(trained_checkpoint, device=device)
    torch.manual_seed(seed)
    trained = held_out_pairwise_accuracy(trained_model, held_out_pairs)
    verdict = (
        "benefit_observed_fixture_scale"
        if trained["pairwise_preference_accuracy"]
        > baseline["pairwise_preference_accuracy"]
        else "no_benefit_fixture_scale"
    )
    return {
        "baseline": baseline,
        "trained": trained,
        "verdict": verdict,
        "notes": (
            f"fixture-scale synthetic held-out corpus ({len(held_out_pairs)} "
            "pairs); not a powered or certified held-out-benefit claim "
            "either way -- see docs/design/dsh5-10-replay-preference-rows.md."
        ),
    }
