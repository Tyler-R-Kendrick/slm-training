"""DSpark-lite trajectory-survival head training (E73 / V7).

Freeze the TwoTower denoiser, simulate one MaskGIT commit step on train
records, and label each committed token by whether the commitment *survives*:
the next pass still predicts the committed token AND the committed token
matches gold. Trains the plug-in ``survival_head`` (same architecture as the
E31 trust gate) with BCE on committed positions only.

This is the calibrated survival signal the V7 scheduler uses instead of raw
confidence — see ``docs/design/speculative-denoising.md`` §3.4 and the DSpark
row in ``docs/design/research-lineage.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.twotower import TwoTowerModel


def mine_survival_batch(
    model: TwoTowerModel,
    records: list[ExampleRecord],
    *,
    mask_rate: float = 0.6,
    commit_frac: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate one frozen-model commit step and build survival labels.

    1. Mask a random ``mask_rate`` fraction of gold target positions.
    2. Forward: commit the top ``commit_frac`` most-confident masked positions
       with the model's own predictions (a MaskGIT step, on-policy).
    3. Forward again on the updated canvas.
    4. A committed position *survives* iff the second pass still argmaxes the
       committed token and the committed token equals gold.

    Returns ``(hidden, labels, weights)`` — hidden states from the second
    pass, {0,1} survival labels, and a loss mask selecting committed
    positions.
    """
    model.eval()
    device = model.device_name
    prompts = []
    targets: list[list[int]] = []
    for r in records:
        prompts.append(
            model._format_one_context(
                r.prompt,
                r.design_md,
                query_prompt=r.prompt,
                slot_contract=model._resolve_slot_contract(r.prompt, r, r.design_md)
                if getattr(model.config, "slot_contract_in_context", False)
                else None,
            )
        )
        targets.append(model.tokenizer.encode(r.openui)[: model.config.max_target_len])

    from slm_training.models.twotower import _pad_batch

    target_ids = _pad_batch(targets, model.tokenizer.pad_id, device=device)
    ctx, ctx_pad = model._encode_context(prompts)
    bsz, seq = target_ids.shape
    frozen = target_ids.eq(model.tokenizer.pad_id) | target_ids.eq(
        model.tokenizer.bos_id
    )
    noise = (torch.rand(bsz, seq, device=device) < mask_rate) & (~frozen)
    for i in range(bsz):
        if frozen[i].all():
            continue
        if not bool(noise[i].any()):
            valid = (~frozen[i]).nonzero(as_tuple=False).view(-1)
            if valid.numel():
                noise[i, int(valid[0])] = True
    noisy = target_ids.clone()
    noisy[noise] = model.tokenizer.mask_id

    with torch.no_grad():
        logits = model.denoiser(
            noisy,
            ctx,
            pad_id=model.tokenizer.pad_id,
            ctx_pad_mask=ctx_pad,
        )
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        # Commit the top commit_frac most-confident masked positions per row.
        committed = torch.zeros_like(noise)
        canvas = noisy.clone()
        for i in range(bsz):
            masked_idx = noise[i].nonzero(as_tuple=False).view(-1)
            if masked_idx.numel() == 0:
                continue
            k = max(1, int(math.ceil(float(masked_idx.numel()) * commit_frac)))
            row_conf = conf[i, masked_idx]
            top = row_conf.topk(min(k, masked_idx.numel())).indices
            pick = masked_idx[top]
            canvas[i, pick] = pred[i, pick]
            committed[i, pick] = True

        logits2, hidden2 = model.denoiser(
            canvas,
            ctx,
            pad_id=model.tokenizer.pad_id,
            ctx_pad_mask=ctx_pad,
            return_hidden=True,
        )
        pred2 = logits2.argmax(dim=-1)

    stable = pred2.eq(canvas)  # next pass keeps the commitment
    correct = canvas.eq(target_ids)  # commitment matches gold
    labels = (stable & correct).float()
    weights = committed.float()
    return hidden2.detach(), labels.detach(), weights.detach()


def train_survival_gate(
    model: TwoTowerModel,
    records: list[ExampleRecord],
    *,
    steps: int = 40,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str | None = None,
    mask_rate: float = 0.6,
    commit_frac: float = 0.25,
) -> dict[str, Any]:
    """Freeze denoiser/context; optimize the survival head only (BCE)."""
    if device:
        model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.survival_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.survival_head.parameters(), lr=lr)
    losses: list[float] = []
    n = max(1, len(records))
    for step in range(max(1, steps)):
        start = (step * batch_size) % n
        batch = records[start : start + batch_size]
        if len(batch) < batch_size:
            batch = batch + records[: batch_size - len(batch)]
        hidden, labels, weights = mine_survival_batch(
            model, batch, mask_rate=mask_rate, commit_frac=commit_frac
        )
        if float(weights.sum().item()) <= 0:
            continue
        model.survival_head.train()
        survival = model.survival_head(hidden)
        loss = F.binary_cross_entropy(survival, labels, weight=weights)
        # Normalize by the committed fraction so sparse batches still learn.
        loss = loss * (weights.numel() / weights.sum().clamp(min=1.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    for p in model.parameters():
        p.requires_grad_(True)
    model.config.survival_gate_train = True
    model.config.survival_gate = True
    return {
        "steps": steps,
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / max(1, len(losses)) if losses else None,
        "config": {
            k: v
            for k, v in asdict(model.config).items()
            if k.startswith("survival") or k.startswith("cluster")
        },
    }


def train_survival_gate_from_paths(
    checkpoint: Path | str,
    train_records: Path | str,
    *,
    out_dir: Path | str,
    steps: int = 40,
    batch_size: int = 4,
    device: str = "cpu",
    limit: int = 64,
) -> dict[str, Any]:
    """Load a checkpoint, train the survival head, save alongside summary."""
    ckpt = Path(checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = TwoTowerModel.from_checkpoint(ckpt, device=device)
    records = load_jsonl(Path(train_records))[: max(1, limit)]
    summary = train_survival_gate(
        model, records, steps=steps, batch_size=batch_size, device=device
    )
    dest = out / "checkpoints" / "last.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(dest)
    summary["checkpoint"] = str(dest)
    (out / "survival_gate_summary.json").write_text(
        __import__("json").dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
