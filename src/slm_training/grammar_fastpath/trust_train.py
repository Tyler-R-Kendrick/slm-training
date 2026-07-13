"""BackPlay-lite training for FastPathGate (E31 / E52).

Freeze the TwoTower denoiser, mine its own token errors, and train the
plug-in trust head with BCE so remask can prefer low-trust positions.

E52: when ``slot_aware_trust_gate`` is set, also mark placeholder / slot-binding
positions as untrusted when the greedy prediction does not match gold.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.twotower import TwoTowerModel


def _placeholder_token_mask(
    model: TwoTowerModel,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Boolean mask of positions whose decoded piece looks like a placeholder."""
    mask = torch.zeros_like(target_ids, dtype=torch.bool)
    tok = model.tokenizer
    for b in range(target_ids.size(0)):
        for t in range(target_ids.size(1)):
            tid = int(target_ids[b, t].item())
            if tid in (tok.pad_id, tok.bos_id, tok.eos_id, tok.mask_id):
                continue
            try:
                piece = tok.id_to_token.get(tid, "") if hasattr(tok, "id_to_token") else ""
                if not piece and hasattr(tok, "decode"):
                    piece = tok.decode([tid])
            except Exception:  # noqa: BLE001
                piece = ""
            text = str(piece)
            if ":" in text or text.startswith("<SYM") or text.startswith("<BIND"):
                mask[b, t] = True
    return mask


def mine_gate_batch(
    model: TwoTowerModel,
    records: list[ExampleRecord],
    *,
    mask_rate: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a frozen denoiser forward and build trust labels.

    Label = 1.0 where the model's greedy prediction matches gold on masked
    positions (trusted); 0.0 where it errs (should remask later).

    E52 slot-aware mode additionally forces label=0 on placeholder positions
    whose predicted token differs from gold (binding / namespace errors).
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
    frozen = target_ids.eq(model.tokenizer.pad_id) | target_ids.eq(model.tokenizer.bos_id)
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
        logits, hidden = model.denoiser(
            noisy,
            ctx,
            pad_id=model.tokenizer.pad_id,
            ctx_pad_mask=ctx_pad,
            return_hidden=True,
        )
        pred = logits.argmax(dim=-1)
        correct = pred.eq(target_ids)
    # Trust labels: 1 on correct visible or correctly filled masks; 0 on errors.
    labels = correct.float()
    labels = labels.masked_fill(frozen, 1.0)

    if bool(getattr(model.config, "slot_aware_trust_gate", False)):
        ph_mask = _placeholder_token_mask(model, target_ids)
        # Any wrong placeholder prediction is untrusted.
        labels = torch.where(ph_mask & (~correct), torch.zeros_like(labels), labels)
        # Also untrust if decoded placeholders diverge from gold inventory set.
        for i, r in enumerate(records):
            gold_set = set(r.placeholders or extract_placeholders(r.openui))
            if not gold_set:
                continue
            try:
                pred_text = model.tokenizer.decode(
                    [int(x) for x in pred[i].tolist() if int(x) != model.tokenizer.pad_id]
                )
                pred_set = set(extract_placeholders(pred_text))
            except Exception:  # noqa: BLE001
                continue
            if pred_set and pred_set != gold_set:
                # Soft: mark all placeholder positions in this row as untrusted.
                labels[i] = torch.where(ph_mask[i], torch.zeros_like(labels[i]), labels[i])

    return hidden.detach(), labels.detach()


def train_trust_gate(
    model: TwoTowerModel,
    records: list[ExampleRecord],
    *,
    steps: int = 40,
    batch_size: int = 4,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    """Freeze denoiser/context; optimize trust_gate only."""
    if device:
        model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.trust_gate.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.trust_gate.parameters(), lr=lr)
    losses: list[float] = []
    n = max(1, len(records))
    for step in range(max(1, steps)):
        start = (step * batch_size) % n
        batch = records[start : start + batch_size]
        if len(batch) < batch_size:
            batch = batch + records[: batch_size - len(batch)]
        hidden, labels = mine_gate_batch(model, batch)
        model.trust_gate.train()
        trust = model.trust_gate(hidden)
        loss = F.binary_cross_entropy(trust, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    # Restore trainability for non-gate params for subsequent stages.
    for p in model.parameters():
        p.requires_grad_(True)
    model.config.trust_gate_train = True
    model.config.remask_use_gate = True
    return {
        "steps": steps,
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "config": {k: v for k, v in asdict(model.config).items() if k.startswith("fastpath") or k.startswith("remask") or k.startswith("trust")},
    }


def train_trust_gate_from_paths(
    checkpoint: Path | str,
    train_records: Path | str,
    *,
    out_dir: Path | str,
    steps: int = 40,
    batch_size: int = 4,
    device: str = "cpu",
    limit: int = 64,
    slot_aware: bool = False,
) -> dict[str, Any]:
    ckpt = Path(checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = TwoTowerModel.from_checkpoint(ckpt, device=device)
    if slot_aware:
        model.config.slot_aware_trust_gate = True
    records = load_jsonl(Path(train_records))[: max(1, limit)]
    summary = train_trust_gate(
        model, records, steps=steps, batch_size=batch_size, device=device
    )
    dest = out / "checkpoints" / "last.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(dest)
    summary["checkpoint"] = str(dest)
    summary["slot_aware_trust_gate"] = bool(
        getattr(model.config, "slot_aware_trust_gate", False)
    )
    (out / "trust_gate_summary.json").write_text(
        __import__("json").dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
