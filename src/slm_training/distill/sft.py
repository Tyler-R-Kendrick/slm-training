"""Anchor-mixed self-distillation SFT (P2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerModel, _pad_batch
from slm_training.telemetry import CycleTelemetry, bind_telemetry, timed


@dataclass
class DistillSFTConfig:
    steps: int = 100
    lr: float = 5e-5
    batch_size: int = 2
    lambda_traj: float = 1.0
    lambda_anchor: float = 0.3
    dropout: float = 0.0
    seed: int = 0
    max_grad_norm: float = 1.0


def traces_to_records(traces: list[dict[str, Any]]) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for idx, trace in enumerate(traces):
        text = ((trace.get("final") or {}).get("text") or "").strip()
        prompt = str((trace.get("meta") or {}).get("prompt") or "")
        if not text or not prompt:
            continue
        labels = dict(trace.get("labels") or {})
        records.append(
            ExampleRecord(
                id=str(trace.get("trace_id") or f"distill_{idx}"),
                prompt=prompt,
                openui=text,
                split="train",
                source="self_distilled_success",
                meta={
                    "source_family": "self_distilled_success",
                    "trace_id": trace.get("trace_id"),
                    "policy_checkpoint_sha": (trace.get("meta") or {}).get(
                        "policy_checkpoint_sha"
                    ),
                    "accepted": bool(labels.get("accepted")),
                },
            )
        )
    return records


def trajectory_action_loss(
    model: TwoTowerModel,
    trace: dict[str, Any],
) -> torch.Tensor | None:
    """Teacher-force recorded commits: canvas → chosen token ids (L_traj)."""
    prompt = str((trace.get("meta") or {}).get("prompt") or "")
    steps = list(trace.get("steps") or [])
    if not prompt or not steps:
        return None
    device = model.device_name
    losses: list[torch.Tensor] = []
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
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for commit in commits:
            pos = int(commit["t"])
            tid = int(commit["id"])
            if pos < 0 or pos >= log_probs.size(1):
                continue
            # Optional grammar-support restriction when recorded.
            allowed = commit.get("allowed_id_set")
            if allowed:
                allowed_set = {int(x) for x in allowed}
                if tid not in allowed_set:
                    continue
            losses.append(-log_probs[0, pos, tid])
    if not losses:
        return None
    return torch.stack(losses).mean()


def train_self_distill(
    model: TwoTowerModel,
    traces: list[dict[str, Any]],
    *,
    anchor_records: list[ExampleRecord] | None = None,
    config: DistillSFTConfig | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """L = L_final + λ_traj · L_next_action + λ_anchor · L_anchor."""
    cfg = config or DistillSFTConfig()
    distill_records = traces_to_records(traces)
    if not distill_records and not traces:
        raise ValueError("no distill traces/records")
    out_dir = Path(out_dir) if out_dir else Path("outputs/runs/self_distill")
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dropout > 0 and hasattr(model, "denoiser"):
        # Best-effort: set dropout modules if present.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = float(cfg.dropout)

    tel = CycleTelemetry(
        enabled=True,
        meta={
            "algo": "self_distill_sft",
            "lambda_traj": cfg.lambda_traj,
            "lambda_anchor": cfg.lambda_anchor,
            "dropout": cfg.dropout,
        },
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr)
    history: list[dict[str, Any]] = []
    import random

    rng = random.Random(cfg.seed)
    anchor_records = list(anchor_records or [])

    with bind_telemetry(tel):
        for step in range(cfg.steps):
            opt.zero_grad(set_to_none=True)
            loss_terms: list[torch.Tensor] = []
            # Final-program CE via standard training_loss.
            if distill_records:
                batch = [
                    distill_records[rng.randrange(len(distill_records))]
                    for _ in range(max(1, cfg.batch_size))
                ]
                with timed("final_loss"):
                    loss_terms.append(model.training_loss(batch))
            # Trajectory action loss.
            if traces and cfg.lambda_traj > 0:
                trace = traces[rng.randrange(len(traces))]
                with timed("traj_loss"):
                    traj = trajectory_action_loss(model, trace)
                if traj is not None:
                    loss_terms.append(cfg.lambda_traj * traj)
            # Anchor mix.
            if anchor_records and cfg.lambda_anchor > 0:
                abatch = [
                    anchor_records[rng.randrange(len(anchor_records))]
                    for _ in range(max(1, cfg.batch_size))
                ]
                with timed("anchor_loss"):
                    loss_terms.append(cfg.lambda_anchor * model.training_loss(abatch))
            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.trainable_parameters()), cfg.max_grad_norm
            )
            opt.step()
            history.append({"step": step + 1, "loss": float(loss.detach().cpu())})

    model.save(out_dir / "model.pt")
    summary = {
        "algo": "self_distill_sft",
        "steps": cfg.steps,
        "n_traces": len(traces),
        "n_distill_records": len(distill_records),
        "n_anchor_records": len(anchor_records),
        "lambda_traj": cfg.lambda_traj,
        "lambda_anchor": cfg.lambda_anchor,
        "dropout": cfg.dropout,
        "history": history[-50:],
        "telemetry": tel.summary(),
    }
    (out_dir / "distill_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def train_self_distill_from_paths(
    checkpoint: Path,
    traces_dir: Path,
    *,
    out_dir: Path,
    anchor_train_dir: Path | None = None,
    steps: int = 100,
    device: str = "cpu",
    budget: int = 500,
    lambda_traj: float = 1.0,
    lambda_anchor: float = 0.3,
    dropout: float = 0.0,
) -> dict[str, Any]:
    from slm_training.distill.select import SelectConfig, select_traces
    from slm_training.distill.trace_store import TraceStore
    from slm_training.harnesses.model_build.data import load_train_records

    model = TwoTowerModel.from_checkpoint(checkpoint, device=device)
    store = TraceStore(traces_dir)
    selected = select_traces(
        store.iter_traces(),
        config=SelectConfig(budget=budget, corpus="self_distilled_success"),
    )
    anchor = load_train_records(anchor_train_dir) if anchor_train_dir else []
    return train_self_distill(
        model,
        selected,
        anchor_records=anchor,
        config=DistillSFTConfig(
            steps=steps,
            lambda_traj=lambda_traj,
            lambda_anchor=lambda_anchor,
            dropout=dropout,
        ),
        out_dir=out_dir,
    )
