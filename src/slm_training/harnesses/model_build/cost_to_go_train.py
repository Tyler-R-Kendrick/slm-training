"""Head-only trainer for the VSS3-02 cost-to-go energy scorer (SLM-70).

Freezes the TwoTower backbone and trains only ``cost_to_go_head`` from
replay-verified solver supervision rows.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from slm_training.harnesses.distill.solver_supervision import CandidateCostRow
from slm_training.models.twotower import TwoTowerModel


def _load_rows(path: Path) -> list[CandidateCostRow]:
    rows: list[CandidateCostRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("row_kind") == "candidate_cost":
                rows.append(CandidateCostRow.from_dict(data))
    return rows


def train_cost_to_go(
    model: TwoTowerModel,
    rows: list[CandidateCostRow],
    *,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    """Freeze context/denoiser/base; optimize only ``cost_to_go_head``."""
    if device:
        model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    if model.cost_to_go_head is None:
        raise RuntimeError("model has no cost_to_go_head; set cost_to_go_hidden_dim > 0")
    for p in model.cost_to_go_head.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(model.cost_to_go_head.parameters(), lr=lr)
    losses: list[float] = []
    n = max(1, len(rows))
    rng = torch.Generator()
    rng.manual_seed(int(model.config.seed))

    for step in range(max(1, steps)):
        # Deterministic cyclic mini-batch.
        start = (step * batch_size) % n
        batch_rows = rows[start : start + batch_size]
        if len(batch_rows) < batch_size:
            batch_rows = batch_rows + rows[: batch_size - len(batch_rows)]
        loss = model.cost_to_go_loss(batch_rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    # Restore trainability for non-head params for subsequent stages.
    for p in model.parameters():
        p.requires_grad_(True)
    model.config.cost_to_go_loss_weight = 1.0
    return {
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "last_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "row_count": n,
    }


def train_cost_to_go_from_paths(
    checkpoint: Path | str,
    rows_path: Path | str,
    *,
    out_dir: Path | str,
    steps: int = 40,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and a solver-supervision JSONL, train the head, save result."""
    ckpt = Path(checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = TwoTowerModel.from_checkpoint(ckpt, device=device)
    rows = _load_rows(Path(rows_path))
    if not rows:
        raise ValueError(f"no candidate_cost rows found in {rows_path}")
    summary = train_cost_to_go(
        model, rows, steps=steps, batch_size=batch_size, lr=lr, device=device
    )
    dest = out / "checkpoints" / "last.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(dest)
    meta = {
        "scorer_id": "twotower-cost-to-go-v1",
        "source_checkpoint": str(ckpt),
        "source_rows": str(rows_path),
        "training": summary,
        "config": {k: v for k, v in asdict(model.config).items() if k.startswith("cost_to_go")},
    }
    (out / "cost_to_go_train_summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return {**summary, "checkpoint": str(dest), "summary_path": str(out / "cost_to_go_train_summary.json")}


__all__ = ["train_cost_to_go", "train_cost_to_go_from_paths"]
