"""Training loop for ModelPlugin implementations."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import batched, load_train_records
from slm_training.harnesses.model_build.factory import build_model


def train(config: ModelBuildConfig, model=None) -> dict:
    records = load_train_records(config.train_dir)
    if not records:
        raise ValueError("train records empty")

    rng = random.Random(config.seed)
    records = list(records)
    rng.shuffle(records)

    plugin = model or build_model(config, records)
    run_dir = config.run_dir
    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    batches = batched(records, config.batch_size)
    if not batches:
        raise ValueError("no batches")

    optimizer = None
    is_twotower = hasattr(plugin, "training_loss")
    if is_twotower:
        import torch

        optimizer = torch.optim.AdamW(
            plugin.trainable_parameters(),
            lr=config.lr,
        )

    step = 0
    last_loss = 0.0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < config.steps:
            for batch in batches:
                if step >= config.steps:
                    break
                if is_twotower and optimizer is not None:
                    import torch

                    plugin.train()
                    optimizer.zero_grad(set_to_none=True)
                    loss_t = plugin.training_loss(batch)
                    loss_t.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(plugin.trainable_parameters()), 1.0
                    )
                    optimizer.step()
                    last_loss = float(loss_t.detach().cpu())
                else:
                    last_loss = float(plugin.forward(batch))

                row = {
                    "step": step,
                    "loss": last_loss,
                    "batch_size": len(batch),
                    "model": config.model_name,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                metrics_file.write(json.dumps(row) + "\n")
                step += 1

    ckpt_path = ckpt_dir / "last.pt"
    plugin.save(ckpt_path)

    summary = {
        "run_id": config.run_id,
        "steps": step,
        "last_loss": last_loss,
        "checkpoint": str(ckpt_path.as_posix()),
        "train_dir": str(config.train_dir),
        "record_count": len(records),
        "model": config.model_name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
