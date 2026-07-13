"""Training loop for ModelPlugin implementations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import batched, load_train_records
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel


def train(
    config: ModelBuildConfig,
    model: ModelPlugin | None = None,
) -> dict:
    records = load_train_records(config.train_dir)
    if not records:
        raise ValueError("train records empty")

    plugin: ModelPlugin = model or StubModel(
        noise_rate=config.noise_rate,
        seed=config.seed,
    )

    run_dir = config.run_dir
    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    batches = batched(records, config.batch_size)
    if not batches:
        raise ValueError("no batches")

    step = 0
    last_loss = 0.0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < config.steps:
            for batch in batches:
                if step >= config.steps:
                    break
                last_loss = float(plugin.forward(batch))
                row = {
                    "step": step,
                    "loss": last_loss,
                    "batch_size": len(batch),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                metrics_file.write(json.dumps(row) + "\n")
                step += 1

    ckpt_path = ckpt_dir / "last.pt"
    plugin.save(ckpt_path)

    # Convenience pointer for scripts using --run-id latest
    summary = {
        "run_id": config.run_id,
        "steps": step,
        "last_loss": last_loss,
        "checkpoint": str(ckpt_path.as_posix()),
        "train_dir": str(config.train_dir),
        "record_count": len(records),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
