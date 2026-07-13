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
    if getattr(config, "use_curriculum", False):
        from slm_training.quality import apply_curriculum_tags

        records = apply_curriculum_tags(records)
    rng.shuffle(records)

    plugin = model or build_model(config, records)
    if int(getattr(config, "retrieval_k", 0) or 0) > 0 and hasattr(plugin, "skeleton_bank"):
        from slm_training.retrieval import build_skeleton_bank

        plugin.skeleton_bank = build_skeleton_bank(records)

    run_dir = config.run_dir
    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    eval_history: list[dict] = []

    def _batches_for_step(step: int) -> list[list]:
        pool = records
        if getattr(config, "use_curriculum", False):
            from slm_training.quality import curriculum_schedule

            stage = curriculum_schedule(step, config.steps)
            staged = [
                r
                for r in records
                if (r.meta or {}).get("curriculum") == stage
            ]
            if staged:
                pool = staged
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return batched(shuffled, config.batch_size)

    batches = _batches_for_step(0)
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

    def _maybe_eval(step: int, force: bool = False) -> dict | None:
        if config.test_dir is None:
            return None
        if not force and (config.eval_every <= 0 or step <= 0 or step % config.eval_every != 0):
            return None
        from dataclasses import replace

        from slm_training.harnesses.model_build.eval_runner import evaluate

        # Persist mid-run checkpoint so evaluate can reload if needed.
        mid_ckpt = ckpt_dir / "last.pt"
        plugin.save(mid_ckpt)
        eval_cfg = replace(config, suite=config.eval_suite)
        metrics = evaluate(eval_cfg, model=plugin, checkpoint=mid_ckpt)
        row = {
            "step": step,
            "parse_rate": metrics.get("parse_rate"),
            "placeholder_fidelity": metrics.get("placeholder_fidelity"),
            "structural_similarity": metrics.get("structural_similarity"),
            "reward_score": metrics.get("reward_score"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        eval_history.append(row)
        (run_dir / "eval_history.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in eval_history),
            encoding="utf-8",
        )
        return row

    step = 0
    last_loss = 0.0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < config.steps:
            batches = _batches_for_step(step)
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
                metrics_file.flush()
                step += 1
                _maybe_eval(step)

    ckpt_path = ckpt_dir / "last.pt"
    plugin.save(ckpt_path)
    final_eval = _maybe_eval(step, force=bool(config.test_dir and config.eval_every > 0))

    summary = {
        "run_id": config.run_id,
        "steps": step,
        "last_loss": last_loss,
        "checkpoint": str(ckpt_path.as_posix()),
        "train_dir": str(config.train_dir),
        "record_count": len(records),
        "model": config.model_name,
        "eval_history": eval_history,
        "final_eval": final_eval,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
