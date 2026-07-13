"""Training loop for ModelPlugin implementations."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import batched, load_train_records
from slm_training.harnesses.model_build.factory import build_model
from slm_training.telemetry import CycleTelemetry, bind_telemetry, timed


def _parse_eval_suites(config: ModelBuildConfig) -> list[str]:
    raw = str(getattr(config, "eval_suites", "") or "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    suite = str(getattr(config, "eval_suite", "smoke") or "smoke")
    return [suite]


def train(config: ModelBuildConfig, model=None) -> dict:
    from slm_training.accel import (
        autocast_context,
        detect_device,
        grad_scaler,
        maybe_compile,
        sync_device,
    )

    accel = detect_device(config.device)
    # Honor explicit device but adopt accel threading / amp defaults.
    if config.device in {"auto", "best"}:
        config.device = accel.device

    records = load_train_records(config.train_dir)
    if not records:
        raise ValueError("train records empty")

    rng = random.Random(config.seed)
    records = list(records)
    if getattr(config, "use_curriculum", False):
        from slm_training.quality import apply_curriculum_tags

        records = apply_curriculum_tags(records, sanitize=True)
    rng.shuffle(records)

    plugin = model or build_model(config, records)
    if int(getattr(config, "retrieval_k", 0) or 0) > 0 and hasattr(plugin, "skeleton_bank"):
        from slm_training.retrieval import build_skeleton_bank

        plugin.skeleton_bank = build_skeleton_bank(records)

    use_compile = bool(getattr(config, "use_compile", False))
    if use_compile and hasattr(plugin, "denoiser"):
        plugin.denoiser = maybe_compile(
            plugin.denoiser,
            enabled=True,
            mode=str(getattr(config, "compile_mode", "default") or "default"),
        )

    run_dir = config.run_dir
    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    eval_history: list[dict] = []
    tel = CycleTelemetry(
        enabled=bool(getattr(config, "telemetry", True)),
        meta={
            "run_id": config.run_id,
            "device": config.device,
            "model": config.model_name,
            "context_backend": getattr(config, "context_backend", None),
        },
    )
    mix_curriculum = bool(getattr(config, "mix_curriculum", True))

    def _batches_for_step(step: int) -> list[list]:
        if getattr(config, "use_curriculum", False):
            from slm_training.quality import sample_curriculum_batch

            # One shuffled epoch worth of mixed batches.
            drawn: list = []
            target = max(config.batch_size, len(records))
            while len(drawn) < target:
                drawn.extend(
                    sample_curriculum_batch(
                        records,
                        batch_size=config.batch_size,
                        step=step,
                        total_steps=config.steps,
                        rng=rng,
                        mix=mix_curriculum,
                    )
                )
            return batched(drawn[: max(config.batch_size * 8, config.batch_size)], config.batch_size)
        shuffled = list(records)
        rng.shuffle(shuffled)
        return batched(shuffled, config.batch_size)

    batches = _batches_for_step(0)
    if not batches:
        raise ValueError("no batches")

    optimizer = None
    is_twotower = hasattr(plugin, "training_loss")
    scaler = None
    use_amp = bool(getattr(config, "use_amp", False)) and accel.amp
    grad_accum = max(1, int(getattr(config, "grad_accum_steps", 1) or 1))
    if is_twotower:
        import torch

        optimizer = torch.optim.AdamW(
            plugin.trainable_parameters(),
            lr=config.lr,
        )
        scaler = grad_scaler(config.device, enabled=use_amp)

    def _maybe_eval(step: int, force: bool = False) -> dict | None:
        if config.test_dir is None:
            return None
        if not force and (config.eval_every <= 0 or step <= 0 or step % config.eval_every != 0):
            return None
        from dataclasses import replace

        from slm_training.harnesses.model_build.eval_runner import evaluate

        mid_ckpt = ckpt_dir / "last.pt"
        with timed("eval_save_ckpt"):
            plugin.save(mid_ckpt)
        suites = _parse_eval_suites(config)
        with timed("eval_suites"):
            if len(suites) == 1:
                eval_cfg = replace(config, suite=suites[0])
                metrics = evaluate(eval_cfg, model=plugin, checkpoint=mid_ckpt)
                row = {
                    "step": step,
                    "suite": suites[0],
                    "parse_rate": metrics.get("parse_rate"),
                    "placeholder_fidelity": metrics.get("placeholder_fidelity"),
                    "structural_similarity": metrics.get("structural_similarity"),
                    "reward_score": metrics.get("reward_score"),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            else:
                board: dict[str, dict] = {}
                for suite in suites:
                    eval_cfg = replace(config, suite=suite)
                    metrics = evaluate(eval_cfg, model=plugin, checkpoint=mid_ckpt)
                    board[suite] = {
                        "parse_rate": metrics.get("parse_rate"),
                        "placeholder_fidelity": metrics.get("placeholder_fidelity"),
                        "structural_similarity": metrics.get("structural_similarity"),
                        "reward_score": metrics.get("reward_score"),
                    }
                row = {
                    "step": step,
                    "suites": suites,
                    "board": board,
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
    micro = 0
    with bind_telemetry(tel), metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < config.steps:
            with timed("batch_build"):
                batches = _batches_for_step(step)
            for batch in batches:
                if step >= config.steps:
                    break
                if is_twotower and optimizer is not None:
                    import torch

                    plugin.train()
                    if micro == 0:
                        optimizer.zero_grad(set_to_none=True)
                    with timed("forward"):
                        with autocast_context(config.device, enabled=use_amp):
                            loss_t = plugin.training_loss(batch) / grad_accum
                    with timed("backward"):
                        scaler.scale(loss_t).backward()
                    micro += 1
                    if micro >= grad_accum:
                        with timed("optim_step"):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                list(plugin.trainable_parameters()), 1.0
                            )
                            scaler.step(optimizer)
                            scaler.update()
                        micro = 0
                        last_loss = float(loss_t.detach().cpu()) * grad_accum
                        step += 1
                        row = {
                            "step": step,
                            "loss": last_loss,
                            "batch_size": len(batch) * grad_accum,
                            "model": config.model_name,
                            "device": config.device,
                            "amp": use_amp,
                            "compile": use_compile,
                            "grad_accum": grad_accum,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                        metrics_file.write(json.dumps(row) + "\n")
                        metrics_file.flush()
                        _maybe_eval(step)
                else:
                    with timed("forward"):
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

        # Flush partial accum.
        if is_twotower and optimizer is not None and micro > 0:
            import torch

            with timed("optim_step"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(plugin.trainable_parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()

        with timed("device_sync"):
            sync_device(config.device)
        ckpt_path = ckpt_dir / "last.pt"
        with timed("final_save"):
            plugin.save(ckpt_path)
        final_eval = _maybe_eval(step, force=bool(config.test_dir and config.eval_every > 0))

    tel_path = tel.write(run_dir / "train_telemetry.json")
    summary = {
        "run_id": config.run_id,
        "steps": step,
        "last_loss": last_loss,
        "checkpoint": str(ckpt_path.as_posix()),
        "train_dir": str(config.train_dir),
        "record_count": len(records),
        "model": config.model_name,
        "device": config.device,
        "accel": {
            "backend": accel.backend,
            "amp": use_amp,
            "compile": use_compile,
            "grad_accum": grad_accum,
            "num_threads": accel.num_threads,
            "note": accel.note,
        },
        "curriculum": {
            "enabled": bool(getattr(config, "use_curriculum", False)),
            "mix": mix_curriculum,
        },
        "eval_history": eval_history,
        "final_eval": final_eval,
        "telemetry": tel.summary(),
        "telemetry_path": str(tel_path.as_posix()),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
