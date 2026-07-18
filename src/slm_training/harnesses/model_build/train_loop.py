"""Training loop for ModelPlugin implementations."""

from __future__ import annotations

import json
import math
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import batched, load_train_records
from slm_training.harnesses.model_build.factory import build_model
from slm_training.runtime.telemetry import CycleTelemetry, bind_telemetry, timed


def _parse_eval_suites(config: ModelBuildConfig) -> list[str]:
    raw = str(getattr(config, "eval_suites", "") or "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    suite = str(getattr(config, "eval_suite", "smoke") or "smoke")
    return [suite]


def _ship_score(metrics: dict) -> float | None:
    """Composite generated-eval score used for best_ship_score.pt selection.

    Matches the grammar-matrix successive-halving weights so NLL-best and
    ship-best checkpoints remain independently trackable.
    """
    keys = (
        ("meaningful_program_rate", 2.0),
        ("placeholder_fidelity", 2.0),
        ("structural_similarity", 1.0),
        ("reward_score", 0.5),
    )
    total = 0.0
    weight = 0.0
    for key, w in keys:
        value = metrics.get(key)
        if value is None:
            continue
        total += w * float(value)
        weight += w
    return total / weight if weight else None


def _clip_optimizer_parameter_groups(optimizer, max_norm: float) -> None:
    """Clip independent optimizer groups without coupling auxiliary heads."""
    import torch

    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group["params"], max_norm)


def train(config: ModelBuildConfig, model=None) -> dict:
    from slm_training.runtime.accel import (
        autocast_context,
        detect_device,
        grad_scaler,
        maybe_compile,
        sync_device,
    )
    from slm_training.harnesses.model_build.full_state import (
        data_manifest_sha,
        load_full_state,
        restore_rng_states,
        save_full_state,
    )

    max_wall_minutes = getattr(config, "max_wall_minutes", None)
    if max_wall_minutes is not None and not 0 < float(max_wall_minutes) <= 5:
        raise ValueError("max_wall_minutes must be positive and at most 5")
    wall_started = time.monotonic()
    wall_deadline = (
        wall_started + float(max_wall_minutes) * 60
        if max_wall_minutes is not None
        else None
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
        from slm_training.harnesses.quality import apply_curriculum_tags

        records = apply_curriculum_tags(records, sanitize=True)
    rng.shuffle(records)
    records_by_id = {r.id: r for r in records}

    plugin = model or build_model(config, records)
    if int(getattr(config, "retrieval_k", 0) or 0) > 0 and hasattr(
        plugin, "skeleton_bank"
    ):
        from slm_training.harnesses.quality import build_skeleton_bank

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
    nll_history: list[dict] = []
    tel = CycleTelemetry(
        enabled=bool(getattr(config, "telemetry", True)),
        meta={
            "run_id": config.run_id,
            "device": config.device,
            "model": config.model_name,
            "context_backend": getattr(config, "context_backend", None),
            "batch_size": int(config.batch_size),
            "grad_accum": int(getattr(config, "grad_accum_steps", 1) or 1),
            "effective_batch_size": int(config.batch_size)
            * int(getattr(config, "grad_accum_steps", 1) or 1),
            "max_wall_minutes": max_wall_minutes,
        },
    )
    mix_curriculum = bool(getattr(config, "mix_curriculum", True))
    curriculum_pools = None
    if getattr(config, "use_curriculum", False):
        from slm_training.harnesses.quality import index_curriculum_stages

        curriculum_pools = index_curriculum_stages(records)

    mixture_weights: dict[str, float] | None = None
    mixture_task_weights: dict[str, float] | None = None
    mixture_meta: dict | None = None
    family_pools = None
    task_family_pools = None
    mixture_path = getattr(config, "mixture_manifest", None)
    if mixture_path:
        from slm_training.data.mixture import (
            index_family_pools,
            index_task_family_pools,
            load_mixture_manifest,
            mixture_hash,
        )

        min_quality = float(getattr(config, "mixture_min_quality_score", 0.0) or 0.0)
        if min_quality > 0.0:
            filtered = [
                record
                for record in records
                if float((record.meta or {}).get("quality", {}).get("score") or 0.0)
                >= min_quality
            ]
            if not filtered:
                raise ValueError(
                    f"mixture_min_quality_score={min_quality:g} removed all records"
                )
            records = filtered
        manifest = load_mixture_manifest(mixture_path)
        mixture_weights = dict(manifest.weights)
        mixture_task_weights = dict(manifest.task_weights or {}) or None
        mixture_sampling_policy = str(
            getattr(config, "mixture_sampling_policy", "with_replacement")
        )
        manifest_hash = mixture_hash(manifest)
        effective_hash = manifest_hash
        if mixture_sampling_policy != "with_replacement":
            import hashlib

            effective_hash = hashlib.sha256(
                f"{manifest_hash}:{mixture_sampling_policy}".encode()
            ).hexdigest()
        mixture_meta = {
            "mixture_id": manifest.mixture_id,
            "weights": mixture_weights,
            "task_weights": mixture_task_weights,
            "hash": effective_hash,
            "manifest_hash": manifest_hash,
            "sampling_policy": mixture_sampling_policy,
            "path": str(mixture_path),
            "min_quality_score": min_quality,
            "filtered_record_count": len(records),
        }
        family_pools = index_family_pools(records)
        if mixture_task_weights:
            task_family_pools = index_task_family_pools(records)

    def _batches_for_step(step: int) -> list[list]:
        if mixture_weights is not None:
            from slm_training.data.mixture import sample_mixture_batch

            target = config.batch_size * 8
            drawn = sample_mixture_batch(
                records,
                weights=mixture_weights,
                batch_size=target,
                rng=rng,
                pools=family_pools,
                task_weights=mixture_task_weights,
                task_pools=task_family_pools,
                sampling_policy=mixture_sampling_policy,
            )
            return batched(drawn, config.batch_size)
        if getattr(config, "use_curriculum", False):
            from slm_training.harnesses.quality import sample_curriculum_batch

            # Generate only the bounded batch window consumed before refresh.
            target = config.batch_size * 8
            drawn = sample_curriculum_batch(
                records,
                batch_size=target,
                step=step,
                total_steps=config.steps,
                rng=rng,
                mix=mix_curriculum,
                stage_pools=curriculum_pools,
            )
            return batched(drawn, config.batch_size)
        shuffled = list(records)
        rng.shuffle(shuffled)
        return batched(shuffled, config.batch_size)

    optimizer = None
    is_twotower = hasattr(plugin, "training_loss")
    supports_loss_suites = all(
        hasattr(plugin, name)
        for name in ("_encode_openui", "_format_one_context", "_encode_context")
    )
    scaler = None
    use_amp = bool(getattr(config, "use_amp", False)) and accel.amp
    grad_accum = max(1, int(getattr(config, "grad_accum_steps", 1) or 1))
    if is_twotower:
        import torch

        parameters = (
            plugin.optimizer_parameter_groups()
            if hasattr(plugin, "optimizer_parameter_groups")
            else plugin.trainable_parameters()
        )
        optimizer = torch.optim.AdamW(parameters, lr=config.lr)
        scaler = grad_scaler(config.device, enabled=use_amp)

    # ── Token accounting / full-state resume ────────────────────────────────
    step = 0
    last_loss = 0.0
    micro = 0
    accum_loss_sum = 0.0
    accum_loss_count = 0
    accum_batch_meta: list[dict] = []
    accum_example_losses: list[float] = []
    seen_prompt_tokens = 0
    seen_target_tokens = 0
    best_weighted_nll = math.inf
    best_ship_score = -math.inf
    pending: list[list] = []
    manifest_sha = data_manifest_sha(config.train_dir)
    resumed_from: str | None = None

    resume_path = getattr(config, "resume_from", None)
    if resume_path:
        resume_path = Path(resume_path)
        payload = load_full_state(resume_path)
        previous_mixture_hash = payload.get("mixture_hash")
        current_mixture_hash = (mixture_meta or {}).get("hash")
        if previous_mixture_hash and previous_mixture_hash != current_mixture_hash:
            raise ValueError(
                "resume_from mixture mismatch: checkpoint used "
                f"{previous_mixture_hash[:12]}… but current policy uses "
                f"{str(current_mixture_hash)[:12]}…"
            )
        prev_sha = payload.get("data_manifest_sha")
        if prev_sha and manifest_sha and prev_sha != manifest_sha:
            raise ValueError(
                "resume_from data mismatch: checkpoint was trained on "
                f"manifest {prev_sha[:12]}… but train_dir has {manifest_sha[:12]}…"
            )
        if payload.get("model") is not None and hasattr(plugin, "load_state_dict"):
            if hasattr(plugin, "_state_dict_for_checkpoint"):
                # Reject silent trainable-weight mismatches (TwoTower-style).
                from slm_training.models.twotower import _load_checkpoint_state

                _load_checkpoint_state(plugin, payload["model"])
            else:
                plugin.load_state_dict(payload["model"], strict=False)
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if (
            scaler is not None
            and payload.get("scaler") is not None
            and hasattr(scaler, "load_state_dict")
        ):
            scaler.load_state_dict(payload["scaler"])
        restore_rng_states(payload, plugin=plugin, loop_rng=rng)
        step = int(payload.get("step") or 0)
        seen_prompt_tokens = int(payload.get("seen_prompt_tokens") or 0)
        seen_target_tokens = int(payload.get("seen_target_tokens") or 0)
        if payload.get("best_weighted_nll") is not None:
            best_weighted_nll = float(payload["best_weighted_nll"])
        if payload.get("best_ship_score") is not None:
            best_ship_score = float(payload["best_ship_score"])
        pending = []
        for batch_ids in payload.get("pending_batch_ids") or []:
            batch = []
            for rid in batch_ids:
                record = records_by_id.get(rid)
                if record is None:
                    raise ValueError(f"resume_from pending record missing: {rid!r}")
                batch.append(record)
            if batch:
                pending.append(batch)
        resumed_from = str(resume_path)

    def _count_tokens(batch: list) -> None:
        nonlocal seen_prompt_tokens, seen_target_tokens
        if hasattr(plugin, "count_batch_tokens"):
            pt, tt = plugin.count_batch_tokens(batch)
            seen_prompt_tokens += int(pt)
            seen_target_tokens += int(tt)

    def _batch_meta(batch: list) -> list[dict]:
        return [
            {
                "id": str(record.id),
                "source": str(record.source),
                "source_family": str((record.meta or {}).get("source_family") or record.source),
                "prompt_chars": len(record.prompt),
                "target_chars": len(record.openui),
            }
            for record in batch
        ]

    def _budget_exhausted() -> bool:
        budget = getattr(config, "target_token_budget", None)
        return (
            budget is not None
            and int(budget) > 0
            and (seen_target_tokens >= int(budget))
        )

    def _wall_budget_exhausted() -> bool:
        return wall_deadline is not None and time.monotonic() >= wall_deadline

    def _save_full_state_now() -> None:
        if not is_twotower or not bool(getattr(config, "full_state_checkpoint", True)):
            return
        with timed("full_state_save"):
            save_full_state(
                ckpt_dir / "last_full_state.pt",
                plugin=plugin,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                seen_prompt_tokens=seen_prompt_tokens,
                seen_target_tokens=seen_target_tokens,
                loop_rng=rng,
                pending_batches=pending,
                config=config,
                manifest_sha=manifest_sha,
                best_weighted_nll=(
                    None if math.isinf(best_weighted_nll) else best_weighted_nll
                ),
                best_ship_score=(
                    None if math.isinf(best_ship_score) else best_ship_score
                ),
                mixture_hash=(mixture_meta or {}).get("hash") if mixture_meta else None,
            )

    def _maybe_eval(step: int, force: bool = False) -> dict | None:
        nonlocal best_ship_score
        if config.test_dir is None:
            return None
        if eval_history and eval_history[-1].get("step") == step:
            return eval_history[-1]
        if not force and (
            config.eval_every <= 0 or step <= 0 or step % config.eval_every != 0
        ):
            return None
        from slm_training.harnesses.model_build.eval_runner import evaluate_suites

        mid_ckpt = ckpt_dir / "last.pt"
        with timed("eval_save_ckpt"):
            plugin.save(mid_ckpt)
        suites = _parse_eval_suites(config)
        ship: float | None = None
        with timed("eval_suites"):
            scoreboard = evaluate_suites(config, suites, model=plugin)
            if len(suites) == 1:
                metrics = scoreboard["suites"][suites[0]]
                ship = _ship_score(metrics)
                row = {
                    "step": step,
                    "suite": suites[0],
                    "parse_rate": metrics.get("parse_rate"),
                    "meaningful_program_rate": metrics.get(
                        "meaningful_program_rate"
                    ),
                    "placeholder_fidelity": metrics.get("placeholder_fidelity"),
                    "structural_similarity": metrics.get("structural_similarity"),
                    "reward_score": metrics.get("reward_score"),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            else:
                board = {
                    suite: {
                        "parse_rate": metrics.get("parse_rate"),
                        "meaningful_program_rate": metrics.get(
                            "meaningful_program_rate"
                        ),
                        "placeholder_fidelity": metrics.get("placeholder_fidelity"),
                        "structural_similarity": metrics.get("structural_similarity"),
                        "reward_score": metrics.get("reward_score"),
                    }
                    for suite, metrics in scoreboard["suites"].items()
                }
                # Mean of per-suite ship scores keeps multi-suite boards comparable.
                scores = [
                    s for s in (_ship_score(m) for m in board.values()) if s is not None
                ]
                ship = sum(scores) / len(scores) if scores else None
                row = {
                    "step": step,
                    "suites": suites,
                    "board": board,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
        if ship is not None and ship > best_ship_score:
            best_ship_score = float(ship)
            with timed("ship_best_ckpt"):
                plugin.save(ckpt_dir / "best_ship_score.pt")
            row["ship_score"] = best_ship_score
            row["ship_best"] = True
        elif ship is not None:
            row["ship_score"] = float(ship)
            row["ship_best"] = False
        eval_history.append(row)
        (run_dir / "eval_history.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in eval_history),
            encoding="utf-8",
        )
        return row

    def _maybe_loss_eval(step: int, force: bool = False) -> dict | None:
        """Deterministic denoising-NLL suites (cheap teacher-forced signal)."""
        nonlocal best_weighted_nll
        if config.test_dir is None or not is_twotower or not supports_loss_suites:
            return None
        if nll_history and nll_history[-1].get("step") == step:
            return nll_history[-1]
        every = int(getattr(config, "loss_eval_every", 0) or 0)
        if not force and (every <= 0 or step <= 0 or step % every != 0):
            return None
        if not (hasattr(plugin, "denoiser") and hasattr(plugin, "tokenizer")):
            return None
        from slm_training.evals.denoising_nll import DenoisingNLLConfig
        from slm_training.evals.loss_suites import (
            evaluate_loss_suites,
            write_loss_suite_report,
        )
        from slm_training.harnesses.model_build.data import load_suite_records

        base_suite = "held_out"
        try:
            load_suite_records(config.test_dir, base_suite)
        except FileNotFoundError:
            base_suite = _parse_eval_suites(config)[0]
        nll_cfg = DenoisingNLLConfig(
            suite_version=str(getattr(config, "loss_suite_version", "v1") or "v1"),
            mask_seed=int(getattr(config, "loss_mask_seed", 0) or 0),
        )
        with timed("loss_suites"):
            report = evaluate_loss_suites(
                plugin,
                config.test_dir,
                nll_config=nll_cfg,
                base_suite=base_suite,
            )
            write_loss_suite_report(run_dir / f"loss_suites_step_{step}.json", report)
        aggregate = report.get("aggregate") or {}
        broad = (report.get("categories") or {}).get("broad") or {}
        row = {
            "step": step,
            "weighted_nll": aggregate.get("weighted_nll"),
            "complete": aggregate.get("complete"),
            "missing_categories": aggregate.get("missing_categories"),
            "broad_mean_nll": (broad.get("aggregate") or {}).get("mean_nll"),
            "broad_constraint_rescue_gap": (broad.get("aggregate") or {}).get(
                "constraint_rescue_gap"
            ),
            "bits_per_char": broad.get("bits_per_char"),
            "base_suite": base_suite,
            "seen_target_tokens": seen_target_tokens,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        nll_history.append(row)
        (run_dir / "nll_history.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in nll_history),
            encoding="utf-8",
        )
        (run_dir / "loss_suites.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        weighted = aggregate.get("weighted_nll")
        if weighted is not None and float(weighted) < best_weighted_nll:
            best_weighted_nll = float(weighted)
            with timed("loss_best_ckpt"):
                plugin.save(ckpt_dir / "best_weighted_nll.pt")
        return row

    stopped_on = "steps"
    mode = "a" if resumed_from else "w"
    with bind_telemetry(tel), metrics_path.open(mode, encoding="utf-8") as metrics_file:
        while step < config.steps:
            if _wall_budget_exhausted():
                stopped_on = "wall_time_budget"
                break
            if _budget_exhausted():
                stopped_on = "token_budget"
                break
            if not pending:
                with timed("batch_build"):
                    pending = _batches_for_step(step)
                if not pending:
                    raise ValueError("no batches")
            batch = pending.pop(0)
            if is_twotower and optimizer is not None:
                import torch

                plugin.train()
                if micro == 0:
                    optimizer.zero_grad(set_to_none=True)
                with timed("forward"):
                    with autocast_context(config.device, enabled=use_amp):
                        raw_loss_t = plugin.training_loss(batch)
                        auxiliary_loss_t = (
                            plugin.take_detached_auxiliary_loss()
                            if hasattr(plugin, "take_detached_auxiliary_loss")
                            else None
                        )
                        loss_t = raw_loss_t / grad_accum
                with timed("backward"):
                    scaler.scale(loss_t).backward()
                    if auxiliary_loss_t is not None:
                        scaler.scale(auxiliary_loss_t / grad_accum).backward()
                _count_tokens(batch)
                reported_loss_t = raw_loss_t.detach()
                if auxiliary_loss_t is not None:
                    reported_loss_t = reported_loss_t + auxiliary_loss_t.detach()
                accum_loss_sum += float(reported_loss_t.cpu())
                accum_loss_count += 1
                accum_batch_meta.extend(_batch_meta(batch))
                accum_example_losses.extend(
                    float(value)
                    for value in (getattr(plugin, "_last_example_token_losses", None) or [])
                )
                micro += 1
                if micro >= grad_accum:
                    with timed("optim_step"):
                        scaler.unscale_(optimizer)
                        _clip_optimizer_parameter_groups(optimizer, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    micro = 0
                    last_loss = accum_loss_sum / max(1, accum_loss_count)
                    accum_loss_sum = 0.0
                    accum_loss_count = 0
                    step += 1
                    row = {
                        "step": step,
                        "loss": last_loss,
                        "batch_size": len(batch) * grad_accum,
                        "seen_prompt_tokens": seen_prompt_tokens,
                        "seen_target_tokens": seen_target_tokens,
                        "model": config.model_name,
                        "device": config.device,
                        "amp": use_amp,
                        "compile": use_compile,
                        "grad_accum": grad_accum,
                        "batches": accum_batch_meta,
                        "example_token_loss_proxy": accum_example_losses,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    extra_metrics = getattr(plugin, "last_training_metrics", None)
                    if isinstance(extra_metrics, dict):
                        row.update(extra_metrics)
                    accum_batch_meta = []
                    accum_example_losses = []
                    metrics_file.write(json.dumps(row) + "\n")
                    metrics_file.flush()
                    did_eval = _maybe_eval(step)
                    did_loss_eval = _maybe_loss_eval(step)
                    if did_eval or did_loss_eval:
                        _save_full_state_now()
            else:
                with timed("forward"):
                    last_loss = float(plugin.forward(batch))
                _count_tokens(batch)
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
                _clip_optimizer_parameter_groups(optimizer, 1.0)
                scaler.step(optimizer)
                scaler.update()
            micro = 0

        with timed("device_sync"):
            sync_device(config.device)
        ckpt_path = ckpt_dir / "last.pt"
        with timed("final_save"):
            plugin.save(ckpt_path)
        final_eval = None
        final_loss_eval = None
        if not _wall_budget_exhausted():
            final_eval = _maybe_eval(
                step, force=bool(config.test_dir and config.eval_every > 0)
            )
            final_loss_eval = _maybe_loss_eval(
                step,
                # Cadence 0 disables intermediate checks, but never disables the
                # final feedback artifact for a testable TwoTower run.
                force=bool(config.test_dir),
            )
        _save_full_state_now()

    if bool(getattr(config, "register_promoted", False)):
        from slm_training.harnesses.experiments.promotion import (
            register_promoted_checkpoint,
        )

        source = ckpt_dir / "best_weighted_nll.pt"
        if not source.exists():
            source = ckpt_dir / "best_ship_score.pt"
        if not source.exists():
            source = ckpt_path
        register_promoted_checkpoint(
            ckpt_dir,
            source=source,
            meta={
                "step": step,
                "best_weighted_nll": (
                    None if math.isinf(best_weighted_nll) else best_weighted_nll
                ),
                "best_ship_score": (
                    None if math.isinf(best_ship_score) else best_ship_score
                ),
                "mixture": mixture_meta,
            },
        )

    trainable_params: int | None = None
    frozen_params: int | None = None
    if hasattr(plugin, "parameters"):
        try:
            trainable_params = sum(
                p.numel() for p in plugin.parameters() if p.requires_grad
            )
            frozen_params = sum(
                p.numel() for p in plugin.parameters() if not p.requires_grad
            )
        except Exception:  # noqa: BLE001
            trainable_params = None
            frozen_params = None

    tel_path = tel.write(run_dir / "train_telemetry.json")
    effective_plugin_config = getattr(plugin, "config", config)
    summary = {
        "run_id": config.run_id,
        "steps": step,
        "stopped_on": stopped_on,
        "last_loss": last_loss,
        "checkpoint": str(ckpt_path.as_posix()),
        "train_dir": str(config.train_dir),
        "record_count": len(records),
        "model": config.model_name,
        "device": config.device,
        "seen_prompt_tokens": seen_prompt_tokens,
        "seen_target_tokens": seen_target_tokens,
        "target_token_budget": getattr(config, "target_token_budget", None),
        "max_wall_minutes": max_wall_minutes,
        "elapsed_wall_seconds": time.monotonic() - wall_started,
        "resumed_from": resumed_from,
        "data_manifest_sha": manifest_sha,
        # Scratch-context and frozen-HF runs are different scientific tracks —
        # never pool their results on one scaling curve.
        "track": {
            "context_backend": getattr(config, "context_backend", None),
            "freeze_context": bool(getattr(config, "freeze_context", False)),
            "hf_model_name": (
                getattr(config, "hf_model_name", None)
                if str(getattr(config, "context_backend", "")).lower() == "hf"
                else None
            ),
            "output_tokenizer": getattr(config, "output_tokenizer", None),
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "tokens_per_trainable_param": (
                seen_target_tokens / trainable_params if trainable_params else None
            ),
        },
        "accel": {
            "backend": accel.backend,
            "amp": use_amp,
            "compile": use_compile,
            "grad_accum": grad_accum,
            "effective_batch_size": int(config.batch_size) * grad_accum,
            "num_threads": accel.num_threads,
            "note": accel.note,
        },
        "curriculum": {
            "enabled": bool(getattr(config, "use_curriculum", False)),
            "mix": mix_curriculum,
        },
        "recipe": {
            "learning_rate": config.lr,
            "seed": config.seed,
            "steps_requested": config.steps,
            "batch_size": config.batch_size,
            "ltr_loss_weight": getattr(config, "ltr_loss_weight", 0.0),
            "compiler_alignment_loss_weight": getattr(
                config, "compiler_alignment_loss_weight", 0.0
            ),
            "compiler_alignment_margin": getattr(
                config, "compiler_alignment_margin", 0.0
            ),
            "compiler_alignment_stratified": bool(
                getattr(config, "compiler_alignment_stratified", False)
            ),
            "compiler_alignment_semantic_exhaustive": bool(
                getattr(config, "compiler_alignment_semantic_exhaustive", False)
            ),
            "component_inventory_loss_weight": getattr(
                config, "component_inventory_loss_weight", 0.0
            ),
            "component_inventory_decode_weight": getattr(
                config, "component_inventory_decode_weight", 0.0
            ),
            "component_plan_loss_weight": getattr(
                config, "component_plan_loss_weight", 0.0
            ),
            "component_plan_decode_weight": getattr(
                config, "component_plan_decode_weight", 0.0
            ),
            "component_plan_attention_pool": bool(
                getattr(config, "component_plan_attention_pool", False)
            ),
            "component_plan_token_pool": bool(
                getattr(config, "component_plan_token_pool", False)
            ),
            "slot_component_loss_weight": getattr(
                config, "slot_component_loss_weight", 0.0
            ),
            "slot_component_focal_gamma": getattr(
                config, "slot_component_focal_gamma", 0.0
            ),
            "slot_component_class_balance_power": getattr(
                config, "slot_component_class_balance_power", 0.0
            ),
            "slot_component_decode_weight": getattr(
                config, "slot_component_decode_weight", 0.0
            ),
            "slot_component_prompt_context": bool(
                getattr(config, "slot_component_prompt_context", True)
            ),
            "slot_component_next_context": bool(
                getattr(config, "slot_component_next_context", False)
            ),
            "slot_component_pair_interaction": bool(
                getattr(config, "slot_component_pair_interaction", False)
            ),
            "slot_component_lexeme_prior_weight": float(
                getattr(config, "slot_component_lexeme_prior_weight", 0.0) or 0.0
            ),
            "slot_component_span_prior_weight": float(
                getattr(config, "slot_component_span_prior_weight", 0.0) or 0.0
            ),
            "slot_component_content_arity": bool(
                getattr(config, "slot_component_content_arity", False)
            ),
            "component_edge_loss_weight": getattr(
                config, "component_edge_loss_weight", 0.0
            ),
            "component_edge_alignment_loss_weight": getattr(
                config, "component_edge_alignment_loss_weight", 0.0
            ),
            "component_edge_decode_weight": getattr(
                config, "component_edge_decode_weight", 0.0
            ),
            "binder_component_plan_loss_weight": getattr(
                config, "binder_component_plan_loss_weight", 0.0
            ),
            "binder_component_plan_decode_weight": getattr(
                config, "binder_component_plan_decode_weight", 0.0
            ),
            "binder_topology_loss_weight": getattr(
                config, "binder_topology_loss_weight", 0.0
            ),
            "binder_topology_decode_weight": getattr(
                config, "binder_topology_decode_weight", 0.0
            ),
            "binder_arity_loss_weight": getattr(
                config, "binder_arity_loss_weight", 0.0
            ),
            "binder_arity_decode_weight": getattr(
                config, "binder_arity_decode_weight", 0.0
            ),
            "fuse_ltr_loss": bool(getattr(config, "fuse_ltr_loss", True)),
            "fidelity_loss_weight": getattr(config, "fidelity_loss_weight", 0.0),
            "fastpath_aux_weight": getattr(config, "fastpath_aux_weight", 0.0),
            "schema_in_context": bool(getattr(config, "schema_in_context", False)),
            "slot_contract_in_context": bool(
                getattr(config, "slot_contract_in_context", False)
            ),
            "slot_contract_constrained_decode": bool(
                getattr(config, "slot_contract_constrained_decode", False)
            ),
            "retrieval_k": getattr(config, "retrieval_k", 0),
            "grammar_constrained": bool(getattr(config, "grammar_constrained", False)),
            "design_md_dropout": float(
                getattr(effective_plugin_config, "design_md_dropout", 0.0) or 0.0
            ),
            "honesty_mode": (
                "no-design-md-context"
                if not getattr(
                    effective_plugin_config, "design_md_in_context", True
                )
                else "design-md-context"
            ),
        },
        "mixture": mixture_meta,
        "eval_history": eval_history,
        "final_eval": final_eval,
        "nll_history": nll_history,
        "final_loss_eval": final_loss_eval,
        "best_weighted_nll": (
            None if math.isinf(best_weighted_nll) else best_weighted_nll
        ),
        "best_ship_score": (None if math.isinf(best_ship_score) else best_ship_score),
        "telemetry": tel.summary(),
        "telemetry_path": str(tel_path.as_posix()),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Durable remote copy for real full training runs (HF-context track).
    try:
        from slm_training.harnesses.model_build.checkpoint_bucket import (
            maybe_sync_train_checkpoints,
        )

        bucket_report = maybe_sync_train_checkpoints(config, ckpt_dir)
    except Exception as exc:  # noqa: BLE001
        # Surface clearly — full runs must not silently keep checkpoints local-only.
        raise RuntimeError(
            f"checkpoint bucket sync failed for run_id={config.run_id!r}: {exc}"
        ) from exc
    if bucket_report is not None:
        summary["checkpoint_bucket"] = bucket_report
        (run_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "checkpoint_bucket.json").write_text(
            json.dumps(bucket_report, indent=2) + "\n", encoding="utf-8"
        )

    try:
        from slm_training.autoresearch.run_insights import load_run_insights

        load_run_insights(run_dir, run_id=config.run_id)
    except Exception as exc:  # noqa: BLE001 - analysis must never fail training
        warnings.warn(f"run insight analysis failed: {exc}", stacklevel=2)

    return summary
