"""Training loop for ModelPlugin implementations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.data import batched, load_train_records
from slm_training.harnesses.model_build.factory import build_model
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.runtime.telemetry import (
    CycleTelemetry,
    bind_telemetry,
    current_trace,
    timed,
)


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


def _strict_root_reference_identity_records(records, tokenizer) -> list:
    """Find records whose terminal root uses a nonempty strict section subset."""
    from slm_training.models.choice_tokenizer import (
        structural_root_reference_identity_target,
    )

    strict = []
    for record in records:
        token_ids = tokenizer.encode(
            record.openui, placeholders=list(record.placeholders or ())
        )
        target = structural_root_reference_identity_target(
            tokenizer,
            token_ids,
            slot_count=len(record.placeholders or ()),
        )
        if target is None:
            continue
        references, section_count = target
        if references and len(references) < section_count:
            strict.append(record)
    return strict


def _rare_slot_component_owner_records(
    records, owner_for_source, threshold: int
) -> tuple[list, dict[str, int], list[str]]:
    """Find records containing visible slot-owner labels below a corpus ceiling."""
    from collections import Counter

    counts: Counter[str] = Counter()
    owners_by_record: list[set[str]] = []
    for record in records:
        owners = owner_for_source(record.openui)
        visible = [
            owners[slot]
            for slot in record.placeholders
            if slot in owners
        ]
        counts.update(visible)
        owners_by_record.append(set(visible))
    rare = sorted(owner for owner, count in counts.items() if count <= threshold)
    rare_set = set(rare)
    selected = [
        record
        for record, owners in zip(records, owners_by_record, strict=True)
        if owners & rare_set
    ]
    return selected, dict(sorted(counts.items())), rare


def _write_record_nll(run_dir: Path, plugin, records) -> Path:
    """Per-record NLL under the final model — Superfiltering difficulty evidence.

    Consumed by derived-data builds (`build_train_data --difficulty-from`) to
    weight curation scores; a scoring failure records the error per row and
    never fails the run.
    """
    try:
        import torch

        no_grad = torch.no_grad
    except ImportError:  # pragma: no cover - torch is a train-time dependency
        from contextlib import nullcontext as no_grad
    if callable(getattr(plugin, "eval", None)):
        try:
            plugin.eval()
        except Exception:  # noqa: BLE001 - stub plugins may not support eval()
            pass
    rows: list[dict] = []
    for record in records:
        try:
            with no_grad():
                loss = plugin.training_loss([record])
            value = float(loss.item() if hasattr(loss, "item") else loss)
            rows.append({"id": record.id, "nll": round(value, 6)})
        except Exception as exc:  # noqa: BLE001 - evidence stays per-row honest
            rows.append({"id": record.id, "nll": None, "error": str(exc)[:200]})
    path = run_dir / "record_nll.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def train(config: ModelBuildConfig, model=None) -> dict:
    from slm_training.harnesses.model_build.feature_flags import resolve, save_snapshot
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

    config, flag_snapshot = resolve(config, phase="training")
    from slm_training.harnesses.capability_gates import require_training_authorized

    require_training_authorized(config)

    campaign_governance = None
    if bool(getattr(config, "register_promoted", False)):
        from slm_training.harnesses.experiments.promotion import (
            load_campaign_governance,
        )

        campaign_paths = (
            config.campaign_manifest,
            config.campaign_result,
            config.campaign_store_root,
            config.campaign_artifact_root,
        )
        if any(path is None for path in campaign_paths):
            raise ValueError(
                "register_promoted requires campaign manifest, result, store root, "
                "and artifact root"
            )
        campaign_governance = load_campaign_governance(
            manifest_path=config.campaign_manifest,
            result_path=config.campaign_result,
            store_root=config.campaign_store_root,
            artifact_root=config.campaign_artifact_root,
        )

    max_wall_minutes = getattr(config, "max_wall_minutes", None)
    max_wall_minutes = (
        float(MAX_RUN_MINUTES) if max_wall_minutes is None else float(max_wall_minutes)
    )
    if not 0 < max_wall_minutes <= MAX_RUN_MINUTES:
        raise ValueError(
            f"max_wall_minutes must be positive and at most {MAX_RUN_MINUTES}"
        )
    wall_started = time.monotonic()
    wall_deadline = wall_started + max_wall_minutes * 60

    accel = detect_device(config.device)
    # Honor explicit device but adopt accel threading / amp defaults.
    if config.device in {"auto", "best"}:
        config.device = accel.device

    primary_records = load_train_records(config.train_dir)
    if not primary_records:
        raise ValueError("train records empty")

    rng = random.Random(config.seed)
    records = list(primary_records)
    if getattr(config, "use_curriculum", False):
        from slm_training.harnesses.quality import apply_curriculum_tags

        records = apply_curriculum_tags(records, sanitize=True)
    rng.shuffle(records)

    resume_path = getattr(config, "resume_from", None)
    initialize_path = getattr(config, "initialize_from", None)
    replay_fraction = float(getattr(config, "replay_fraction", 0.0) or 0.0)
    replay_dir = getattr(config, "replay_train_dir", None)
    if not 0.0 <= replay_fraction <= 1.0:
        raise ValueError("replay_fraction must be between 0 and 1")
    if replay_fraction and replay_dir is None:
        raise ValueError("replay_fraction requires replay_train_dir")
    if replay_fraction and (
        getattr(config, "use_curriculum", False)
        or getattr(config, "mixture_manifest", None)
    ):
        raise ValueError(
            "replay sampling cannot be combined with curriculum or mixture sampling"
        )
    replay_records = []
    replay_manifest_sha: str | None = None
    if replay_fraction:
        replay_dir = Path(replay_dir)
        loaded_replay = load_train_records(replay_dir)
        if not loaded_replay:
            raise ValueError("replay train records empty")
        replay_records = [
            replace(record, id=f"replay::{record.id}") for record in loaded_replay
        ]
        replay_manifest_sha = data_manifest_sha(replay_dir)
    records_by_id = {r.id: r for r in [*records, *replay_records]}

    initialization_weight_retention = float(
        getattr(config, "initialization_weight_retention", 0.0) or 0.0
    )
    if resume_path and initialize_path:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    if not 0.0 <= initialization_weight_retention <= 1.0:
        raise ValueError("initialization_weight_retention must be between 0 and 1")
    if initialization_weight_retention and not initialize_path:
        raise ValueError("initialization_weight_retention requires initialize_from")

    plugin = model or build_model(config, records)
    strict_subset_multiplier = int(
        getattr(config, "root_reference_identity_strict_subset_multiplier", 1) or 1
    )
    if strict_subset_multiplier < 1:
        raise ValueError(
            "root_reference_identity_strict_subset_multiplier must be at least 1"
        )
    strict_subset_records = []
    audit_strict_subsets = strict_subset_multiplier > 1 or float(
        getattr(config, "root_reference_identity_loss_weight", 0.0) or 0.0
    ) > 0.0
    if audit_strict_subsets:
        tokenizer = getattr(plugin, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "root-reference strict-subset sampling requires a tokenizer"
            )
        strict_subset_records = _strict_root_reference_identity_records(
            records, tokenizer
        )
        if strict_subset_multiplier > 1 and not strict_subset_records:
            raise ValueError("no strict-subset root-reference records found")
    owner_rare_threshold = int(
        getattr(config, "slot_component_owner_rare_threshold", 0) or 0
    )
    owner_rare_multiplier = int(
        getattr(config, "slot_component_owner_rare_multiplier", 1) or 1
    )
    if owner_rare_threshold < 0:
        raise ValueError("slot_component_owner_rare_threshold must be nonnegative")
    if owner_rare_multiplier < 1:
        raise ValueError("slot_component_owner_rare_multiplier must be at least 1")
    if owner_rare_multiplier > 1 and owner_rare_threshold < 1:
        raise ValueError(
            "slot_component_owner_rare_multiplier requires a positive threshold"
        )
    if owner_rare_multiplier > 1 and (
        replay_fraction
        or getattr(config, "use_curriculum", False)
        or getattr(config, "mixture_manifest", None)
    ):
        raise ValueError(
            "slot-owner rare sampling cannot be combined with replay, curriculum, "
            "or mixture sampling"
        )
    owner_rare_records = []
    owner_counts: dict[str, int] = {}
    rare_owners: list[str] = []
    if owner_rare_threshold > 0:
        owner_for_source = getattr(plugin, "_slot_component_owners", None)
        if not callable(owner_for_source):
            raise ValueError("slot-owner rare sampling requires owner extraction")
        owner_rare_records, owner_counts, rare_owners = (
            _rare_slot_component_owner_records(
                records, owner_for_source, owner_rare_threshold
            )
        )
        if owner_rare_multiplier > 1 and not owner_rare_records:
            raise ValueError("no rare slot-owner records found")
    initialized_from: str | None = None
    initialized_prior_fields: list[str] = []
    rebuilt_prior_fields: list[str] = []
    if initialize_path:
        initialize_path = Path(initialize_path)
        if not initialize_path.is_file():
            raise FileNotFoundError(
                f"initialize_from checkpoint not found: {initialize_path}"
            )
        loader = getattr(plugin, "load", None)
        if not callable(loader):
            raise TypeError(f"{type(plugin).__name__} does not support initialize_from")
        plugin_config = getattr(plugin, "config", None)
        corpus_priors = {
            field_name: getattr(plugin_config, field_name)
            for field_name in (
                "slot_component_lexeme_priors",
                "slot_component_span_priors",
            )
            if plugin_config is not None and hasattr(plugin_config, field_name)
        }
        loader(initialize_path)
        initialized_from = str(initialize_path)
        initialized_prior_fields = list(getattr(plugin, "initialized_prior_fields", ()))
        # These priors are deterministic corpus statistics, not learned weights.
        # Warm starts must keep the current corpus/model formula rather than
        # silently restoring stale values cached in the parent checkpoint.
        for field_name, values in corpus_priors.items():
            setattr(plugin.config, field_name, values)
            if field_name in initialized_prior_fields:
                rebuilt_prior_fields.append(field_name)
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
    flag_payload = save_snapshot(run_dir, flag_snapshot)
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
            "replay_fraction": replay_fraction,
        },
    )
    run_otel_trace = current_trace()
    try:
        progress_seconds = float(os.getenv("SLM_OTEL_PROGRESS_SECONDS", "20"))
    except ValueError:
        progress_seconds = 20.0
    last_progress_emit = 0.0

    def _emit_progress(step_value: int, loss_value: float) -> None:
        # Throttled OTLP heartbeat so telemetry peers can list and stream this
        # run live. No-op unless the CLI wrapped training in run_trace(), and
        # SLM_OTEL_PROGRESS_SECONDS=0 is the kill switch.
        nonlocal last_progress_emit
        if run_otel_trace is None or progress_seconds <= 0:
            return
        now_monotonic = time.monotonic()
        if (
            last_progress_emit
            and now_monotonic - last_progress_emit < progress_seconds
        ):
            return
        last_progress_emit = now_monotonic
        try:
            run_otel_trace.log(
                "train.progress",
                attributes={
                    "slm.step": step_value,
                    "slm.loss": loss_value,
                    "slm.tokens.target": seen_target_tokens,
                    "slm.steps.total": int(config.steps),
                },
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never abort training
            warnings.warn(f"train.progress heartbeat failed: {exc}", stacklevel=2)
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
        if mixture_sampling_policy == "exposure_targeted":
            mixture_meta["exposure_target_profile"] = getattr(
                config, "mixture_exposure_target_profile", None
            )
            mixture_meta["total_decision_budget"] = getattr(
                config, "mixture_total_decision_budget", None
            )
            mixture_meta["per_root_cap"] = getattr(
                config, "mixture_per_root_cap", None
            )
            mixture_meta["per_template_cap"] = getattr(
                config, "mixture_per_template_cap", None
            )
            mixture_meta["max_importance_weight"] = getattr(
                config, "mixture_max_importance_weight", None
            )
        family_pools = index_family_pools(records)
        if mixture_task_weights:
            task_family_pools = index_task_family_pools(records)

    def _batches_for_step(step: int) -> list[list]:
        if replay_records:
            target = config.batch_size * 8
            replay_count = round(target * replay_fraction)
            primary_count = target - replay_count
            drawn = [rng.choice(records) for _ in range(primary_count)]
            drawn.extend(rng.choice(replay_records) for _ in range(replay_count))
            rng.shuffle(drawn)
            return batched(drawn, config.batch_size)
        if mixture_weights is not None:
            from slm_training.data.mixture import sample_mixture_batch

            target = config.batch_size * 8
            sample_kwargs: dict[str, object] = {
                "weights": mixture_weights,
                "batch_size": target,
                "rng": rng,
                "pools": family_pools,
                "task_weights": mixture_task_weights,
                "task_pools": task_family_pools,
                "sampling_policy": mixture_sampling_policy,
            }
            if mixture_sampling_policy == "exposure_targeted":
                sample_kwargs.update(
                    {
                        "total_decision_budget": (
                            getattr(config, "mixture_total_decision_budget", None)
                            or target
                        ),
                        "per_root_cap": getattr(
                            config, "mixture_per_root_cap", None
                        ),
                        "per_template_cap": getattr(
                            config, "mixture_per_template_cap", None
                        ),
                        "max_importance_weight": getattr(
                            config, "mixture_max_importance_weight", None
                        ),
                    }
                )
            drawn = sample_mixture_batch(records, **sample_kwargs)
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
        if strict_subset_multiplier > 1:
            shuffled.extend(
                strict_subset_records * (strict_subset_multiplier - 1)
            )
        if owner_rare_multiplier > 1:
            shuffled.extend(owner_rare_records * (owner_rare_multiplier - 1))
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
    initialized_weight_anchor: list[tuple] = []
    if is_twotower:
        import torch

        optimizer_name = str(getattr(config, "optimizer_name", "adamw") or "adamw").lower()
        if optimizer_name == "muon_hybrid":
            from slm_training.optimizers.muon import build_muon_hybrid

            # Muon uses deterministic parameter ownership by canonical name.
            optimizer = build_muon_hybrid(
                plugin.named_parameters(),
                lr=float(config.lr),
                muon_lr=getattr(config, "muon_lr", None),
                adamw_lr=getattr(config, "adamw_lr", None),
                weight_decay=float(getattr(config, "weight_decay", 0.0) or 0.0),
                muon_momentum=float(getattr(config, "muon_momentum", 0.9) or 0.9),
                muon_nesterov=bool(getattr(config, "muon_nesterov", False)),
                muon_ns_steps=int(getattr(config, "muon_ns_steps", 5) or 5),
            )
        else:
            parameters = (
                plugin.optimizer_parameter_groups()
                if hasattr(plugin, "optimizer_parameter_groups")
                else plugin.trainable_parameters()
            )
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(config.lr),
                weight_decay=float(getattr(config, "weight_decay", 0.0) or 0.0),
            )
        scaler = grad_scaler(config.device, enabled=use_amp)
        if initialize_path:
            initialized_weight_anchor = [
                (parameter, parameter.detach().clone())
                for parameter in plugin.parameters()
                if parameter.requires_grad
            ]

    def _retain_initialized_weights() -> None:
        if not initialized_weight_anchor or not initialization_weight_retention:
            return
        with torch.no_grad():
            for parameter, anchor in initialized_weight_anchor:
                parameter.lerp_(anchor, initialization_weight_retention)

    def _initialized_weight_rms_drift() -> float | None:
        if not initialized_weight_anchor:
            return None
        squared = 0.0
        count = 0
        with torch.no_grad():
            for parameter, anchor in initialized_weight_anchor:
                squared += float((parameter - anchor).square().sum())
                count += parameter.numel()
        return math.sqrt(squared / count) if count else None

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
    seen_primary_examples = 0
    seen_replay_examples = 0
    source_loss_proxy = {
        "primary": {"count": 0, "sum": 0.0, "first": [], "last": []},
        "replay": {"count": 0, "sum": 0.0, "first": [], "last": []},
    }
    best_weighted_nll = math.inf
    best_ship_score = -math.inf
    pending: list[list] = []
    primary_manifest_sha = data_manifest_sha(config.train_dir)
    manifest_sha = primary_manifest_sha
    if replay_records:
        manifest_sha = hashlib.sha256(
            json.dumps(
                {
                    "primary": primary_manifest_sha,
                    "replay": replay_manifest_sha,
                    "replay_fraction": replay_fraction,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    resumed_from: str | None = None

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
            saved_fp = payload.get("optimizer_fingerprint")
            current_fp = (
                optimizer.fingerprint if hasattr(optimizer, "fingerprint") else None
            )
            if saved_fp is not None and saved_fp != current_fp:
                raise ValueError(
                    "optimizer fingerprint mismatch: "
                    f"checkpoint={saved_fp}, current={current_fp}"
                )
            if saved_fp is None and current_fp is not None:
                raise ValueError(
                    "checkpoint has no optimizer fingerprint but current optimizer "
                    f"is {current_fp}; resume across optimizer families is not allowed"
                )
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
        nonlocal seen_primary_examples, seen_prompt_tokens
        nonlocal seen_replay_examples, seen_target_tokens
        replay_examples = sum(record.id.startswith("replay::") for record in batch)
        seen_replay_examples += replay_examples
        seen_primary_examples += len(batch) - replay_examples
        if hasattr(plugin, "count_batch_tokens"):
            pt, tt = plugin.count_batch_tokens(batch)
            seen_prompt_tokens += int(pt)
            seen_target_tokens += int(tt)

    def _batch_meta(batch: list) -> list[dict]:
        return [
            {
                "id": str(record.id),
                "source": str(record.source),
                "source_family": str(
                    (record.meta or {}).get("source_family") or record.source
                ),
                "prompt_chars": len(record.prompt),
                "target_chars": len(record.openui),
            }
            for record in batch
        ]

    def _record_source_loss_proxy(batch: list, values: list[float]) -> None:
        if len(batch) != len(values):
            return
        for record, value in zip(batch, values, strict=True):
            group = "replay" if record.id.startswith("replay::") else "primary"
            stats = source_loss_proxy[group]
            stats["count"] += 1
            stats["sum"] += value
            if len(stats["first"]) < 20:
                stats["first"].append(value)
            stats["last"].append(value)
            if len(stats["last"]) > 20:
                stats["last"].pop(0)

    def _source_loss_proxy_summary(group: str) -> dict:
        stats = source_loss_proxy[group]
        count = stats["count"]
        first = stats["first"]
        last = stats["last"]
        return {
            "count": count,
            "mean": stats["sum"] / count if count else None,
            "first_20_mean": sum(first) / len(first) if first else None,
            "last_20_mean": sum(last) / len(last) if last else None,
        }

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
                    "meaningful_program_rate": metrics.get("meaningful_program_rate"),
                    "meaningful_program_v1_rate": metrics.get(
                        "meaningful_program_v1_rate"
                    ),
                    "binding_aware_meaningful_v2_rate_strict": metrics.get(
                        "binding_aware_meaningful_v2_rate_strict"
                    ),
                    "binding_aware_meaningful_v2_rate_coverage_conditioned": metrics.get(
                        "binding_aware_meaningful_v2_rate_coverage_conditioned"
                    ),
                    "binding_aware_meaningful_v2_coverage": metrics.get(
                        "binding_aware_meaningful_v2_coverage"
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
                        "meaningful_program_v1_rate": metrics.get(
                            "meaningful_program_v1_rate"
                        ),
                        "binding_aware_meaningful_v2_rate_strict": metrics.get(
                            "binding_aware_meaningful_v2_rate_strict"
                        ),
                        "binding_aware_meaningful_v2_rate_coverage_conditioned": metrics.get(
                            "binding_aware_meaningful_v2_rate_coverage_conditioned"
                        ),
                        "binding_aware_meaningful_v2_coverage": metrics.get(
                            "binding_aware_meaningful_v2_coverage"
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
        from slm_training.versioning import build_version_stamp

        report["version_stamp"] = build_version_stamp("evals.loss_suite")
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
                accum_loss_sum += float(reported_loss_t)
                accum_loss_count += 1
                accum_batch_meta.extend(_batch_meta(batch))
                batch_example_losses = [
                    float(value)
                    for value in (
                        getattr(plugin, "_last_example_token_losses", None) or []
                    )
                ]
                accum_example_losses.extend(batch_example_losses)
                _record_source_loss_proxy(batch, batch_example_losses)
                micro += 1
                if micro >= grad_accum:
                    with timed("optim_step"):
                        scaler.unscale_(optimizer)
                        _clip_optimizer_parameter_groups(optimizer, 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        _retain_initialized_weights()
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
                    _emit_progress(step, last_loss)
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
                _emit_progress(int(row["step"]), last_loss)
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
                _retain_initialized_weights()
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
            evaluate_promotion,
            register_promoted_checkpoint,
        )

        assert campaign_governance is not None
        manifest, result, store, artifact_root = campaign_governance
        promotion = evaluate_promotion(
            campaign_manifest=manifest,
            campaign_result=result,
            campaign_store=store,
            artifact_root=artifact_root,
        )
        source = ckpt_dir / "best_weighted_nll.pt"
        if not source.exists():
            source = ckpt_dir / "best_ship_score.pt"
        if not source.exists():
            source = ckpt_path
        register_promoted_checkpoint(
            ckpt_dir,
            source=source,
            promotion_result=promotion,
            campaign_manifest=manifest,
            campaign_result=result,
            campaign_store=store,
            artifact_root=artifact_root,
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

    record_nll_path: Path | None = None
    if getattr(config, "emit_record_nll", False):
        record_nll_path = _write_record_nll(run_dir, plugin, records)

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
        "replay": {
            "enabled": bool(replay_records),
            "train_dir": str(replay_dir) if replay_records else None,
            "requested_fraction": replay_fraction,
            "effective_fraction": (
                seen_replay_examples / (seen_primary_examples + seen_replay_examples)
                if seen_primary_examples + seen_replay_examples
                else 0.0
            ),
            "primary_record_count": len(records),
            "replay_record_count": len(replay_records),
            "seen_primary_examples": seen_primary_examples,
            "seen_replay_examples": seen_replay_examples,
            "primary_data_manifest_sha": primary_manifest_sha,
            "replay_data_manifest_sha": replay_manifest_sha,
            "combined_data_manifest_sha": manifest_sha,
            "example_token_loss_proxy": {
                "primary": _source_loss_proxy_summary("primary"),
                "replay": _source_loss_proxy_summary("replay"),
            },
        },
        "model": config.model_name,
        "device": config.device,
        "seen_prompt_tokens": seen_prompt_tokens,
        "seen_target_tokens": seen_target_tokens,
        "target_token_budget": getattr(config, "target_token_budget", None),
        "max_wall_minutes": max_wall_minutes,
        "elapsed_wall_seconds": time.monotonic() - wall_started,
        "resumed_from": resumed_from,
        "initialized_from": initialized_from,
        "initialized_prior_fields": initialized_prior_fields,
        "rebuilt_prior_fields": rebuilt_prior_fields,
        "initialized_weight_count": sum(
            parameter.numel() for parameter, _anchor in initialized_weight_anchor
        ),
        "initialized_weight_rms_drift": _initialized_weight_rms_drift(),
        "data_manifest_sha": manifest_sha,
        "record_nll": str(record_nll_path.as_posix()) if record_nll_path else None,
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
            "optimizer_name": getattr(config, "optimizer_name", "adamw"),
            "muon_lr": getattr(config, "muon_lr", None),
            "adamw_lr": getattr(config, "adamw_lr", None),
            "weight_decay": getattr(config, "weight_decay", 0.0),
            "muon_momentum": getattr(config, "muon_momentum", 0.9),
            "muon_nesterov": getattr(config, "muon_nesterov", False),
            "muon_ns_steps": getattr(config, "muon_ns_steps", 5),
            "replay_fraction": replay_fraction,
            "initialization_weight_retention": initialization_weight_retention,
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
            "slot_component_loss_weight": getattr(
                config, "slot_component_loss_weight", 0.0
            ),
            "slot_component_class_balance_power": getattr(
                config, "slot_component_class_balance_power", 0.0
            ),
            "slot_component_owner_rare_threshold": owner_rare_threshold,
            "slot_component_owner_rare_multiplier": owner_rare_multiplier,
            "slot_component_owner_counts": owner_counts,
            "slot_component_owner_rare_classes": rare_owners,
            "slot_component_owner_rare_records": len(owner_rare_records),
            "slot_component_owner_sampling_records": (
                len(records)
                + len(owner_rare_records) * (owner_rare_multiplier - 1)
            ),
            "slot_component_decode_weight": getattr(
                config, "slot_component_decode_weight", 0.0
            ),
            "slot_component_prompt_context": bool(
                getattr(config, "slot_component_prompt_context", False)
            ),
            "slot_component_next_context": bool(
                getattr(config, "slot_component_next_context", False)
            ),
            "slot_component_pair_interaction": bool(
                getattr(config, "slot_component_pair_interaction", False)
            ),
            "slot_contract_in_context": bool(
                getattr(config, "slot_contract_in_context", False)
            ),
            "honest_slot_contract": bool(
                getattr(config, "honest_slot_contract", False)
            ),
            "slot_component_lexeme_prior_weight": getattr(
                config, "slot_component_lexeme_prior_weight", 0.0
            ),
            "slot_component_span_prior_weight": getattr(
                config, "slot_component_span_prior_weight", 0.0
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
            "root_reference_arity_loss_weight": getattr(
                config, "root_reference_arity_loss_weight", 0.0
            ),
            "root_reference_arity_decode_weight": getattr(
                config, "root_reference_arity_decode_weight", 0.0
            ),
            "root_reference_identity_loss_weight": getattr(
                config, "root_reference_identity_loss_weight", 0.0
            ),
            "root_reference_identity_negative_weight": getattr(
                config, "root_reference_identity_negative_weight", 1.0
            ),
            "root_reference_identity_strict_subset_multiplier": (
                strict_subset_multiplier
            ),
            "root_reference_identity_strict_subset_records": len(
                strict_subset_records
            ),
            "root_reference_identity_sampling_records": (
                len(records)
                + len(strict_subset_records) * (strict_subset_multiplier - 1)
            ),
            "root_reference_identity_decode_weight": getattr(
                config, "root_reference_identity_decode_weight", 0.0
            ),
            "fuse_ltr_loss": bool(getattr(config, "fuse_ltr_loss", True)),
            "fidelity_loss_weight": getattr(config, "fidelity_loss_weight", 0.0),
            "fastpath_aux_weight": getattr(config, "fastpath_aux_weight", 0.0),
            "schema_in_context": bool(getattr(config, "schema_in_context", False)),
            "retrieval_k": getattr(config, "retrieval_k", 0),
            "grammar_constrained": bool(getattr(config, "grammar_constrained", False)),
            "design_md_dropout": float(
                getattr(effective_plugin_config, "design_md_dropout", 0.0) or 0.0
            ),
            "honesty_mode": (
                "no-design-md-context"
                if not getattr(effective_plugin_config, "design_md_in_context", True)
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
        "feature_flags": flag_payload,
    }
    from slm_training.versioning import build_version_stamp

    summary["version_stamp"] = build_version_stamp(
        "harness.model_build.train", "harness.experiment_feature_flags"
    )
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
