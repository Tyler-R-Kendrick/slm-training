"""Resolve real compiled training commands and lock paired scientific identities.

Controller-only local CPU preflight. Construction counts actual parameters but
does not train, score, publish a model, or authorize scientific acceptance.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from slm_training.autoresearch.experiment_identity import (
    InterventionContract,
    attempt_identity,
    digest,
    replicate_identity,
    treatment_identity,
)
from slm_training.autoresearch.intervention_matching import assert_intervention_match
from slm_training.autoresearch.preflight.treatment_design import CHECK
from slm_training.autoresearch.preflight.loss_binding import (
    bind_endpoint,
)
from slm_training.autoresearch.preflight.search_binding import (
    bind_search_manifest,
    lock_search_manifests,
    require_attempt_locks,
)

_LOCATIONS = frozenset(
    {
        "run_id",
        "run_root",
        "checkpoint_every_steps",
        "telemetry",
        "telemetry_sample_interval_ms",
        "campaign_manifest",
        "campaign_result",
        "campaign_store_root",
        "campaign_artifact_root",
    }
)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported resolved configuration value: {type(value).__name__}")


def _prepare(commands):
    import torch
    from scripts.train_model import resolve_config
    from slm_training.harnesses.model_build.data import load_train_records
    from slm_training.harnesses.model_build.factory import build_model
    from slm_training.harnesses.model_build.feature_flags import resolve
    from slm_training.harnesses.model_build.resume_contract import resume_identity

    train = [
        cmd for cmd in commands if len(cmd) > 2 and cmd[2] == "scripts.train_model"
    ]
    if len(train) != 1 or any("scripts.build_train_data" in cmd for cmd in commands):
        raise ValueError("treatment_design:requires_published_training_snapshot")
    config, _ = resolve(resolve_config(train[0][3:]), phase="training")
    if (
        config.device != "cpu"
        or config.context_backend != "scratch"
        or config.denoiser_backend != "scratch"
    ):
        raise ValueError(
            "capability_unavailable:compiled_preflight_local_cpu_scratch_only"
        )
    records = load_train_records(config.train_dir)
    if not records:
        raise ValueError("treatment_design:empty_training_snapshot")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_model(config, records)
        identity = resume_identity(config, model)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        architecture = digest(
            {
                "type": type(model).__qualname__,
                "shapes": {
                    key: list(value.shape) for key, value in model.state_dict().items()
                },
            }
        )
    resolved = json.loads(
        json.dumps(asdict(config), default=_json_value, allow_nan=False)
    )
    resolved = {key: value for key, value in resolved.items() if key not in _LOCATIONS}
    ancestor = _ancestor(config)
    # Paths remain for actual compiler matching; treatment_identity excludes
    # invocation locations and binds the corresponding content identities.
    return {
        "config": resolved,
        "params": params,
        "architecture": architecture,
        "record_count": len(records),
        "tokenizers": digest(identity["tokenizers"]),
        "ancestor": ancestor,
        "data": digest(identity["data_content"]),
        "source": identity["source"],
        "runtime": identity["runtime"],
        "commands": commands,
    }


def _ancestor(config):
    from slm_training.harness_core.checkpoint_bundle import validate_bundle

    path = config.initialize_from or config.resume_from
    if not path:
        return "fresh_initialization"
    directory = Path(path).parent
    if not (directory / "manifest.json").is_file():
        raise ValueError(
            "treatment_design:starting_checkpoint_requires_validated_bundle"
        )
    _, manifest = validate_bundle(directory.parent.parent, directory.name)
    if Path(path).name not in manifest["files"]:
        raise ValueError("treatment_design:checkpoint_not_a_bundle_component")
    return directory.name


def _contract(candidate, control_config, candidate_config):
    declared = getattr(candidate, "intervention", None)
    if declared is not None:
        return InterventionContract.model_validate(declared)
    varied = tuple(
        sorted(
            key
            for key in control_config.keys() | candidate_config.keys()
            if control_config.get(key) != candidate_config.get(key)
        )
    )
    # Ordinary mechanism remains default. Data/duration/capacity differences
    # fail matching unless explicitly declared in the locked ExperimentSpec.
    return InterventionContract(
        kind="mechanism", varied_fields=varied, resource_basis="updates"
    )


def _bindings(prepared, *, endpoint, contract, totals):
    cfg = prepared["config"]
    return {
        "architecture": prepared["architecture"],
        "tokenizer_layout": prepared["tokenizers"],
        "training_snapshot": prepared["data"],
        "preprocessing": prepared["source"],
        "starting_checkpoint": prepared["ancestor"],
        "starting_checkpoint_role": "exact_resume"
        if cfg.get("resume_from")
        else "warm_start"
        if cfg.get("initialize_from")
        else "fresh",
        "endpoint": endpoint,
        "resource_contract": {
            "basis": contract.resource_basis,
            "planned_totals": totals,
        },
    }


def prepare_pair(
    campaign, control, candidate, *, output_root, endpoint, search_slug=None
):
    """Real compiler -> CLI defaults -> actual model shape -> mandatory preflight."""
    from slm_training.autoresearch.engine import compile_commands
    from slm_training.autoresearch.hillclimb import assert_warm_start_launch

    arms = [
        _prepare(compile_commands(campaign, exp, output_root=output_root))
        for exp in (control, candidate)
    ]
    endpoint = bind_endpoint(arms, endpoint, search_slug)
    left, right = (arm["config"] for arm in arms)
    contract = _contract(candidate, left, right)
    totals = getattr(candidate, "planned_resource_totals", None)
    if totals is None:
        if contract.resource_basis != "updates":
            raise ValueError(
                "treatment_design:explicit_common_resource_totals_required"
            )
        totals = (left["steps"], right["steps"])
    if contract.resource_basis == "examples" and tuple(totals) != tuple(
        _planned_examples(arm) for arm in arms
    ):
        raise ValueError(
            "treatment_design:planned_examples_disagree_with_actual_batch_schedule"
        )
    params = tuple(arm["params"] for arm in arms)
    if arms[0]["tokenizers"] != arms[1]["tokenizers"]:
        raise ValueError("treatment_design:unmatched_tokenizer_layout")
    if contract.kind == "mechanism":
        assert_warm_start_launch(left, right, trainable_params=params)
    assert_intervention_match(
        left, right, contract=contract, resource_totals=totals, trainable_params=params
    )
    bindings = [
        _bindings(arm, endpoint=endpoint, contract=contract, totals=totals)
        for arm in arms
    ]
    design = {
        "control": left,
        "candidate": right,
        "intervention": contract.model_dump(),
        "resource_totals": totals,
        "trainable_params": params,
        "endpoint": endpoint["kind"],
        "compiled_commands": arms[1]["commands"],
        "bindings": bindings[1],
    }
    verdict = CHECK.run({"treatment_design": design})
    if verdict.verdict != "pass":
        raise ValueError(f"treatment_design:mandatory_preflight:{verdict.reasons}")
    treatments = [
        treatment_identity(arm["config"], bindings=binding, intervention=contract)
        for arm, binding in zip(arms, bindings, strict=True)
    ]
    hypothesis = getattr(candidate, "hypothesis_id", None) or digest(
        candidate.hypothesis
    )
    # Nuisance seeds do not enter treatment identity; the design records their
    # realization separately. A learned ancestor is never a fresh initialization.
    seed = right["seed"]
    seeds = {
        key: value
        for key, value in right.items()
        if (key == "seed" or key.endswith("_seed")) and value is not None
    }
    if search_slug is not None:
        seeds["loss_mask_seed"] = endpoint["loss_measurement"]["mask_seed"]
    design_digest = digest(
        {
            "treatments": treatments,
            "hypothesis": hypothesis,
            "endpoint": endpoint,
            "intervention": contract.model_dump(),
        }
    )
    independence = (
        "conditional_on_shared_ancestor"
        if arms[1]["ancestor"] != "fresh_initialization"
        else "seed_realization_not_independence_certificate"
    )
    replicate = replicate_identity(
        hypothesis_id=hypothesis,
        design_digest=design_digest,
        unit_id=f"planned-seed-{seed}",
        seeds=seeds,
        ancestor_digest=arms[1]["ancestor"],
        independence=independence,
    )
    return {
        "schema": "compiled_treatment_design/v1",
        **({"search_slug": search_slug} if search_slug is not None else {}),
        "hypothesis_id": hypothesis,
        "design_digest": design_digest,
        "replicate_id": replicate,
        "randomness": seeds,
        "independence": independence,
        "arm_ids": [control.experiment_id, candidate.experiment_id],
        "treatment_ids": treatments,
        "preflight": verdict.model_dump(),
        "design": design,
        "compiled_commands": [arm["commands"] for arm in arms],
        "source": [arm["source"] for arm in arms],
        "runtime": [arm["runtime"] for arm in arms],
    }


def _planned_examples(arm):
    cfg, count = arm["config"], arm["record_count"]
    if any(
        cfg.get(key)
        for key in (
            "replay_fraction",
            "mixture_manifest",
            "semantic_contrast_dir",
            "use_curriculum",
            "resume_from",
        )
    ):
        raise ValueError(
            "treatment_design:nonuniform_example_schedule_requires_explicit_supported_planner"
        )
    batch = cfg["batch_size"]
    batches_per_epoch = (count + batch - 1) // batch
    batches = cfg["steps"] * cfg["grad_accum_steps"]
    epochs, remaining = divmod(batches, batches_per_epoch)
    return epochs * count + min(remaining * batch, count)


def persist_pair(store, pair):
    """Content-addressed artifact plus replay-safe canonical event, no shadow index."""
    path = store.write_artifact("treatment_designs", pair)
    store.append_event(
        "experiment_design_locked",
        artifact_sha256=path.stem,
        detail={
            "design_digest": pair["design_digest"],
            "replicate_id": pair["replicate_id"],
            "arm_ids": pair["arm_ids"],
            "path": str(path),
        },
        idempotency_key=f"design:{pair['design_digest']}:{pair['replicate_id']}",
    )
    return path


def begin_attempt(store, pair, arm_id):
    require_attempt_locks(store, pair, arm_id)
    index = pair["arm_ids"].index(arm_id)
    treatment = pair["treatment_ids"][index]
    previous = [
        event
        for event in store.verify_event_chain()
        if event["event_type"] == "experiment_attempt_started"
        and event["detail"].get("treatment_id") == treatment
        and event["detail"].get("replicate_id") == pair["replicate_id"]
    ]
    identity = attempt_identity(
        treatment_id=treatment, replicate_id=pair["replicate_id"], ordinal=len(previous)
    )
    detail = {
        "hypothesis_id": pair["hypothesis_id"],
        "treatment_id": treatment,
        "replicate_id": pair["replicate_id"],
        "attempt_id": identity,
        "attempt_ordinal": len(previous),
        "independent_sample_increment": 0,
        "design_digest": pair["design_digest"],
    }
    store.append_event(
        "experiment_attempt_started",
        experiment_id=arm_id,
        detail=detail,
        idempotency_key=f"attempt:{identity}",
    )
    return detail


def lock_driver_designs(
    store,
    by_id,
    control_id,
    candidate_ids,
    *,
    endpoint,
    manifest_templates=None,
    search_slugs=None,
):
    from slm_training.autoresearch.schemas import ExperimentSpec

    campaign = store.load_campaign()
    if manifest_templates is not None and len(candidate_ids) != 1:
        raise ValueError("treatment_design:search_requires_one_paired_candidate")
    control = ExperimentSpec.model_validate_json(by_id[control_id].read_text())
    pairs = {}
    for candidate_id in candidate_ids:
        candidate = ExperimentSpec.model_validate_json(by_id[candidate_id].read_text())
        template = (
            manifest_templates[candidate_id] if manifest_templates is not None else None
        )
        resolved_endpoint = dict(endpoint)
        if template is not None:
            primary = next(e for e in template.endpoints if e.role == "primary")
            resolved_endpoint["primary"] = primary.model_dump(mode="json")
        pair = prepare_pair(
            campaign,
            control,
            candidate,
            output_root=store.root.parent,
            endpoint=resolved_endpoint,
            search_slug=search_slugs[candidate_id]
            if search_slugs is not None
            else None,
        )
        if template is not None:
            manifest = bind_search_manifest(template, pair, control, candidate)
            lock_search_manifests(store, manifest, pair)
        persist_pair(store, pair)
        pairs[candidate_id] = pair
    return pairs
