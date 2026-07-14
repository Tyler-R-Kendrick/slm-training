#!/usr/bin/env python3
"""Orchestrate immutable branch/train/evaluate/promote/merge/deploy cycles."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from slm_training.lineage.data_cycle import snapshot_directory
from slm_training.lineage.evaluation_snapshot import build_evaluation_snapshot
from slm_training.lineage.merge import merge_checkpoints, validate_merge_manifests
from slm_training.lineage.promotion import (
    deployment_failures,
    promotion_failures,
    select_causal_base,
)
from slm_training.lineage.records import (
    ChampionPointer,
    EvaluationReport,
    MergeManifest,
    RunManifest,
    content_sha,
)
from slm_training.lineage.store import LineageStore, utc_now
from slm_training.lineage.tracks import (
    CAUSAL_BASE_CANDIDATES,
    CAUSAL_LORA_RECIPE,
    IMMUTABLE_BRANCH_RECIPE_KEYS,
    TOKEN_RUNGS,
    TWOTOWER_BASE_ID,
    TWOTOWER_BASE_REVISION,
    TWOTOWER_E53_RECIPE,
)


def _json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    value = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else json.loads(raw)
    )
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return f"{sha}+dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _store(args: argparse.Namespace) -> LineageStore:
    return LineageStore(args.lineage_root)


def _new_pointer_id(track: str, run_id: str) -> str:
    return f"{track}-{run_id}-{uuid.uuid4().hex[:12]}"


def cmd_snapshot(args: argparse.Namespace) -> int:
    snapshot = snapshot_directory(
        args.snapshot_id,
        args.source,
        target_token_count=args.target_token_count,
        annotations_sha=args.annotations_sha,
        metadata=_json_object(args.metadata_json),
    )
    path = _store(args).write_snapshot(snapshot)
    print(json.dumps({"snapshot_sha": snapshot.sha, "path": str(path)}, indent=2))
    return 0


def cmd_snapshot_eval(args: argparse.Namespace) -> int:
    suites = dict(item.split("=", 1) for item in args.suite)
    training_ids = set(_json_object(args.training_ids_json).get("ids", []))
    snapshot = build_evaluation_snapshot(
        args.snapshot_id,
        suites,
        args.human_feedback_holdout,
        training_ids=training_ids,
    )
    path = _store(args).write_snapshot(snapshot)
    print(json.dumps({"snapshot_sha": snapshot.sha, "path": str(path)}, indent=2))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    if args.track == "twotower":
        recipe = dict(TWOTOWER_E53_RECIPE)
        base_id = args.base_model_id or TWOTOWER_BASE_ID
        base_revision = args.base_model_revision or TWOTOWER_BASE_REVISION
    else:
        recipe = dict(CAUSAL_LORA_RECIPE)
        base_id = args.base_model_id
        if base_id not in CAUSAL_BASE_CANDIDATES:
            raise ValueError(
                "causal init requires one bakeoff base: "
                + ", ".join(CAUSAL_BASE_CANDIDATES)
            )
        expected = CAUSAL_BASE_CANDIDATES[base_id]
        base_revision = args.base_model_revision or expected
        if base_revision != expected:
            raise ValueError(f"unapproved base revision for {base_id}: {base_revision}")
    recipe.update(_json_object(args.recipe_json))
    recipe_sha = content_sha(recipe)
    manifest = RunManifest(
        run_id=args.run_id,
        track=args.track,
        parent_ids=(),
        base_model_id=base_id,
        base_model_revision=base_revision,
        architecture_sha=content_sha({"track": args.track, "recipe": recipe}),
        tokenizer_sha=content_sha(
            {"base": base_id, "revision": base_revision, "kind": "tokenizer"}
        ),
        parameter_shapes_sha=content_sha(
            {"track": args.track, "recipe_sha": recipe_sha}
        ),
        data_snapshot_sha=args.data_snapshot_sha,
        eval_snapshot_sha=args.eval_snapshot_sha,
        recipe_sha=recipe_sha,
        code_sha=_git_sha(),
        seed=args.seed,
        hardware=_json_object(args.hardware_json),
        artifact_uris=(),
        metrics={},
        lifecycle_state="running",
        initialization="scratch",
        recipe=recipe,
        created_at=utc_now(),
    )
    path = _store(args).create_run(manifest)
    print(json.dumps({"manifest_sha": manifest.sha, "run_dir": str(path)}, indent=2))
    return 0


def cmd_branch(args: argparse.Namespace) -> int:
    store = _store(args)
    parent = store.load_run(args.parent)
    if parent.lifecycle_state not in {"validated", "champion", "deployed"}:
        raise ValueError("branches require a validated/champion/deployed parent")
    overrides = _json_object(args.recipe_json)
    changed_layout = sorted(
        key
        for key in IMMUTABLE_BRANCH_RECIPE_KEYS & overrides.keys()
        if overrides[key] != parent.recipe.get(key)
    )
    if changed_layout:
        raise ValueError(
            "branch cannot change architecture/tokenizer keys: "
            + ", ".join(changed_layout)
        )
    recipe = dict(parent.recipe)
    recipe.update(overrides)
    child = replace(
        parent,
        run_id=args.run_id,
        parent_ids=(parent.run_id,),
        data_snapshot_sha=args.data_snapshot_sha or parent.data_snapshot_sha,
        recipe_sha=content_sha(recipe),
        code_sha=_git_sha(),
        seed=args.seed,
        hardware=_json_object(args.hardware_json),
        artifact_uris=(),
        metrics={},
        lifecycle_state="running",
        initialization="parent",
        recipe=recipe,
        created_at=utc_now(),
        legacy_kind=None,
    )
    path = store.create_run(child)
    print(json.dumps({"manifest_sha": child.sha, "run_dir": str(path)}, indent=2))
    return 0


def _parent_artifact(store: LineageStore, manifest: RunManifest) -> Path | None:
    if not manifest.parent_ids:
        return None
    parent = store.load_run(manifest.parent_ids[0])
    local = [Path(uri) for uri in parent.artifact_uris if "://" not in uri]
    if not local:
        raise ValueError(f"parent {parent.run_id} has no local restorable artifact")
    if not local[0].exists():
        raise FileNotFoundError(local[0])
    return local[0]


def cmd_train(args: argparse.Namespace) -> int:
    store = _store(args)
    manifest = store.load_run(args.run_id)
    if manifest.lifecycle_state != "running":
        raise ValueError("only running manifests can train")
    if args.token_rung not in TOKEN_RUNGS:
        raise ValueError(f"token rung must be one of {TOKEN_RUNGS}")
    target_budget = max(1, int(args.target_token_count * args.token_rung))
    run_root = Path(args.run_root)
    run_dir = run_root / manifest.run_id
    if (run_dir / "checkpoints" / "last.pt").exists() and not args.resume:
        raise FileExistsError(
            "run artifact exists; use resume for this same run or choose a new run id"
        )
    if (run_dir / "adapter").exists() and not args.resume:
        raise FileExistsError(
            "run adapter exists; use resume for this same run or choose a new run id"
        )
    parent_artifact = None if args.resume else _parent_artifact(store, manifest)
    if args.resume:
        resume = Path(args.resume).resolve()
        if run_dir.resolve() not in resume.parents:
            raise ValueError(
                "resume checkpoint must belong to this immutable run directory"
            )
        if manifest.track == "twotower" and resume.name != "last_full_state.pt":
            raise ValueError("TwoTower resume must use this run's last_full_state.pt")
        if manifest.track == "causal_lm" and (
            not resume.is_dir() or not resume.name.startswith("checkpoint-")
        ):
            raise ValueError(
                "causal resume must use this run's Trainer checkpoint directory"
            )
    else:
        resume = None

    if manifest.track == "twotower":
        from slm_training.harnesses.model_build import (
            ModelBuildConfig,
            build_model,
            train,
        )
        from slm_training.harnesses.model_build.data import load_train_records

        valid = {item.name for item in fields(ModelBuildConfig)}
        recipe = {key: value for key, value in manifest.recipe.items() if key in valid}
        config = ModelBuildConfig(
            train_dir=Path(args.train_dir),
            run_root=run_root,
            run_id=manifest.run_id,
            seed=manifest.seed,
            device=args.device,
            target_token_budget=target_budget,
            steps=args.max_steps,
            resume_from=resume,
            sync_checkpoints=args.sync_checkpoints,
            **recipe,
        )
        model = None
        if parent_artifact is not None:
            records = load_train_records(config.train_dir)
            model = build_model(config, records, checkpoint=parent_artifact)
        summary = train(config, model=model)
        artifact = str(summary["checkpoint"])
        metrics = {
            "last_loss": float(summary.get("last_loss", 0.0)),
            "token_rung": args.token_rung,
        }
    else:
        from slm_training.harnesses.model_build.data import load_train_records
        from slm_training.models.causal_lm_openui import (
            CausalLMOpenUIConfig,
            CausalLMOpenUIPlugin,
        )

        plugin = CausalLMOpenUIPlugin.from_pretrained(
            CausalLMOpenUIConfig(
                base_model_id=manifest.base_model_id,
                base_model_revision=manifest.base_model_revision,
                device=args.device,
                local_files_only=args.local_files_only,
            )
        )
        if parent_artifact is None:
            plugin.enable_lora()
        else:
            plugin.load_parent_weights(parent_artifact)
        output = run_dir / "adapter"
        summary = plugin.train_sft(
            load_train_records(Path(args.train_dir)),
            output,
            target_token_budget=target_budget,
            seed=manifest.seed,
            resume_from_checkpoint=resume,
        )
        artifact = str(output)
        metrics = {
            key: float(value)
            for key, value in summary.get("metrics", {}).items()
            if isinstance(value, (int, float))
        }
        metrics["token_rung"] = args.token_rung
    updated = store.transition_run(
        manifest.run_id, "screened", artifact_uris=(artifact,), metrics=metrics
    )
    print(
        json.dumps(
            {"run_id": updated.run_id, "artifact": artifact, "metrics": metrics},
            indent=2,
        )
    )
    return 0


def _report_from_scoreboard(
    manifest: RunManifest,
    scoreboard: dict[str, Any],
    args: argparse.Namespace,
) -> EvaluationReport:
    suites = scoreboard.get("suites") or {}
    metrics: dict[str, float] = {}
    for key in (
        "parse_rate",
        "placeholder_fidelity",
        "request_coverage",
        "structural_similarity",
        "reward_score",
    ):
        values = [
            float(row[key]) for row in suites.values() if row.get(key) is not None
        ]
        if values:
            metrics[key] = min(values)
    metrics.update(
        {key: float(value) for key, value in _json_object(args.metrics_json).items()}
    )
    loss_path = Path(args.run_root) / manifest.run_id / "loss_suites.json"
    loss_payload = _json_object(str(loss_path)) if loss_path.exists() else {}
    aggregate = loss_payload.get("aggregate") or {}
    weighted_nll = args.weighted_nll
    if weighted_nll is None and aggregate.get("weighted_nll") is not None:
        weighted_nll = float(aggregate["weighted_nll"])
    category_nll = {
        key: float(value) for key, value in _json_object(args.category_nll_json).items()
    }
    if not category_nll:
        for key, value in (loss_payload.get("categories") or {}).items():
            mean = (value or {}).get("aggregate", {}).get("mean_nll")
            if mean is not None:
                category_nll[key] = float(mean)
    artifact = Path(manifest.artifact_uris[0]) if manifest.artifact_uris else None
    size = (
        sum(item.stat().st_size for item in artifact.rglob("*") if item.is_file())
        if artifact and artifact.is_dir()
        else artifact.stat().st_size
        if artifact and artifact.is_file()
        else None
    )
    gates = scoreboard.get("gates") or {}
    return EvaluationReport(
        report_id=args.report_id
        or f"{manifest.run_id}-s{manifest.seed}-r{args.token_rung:g}",
        run_id=manifest.run_id,
        eval_snapshot_sha=manifest.eval_snapshot_sha,
        created_at=utc_now(),
        ship_gates_pass=bool(gates.get("pass", args.ship_gates_pass)),
        weighted_nll=weighted_nll,
        category_nll=category_nll,
        metrics=metrics,
        suite_sizes={key: int(value.get("n", 0)) for key, value in suites.items()},
        seed=manifest.seed,
        token_rung=args.token_rung,
        artifact_size_bytes=args.artifact_size_bytes or size,
        warm_p95_seconds=args.warm_p95_seconds,
        hardware=_json_object(args.hardware_json),
        comparisons={
            key: int(value)
            for key, value in _json_object(args.comparisons_json).items()
        },
        metadata={
            "ranking_stable": args.ranking_stable,
            "loss_suite_complete": aggregate.get("complete"),
        },
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    store = _store(args)
    manifest = store.load_run(args.run_id)
    if manifest.lifecycle_state not in {"screened", "validated"}:
        raise ValueError("evaluate requires a screened or validated run")
    if args.scoreboard:
        scoreboard = _json_object(args.scoreboard)
    else:
        from slm_training.harnesses.model_build import ModelBuildConfig
        from slm_training.harnesses.model_build.eval_runner import evaluate_suites

        artifact = Path(manifest.artifact_uris[0])
        config = ModelBuildConfig(
            train_dir=Path(args.train_dir),
            test_dir=Path(args.test_dir),
            run_root=Path(args.run_root),
            run_id=manifest.run_id,
            seed=manifest.seed,
            device=args.device,
            model_name="twotower" if manifest.track == "twotower" else "stub",
        )
        if manifest.track == "twotower":
            scoreboard = evaluate_suites(
                config,
                ["smoke", "held_out", "adversarial", "ood", "rico_held"],
                checkpoint=artifact,
                write_gates=True,
            )
        else:
            from slm_training.models.causal_lm_openui import (
                CausalLMOpenUIConfig,
                CausalLMOpenUIPlugin,
            )

            plugin = CausalLMOpenUIPlugin.from_pretrained(
                CausalLMOpenUIConfig(
                    manifest.base_model_id,
                    manifest.base_model_revision,
                    device=args.device,
                    local_files_only=args.local_files_only,
                )
            )
            plugin.load(artifact)
            scoreboard = evaluate_suites(
                config,
                ["smoke", "held_out", "adversarial", "ood", "rico_held"],
                model=plugin,
                write_gates=True,
            )
    report = _report_from_scoreboard(manifest, scoreboard, args)
    path = store.write_report(report)
    if manifest.lifecycle_state == "screened":
        store.transition_run(
            manifest.run_id, "validated" if report.ship_gates_pass else "rejected"
        )
    print(
        json.dumps(
            {
                "report_sha": report.sha,
                "path": str(path),
                "ship_gates_pass": report.ship_gates_pass,
            },
            indent=2,
        )
    )
    return 0 if report.ship_gates_pass else 2


def cmd_promote(args: argparse.Namespace) -> int:
    store = _store(args)
    manifest = store.load_run(args.run_id)
    if manifest.lifecycle_state != "validated":
        raise ValueError("promotion requires a validated run")
    eval_snapshot = store.load_snapshot(manifest.eval_snapshot_sha)
    if eval_snapshot.metadata.get("kind") != "frozen_production_evaluation":
        raise ValueError("promotion requires a frozen production evaluation snapshot")
    if int(eval_snapshot.metadata.get("human_feedback_holdout_n", 0)) < 1:
        raise ValueError("promotion requires a never-trained human-feedback holdout")
    candidate = store.load_report(args.report)
    deployment_artifact = Path(args.deployment_artifact or manifest.artifact_uris[-1])
    if not deployment_artifact.exists():
        raise FileNotFoundError(deployment_artifact)
    deployment_size = (
        sum(
            item.stat().st_size
            for item in deployment_artifact.rglob("*")
            if item.is_file()
        )
        if deployment_artifact.is_dir()
        else deployment_artifact.stat().st_size
    )
    candidate = replace(candidate, artifact_size_bytes=deployment_size)
    reports = [store.load_report(item) for item in args.finalist_report]
    if candidate.run_id != manifest.run_id:
        raise ValueError("candidate report does not belong to the promoted run")
    finalist_manifests = [store.load_run(item.run_id) for item in [candidate, *reports]]
    family_fields = (
        "track",
        "parent_ids",
        "base_model_id",
        "base_model_revision",
        "architecture_sha",
        "tokenizer_sha",
        "parameter_shapes_sha",
        "data_snapshot_sha",
        "eval_snapshot_sha",
        "recipe_sha",
    )
    for field_name in family_fields:
        if len({repr(getattr(item, field_name)) for item in finalist_manifests}) != 1:
            raise ValueError(f"finalist reports mix incompatible {field_name}")
    if any(
        report.seed != item.seed
        for report, item in zip([candidate, *reports], finalist_manifests, strict=True)
    ):
        raise ValueError("evaluation report seed must match its run manifest")
    if any(
        report.eval_snapshot_sha != item.eval_snapshot_sha
        for report, item in zip([candidate, *reports], finalist_manifests, strict=True)
    ):
        raise ValueError("evaluation report snapshot must match its run manifest")
    failures: list[str] = []
    if args.parent_report:
        parent_report = store.load_report(args.parent_report)
        if manifest.parent_ids != (parent_report.run_id,):
            raise ValueError(
                "parent report must belong to the candidate's direct parent"
            )
        if parent_report.eval_snapshot_sha != candidate.eval_snapshot_sha:
            raise ValueError(
                "parent and candidate reports must use the same frozen evaluation"
            )
        failures.extend(promotion_failures(candidate, parent_report, reports))
    else:
        baseline_reports = [candidate, *reports]
        if any(not item.ship_gates_pass for item in baseline_reports):
            failures.append("baseline champion must pass every honest ship gate")
        if any(
            item.metadata.get("loss_suite_complete") is not True
            for item in baseline_reports
        ):
            failures.append("baseline champion requires complete loss suites")
        if any(
            item.metadata.get("ranking_stable") is not True for item in baseline_reports
        ):
            failures.append("baseline champion ranking must be stable")
        if candidate.weighted_nll is None:
            failures.append("baseline champion requires weighted NLL")
        for category in ("binding", "structural", "repair"):
            if category not in candidate.category_nll:
                failures.append(f"baseline champion requires {category} NLL")
        if any(
            int(item.suite_sizes.get("rico_held", 0)) < 1500
            for item in baseline_reports
        ):
            failures.append(
                "baseline champion requires rico_held n>=1500 for every finalist"
            )
        for metric in (
            "parse_rate",
            "placeholder_fidelity",
            "request_coverage",
            "structural_similarity",
        ):
            if metric not in candidate.metrics:
                failures.append(f"baseline champion requires {metric}")
        if len({item.seed for item in baseline_reports}) < 3:
            failures.append("baseline champion requires three seeds")
        if not {1.0, 3.0}.issubset({item.token_rung for item in baseline_reports}):
            failures.append("baseline champion requires 1x and 3x token rungs")
        if (
            candidate.artifact_size_bytes is None
            or candidate.artifact_size_bytes > 1_000_000_000
        ):
            failures.append("quantized artifact must be at most 1GB")
        if candidate.warm_p95_seconds is None or candidate.warm_p95_seconds > 15:
            failures.append("Windows warm p95 must be at most 15 seconds")
    if failures:
        raise ValueError("promotion rejected: " + "; ".join(failures))
    champion = store.champion(manifest.track)
    promoted = store.transition_run(manifest.run_id, "champion")
    pointer = ChampionPointer(
        pointer_id=_new_pointer_id(manifest.track, manifest.run_id),
        track=manifest.track,
        run_id=manifest.run_id,
        artifact_uri=str(deployment_artifact),
        manifest_sha=promoted.sha,
        evaluation_report_sha=candidate.sha,
        created_at=utc_now(),
        previous_run_id=champion.run_id if champion else None,
    )
    path = store.promote(pointer)
    print(json.dumps({"pointer_sha": pointer.sha, "path": str(path)}, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = _store(args)
    manifest = store.load_run(args.run_id)
    if manifest.lifecycle_state not in {"validated", "champion"}:
        raise ValueError("export requires a validated or champion run")
    if not manifest.artifact_uris:
        raise ValueError("run has no trained artifact")
    source = Path(manifest.artifact_uris[0])
    if manifest.track == "twotower":
        from slm_training.models.twotower import TwoTowerModel

        plugin = TwoTowerModel.from_checkpoint(source, device=args.device)
    else:
        from slm_training.models.causal_lm_openui import (
            CausalLMOpenUIConfig,
            CausalLMOpenUIPlugin,
        )

        plugin = CausalLMOpenUIPlugin.from_pretrained(
            CausalLMOpenUIConfig(
                manifest.base_model_id,
                manifest.base_model_revision,
                device=args.device,
                local_files_only=args.local_files_only,
            )
        )
        plugin.load(source)
    artifacts = plugin.export(args.output, format=args.format)
    size = sum(path.stat().st_size for path in artifacts)
    store.record_artifacts(manifest.run_id, (*manifest.artifact_uris, str(args.output)))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "size_bytes": size,
                "artifacts": [str(path) for path in artifacts],
            },
            indent=2,
        )
    )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    store = _store(args)
    parent = store.load_run(args.parent)
    children = validate_merge_manifests(
        parent, [store.load_run(item) for item in args.child]
    )
    paths = [Path(child.artifact_uris[0]) for child in children]
    output = merge_checkpoints(
        args.parent_checkpoint,
        paths,
        args.output,
        method=args.method,
        density=args.density,
    )
    merge = MergeManifest(
        merge_id=args.run_id,
        track=parent.track,
        parent_id=parent.run_id,
        child_ids=tuple(child.run_id for child in children),
        method=args.method,
        compatibility_sha=parent.compatibility_sha,
        output_uri=str(output),
        density=args.density,
        created_at=utc_now(),
    )
    path = store.write_merge(merge)
    challenger = replace(
        parent,
        run_id=args.run_id,
        parent_ids=(parent.run_id,),
        artifact_uris=(str(output),),
        metrics={},
        lifecycle_state="screened",
        initialization="parent",
        recipe={**dict(parent.recipe), "merge": merge.to_dict()},
        recipe_sha=merge.sha,
        code_sha=_git_sha(),
        created_at=utc_now(),
    )
    store.create_run(challenger)
    print(
        json.dumps(
            {"merge_sha": merge.sha, "path": str(path), "artifact": str(output)},
            indent=2,
        )
    )
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    store = _store(args)
    pointer = store.champion(args.track)
    if pointer is None:
        raise ValueError(f"no champion for track {args.track}")
    report = store.load_report(pointer.evaluation_report_sha)
    failures = deployment_failures(report)
    if failures and not args.rollback:
        raise ValueError("deployment rejected: " + "; ".join(failures))
    manifest = store.load_run(pointer.run_id)
    if manifest.lifecycle_state == "champion":
        store.transition_run(pointer.run_id, "deployed")
    path = store.deploy(pointer)
    print(json.dumps({"deployed": pointer.run_id, "path": str(path)}, indent=2))
    return 0


def cmd_lock_causal_base(args: argparse.Namespace) -> int:
    store = _store(args)
    reports = [store.load_report(item) for item in args.report]
    manifests = [store.load_run(report.run_id) for report in reports]
    expected = set(CAUSAL_BASE_CANDIDATES)
    actual = {manifest.base_model_id for manifest in manifests}
    if len(reports) != len(expected) or actual != expected:
        raise ValueError(
            "causal base bakeoff requires exactly one report for each pinned base: "
            + ", ".join(sorted(expected))
        )
    if any(manifest.track != "causal_lm" for manifest in manifests):
        raise ValueError("causal bakeoff reports must belong to causal_lm runs")
    for manifest in manifests:
        expected_revision = CAUSAL_BASE_CANDIDATES[manifest.base_model_id]
        if manifest.base_model_revision != expected_revision:
            raise ValueError(
                f"unpinned causal base revision for {manifest.base_model_id}"
            )
    if len({manifest.data_snapshot_sha for manifest in manifests}) != 1:
        raise ValueError("causal bakeoff runs must share a data snapshot")
    if len({manifest.eval_snapshot_sha for manifest in manifests}) != 1:
        raise ValueError("causal bakeoff runs must share an evaluation snapshot")
    if len({manifest.recipe_sha for manifest in manifests}) != 1:
        raise ValueError("causal bakeoff runs must share an identical LoRA recipe")
    for report in reports:
        required_metrics = {"semantic_score", "structural_similarity"}
        if not report.ship_gates_pass:
            raise ValueError(
                "causal base lock requires both candidates to pass full gates"
            )
        if not required_metrics.issubset(report.metrics):
            raise ValueError(
                "causal base reports require semantic and structural metrics"
            )
        if report.warm_p95_seconds is None or report.artifact_size_bytes is None:
            raise ValueError(
                "causal base reports require warm latency and artifact size"
            )
    winner = select_causal_base(reports)
    manifest = store.load_run(winner.run_id)
    path = store.lock_base(manifest)
    print(
        json.dumps(
            {
                "run_id": manifest.run_id,
                "base_model_id": manifest.base_model_id,
                "base_model_revision": manifest.base_model_revision,
                "path": str(path),
            },
            indent=2,
        )
    )
    return 0


def cmd_import_legacy(args: argparse.Namespace) -> int:
    kind = "hardware_smoke" if args.hardware_smoke else "legacy_evidence"
    manifest = RunManifest(
        run_id=args.run_id,
        track=args.track,
        parent_ids=(),
        base_model_id="unknown",
        base_model_revision="unknown",
        architecture_sha="unknown",
        tokenizer_sha="unknown",
        parameter_shapes_sha="unknown",
        data_snapshot_sha="unknown",
        eval_snapshot_sha="unknown",
        recipe_sha=content_sha({"legacy_path": str(args.path)}),
        code_sha="unknown",
        seed=0,
        hardware={},
        artifact_uris=(str(args.path),),
        metrics={},
        lifecycle_state="rejected",
        initialization="legacy",
        recipe={"legacy_path": str(args.path), "deployable": False},
        created_at=utc_now(),
        legacy_kind=kind,
    )
    path = _store(args).create_run(manifest)
    print(json.dumps({"kind": kind, "path": str(path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-root", type=Path, default=Path("outputs/lineage"))
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot-data")
    snap.add_argument("--snapshot-id", required=True)
    snap.add_argument("--source", type=Path, action="append", required=True)
    snap.add_argument("--target-token-count", type=int)
    snap.add_argument("--annotations-sha")
    snap.add_argument("--metadata-json")
    snap.set_defaults(func=cmd_snapshot)

    eval_snapshot = sub.add_parser("snapshot-eval")
    eval_snapshot.add_argument("--snapshot-id", required=True)
    eval_snapshot.add_argument(
        "--suite",
        action="append",
        required=True,
        help="Suite mapping NAME=records.jsonl (all five production suites required).",
    )
    eval_snapshot.add_argument("--human-feedback-holdout", type=Path, required=True)
    eval_snapshot.add_argument("--training-ids-json")
    eval_snapshot.set_defaults(func=cmd_snapshot_eval)

    init = sub.add_parser("init")
    init.add_argument("--track", choices=("twotower", "causal_lm"), required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--data-snapshot-sha", required=True)
    init.add_argument("--eval-snapshot-sha", required=True)
    init.add_argument("--base-model-id")
    init.add_argument("--base-model-revision")
    init.add_argument("--recipe-json")
    init.add_argument("--hardware-json")
    init.add_argument("--seed", type=int, default=0)
    init.set_defaults(func=cmd_init)

    branch = sub.add_parser("branch")
    branch.add_argument("--parent", required=True)
    branch.add_argument("--run-id", required=True)
    branch.add_argument("--data-snapshot-sha")
    branch.add_argument("--recipe-json")
    branch.add_argument("--hardware-json")
    branch.add_argument("--seed", type=int, default=0)
    branch.set_defaults(func=cmd_branch)

    train = sub.add_parser("train")
    train.add_argument("--run-id", required=True)
    train.add_argument("--train-dir", type=Path, required=True)
    train.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    train.add_argument("--target-token-count", type=int, required=True)
    train.add_argument("--token-rung", type=float, choices=TOKEN_RUNGS, required=True)
    train.add_argument("--max-steps", type=int, default=100000)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume", type=Path)
    train.add_argument("--local-files-only", action="store_true")
    train.add_argument("--sync-checkpoints", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--report-id")
    evaluate.add_argument(
        "--train-dir", type=Path, default=Path("outputs/train_data/v1")
    )
    evaluate.add_argument("--test-dir", type=Path, default=Path("outputs/test_data/v1"))
    evaluate.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--scoreboard")
    evaluate.add_argument("--weighted-nll", type=float)
    evaluate.add_argument("--category-nll-json")
    evaluate.add_argument("--metrics-json")
    evaluate.add_argument("--comparisons-json")
    evaluate.add_argument("--hardware-json")
    evaluate.add_argument("--artifact-size-bytes", type=int)
    evaluate.add_argument("--warm-p95-seconds", type=float)
    evaluate.add_argument("--token-rung", type=float, choices=TOKEN_RUNGS, default=1.0)
    evaluate.add_argument("--ship-gates-pass", action="store_true")
    evaluate.add_argument(
        "--ranking-stable", action=argparse.BooleanOptionalAction, default=True
    )
    evaluate.add_argument("--local-files-only", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    promote = sub.add_parser("promote")
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--report", required=True)
    promote.add_argument("--parent-report")
    promote.add_argument("--finalist-report", action="append", default=[])
    promote.add_argument("--deployment-artifact", type=Path)
    promote.set_defaults(func=cmd_promote)

    export = sub.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--format", choices=("onnx",), default="onnx")
    export.add_argument("--device", default="cpu")
    export.add_argument("--local-files-only", action="store_true")
    export.set_defaults(func=cmd_export)

    merge = sub.add_parser("merge")
    merge.add_argument("--run-id", required=True)
    merge.add_argument("--parent", required=True)
    merge.add_argument("--child", action="append", required=True)
    merge.add_argument("--parent-checkpoint", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--method", choices=("average", "ties"), required=True)
    merge.add_argument("--density", type=float, default=0.2)
    merge.set_defaults(func=cmd_merge)

    deploy = sub.add_parser("deploy")
    deploy.add_argument("--track", choices=("twotower", "causal_lm"), required=True)
    deploy.add_argument("--rollback", action="store_true")
    deploy.set_defaults(func=cmd_deploy)

    lock_base = sub.add_parser("lock-causal-base")
    lock_base.add_argument("--report", action="append", required=True)
    lock_base.set_defaults(func=cmd_lock_causal_base)

    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("--run-id", required=True)
    legacy.add_argument(
        "--track", choices=("twotower", "causal_lm"), default="twotower"
    )
    legacy.add_argument("--path", type=Path, required=True)
    legacy.add_argument("--hardware-smoke", action="store_true")
    legacy.set_defaults(func=cmd_import_legacy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
