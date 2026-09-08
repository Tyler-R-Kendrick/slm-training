"""Freeze all selected driver arms and their original finalization inputs."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from scripts.autoresearch_command_cursor import resolved_continuation_grant
from scripts.autotrain_cycle_context import register, writer
from slm_training.autoresearch.schemas import ExperimentSpec
from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.levers import MAX_RUN_SECONDS

# Explicit driver locals, not a general serialization of a running interpreter.
_FIELDS = (
    "campaign_id",
    "loop_id",
    "cycle",
    "upstream",
    "integration",
    "role",
    "cycle_intent",
    "effective_primary",
    "matrix",
    "promoting_champion",
    "open_champion",
    "replayed_confirmation",
    "control_eid",
    "candidate_eid",
    "order",
    "scheduled_order",
    "arm_seed",
    "arm_wall_minutes",
    "by_id",
    "replay_manifest_paths",
    "replay_manifests",
    "promotion_chunk_plan",
    "promote_formal_status",
    "promote_preflight_sha",
    "screening_multi",
    "screening_candidate_ids",
    "role_primary",
    "multi_arm_cfg",
    "fitted_candidate_count",
    "multi_arm_constraint",
    "selection_rule_locked",
    "multi_arm_skip",
    "train_version",
    "eval_version",
    "claim_for_role",
    "ar",
    "locked_designs",
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported driver input {type(value).__name__}")


def capture_inputs(scope):
    value = {key: scope[key] for key in _FIELDS}
    replay = scope["replay"]
    value["replay"] = (
        None
        if replay is None
        else {key: replay[key] for key in ("handoff", "action", "action_index")}
    )
    value["skip_slugs"] = sorted(
        scope["skip_slugs"]
        | (
            {scope["rec_slug"]}
            if scope["saturation_state"] is not None and scope["rec_slug"]
            else set()
        )
    )
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _write_manifest(path, manifest):
    if path.exists():
        old = ExperimentCampaignV1.model_validate_json(path.read_text())
        if old != manifest:
            raise ValueError("existing driver manifest differs; successor required")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        CampaignStore._replace_durable(path, manifest.model_dump_json(indent=2) + "\n")


def _manifest_path(store, continuous, value, experiment):
    from slm_training.autoresearch.climb_policy import load_climb_policy

    eid = experiment.experiment_id
    path = Path(
        value["replay_manifest_paths"].get(
            eid, store.root / "manifests" / f"{eid}.json"
        )
    )
    if any(
        e["event_type"] == "experiment_campaign_locked" and e["experiment_id"] == eid
        for e in store.verify_event_chain()
    ):
        _write_manifest(path, store.load_experiment_campaign(eid).manifest)
        return path
    if eid in value["replay_manifest_paths"]:
        return path
    if not (
        value["screening_multi"] and eid == value["candidate_eid"] and path.is_file()
    ):
        policy = load_climb_policy()
        manifest = continuous._manifest(
            store.campaign_id,
            experiment.model_dump(mode="json"),
            value["integration"],
            role=value["role"],
            policy=policy,
            cycle_intent=value["cycle_intent"],
            formal_preflight_sha256=value["promote_preflight_sha"]
            if (value["cycle_intent"] == "promote" and "-promote" in eid)
            else None,
            chunk_plan=value["promotion_chunk_plan"],
        )
        _write_manifest(path, manifest)
    return path


def _arm(store, cwd, continuous, value, eid):
    from scripts import autoresearch

    experiment_path = Path(value["by_id"][eid])
    experiment = ExperimentSpec.model_validate_json(experiment_path.read_text())
    path = _manifest_path(store, continuous, value, experiment)
    manifest = ExperimentCampaignV1.model_validate_json(path.read_text())
    lock = store.lock_experiment_campaign(manifest)
    campaign = store.load_campaign()
    commands = autoresearch.compile_commands(
        campaign, experiment, output_root=Path(value["root_arg"])
    )
    cmd = [
        *value["ar"],
        "run",
        "--campaign-id",
        store.campaign_id,
        "--experiment",
        str(experiment_path),
        "--campaign-manifest",
        str(path),
        "--execute",
        "--experiment-wall-seconds",
        f"{value['arm_wall_minutes'] * 60:.6f}",
    ]
    files = [experiment_path, path]
    reuse = value["replay_manifests"].get(eid, {}).get("train_reuse")
    if reuse is not None:
        cmd.extend(["--reuse-train-run", str(reuse["run_dir"])])
        lineage = tuple(Path(p) for p in reuse["manifest_paths"])
        for item in lineage:
            cmd.extend(["--reuse-train-manifest", str(item)])
        commands, _ = autoresearch._prepare_reused_training(
            campaign=campaign,
            experiment=experiment,
            target_manifest=manifest,
            commands=commands,
            source_run=Path(reuse["run_dir"]),
            lineage_paths=lineage,
        )
        files.extend([*lineage, Path(reuse["run_dir"]) / "train_summary.json"])
    return {
        "cmd": cmd,
        "commands": commands,
        "manifest_digest": lock.manifest_sha256,
        "experiment_path": str(experiment_path),
    }, files


def prepare_cycle(cwd, root, continuous, value, *, spent_seconds):
    import time
    from slm_training.autoresearch.climb_policy import load_climb_policy

    started = time.monotonic()
    store = CampaignStore(value["campaign_id"], root)
    total_seconds = store.load_campaign().budget.logical_seconds
    value = {
        **value,
        "schema_version": "driver_cycle/v1",
        "root_arg": str(root),
        "cwd": str(Path(cwd).resolve()),
        "total_seconds": total_seconds,
        "policy_sha256": load_climb_policy().sha256,
        "execution_identity": resolved_continuation_grant(
            cwd, total_seconds
        ).execution_identity,
    }
    value["order"] = list(
        dict.fromkeys(eid for eid in value["order"] if eid in value["by_id"])
    )
    value["arms"], value["files"] = {}, {}
    for eid in value["order"]:
        arm, files = _arm(store, cwd, continuous, value, eid)
        value["arms"][eid] = arm
        value["files"].update(
            {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        )
    value["initial_spent_seconds"] = spent_seconds + time.monotonic() - started
    with writer(root, value["loop_id"]) as runtime:
        register(store, runtime, value)
    return value


def _recorded_pair(store, control_id, candidate_id):
    from scripts.autotrain_cycle_context import read_artifact
    from slm_training.autoresearch.preflight.search_binding import require_attempt_locks

    for event in reversed(store.verify_event_chain()):
        if event["event_type"] != "experiment_design_locked":
            continue
        pair = read_artifact(store, "treatment_designs", event["artifact_sha256"])
        if pair["arm_ids"] == [control_id, candidate_id]:
            for arm_id in pair["arm_ids"]:
                require_attempt_locks(store, pair, arm_id)
            return pair
    raise ValueError("recorded driver requires the original paired design lock")


@dataclass(frozen=True)
class RecordedCycleSelection:
    campaign_id: str
    loop_id: str
    control_id: str
    candidate_ids: tuple[str, ...]
    manifest_paths: dict[str, Path]
    arm_wall_seconds: float


def prepare_recorded_cycle(cwd, root, continuous, selection, *, spent_seconds=0):
    """Adopt one recorded screening pair, without init/research or invented results.

    The caller first persists the canonical campaign, formed hypothesis matrix,
    applicable campaign manifests and paired design. This constructor reuses all
    four producers; it does not manufacture a scientific design from artifacts.
    The full run_cycle constructor remains the owner for multi-arm/promotion work.
    """
    import math
    import sys
    from scripts import autoresearch

    campaign_id, loop_id, control_id = (
        selection.campaign_id,
        selection.loop_id,
        selection.control_id,
    )
    candidate_ids, manifest_paths = selection.candidate_ids, selection.manifest_paths
    arm_wall_seconds = selection.arm_wall_seconds
    candidates = list(candidate_ids)
    if len(candidates) != 1 or control_id in candidates:
        raise ValueError("recorded screening requires exactly one distinct pair")
    if (
        not math.isfinite(arm_wall_seconds)
        or not 0 < arm_wall_seconds <= MAX_RUN_SECONDS
    ):
        raise ValueError("invalid bounded recorded arm allowance")
    store = CampaignStore(campaign_id, root)
    matrix = autoresearch._latest_formed_matrix(store, required=True)
    order = [control_id, *candidates]
    experiments = {h.experiment.experiment_id: h.experiment for h in matrix.hypotheses}
    if set(order) - experiments.keys() or set(manifest_paths) != set(order):
        raise ValueError("recorded pair lacks exact matrix/manifest coverage")
    manifests = {
        eid: ExperimentCampaignV1.model_validate_json(
            Path(manifest_paths[eid]).read_text()
        )
        for eid in order
    }
    for eid, manifest in manifests.items():
        if (
            manifest.experiment_id != eid
            or manifest.campaign_id != campaign_id
            or store.load_experiment_campaign(eid).manifest != manifest
            or manifest.claim_class
            not in {"wiring", "fixture", "diagnostic", "screening"}
        ):
            raise ValueError("recorded applicable manifest differs from original lock")
    primary = next(e for e in manifests[candidates[0]].endpoints if e.role == "primary")
    pair = _recorded_pair(store, control_id, candidates[0])
    value = {name: None for name in _FIELDS}
    value.update(
        campaign_id=campaign_id,
        loop_id=loop_id,
        cycle=1,
        upstream=manifests[control_id].source_commit,
        integration=manifests[control_id].source_commit,
        role="screening",
        cycle_intent="screening",
        effective_primary=primary.metric,
        matrix=matrix.model_dump(mode="json"),
        control_eid=control_id,
        candidate_eid=candidates[0],
        order=order,
        scheduled_order=order,
        arm_seed=experiments[control_id].knobs.seed,
        arm_wall_minutes=arm_wall_seconds / 60,
        by_id={
            eid: str(store.write_artifact("experiments", experiments[eid]))
            for eid in order
        },
        replay_manifest_paths={eid: str(path) for eid, path in manifest_paths.items()},
        replay_manifests={},
        screening_multi=False,
        screening_candidate_ids=candidates,
        role_primary=primary.model_dump(mode="json"),
        multi_arm_skip=[],
        train_version=experiments[control_id].knobs.train_version,
        eval_version=experiments[control_id].knobs.eval_version,
        claim_for_role=manifests[candidates[0]].claim_class,
        ar=[sys.executable, "-m", "scripts.autoresearch", "--root", str(root)],
        locked_designs={candidates[0]: pair},
        skip_slugs=[],
        replay=None,
    )
    return prepare_cycle(cwd, root, continuous, value, spent_seconds=spent_seconds)
