"""SLM-287 locked five-seed/two-config baseline protocol.

This module deliberately separates protocol validation from model execution.
The shared evaluator remains the only source of per-record scores; this owner
freezes the inputs, rejects incomplete cells, and derives paired statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignBudget,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.evals.power_protocol import (
    cluster_bootstrap_ci,
    exact_paired_binary_test,
    mde_simulation,
)
from slm_training.versioning import build_version_stamp

EXPERIMENT_ID = "slm287-locked-power-protocol"
SEEDS = (0, 1, 2, 3, 4)
BACKENDS = ("scratch_design_off", "scratch_design_on")
VARIANTS = ("raw", "constrained", "repaired")
RECORD_METRICS = (
    "binding_aware_meaningful_v2",
    "binder_reference_f1",
    "latency_ms",
)
CELL_METRICS = (
    "compute_proxy_forwards",
    "peak_rss_bytes",
)
METRICS = RECORD_METRICS + CELL_METRICS
LOCKED_MANIFEST = Path(
    "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
)
E297_TRAIN_DIR = Path(
    "src/slm_training/resources/data/train/e297_semantic_contract_judge_v1"
)
E297_RECORD_COUNT = 480
LOCAL_TRAIN_TARGET_TOKENS = 5_000


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_eval_root(
    output_dir: Path, *, seed: int, backend: str, shard_index: int
) -> Path:
    """Return the evaluator workspace owned by exactly one local shard."""
    return output_dir / "locked_eval" / f"seed{seed}_{backend}_shard{shard_index}"


def locked_cell_manifest_path(
    output_dir: Path, *, protocol: "LockedPowerProtocol", seed: int, backend: str
) -> Path:
    """Return the one immutable trained-cell manifest consumed by every shard."""
    return (
        output_dir
        / "trained_cells"
        / str(protocol.to_dict()["protocol_sha256"])
        / f"seed{seed}_{backend}.json"
    )


@dataclass(frozen=True)
class LockedPowerProtocol:
    manifest_sha256: str
    record_ids: tuple[str, ...]
    recipe: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "Slm287LockedPowerProtocolV1",
            "experiment_id": EXPERIMENT_ID,
            "locked_eval_manifest_sha256": self.manifest_sha256,
            "record_ids": list(self.record_ids),
            "seeds": list(SEEDS),
            "backends": list(BACKENDS),
            "variants": list(VARIANTS),
            "recipe": self.recipe,
            "protocol_sha256": _sha(
                {
                    "manifest": self.manifest_sha256,
                    "record_ids": self.record_ids,
                    "recipe": self.recipe,
                    "seeds": SEEDS,
                    "backends": BACKENDS,
                    "variants": VARIANTS,
                }
            ),
        }


def load_locked_protocol(
    path: Path = LOCKED_MANIFEST, *, limit: int | None = None
) -> LockedPowerProtocol:
    """Load the immutable locked-test records and freeze the local recipe."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("manifest_sha256") or "")
    actual = _sha(
        {
            "schema": payload.get("schema"),
            "rows": payload.get("rows"),
            "metadata": payload.get("metadata"),
        }
    )
    if payload.get("schema") != "LockedPromotionManifestV1" or expected != actual:
        raise ValueError("locked promotion manifest digest mismatch")
    record_ids = tuple(
        str(row["record"]["id"])
        for row in payload.get("rows", [])
        if row.get("partition") == "locked_test"
    )
    if limit is not None:
        record_ids = record_ids[: max(0, int(limit))]
    if not record_ids:
        raise ValueError("locked protocol requires at least one locked_test record")
    return LockedPowerProtocol(
        manifest_sha256=expected,
        record_ids=record_ids,
        recipe={
            "model_name": "twotower",
            "output_tokenizer": "choice",
            "device": "cpu",
            "precision": "float32",
            "optimizer": "adamw",
            "train": {
                "corpus": str(E297_TRAIN_DIR),
                "manifest_sha256": _file_sha256(E297_TRAIN_DIR / "manifest.json"),
                "record_count": E297_RECORD_COUNT,
                "max_updates": 10_000,
                "target_token_budget": LOCAL_TRAIN_TARGET_TOKENS,
                "batch_size": 4,
            },
            "prompt_source": "e297.record.prompt",
            "target_source": "e297.record.openui",
            "decode": {"raw": False, "constrained": True, "repaired": True},
            "backends": {
                "scratch_design_off": {
                    "context_backend": "scratch",
                    "design_md_in_context": False,
                },
                "scratch_design_on": {
                    "context_backend": "scratch",
                    "design_md_in_context": True,
                },
            },
        },
    )


def build_locked_campaign(protocol: LockedPowerProtocol) -> ExperimentCampaignV1:
    """Build the immutable AP-007 diagnostic campaign before shard execution."""
    stamp = build_version_stamp(
        "harness.experiments", "evals.power_protocol", "data.locked_eval_manifest"
    )
    return ExperimentCampaignV1(
        campaign_id="slm287-locked-power",
        experiment_id=EXPERIMENT_ID,
        hypothesis="DESIGN.md context changes locked baseline outcomes under paired local decoding.",
        decision="Measure rather than select a baseline configuration.",
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="meaning_v2", metric="binding_aware_meaningful_v2",
                role="primary", direction="increase", minimum_effect=0.02,
            ),
            CampaignEndpointV1(
                endpoint_id="binder_f1", metric="binder_reference_f1",
                role="secondary", direction="increase", minimum_effect=0.02,
            ),
        ),
        arms=tuple(
            CampaignArmV1(
                arm_id=backend,
                role="control" if backend == BACKENDS[0] else "candidate",
                config_sha256=_sha(protocol.recipe["backends"][backend]),
            )
            for backend in BACKENDS
        ),
        seeds=SEEDS,
        budget=CampaignBudget(max_experiments=10, max_wall_minutes=2),
        stopping_rules=("Reject any timed-out shard; merge only exact complete coverage.",),
        controls=(
            CampaignControlV1(
                control_id="paired_design_off",
                description="Same seed and locked records without DESIGN.md context.",
                kind="quality",
            ),
            CampaignControlV1(
                control_id="raw_decode",
                description="Raw decode is retained beside constrained and repaired variants.",
                kind="negative",
            ),
        ),
        negative_controls=("raw_decode",),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="locked_quality", hypothesis_ids=("meaning_v2", "binder_f1"), alpha=0.05
            ),
        ),
        promotion_gates=(
            CampaignGateV1(gate_id="diagnostic_only", endpoint_id="meaning_v2", operator="ge", threshold=-1.0),
        ),
        rollback_gates=(
            CampaignGateV1(gate_id="timeout_rejects", endpoint_id="meaning_v2", operator="lt", threshold=2.0),
        ),
        artifact_requirements=(
            ArtifactRequirementV1(kind="version_stamp"),
            ArtifactRequirementV1(kind="seed_result", minimum_count=10),
            ArtifactRequirementV1(kind="paired_examples"),
            ArtifactRequirementV1(kind="endpoint_result"),
            ArtifactRequirementV1(kind="agentevals"),
            ArtifactRequirementV1(kind="agentv"),
        ),
        claim_class="diagnostic",
        source_commit=str(stamp["code_commit"]),
        source_dirty=bool(stamp["code_dirty"]),
        author="slm-training",
        locked_eval_manifest_sha256=protocol.manifest_sha256,
        created_at="2026-07-25T00:00:00Z",
    )


def validate_cells(protocol: LockedPowerProtocol, cells: list[dict[str, Any]]) -> None:
    """Fail closed unless every declared seed/backend has paired full evidence."""
    expected = {(seed, backend) for seed in SEEDS for backend in BACKENDS}
    observed = {(cell.get("seed"), cell.get("backend")) for cell in cells}
    if observed != expected or len(cells) != len(expected):
        raise ValueError("locked power protocol requires exactly five seeds by two backends")
    ids = set(protocol.record_ids)
    initial_by_seed: dict[int, str] = {}
    for cell in cells:
        if cell.get("locked_eval_manifest_sha256") != protocol.manifest_sha256:
            raise ValueError("cell locked manifest digest mismatch")
        if cell.get("initial_tensor_sha256") != cell.get("repeat_initial_tensor_sha256"):
            raise ValueError("identical initialization was not bit exact")
        seed = int(cell["seed"])
        initial = str(cell["initial_tensor_sha256"])
        if seed in initial_by_seed and initial_by_seed[seed] != initial:
            raise ValueError("paired backends must share bit-exact initialization")
        initial_by_seed[seed] = initial
        if cell.get("train_data_manifest_sha256") != _file_sha256(E297_TRAIN_DIR / "manifest.json"):
            raise ValueError("cell did not retain the locked E297 training digest")
        if not cell.get("checkpoint_sha256") or not cell.get("cell_manifest_sha256"):
            raise ValueError("cell lacks the trained checkpoint manifest identity")
        records = cell.get("records")
        if not isinstance(records, dict) or set(records) != ids:
            raise ValueError("cell records are not the frozen locked record set")
        cell_metrics = cell.get("cell_metrics")
        if not isinstance(cell_metrics, dict) or set(cell_metrics) != set(VARIANTS):
            raise ValueError("cell lacks raw/constrained/repaired cell metrics")
        for record in records.values():
            if not isinstance(record, dict) or set(record) != set(VARIANTS):
                raise ValueError("cell lacks raw/constrained/repaired evidence")
            for values in record.values():
                if any(
                    metric not in values or not math.isfinite(float(values[metric]))
                    for metric in RECORD_METRICS
                ):
                    raise ValueError("cell metric is missing or non-finite")
        for values in cell_metrics.values():
            if any(
                metric not in values or not math.isfinite(float(values[metric]))
                for metric in CELL_METRICS
            ):
                raise ValueError("cell metric is missing or non-finite")


def locked_shard_ids(
    protocol: LockedPowerProtocol, *, shard_index: int, shard_count: int
) -> tuple[str, ...]:
    """Return the deterministic locked IDs assigned to one capped local shard."""
    if not 0 < shard_count <= len(protocol.record_ids):
        raise ValueError("locked shard count must be between one and locked record count")
    if not 0 <= shard_index < shard_count:
        raise IndexError(f"shard_index {shard_index} out of range {shard_count}")
    seed = str(protocol.to_dict()["protocol_sha256"])
    ordered = sorted(
        protocol.record_ids,
        key=lambda record_id: hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest(),
    )
    return tuple(record_id for rank, record_id in enumerate(ordered) if rank % shard_count == shard_index)


def merge_locked_shards(
    protocol: LockedPowerProtocol,
    shards: list[dict[str, Any]],
    *,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Merge exact shard artifacts, rejecting any missing or duplicated evidence."""
    expected = {
        (seed, backend, shard_index)
        for seed in SEEDS
        for backend in BACKENDS
        for shard_index in range(shard_count)
    }
    observed = {
        (int(shard.get("seed", -1)), str(shard.get("backend")), int(shard.get("shard_index", -1)))
        for shard in shards
    }
    if observed != expected or len(shards) != len(expected):
        raise ValueError("locked power protocol requires every seed/backend/shard artifact")
    protocol_sha = protocol.to_dict()["protocol_sha256"]
    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for shard in shards:
        if shard.get("protocol_sha256") != protocol_sha:
            raise ValueError("shard protocol digest mismatch")
        if int(shard.get("shard_count", -1)) != shard_count:
            raise ValueError("shard count mismatch")
        expected_ids = set(
            locked_shard_ids(
                protocol,
                shard_index=int(shard["shard_index"]),
                shard_count=shard_count,
            )
        )
        if set(shard.get("record_ids") or ()) != expected_ids:
            raise ValueError("shard record IDs do not match deterministic assignment")
        by_cell.setdefault((int(shard["seed"]), str(shard["backend"])), []).append(shard)

    campaign_digests = {str(shard.get("campaign_manifest_sha256") or "") for shard in shards}
    if len(campaign_digests) != 1 or not next(iter(campaign_digests)):
        raise ValueError("shards must share one locked campaign manifest")

    cells: list[dict[str, Any]] = []
    for (seed, backend), cell_shards in sorted(by_cell.items()):
        first = cell_shards[0]
        initial = str(first.get("initial_tensor_sha256"))
        repeated = str(first.get("repeat_initial_tensor_sha256"))
        if any(
            str(shard.get("initial_tensor_sha256")) != initial
            or str(shard.get("repeat_initial_tensor_sha256")) != repeated
            for shard in cell_shards
        ):
            raise ValueError("shards rebuilt a cell with non-identical initialization")
        for key in (
            "train_data_manifest_sha256",
            "checkpoint_sha256",
            "cell_manifest_sha256",
        ):
            value = first.get(key)
            if not value or any(shard.get(key) != value for shard in cell_shards):
                raise ValueError("shards did not reuse one immutable trained cell")
        records: dict[str, Any] = {}
        cell_metrics = {
            variant: {"compute_proxy_forwards": 0.0, "peak_rss_bytes": 0.0}
            for variant in VARIANTS
        }
        for shard in cell_shards:
            shard_records = shard.get("records")
            if not isinstance(shard_records, dict) or set(records) & set(shard_records):
                raise ValueError("shard records are missing or duplicated")
            records.update(shard_records)
            shard_metrics = shard.get("cell_metrics")
            if not isinstance(shard_metrics, dict) or set(shard_metrics) != set(VARIANTS):
                raise ValueError("shard cell metrics are missing")
            for variant in VARIANTS:
                cell_metrics[variant]["compute_proxy_forwards"] += float(
                    shard_metrics[variant]["compute_proxy_forwards"]
                )
                cell_metrics[variant]["peak_rss_bytes"] = max(
                    cell_metrics[variant]["peak_rss_bytes"],
                    float(shard_metrics[variant]["peak_rss_bytes"]),
                )
        cells.append(
            {
                "seed": seed,
                "backend": backend,
                "locked_eval_manifest_sha256": protocol.manifest_sha256,
                "initial_tensor_sha256": initial,
                "repeat_initial_tensor_sha256": repeated,
                "train_data_manifest_sha256": first["train_data_manifest_sha256"],
                "checkpoint_sha256": first["checkpoint_sha256"],
                "cell_manifest_sha256": first["cell_manifest_sha256"],
                "records": records,
                "cell_metrics": cell_metrics,
            }
        )
    return cells


def summarize_cells(protocol: LockedPowerProtocol, cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a complete protocol without selecting a best seed."""
    validate_cells(protocol, cells)
    by_key = {(int(cell["seed"]), str(cell["backend"])): cell for cell in cells}
    summaries: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        rows = [
            cell["records"][record_id][variant]
            for cell in cells
            for record_id in protocol.record_ids
        ]
        summaries[variant] = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in RECORD_METRICS
        }
        summaries[variant].update(
            {
                metric: float(
                    np.mean([float(cell["cell_metrics"][variant][metric]) for cell in cells])
                )
                for metric in CELL_METRICS
            }
        )
    control, candidate = BACKENDS
    left: list[float] = []
    right: list[float] = []
    differences: list[float] = []
    seed_ids: list[int] = []
    target_ids: list[str] = []
    for seed in SEEDS:
        for record_id in protocol.record_ids:
            left_value = float(by_key[seed, control]["records"][record_id]["repaired"]["binding_aware_meaningful_v2"])
            right_value = float(by_key[seed, candidate]["records"][record_id]["repaired"]["binding_aware_meaningful_v2"])
            left.append(left_value)
            right.append(right_value)
            differences.append(right_value - left_value)
            seed_ids.append(seed)
            target_ids.append(record_id)
    seed_effects = [
        float(np.mean([difference for difference, row_seed in zip(differences, seed_ids) if row_seed == seed]))
        for seed in SEEDS
    ]
    target_effects = [
        float(np.mean([difference for difference, row_target in zip(differences, target_ids) if row_target == record_id]))
        for record_id in protocol.record_ids
    ]
    paired = {
        "primary_metric": "binding_aware_meaningful_v2",
        "bootstrap_ci": cluster_bootstrap_ci(
            differences,
            target_ids,
            lambda values: float(np.mean(values)),
        ),
        "exact_paired_test": exact_paired_binary_test([int(v) for v in left], [int(v) for v in right]),
        "pairing": "record_id paired within seed; target-cluster bootstrap",
    }
    endpoints: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        endpoint_intervals: dict[str, Any] = {}
        for metric in RECORD_METRICS:
            differences = [
                float(by_key[seed, candidate]["records"][record_id][variant][metric])
                - float(by_key[seed, control]["records"][record_id][variant][metric])
                for seed in SEEDS
                for record_id in protocol.record_ids
            ]
            endpoint_intervals[metric] = cluster_bootstrap_ci(
                differences,
                [record_id for _seed in SEEDS for record_id in protocol.record_ids],
                lambda values: float(np.mean(values)),
            )
        for metric in CELL_METRICS:
            differences = [
                float(by_key[seed, candidate]["cell_metrics"][variant][metric])
                - float(by_key[seed, control]["cell_metrics"][variant][metric])
                for seed in SEEDS
            ]
            endpoint_intervals[metric] = cluster_bootstrap_ci(
                differences,
                list(SEEDS),
                lambda values: float(np.mean(values)),
            )
        endpoints[variant] = endpoint_intervals
    paired["endpoints"] = endpoints
    mde = mde_simulation(
        base_rate=float(np.mean(left)),
        sigma_seed=float(np.std(seed_effects, ddof=0)),
        sigma_target=float(np.std(target_effects, ddof=0)),
        n_targets=len(protocol.record_ids),
        paths_per_target=1,
        n_seeds=len(SEEDS),
        n_simulations=100,
        effect_sizes=[0.02, 0.05, 0.08, 0.12, 0.2],
        seed=0,
    )
    return {
        "schema": "Slm287LockedPowerResultV1",
        "claim_class": "diagnostic",
        "protocol": protocol.to_dict(),
        "cells": cells,
        "aggregate": summaries,
        "paired": paired,
        "mde": mde,
        "status": "complete_local_grid",
        "ship_gate": "not_evaluated",
        "human_rating_gate": "not_required",
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "evals.power_protocol",
            "data.locked_eval_manifest",
        ),
    }


def _locked_train_config(
    protocol: LockedPowerProtocol, *, output_dir: Path, seed: int, backend: str
):
    """Freeze the one local E297 training recipe for a protocol cell."""
    from slm_training.harnesses.model_build import ModelBuildConfig

    overrides = protocol.recipe["backends"][backend]
    protocol_sha = str(protocol.to_dict()["protocol_sha256"])
    return ModelBuildConfig(
        train_dir=E297_TRAIN_DIR,
        run_root=output_dir / "trained_cells" / protocol_sha / f"seed{seed}_{backend}",
        run_id="train",
        seed=seed,
        device="cpu",
        model_name="twotower",
        output_tokenizer="choice",
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend=str(overrides["context_backend"]),
        local_files_only=True,
        design_md_in_context=bool(overrides["design_md_in_context"]),
        optimizer_name="adamw",
        steps=int(protocol.recipe["train"]["max_updates"]),
        batch_size=int(protocol.recipe["train"]["batch_size"]),
        target_token_budget=int(protocol.recipe["train"]["target_token_budget"]),
        grammar_ltr_max_tokens=8,
        grammar_ltr_primary=True,
        gen_steps=1,
        run_class="scratch_matrix",
    )


def _validate_locked_cell_manifest(
    protocol: LockedPowerProtocol,
    payload: dict[str, Any],
    *,
    seed: int,
    backend: str,
) -> dict[str, Any]:
    """Reject stale, partial, or mismatched training before evaluating a shard."""
    if payload.get("schema") != "Slm287LockedTrainedCellV1":
        raise ValueError("trained cell manifest schema mismatch")
    if payload.get("protocol_sha256") != protocol.to_dict()["protocol_sha256"]:
        raise ValueError("trained cell protocol digest mismatch")
    if payload.get("locked_eval_manifest_sha256") != protocol.manifest_sha256:
        raise ValueError("trained cell locked manifest digest mismatch")
    if int(payload.get("seed", -1)) != seed or payload.get("backend") != backend:
        raise ValueError("trained cell does not match requested seed/backend")
    if payload.get("train_record_count") != E297_RECORD_COUNT:
        raise ValueError("trained cell did not use the complete E297 corpus")
    expected_data_sha = _file_sha256(E297_TRAIN_DIR / "manifest.json")
    if payload.get("train_data_manifest_sha256") != expected_data_sha:
        raise ValueError("trained cell E297 manifest digest mismatch")
    if payload.get("target_token_budget") != LOCAL_TRAIN_TARGET_TOKENS:
        raise ValueError("trained cell target-token budget mismatch")
    if payload.get("initial_tensor_sha256") != payload.get("repeat_initial_tensor_sha256"):
        raise ValueError("trained cell initialization was not bit exact")
    checkpoint = Path(str(payload.get("checkpoint_path") or ""))
    if not checkpoint.is_file() or _file_sha256(checkpoint) != payload.get("checkpoint_sha256"):
        raise ValueError("trained cell checkpoint is missing or has changed")
    return payload


def load_locked_trained_cell(
    protocol: LockedPowerProtocol, *, output_dir: Path, seed: int, backend: str
) -> tuple[dict[str, Any], Path]:
    """Load the required frozen cell checkpoint; shards are evaluation-only."""
    path = locked_cell_manifest_path(
        output_dir, protocol=protocol, seed=seed, backend=backend
    )
    if not path.is_file():
        raise ValueError("locked shard requires a completed locked-train cell manifest")
    payload = _validate_locked_cell_manifest(
        protocol, json.loads(path.read_text(encoding="utf-8")), seed=seed, backend=backend
    )
    return payload, path


def prepare_locked_trained_cell(
    protocol: LockedPowerProtocol, *, output_dir: Path, seed: int, backend: str
) -> dict[str, Any]:
    """Train exactly one frozen local E297 checkpoint for a seed/backend cell."""
    from slm_training.harnesses.model_build import build_model, train
    from slm_training.harnesses.model_build.data import load_train_records

    if backend not in BACKENDS or seed not in SEEDS:
        raise ValueError("locked train must name a declared seed and backend")
    path = locked_cell_manifest_path(
        output_dir, protocol=protocol, seed=seed, backend=backend
    )
    if path.is_file():
        return load_locked_trained_cell(
            protocol, output_dir=output_dir, seed=seed, backend=backend
        )[0]
    records = load_train_records(E297_TRAIN_DIR)
    if len(records) != E297_RECORD_COUNT:
        raise ValueError("E297 record count no longer matches the locked recipe")
    if {record.id for record in records} & set(protocol.record_ids):
        raise ValueError("locked evaluation rows must never enter local training")
    config = _locked_train_config(protocol, output_dir=output_dir, seed=seed, backend=backend)
    model = build_model(config, records)

    def state_sha(value: Any) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(value.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    initial = state_sha(model)
    repeated = state_sha(build_model(config, records))
    summary = train(config, model=model)
    checkpoint = Path(str(summary["checkpoint"]))
    payload = {
        "schema": "Slm287LockedTrainedCellV1",
        "protocol_sha256": protocol.to_dict()["protocol_sha256"],
        "locked_eval_manifest_sha256": protocol.manifest_sha256,
        "seed": seed,
        "backend": backend,
        "train_data_manifest": str(E297_TRAIN_DIR / "manifest.json"),
        "train_data_manifest_sha256": _file_sha256(E297_TRAIN_DIR / "manifest.json"),
        "train_record_count": len(records),
        "target_token_budget": LOCAL_TRAIN_TARGET_TOKENS,
        "initial_tensor_sha256": initial,
        "repeat_initial_tensor_sha256": repeated,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "train_summary": {
            key: summary.get(key)
            for key in ("steps", "seen_target_tokens", "stopped_on", "data_manifest_sha")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return _validate_locked_cell_manifest(protocol, payload, seed=seed, backend=backend)


def execute_local_shard(
    protocol: LockedPowerProtocol,
    *,
    manifest_path: Path,
    output_dir: Path,
    seed: int,
    backend: str,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Run one deterministic, bounded local cell shard through the evaluator.

    The temporary suite is a byte-for-byte projection of the locked rows.  It
    exists only under the run root so the shared evaluator can score it; the
    manifest digest remains the authority for membership.
    """
    from slm_training.dsl.schema import ExampleRecord
    from dataclasses import replace

    from slm_training.harnesses.model_build import build_model
    from slm_training.harnesses.model_build.eval_runner import (
        evaluate_grammar_leakage_audit,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_by_id = {
        str(row["record"]["id"]): ExampleRecord.from_dict(row["record"])
        for row in payload["rows"]
        if row.get("partition") == "locked_test"
    }
    shard_ids = locked_shard_ids(
        protocol,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    records = [record_by_id[record_id] for record_id in shard_ids]
    if backend not in BACKENDS or seed not in SEEDS:
        raise ValueError("locked shard must name a declared seed and backend")
    # Every shard owns its evaluator root.  Concurrent local cells must never
    # race on a shared manifest and accidentally score another shard's records.
    eval_root = locked_eval_root(
        output_dir, seed=seed, backend=backend, shard_index=shard_index
    )
    suite_dir = eval_root / "suites" / f"locked_power_{seed}_{backend}_{shard_index}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    records_path = suite_dir / "records.jsonl"
    records_path.write_text(
        "".join(_canonical(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    suite_name = f"locked_power_{seed}_{backend}_{shard_index}"
    (eval_root / "manifest.json").write_text(
        json.dumps({"suites": {suite_name: str(records_path)}}, indent=2) + "\n",
        encoding="utf-8",
    )
    cell, cell_path = load_locked_trained_cell(
        protocol, output_dir=output_dir, seed=seed, backend=backend
    )
    config = replace(
        _locked_train_config(protocol, output_dir=output_dir, seed=seed, backend=backend),
        test_dir=eval_root,
        suite=suite_name,
        run_id=f"{EXPERIMENT_ID}/{backend}/seed{seed}/shard{shard_index}",
        decode_timeout_seconds=5.0,
        runtime_override_fields=frozenset(),
    )
    # Shards load the sole trained checkpoint for this cell; they never rebuild
    # or fit a model from frozen evaluation records.
    model = build_model(config, [], checkpoint=Path(cell["checkpoint_path"]))
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    audit = evaluate_grammar_leakage_audit(
        config,
        model=model,
        publish_agentv=False,
        variant_names=VARIANTS,
    )
    if any(
        int(audit["variants"][variant].get("decode_timeout_count") or 0)
        for variant in VARIANTS
    ):
        raise TimeoutError("locked local shard timed out; no numeric power result is valid")
    peak_rss = max(before_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    scored: dict[str, dict[str, dict[str, float]]] = {}
    cell_metrics: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        details = audit["variants"][variant].get("details") or []
        values: dict[str, dict[str, float]] = {}
        for detail in details:
            values[str(detail["id"])] = {
                "binding_aware_meaningful_v2": float(
                    bool(detail.get("binding_aware_meaningful_v2"))
                ),
                "binder_reference_f1": float(detail.get("binder_reference_f1") or 0.0),
                "latency_ms": float(detail.get("latency_ms") or 0.0),
            }
        scored[variant] = values
        cell_metrics[variant] = {
            "compute_proxy_forwards": float(
                (audit["variants"][variant].get("decode_telemetry") or {}).get("forwards", 0)
            ),
            "peak_rss_bytes": float(peak_rss * 1024),
        }
    return {
        "schema": "Slm287LockedPowerShardV1",
        "protocol_sha256": protocol.to_dict()["protocol_sha256"],
        "shard_count": shard_count,
        "shard_index": shard_index,
        "record_ids": list(shard_ids),
        "seed": seed,
        "backend": backend,
        "locked_eval_manifest_sha256": protocol.manifest_sha256,
        "train_data_manifest_sha256": cell["train_data_manifest_sha256"],
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "cell_manifest_sha256": _file_sha256(cell_path),
        "initial_tensor_sha256": cell["initial_tensor_sha256"],
        "repeat_initial_tensor_sha256": cell["repeat_initial_tensor_sha256"],
        "records": {
            record_id: {variant: scored[variant][record_id] for variant in VARIANTS}
            for record_id in shard_ids
        },
        "cell_metrics": cell_metrics,
    }


def execute_local_grid(
    protocol: LockedPowerProtocol, *, manifest_path: Path, output_dir: Path
) -> list[dict[str, Any]]:
    """Compatibility owner for the unsharded local grid."""
    shards = [
        execute_local_shard(
            protocol,
            manifest_path=manifest_path,
            output_dir=output_dir,
            seed=seed,
            backend=backend,
            shard_index=0,
            shard_count=1,
        )
        for seed in SEEDS
        for backend in BACKENDS
    ]
    return merge_locked_shards(protocol, shards, shard_count=1)
