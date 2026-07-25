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
METRICS = (
    "binding_aware_meaningful_v2",
    "binder_reference_f1",
    "latency_ms",
    "compute_proxy_forwards",
    "peak_rss_bytes",
)
LOCKED_MANIFEST = Path(
    "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


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
            "updates": 0,
            "prompt_source": "locked_manifest.record.prompt",
            "target_source": "locked_manifest.record.openui",
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
        records = cell.get("records")
        if not isinstance(records, dict) or set(records) != ids:
            raise ValueError("cell records are not the frozen locked record set")
        for record in records.values():
            if not isinstance(record, dict) or set(record) != set(VARIANTS):
                raise ValueError("cell lacks raw/constrained/repaired evidence")
            for values in record.values():
                if any(metric not in values or not math.isfinite(float(values[metric])) for metric in METRICS):
                    raise ValueError("cell metric is missing or non-finite")


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
            for metric in METRICS
        }
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


def execute_local_grid(
    protocol: LockedPowerProtocol,
    *,
    manifest_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run the canonical evaluator for every declared local zero-update cell.

    The temporary suite is a byte-for-byte projection of the locked rows.  It
    exists only under the run root so the shared evaluator can score it; the
    manifest digest remains the authority for membership.
    """
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.harnesses.model_build import ModelBuildConfig, build_model
    from slm_training.harnesses.model_build.eval_runner import (
        evaluate_grammar_leakage_audit,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_by_id = {
        str(row["record"]["id"]): ExampleRecord.from_dict(row["record"])
        for row in payload["rows"]
        if row.get("partition") == "locked_test"
    }
    records = [record_by_id[record_id] for record_id in protocol.record_ids]
    suite_dir = output_dir / "locked_eval" / "suites" / "locked_power"
    suite_dir.mkdir(parents=True, exist_ok=True)
    records_path = suite_dir / "records.jsonl"
    records_path.write_text(
        "".join(_canonical(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "locked_eval" / "manifest.json").write_text(
        json.dumps({"suites": {"locked_power": str(records_path)}}, indent=2) + "\n",
        encoding="utf-8",
    )

    def state_sha(model: Any) -> str:
        state = getattr(model, "state_dict")()
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        for backend in BACKENDS:
            overrides = protocol.recipe["backends"][backend]
            config = ModelBuildConfig(
                train_dir=output_dir / "empty_train",
                test_dir=output_dir / "locked_eval",
                suite="locked_power",
                run_root=output_dir,
                run_id=f"{EXPERIMENT_ID}/{backend}/seed{seed}",
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
                grammar_ltr_max_tokens=32,
                grammar_ltr_primary=True,
                gen_steps=1,
                decode_timeout_seconds=5.0,
                run_class="scratch_matrix",
            )
            model = build_model(config, records)
            initial = state_sha(model)
            repeated = state_sha(build_model(config, records))
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
                raise TimeoutError(
                    "locked local cell timed out; no numeric power result is valid"
                )
            peak_rss = max(before_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            scored: dict[str, dict[str, dict[str, float]]] = {}
            for variant in VARIANTS:
                details = audit["variants"][variant].get("details") or []
                values: dict[str, dict[str, float]] = {}
                for detail in details:
                    values[str(detail["id"])] = {
                        "binding_aware_meaningful_v2": float(bool(detail.get("binding_aware_meaningful_v2"))),
                        "binder_reference_f1": float(detail.get("binder_reference_f1") or 0.0),
                        "latency_ms": float(detail.get("latency_ms") or 0.0),
                        "compute_proxy_forwards": float(
                            (audit["variants"][variant].get("decode_telemetry") or {}).get("forwards", 0)
                        ),
                        "peak_rss_bytes": float(peak_rss * 1024),
                    }
                scored[variant] = values
            cells.append(
                {
                    "seed": seed,
                    "backend": backend,
                    "locked_eval_manifest_sha256": protocol.manifest_sha256,
                    "initial_tensor_sha256": initial,
                    "repeat_initial_tensor_sha256": repeated,
                    "records": {
                        record_id: {variant: scored[variant][record_id] for variant in VARIANTS}
                        for record_id in protocol.record_ids
                    },
                }
            )
    return cells
