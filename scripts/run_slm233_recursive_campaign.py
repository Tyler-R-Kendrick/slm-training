#!/usr/bin/env python3
"""Run the gate-bounded, matched SLM-233 recursive-depth campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from urllib.parse import quote

import torch

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm233_recursive_campaign import (
    RecursiveCoreVerdict,
    RecursiveFairnessManifestV1,
    classify_recursive_core_gate,
    stable_hash,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    _reward_for_prediction,
    structural_similarity,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.recursive_denoiser import (
    checkpoint_state_dict_bytes,
    estimate_transformer_block_flops,
)
from slm_training.models.rng_contract import (
    derive_seed,
    rng_namespace_report,
    seed_training_corruption,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm233-recursive-campaign-20260724.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm233-recursive-campaign-20260724.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm233-recursive-campaign-agentv-20260724"
DEFAULT_DATA = (
    ROOT / "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722"
)
COMPONENT = "harness.experiments.slm233_recursive_campaign"
SEEDS = (23301, 23302, 23303)
TEST_DEPTHS = (1, 2, 4, 6, 8)
OPTIMIZER_STEPS = 3
TRAIN_IDS = ("smoke_hero_01", "smoke_button_01")
HELDOUT_IDS = ("held_out_form_01", "held_out_dual_card_01")

GATE_PATHS = {
    "floor": ROOT / "docs/design/semantic-floor-gate-v1.json",
    "observability": ROOT
    / "docs/design/iter-slm230-recurrence-observability-20260724.json",
    "dynamics": ROOT / "docs/design/iter-slm231-recurrence-dynamics-20260724.json",
    "z_use": ROOT / "docs/design/iter-slm232-latent-state-use-20260724.json",
    "update": ROOT / "docs/design/iter-slm243-recursive-update-gate-20260724.json",
}


@dataclass(frozen=True)
class ArmSpec:
    arm: str
    label: str
    denoiser_arch: str
    train_r: int
    denoiser_layers: int
    transition_layers: int
    depth_weights: tuple[float, ...]
    depth_mode: str
    depth_aux_weight: float
    block_evaluations: int
    view: str = "equal_block_primary"
    freeze_layerscale: bool = False


ARM_SPECS = (
    ArmSpec("A", "stacked final-only", "stacked", 1, 4, 0, (), "off", 0.0, 4),
    ArmSpec(
        "B",
        "shared recursive z final-only",
        "shared_recursive",
        2,
        2,
        2,
        (),
        "off",
        0.0,
        4,
    ),
    ArmSpec(
        "C",
        "shared recursive z normalized all-depth",
        "shared_recursive",
        2,
        2,
        2,
        (0.5, 0.5),
        "all_depths",
        0.3,
        4,
    ),
    ArmSpec(
        "D",
        "shared y-only normalized all-depth",
        "shared_recursive_y_only",
        2,
        2,
        2,
        (0.5, 0.5),
        "all_depths",
        0.3,
        4,
    ),
    ArmSpec(
        "E",
        "shared recursive z R4 normalized all-depth",
        "shared_recursive",
        4,
        1,
        1,
        (0.25, 0.25, 0.25, 0.25),
        "all_depths",
        0.3,
        4,
    ),
)

PARAMETER_VIEW_SPECS = (
    ArmSpec(
        "P-A",
        "stacked two-block active-parameter reference",
        "stacked",
        1,
        2,
        0,
        (),
        "off",
        0.0,
        2,
        view="equal_active_parameter_secondary",
    ),
    ArmSpec(
        "P-D",
        "layerscale y-only R2 with frozen scales",
        "shared_recursive_y_only",
        2,
        2,
        2,
        (),
        "off",
        0.0,
        4,
        view="equal_active_parameter_secondary",
        freeze_layerscale=True,
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_hash(model: TwoTowerModel) -> str:
    return stable_hash(
        {
            name: _tensor_hash(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        path.write_text(text, encoding="utf-8")


def _load_records(data_dir: Path) -> tuple[list[Any], list[Any]]:
    smoke = {record.id: record for record in load_suite_records(data_dir, "smoke")}
    heldout = {
        record.id: record for record in load_suite_records(data_dir, "held_out")
    }
    return (
        [smoke[record_id] for record_id in TRAIN_IDS],
        [heldout[record_id] for record_id in HELDOUT_IDS],
    )


def _config(spec: ArmSpec, seed: int) -> TwoTowerConfig:
    recursive = spec.denoiser_arch.startswith("shared_recursive")
    return TwoTowerConfig(
        d_model=16,
        n_heads=2,
        context_layers=1,
        denoiser_layers=spec.denoiser_layers,
        denoiser_arch=spec.denoiser_arch,
        recursive_steps=spec.train_r,
        recursive_transition_layers=spec.transition_layers,
        recursive_update_mode="layerscale" if recursive else "current_v1",
        recursive_empty_f_mode="zero" if recursive else "pass_through",
        recursive_norm_mode="shared",
        recursive_depth_supervision_weights=spec.depth_weights,
        recursive_depth_aux_mode=spec.depth_mode,
        recursive_depth_aux_weight=spec.depth_aux_weight,
        grammar_constrained=False,
        ltr_loss_weight=0.0,
        max_target_len=64,
        gen_steps=2,
        dropout=0.0,
        seed=seed,
    )


def _build_model(
    spec: ArmSpec, seed: int, train_records: list[Any]
) -> TwoTowerModel:
    model = TwoTowerModel.from_records(
        train_records,
        config=_config(spec, seed),
        device="cpu",
    )
    if spec.freeze_layerscale:
        for name, parameter in model.denoiser.named_parameters():
            if name in {"f_update_scale", "g_update_scale"}:
                parameter.requires_grad_(False)
    return model


def _accounting(model: TwoTowerModel, spec: ArmSpec) -> dict[str, int | float]:
    seq_len = int(model.config.max_target_len)
    ctx_len = 16
    per_block = estimate_transformer_block_flops(
        seq_len=seq_len,
        ctx_len=ctx_len,
        d_model=int(model.config.d_model),
    )
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "denoiser_parameters_total": sum(
            parameter.numel() for parameter in model.denoiser.parameters()
        ),
        "denoiser_parameters_trainable": sum(
            parameter.numel()
            for parameter in model.denoiser.parameters()
            if parameter.requires_grad
        ),
        "checkpoint_state_dict_bytes": checkpoint_state_dict_bytes(model),
        "denoiser_checkpoint_bytes": checkpoint_state_dict_bytes(model.denoiser),
        "block_evaluations_per_forward": spec.block_evaluations,
        "estimated_denoiser_flops_per_forward": per_block
        * spec.block_evaluations,
    }


def _load_gate_refs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in GATE_PATHS.items()
    }
    refs: dict[str, dict[str, Any]] = {}
    for name, path in GATE_PATHS.items():
        report = reports[name]
        verdict = (
            report.get("gate", {}).get("verdict")
            if name == "update"
            else report.get("verdict")
        )
        refs[name] = {
            "path": str(path.relative_to(ROOT)),
            "file_sha256": _sha256_file(path),
            "scientific_hash": str(
                report.get("gate_hash")
                or report.get("report_hash")
                or stable_hash(report.get("gate", report))
            ),
            "verdict": str(verdict),
            "version_stamp": report.get("version_stamp"),
        }
    update_gate = reports["update"]["gate"]
    if update_gate["allowed_slm233_modes"] != ["layerscale_diagnostic"]:
        raise RuntimeError("SLM-243 does not authorize the layerscale diagnostic")
    if int(update_gate["maximum_authorized_depth"]) < max(TEST_DEPTHS):
        raise RuntimeError("SLM-243 does not authorize the preregistered depth grid")
    return refs, reports


@contextmanager
def _capture_mask(model: TwoTowerModel) -> Iterator[list[dict[str, Any]]]:
    original = model._mask_targets
    captures: list[dict[str, Any]] = []

    def wrapped(target_ids: torch.Tensor) -> Any:
        noisy, predict_mask, row_weights = original(target_ids)
        captures.append(
            {
                "target_ids": _tensor_hash(target_ids),
                "noisy_ids": _tensor_hash(noisy),
                "predict_mask": _tensor_hash(predict_mask),
                "row_weights": (
                    None if row_weights is None else _tensor_hash(row_weights)
                ),
                "target_token_count": int(target_ids.ne(model.tokenizer.pad_id).sum()),
                "predicted_token_count": int(predict_mask.sum()),
            }
        )
        return noisy, predict_mask, row_weights

    had_override = "_mask_targets" in model.__dict__
    prior = model.__dict__.get("_mask_targets")
    model._mask_targets = wrapped  # type: ignore[method-assign]
    try:
        yield captures
    finally:
        if had_override:
            model._mask_targets = prior  # type: ignore[method-assign]
        else:
            delattr(model, "_mask_targets")


def _loss_call(
    model: TwoTowerModel,
    records: list[Any],
    *,
    corruption_seed: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    with _capture_mask(model) as captures:
        seed_training_corruption(
            int(model.config.seed),
            model,
            override_seed=corruption_seed,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
    if len(captures) != 1:
        raise RuntimeError(f"expected one corruption capture, got {len(captures)}")
    metrics = {
        key: value
        for key, value in model.last_training_metrics.items()
        if key.startswith("recursive_")
        or key in {"primary_final_reconstruction_loss", "combined_training_loss"}
    }
    return {
        "loss": float(loss.detach().cpu()),
        "batch": captures[0],
        "metrics": metrics,
    }


def _common_initialization(
    models: dict[str, TwoTowerModel]
) -> tuple[dict[str, str], bool, list[str]]:
    named = {
        arm: dict(model.named_parameters()) for arm, model in models.items()
    }
    common_names = set.intersection(*(set(values) for values in named.values()))
    common_names = {
        name
        for name in common_names
        if len({tuple(named[arm][name].shape) for arm in named}) == 1
    }
    hashes: dict[str, str] = {}
    mismatches: list[str] = []
    for name in sorted(common_names):
        values = {_tensor_hash(named[arm][name]) for arm in named}
        if len(values) == 1:
            hashes[name] = next(iter(values))
        else:
            mismatches.append(name)
    return hashes, not mismatches, mismatches


def _checkpoint_roundtrip(
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    spec: ArmSpec,
    train_records: list[Any],
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="slm233-resume-") as temp_dir:
        path = Path(temp_dir) / "state.pt"
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
            path,
        )
        restored = _build_model(spec, int(model.config.seed), train_records)
        restored_optimizer = torch.optim.AdamW(
            [p for p in restored.parameters() if p.requires_grad],
            lr=1e-3,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        restored.load_state_dict(payload["model"], strict=True)
        restored_optimizer.load_state_dict(payload["optimizer"])
        return {
            "passed": _model_hash(restored) == _model_hash(model),
            "serialized_bytes": path.stat().st_size,
            "optimizer_state_entries": len(restored_optimizer.state),
            "durable_checkpoint_created": False,
        }


def _free_running(
    model: TwoTowerModel, heldout: list[Any]
) -> dict[str, Any]:
    predictions = model.generate_batch(
        [record.prompt for record in heldout],
        golds=heldout,
        max_len=32,
        grammar_constrained=False,
    )
    rows = []
    for prediction, record in zip(predictions, heldout, strict=True):
        meaningful, reason, _ = _is_meaningful_program(prediction, gold=record)
        rows.append(
            {
                "record_id": record.id,
                "meaningful": meaningful,
                "meaningful_failure": reason,
                "structure": structural_similarity(prediction, record.openui),
                "reward": _reward_for_prediction(prediction, record),
                "prediction_sha256": hashlib.sha256(prediction.encode()).hexdigest(),
            }
        )
    return {
        "support": len(rows),
        "meaningful_rate": sum(row["meaningful"] for row in rows) / len(rows),
        "mean_structure": sum(row["structure"] for row in rows) / len(rows),
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "rows": rows,
    }


def _test_depth_rows(
    model: TwoTowerModel,
    spec: ArmSpec,
    train_records: list[Any],
    heldout: list[Any],
    *,
    evaluation_seed: int,
) -> list[dict[str, Any]]:
    if not spec.denoiser_arch.startswith("shared_recursive"):
        return []
    rows = []
    for test_r in TEST_DEPTHS:
        eval_spec = replace(
            spec,
            train_r=test_r,
            depth_weights=(),
            depth_mode="off",
            depth_aux_weight=0.0,
            block_evaluations=test_r * spec.transition_layers,
            freeze_layerscale=False,
        )
        clone = _build_model(eval_spec, int(model.config.seed), train_records)
        clone.load_state_dict(model.state_dict(), strict=True)
        result = _loss_call(
            clone,
            heldout,
            corruption_seed=evaluation_seed,
        )
        primary = result["metrics"].get(
            "primary_final_reconstruction_loss", result["loss"]
        )
        rows.append(
            {
                "arm": spec.arm,
                "seed": int(model.config.seed),
                "train_r": spec.train_r,
                "test_r": test_r,
                "heldout_primary_nll": float(primary),
                "finite": math.isfinite(float(primary)),
                "claim_class": "layerscale_test_depth_diagnostic_not_reasoning",
            }
        )
    return rows


def _aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [cell for cell in cells if cell["view"] == "equal_block_primary"]
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for cell in primary:
        by_arm.setdefault(cell["arm"], []).append(cell)
    arm_summaries = {}
    for arm, rows in sorted(by_arm.items()):
        arm_summaries[arm] = {
            "seed_count": len(rows),
            "mean_train_final_nll": sum(
                row["train_final_primary_nll"] for row in rows
            )
            / len(rows),
            "mean_heldout_final_nll": sum(
                row["heldout_final_primary_nll"] for row in rows
            )
            / len(rows),
            "mean_wall_seconds": sum(row["wall_seconds"] for row in rows)
            / len(rows),
            "mean_structure": sum(
                row["free_running"]["mean_structure"] for row in rows
            )
            / len(rows),
            "mean_reward": sum(
                row["free_running"]["mean_reward"] for row in rows
            )
            / len(rows),
        }
    baseline = arm_summaries["A"]["mean_heldout_final_nll"]
    effects = {
        arm: {
            "heldout_nll_delta_vs_A": values["mean_heldout_final_nll"] - baseline,
            "descriptive_proxy_only": True,
        }
        for arm, values in arm_summaries.items()
        if arm != "A"
    }
    return {"by_arm": arm_summaries, "paired_proxy_effects": effects}


def _build_manifest(
    *,
    seed_manifests: list[dict[str, Any]],
    accounting: dict[str, dict[str, int | float]],
    gate_refs: dict[str, dict[str, Any]],
    train_records: list[Any],
    heldout: list[Any],
    schedules: dict[str, Any],
) -> dict[str, Any]:
    common_hashes = {
        f"seed_{manifest['seed']}::{name}": digest
        for manifest in seed_manifests
        for name, digest in manifest["common_tensor_hashes"].items()
    }
    common_match = all(
        manifest["common_tensor_hashes_match"] for manifest in seed_manifests
    )
    all_records = train_records + heldout
    corpus_payload = [asdict(record) for record in all_records]
    data_order = [record.id for record in train_records]
    exposures = [
        {
            "seed": item["seed"],
            "batch_digests": item["batch_digests"],
        }
        for item in seed_manifests
    ]
    config_stamp = build_version_stamp(COMPONENT, "model.twotower")
    manifest = RecursiveFairnessManifestV1(
        common_tensor_hashes=common_hashes,
        common_tensor_hashes_match=common_match,
        architecture_seed_namespaces={
            key: value
            for key, value in rng_namespace_report(SEEDS[0]).items()
            if key.startswith("arch_specific:")
        },
        stacked_to_shared_layer_mapping={
            "A.denoiser.layers.0": "B/C/D.denoiser.layers.0",
            "A.denoiser.layers.1": "B/C/D.denoiser.layers.1",
            "A.denoiser.layers.0_only": "E.denoiser.layers.0",
        },
        accounting_by_arm=accounting,
        optimizer_contract={
            "name": "AdamW",
            "learning_rate": 1e-3,
            "initial_state": "empty",
            "groups": "all and only requires_grad parameters",
        },
        corpus_hash=stable_hash(corpus_payload),
        data_order_hash=stable_hash(data_order),
        corruption_schedule_hash=stable_hash(schedules),
        exposure_hash=stable_hash(exposures),
        checkpoint_eval_schedule_hash=stable_hash(
            {
                "optimizer_steps": OPTIMIZER_STEPS,
                "test_depths": TEST_DEPTHS,
                "checkpoint": "temporary_full_state_roundtrip_only",
            }
        ),
        decode_evaluator_gate_hashes={
            **{
                name: str(value["file_sha256"])
                for name, value in gate_refs.items()
            },
            "component_versions": stable_hash(config_stamp["components"]),
        },
        hardware_runtime_budget={
            "device": "cpu",
            "platform": platform.platform(),
            "torch": torch.__version__,
            "max_wall_minutes": float(MAX_RUN_MINUTES),
            "torch_threads": torch.get_num_threads(),
        },
    )
    return manifest.to_dict()


def run_campaign(
    *,
    data_dir: Path = DEFAULT_DATA,
    agentv_dir: Path = DEFAULT_AGENTV,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    torch.set_num_threads(1)
    gate_refs, gate_reports = _load_gate_refs()
    train_records, heldout = _load_records(data_dir)
    specs = ARM_SPECS + PARAMETER_VIEW_SPECS
    cells: list[dict[str, Any]] = []
    test_depth_rows: list[dict[str, Any]] = []
    seed_manifests: list[dict[str, Any]] = []
    accounting_by_arm: dict[str, dict[str, int | float]] = {}
    resume_rows: list[dict[str, Any]] = []
    schedules: dict[str, Any] = {}

    for seed_index, seed in enumerate(SEEDS):
        models = {
            spec.arm: _build_model(spec, seed, train_records) for spec in specs
        }
        initial_hashes = {arm: _model_hash(model) for arm, model in models.items()}
        if initial_hashes["B"] != initial_hashes["C"]:
            raise RuntimeError("B/C objective controls did not start identically")
        common_hashes, common_match, mismatches = _common_initialization(models)
        if not common_match:
            raise RuntimeError(f"common initialization mismatch: {mismatches}")
        optimizers = {
            arm: torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1e-3,
            )
            for arm, model in models.items()
        }
        for spec in specs:
            accounting_by_arm.setdefault(
                spec.arm, _accounting(models[spec.arm], spec)
            )
        train_seeds = [
            derive_seed(seed * 100 + step, "training_corruption")
            for step in range(OPTIMIZER_STEPS)
        ]
        eval_seed = derive_seed(seed * 100 + OPTIMIZER_STEPS, "training_corruption")
        schedules[str(seed)] = {
            "train_corruption_seeds": train_seeds,
            "evaluation_corruption_seed": eval_seed,
        }
        batch_digests: dict[str, list[dict[str, Any]]] = {
            spec.arm: [] for spec in specs
        }
        training: dict[str, list[dict[str, Any]]] = {
            spec.arm: [] for spec in specs
        }
        arm_started: dict[str, float] = {spec.arm: time.monotonic() for spec in specs}
        for corruption_seed in train_seeds:
            for spec in specs:
                result = _loss_call(
                    models[spec.arm],
                    train_records,
                    corruption_seed=corruption_seed,
                    optimizer=optimizers[spec.arm],
                )
                training[spec.arm].append(result)
                batch_digests[spec.arm].append(result["batch"])
        reference_batches = batch_digests["A"]
        if any(value != reference_batches for value in batch_digests.values()):
            raise RuntimeError("matched arms did not receive identical corruption")

        for spec in specs:
            model = models[spec.arm]
            heldout_result = _loss_call(
                model,
                heldout,
                corruption_seed=eval_seed,
            )
            train_final = training[spec.arm][-1]
            train_metrics = train_final["metrics"]
            heldout_metrics = heldout_result["metrics"]
            weights = list(spec.depth_weights)
            normalized = not weights or math.isclose(sum(weights), 1.0)
            if not normalized:
                raise RuntimeError(f"unnormalized deep-supervision weights: {spec.arm}")
            free_running = _free_running(model, heldout)
            cell = {
                "schema": "RecursiveCampaignCellV1",
                "arm": spec.arm,
                "label": spec.label,
                "view": spec.view,
                "seed": seed,
                "config_hash": stable_hash(asdict(model.config)),
                "initial_model_hash": initial_hashes[spec.arm],
                "denoiser_arch": spec.denoiser_arch,
                "train_r": spec.train_r,
                "test_r": spec.train_r,
                "recursive_update_mode": model.config.recursive_update_mode,
                "recursive_empty_f_mode": model.config.recursive_empty_f_mode,
                "recursive_norm_mode": model.config.recursive_norm_mode,
                "depth_supervision": {
                    "mode": spec.depth_mode,
                    "weights": weights,
                    "weights_sum": sum(weights),
                    "aux_weight": spec.depth_aux_weight,
                    "normalized": normalized,
                },
                "training_losses": [result["loss"] for result in training[spec.arm]],
                "train_final_primary_nll": float(
                    train_metrics.get(
                        "primary_final_reconstruction_loss", train_final["loss"]
                    )
                ),
                "heldout_final_primary_nll": float(
                    heldout_metrics.get(
                        "primary_final_reconstruction_loss",
                        heldout_result["loss"],
                    )
                ),
                "depth_metrics": train_metrics,
                "batch_digests": batch_digests[spec.arm],
                "heldout_batch_digest": heldout_result["batch"],
                "free_running": free_running,
                "accounting": accounting_by_arm[spec.arm],
                "finite": all(
                    math.isfinite(value)
                    for value in (
                        [result["loss"] for result in training[spec.arm]]
                        + [float(heldout_result["loss"])]
                    )
                ),
                "completed": True,
                "timed_out": False,
                "wall_seconds": time.monotonic() - arm_started[spec.arm],
            }
            cells.append(cell)
            if seed_index == 0:
                resume_rows.append(
                    {
                        "arm": spec.arm,
                        **_checkpoint_roundtrip(
                            model,
                            optimizers[spec.arm],
                            spec,
                            train_records,
                        ),
                    }
                )
            if spec.view == "equal_block_primary":
                test_depth_rows.extend(
                    _test_depth_rows(
                        model,
                        spec,
                        train_records,
                        heldout,
                        evaluation_seed=eval_seed,
                    )
                )
        seed_manifests.append(
            {
                "seed": seed,
                "common_tensor_hashes": common_hashes,
                "common_tensor_hashes_match": common_match,
                "batch_digests": batch_digests,
            }
        )

    manifest = _build_manifest(
        seed_manifests=seed_manifests,
        accounting=accounting_by_arm,
        gate_refs=gate_refs,
        train_records=train_records,
        heldout=heldout,
        schedules=schedules,
    )
    aggregate = _aggregate(cells)
    primary_cells = [cell for cell in cells if cell["view"] == "equal_block_primary"]
    matrix_complete = len(primary_cells) == len(ARM_SPECS) * len(SEEDS)
    controls_matched = all(
        item["common_tensor_hashes_match"] for item in seed_manifests
    )
    all_finite = all(cell["finite"] for cell in cells) and all(
        row["finite"] for row in test_depth_rows
    )
    parameter_view = {
        arm: accounting_by_arm[arm] for arm in ("P-A", "P-D")
    }
    parameter_view["active_parameter_residual"] = (
        int(accounting_by_arm["P-D"]["parameters_trainable"])
        - int(accounting_by_arm["P-A"]["parameters_trainable"])
    )
    parameter_view["total_parameter_residual"] = (
        int(accounting_by_arm["P-D"]["parameters_total"])
        - int(accounting_by_arm["P-A"]["parameters_total"])
    )
    cost_frontier = [
        {
            "arm": arm,
            **accounting,
            "mean_wall_seconds": aggregate["by_arm"].get(arm, {}).get(
                "mean_wall_seconds"
            ),
        }
        for arm, accounting in accounting_by_arm.items()
    ]
    gate = classify_recursive_core_gate(
        floor_verdict=str(gate_refs["floor"]["verdict"]),
        observability_verdict=str(gate_refs["observability"]["verdict"]),
        dynamics_verdict=str(gate_refs["dynamics"]["verdict"]),
        z_verdict=str(gate_refs["z_use"]["verdict"]),
        matrix_complete=matrix_complete,
        controls_matched=controls_matched,
        all_finite=all_finite,
        semantic_outcomes_available=False,
        fairness_manifest_hash=str(manifest["manifest_hash"]),
        gate_refs=gate_refs,
        matched_matrix_ref="agentv-dir://raw_campaign.json",
        primary_effect_sizes=aggregate["paired_proxy_effects"],
        equivalence_margins={
            "heldout_nll_absolute": 0.05,
            "protected_semantic_absolute": 0.02,
        },
        cost_frontier=cost_frontier,
        checkpoint_refs=(),
    ).to_dict()
    if gate["verdict"] != RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE.value:
        raise RuntimeError(f"unexpected gate verdict: {gate['verdict']}")

    raw = {
        "schema": "RecursiveCoreCampaignRawV1",
        "issue": "SLM-233",
        "fairness_manifest": manifest,
        "cells": cells,
        "test_depth_rows": test_depth_rows,
        "resume_rows": resume_rows,
        "seed_manifests": seed_manifests,
        "schedules": schedules,
    }
    agentv_dir.mkdir(parents=True, exist_ok=True)
    raw_path = agentv_dir / "raw_campaign.json"
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_hash = _sha256_file(raw_path)

    version_stamp = build_version_stamp(
        COMPONENT,
        "model.twotower",
        "model.recursive_denoiser",
        "evals.scoring",
    )
    report = {
        "schema": "RecursiveCoreCampaignReportV2",
        "issue": "SLM-233",
        "matrix_set": "slm233-matched-recursive-depth",
        "matrix_version": "slm233-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "bounded_matched_proxy_complete",
        "claim_class": "architecture_not_identifiable_not_ship",
        "activation_gates": gate_refs,
        "preregistration": {
            "seeds": list(SEEDS),
            "primary_arms": [spec.arm for spec in ARM_SPECS],
            "test_depths": list(TEST_DEPTHS),
            "optimizer_steps": OPTIMIZER_STEPS,
            "minimum_effects": {
                "heldout_nll_absolute": 0.05,
                "protected_semantic_absolute": 0.02,
            },
            "stop_rules": [
                "nonfinite_or_timeout",
                "unmatched_initialization_or_exposure",
                "semantic_floor_not_escaped",
            ],
        },
        "recipe": {
            "device": "cpu",
            "backend": "scratch",
            "data": str(data_dir.relative_to(ROOT)),
            "train_ids": list(TRAIN_IDS),
            "heldout_ids": list(HELDOUT_IDS),
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "optimizer_steps": OPTIMIZER_STEPS,
            "max_target_len": 64,
            "honesty_mode": "bounded_proxy_not_ship",
            "max_wall_minutes": float(MAX_RUN_MINUTES),
        },
        "fairness_manifest": manifest,
        "raw_campaign": {
            "schema": raw["schema"],
            "path": "agentv-dir://raw_campaign.json",
            "sha256": raw_hash,
            "cell_count": len(cells),
            "primary_cell_count": len(primary_cells),
            "test_depth_cell_count": len(test_depth_rows),
        },
        "arms": [
            {
                **asdict(spec),
                "accounting": accounting_by_arm[spec.arm],
            }
            for spec in specs
        ],
        "matched_views": {
            "equal_block_primary": {
                "target_block_evaluations": 4,
                "arms": [spec.arm for spec in ARM_SPECS],
                "exact": all(spec.block_evaluations == 4 for spec in ARM_SPECS),
            },
            "equal_active_parameter_secondary": parameter_view,
        },
        "outcomes": {
            "standard": {
                "status": "measured_proxy",
                "support": {
                    "seeds": len(SEEDS),
                    "train_records": len(train_records),
                    "heldout_records": len(heldout),
                },
                "metrics": aggregate,
                "claim_boundary": (
                    "NLL/parse/structure/reward are descriptive proxy diagnostics "
                    "because SemanticFloorGateV1 did not escape"
                ),
            },
            "protected_semantic": {
                "status": "censored",
                "support": 0,
                "reason": "SemanticFloorGateV1 blocks semantic prediction/causal claims",
            },
            "recovery_compositional": {
                "status": "censored",
                "support": 0,
                "reason": "no provenance-compatible recovery suite under proxy-only scope",
            },
            "mechanistic": {
                "status": "activation_gate_join_plus_layerscale_depth_diagnostic",
                "test_depth_rows": len(test_depth_rows),
                "prior_numeric_rows_reused": False,
                "reason": (
                    "SLM-230/231/232 verdicts are scope gates only; their values "
                    "were not transplanted onto these scratch states"
                ),
            },
            "cost": {
                "status": "measured",
                "frontier": cost_frontier,
            },
        },
        "recursive_core_gate": gate,
        "checkpoint_created": False,
        "checkpoint_promoted": False,
        "checkpoint_refs": [],
        "temporary_full_state_resume": resume_rows,
        "production_default_changed": False,
        "ship_gate_claim": False,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "version_stamp": version_stamp,
    }
    report["report_hash"] = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key
            not in {
                "generated_at",
                "elapsed_seconds",
                "peak_rss_bytes",
                "version_stamp",
                "report_hash",
                "agentv",
            }
        }
    )
    cases = [
        {
            "id": "activation-gate-hashes",
            "criteria": "All five prerequisite gate files are hash-bound.",
            "pass": len(gate_refs) == 5
            and all(len(ref["file_sha256"]) == 64 for ref in gate_refs.values()),
            "failures": [],
            "result": gate_refs,
            "metadata": {"honesty": "bounded_proxy_not_ship"},
        },
        {
            "id": "matched-fairness-and-exposure",
            "criteria": "The five-arm, three-seed primary matrix is complete and matched.",
            "pass": matrix_complete and controls_matched,
            "failures": [] if matrix_complete and controls_matched else ["unmatched"],
            "result": {
                "matrix_complete": matrix_complete,
                "controls_matched": controls_matched,
                "manifest_hash": manifest["manifest_hash"],
            },
            "metadata": {"honesty": "bounded_proxy_not_ship"},
        },
        {
            "id": "normalized-supervision-and-accounting",
            "criteria": "Every deep-supervision schedule is normalized and accounting is explicit.",
            "pass": all(
                cell["depth_supervision"]["normalized"] for cell in cells
            )
            and parameter_view["active_parameter_residual"] == 0,
            "failures": [],
            "result": {
                "parameter_view": parameter_view,
                "primary_blocks": 4,
            },
            "metadata": {"honesty": "bounded_proxy_not_ship"},
        },
        {
            "id": "finite-complete-no-timeouts",
            "criteria": "No incomplete, timed-out, or non-finite cell is accepted.",
            "pass": all_finite
            and all(cell["completed"] and not cell["timed_out"] for cell in cells),
            "failures": [],
            "result": {
                "all_finite": all_finite,
                "cell_count": len(cells),
                "test_depth_cell_count": len(test_depth_rows),
            },
            "metadata": {"honesty": "bounded_proxy_not_ship"},
        },
        {
            "id": "full-state-resume-no-durable-checkpoint",
            "criteria": "Temporary full-state resume passes without creating a durable checkpoint.",
            "pass": all(
                row["passed"] and not row["durable_checkpoint_created"]
                for row in resume_rows
            ),
            "failures": [],
            "result": resume_rows,
            "metadata": {"honesty": "bounded_proxy_not_ship"},
        },
        {
            "id": "honest-recursive-core-gate",
            "criteria": "Proxy-only evidence resolves to architecture_not_identifiable and blocks downstream semantic claims.",
            "pass": gate["verdict"]
            == RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE.value
            and not gate["checkpoint_refs"],
            "failures": [],
            "result": gate,
            "metadata": {"honesty": "bounded_proxy_not_ship"},
        },
    ]
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            agentv_dir,
            name="slm233-recursive-campaign",
            claim="bounded_matched_proxy_architecture_not_identifiable_not_ship",
            cases=cases,
        ),
        agentv_dir,
    )
    _rewrite_agentv_paths(agentv_dir)
    return report, raw


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["recursive_core_gate"]
    standard = report["outcomes"]["standard"]["metrics"]["by_arm"]
    lines = [
        "# SLM-233: matched recursive-depth campaign",
        "",
        f"Status: **{report['status']}**  ",
        f"Verdict: **{gate['verdict']}**  ",
        f"Claim class: `{report['claim_class']}`",
        "",
        "## Gate-conditioned scope",
        "",
        (
            "The semantic floor is not escaped, SLM-230 is stagnant, SLM-231 "
            "is expansive-unstable, and SLM-232 is unstable. SLM-243 permits "
            "only the LayerScale diagnostic through R=8. Therefore this is the "
            "required bounded proxy/control campaign, not a semantic architecture "
            "or promotion experiment."
        ),
        "",
        "| gate | verdict | scientific hash |",
        "| --- | --- | --- |",
    ]
    for name, ref in report["activation_gates"].items():
        lines.append(
            f"| {name} | `{ref['verdict']}` | `{ref['scientific_hash']}` |"
        )
    lines.extend(
        [
            "",
            "## Matched primary matrix",
            "",
            "All A-E arms execute exactly four transformer blocks per denoiser call, "
            "use three paired seeds, identical records/order/corruption, common "
            "initialization hashes, and the same decode/evaluator budget.",
            "",
            "| arm | description | train R | params | bytes | heldout NLL | wall s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    arms = {item["arm"]: item for item in report["arms"]}
    for arm in ("A", "B", "C", "D", "E"):
        item = arms[arm]
        summary = standard[arm]
        account = item["accounting"]
        lines.append(
            f"| {arm} | {item['label']} | {item['train_r']} | "
            f"{account['parameters_trainable']} | "
            f"{account['checkpoint_state_dict_bytes']} | "
            f"{summary['mean_heldout_final_nll']:.6f} | "
            f"{summary['mean_wall_seconds']:.3f} |"
        )
    parameter = report["matched_views"]["equal_active_parameter_secondary"]
    lines.extend(
        [
            "",
            "## Fairness and secondary parameter view",
            "",
            f"- Fairness manifest: `{report['fairness_manifest']['manifest_hash']}`",
            f"- Raw matrix: `{report['raw_campaign']['sha256']}` "
            f"({report['raw_campaign']['cell_count']} train cells; "
            f"{report['raw_campaign']['test_depth_cell_count']} test-depth cells)",
            f"- P-D minus P-A active-parameter residual: "
            f"`{parameter['active_parameter_residual']}`",
            f"- P-D minus P-A total-parameter residual: "
            f"`{parameter['total_parameter_residual']}` "
            "(the frozen LayerScale vectors remain serialized and are not hidden)",
            "",
            "## Outcomes and claim boundary",
            "",
            "- Train/held-out NLL, bounded free-running parse/structure/reward, "
            "block/FLOP/parameter/byte/wall accounting, and LayerScale R-test "
            "diagnostics were measured.",
            "- Protected semantic and recovery/compositional outcomes are censored, "
            "not encoded as zero, because the floor gate does not authorize them.",
            "- Prior SLM-230/231/232 numeric rows were not transplanted to these "
            "scratch states; only their authoritative gate verdicts were joined.",
            "- No durable checkpoint was created, no model card update is triggered, "
            "and no production default changed.",
            "",
            "## RecursiveCoreGateV2",
            "",
            f"- Verdict: **{gate['verdict']}**",
            f"- Allowed work: `{gate['allowed_downstream_work']}`",
            f"- Blocked claims: `{gate['blocked_claims']}`",
            f"- Checkpoint refs: `{gate['checkpoint_refs']}`",
            f"- Rationale: {gate['rationale']}",
            "",
            "This verdict is not `no_recursive_gain`: the architecture effect is "
            "unidentifiable under the current semantic floor.",
            "",
            "## AgentEvals / AgentV",
            "",
            f"- SDK: `{report['agentv'].get('sdk')}`",
            f"- Summary: `{report['agentv'].get('summary')}`",
            "",
            f"Report hash: `{report['report_hash']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _check(json_path: Path, markdown_path: Path, agentv_dir: Path) -> None:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    expected_hash = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key
            not in {
                "generated_at",
                "elapsed_seconds",
                "peak_rss_bytes",
                "version_stamp",
                "report_hash",
                "agentv",
            }
        }
    )
    if report.get("report_hash") != expected_hash:
        raise SystemExit("report_hash mismatch")
    raw_path = agentv_dir / "raw_campaign.json"
    if _sha256_file(raw_path) != report["raw_campaign"]["sha256"]:
        raise SystemExit("raw campaign hash mismatch")
    if markdown_path.read_text(encoding="utf-8") != render_markdown(report):
        raise SystemExit("markdown does not match report")
    current_refs, _ = _load_gate_refs()
    if current_refs != report["activation_gates"]:
        raise SystemExit("activation gate reference drift")
    if (
        report["recursive_core_gate"]["verdict"]
        != RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE.value
    ):
        raise SystemExit("unexpected RecursiveCoreGateV2 verdict")


def _clean_tree() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "check"), default="run")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.mode == "check":
        _check(args.json, args.markdown, args.agentv_dir)
        return
    if not args.allow_dirty and not _clean_tree():
        raise SystemExit("refusing to persist experiment evidence from a dirty tree")
    report, _ = run_campaign(data_dir=args.data_dir, agentv_dir=args.agentv_dir)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if time.monotonic() and report["elapsed_seconds"] >= MAX_RUN_MINUTES * 60:
        raise SystemExit("run exceeded the repository hard cap")


if __name__ == "__main__":
    main()
