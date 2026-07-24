"""SLM-200 matched objective attribution over identical legal-edit states."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from slm_training.data.flow.bridge_corpus import (
    LegalEditBridgeRowV1,
    RequestEditContractV1,
    canonical_fingerprint,
    load_corpus,
)
from slm_training.flow.targets import LegalEditRateTargetV1, from_bridge_rows
from slm_training.flow.termination import FixedKPolicy
from slm_training.harnesses.experiments.slm199_legal_edit_flow import _exact_oracle
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.legal_edit_batch import LegalEditBatch
from slm_training.models.legal_edit_flow import (
    LegalEditFlow,
    LegalEditFlowConfig,
    legal_edit_flow_losses,
)
from slm_training.models.legal_edit_scorer import (
    DirectLegalEditPolicy,
    LegalEditScorerConfig,
)

MATRIX_SET = "slm200_flow_objective_attribution"
DEFAULT_CORPUS = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)
DEFAULT_RECORDS = Path("tests/fixtures/slm196_legal_edit_bridge/records.jsonl")
POWER_PROTOCOL = Path("docs/design/iter-slm183-power-protocol-20260720.json")
RESOLUTION_PROTOCOL = Path("docs/design/iter-slm185-judge-resolution-20260720.json")
UTILITY_PROTOCOL = Path("docs/design/iter-slm186-verified-utility-20260721.json")
FIXED_EDIT_BUDGET = 2
ObjectiveKind = Literal["ce", "ce_hazard", "flow"]


@dataclass(frozen=True)
class ObjectiveSpec:
    arm_id: str
    label: str
    kind: ObjectiveKind
    time_conditioned: bool
    row_weighting: Literal["uniform", "bridge", "inverse_time", "rate"]
    shuffled_targets: bool = False
    exact_only: bool = False


OBJECTIVES = (
    ObjectiveSpec("A1", "plain multi-positive CE", "ce", False, "uniform"),
    ObjectiveSpec("A2", "time-conditioned multi-positive CE", "ce", True, "uniform"),
    ObjectiveSpec("A3", "plain CE + bridge-state weighting", "ce", False, "bridge"),
    ObjectiveSpec("A4", "time-conditioned inverse-time weighted CE", "ce", True, "inverse_time"),
    ObjectiveSpec("A5", "rate-target-weighted normalized CE", "ce", True, "rate"),
    ObjectiveSpec("A6", "probability + total-hazard multi-task", "ce_hazard", True, "rate"),
    ObjectiveSpec("A7", "full discrete edge-rate objective", "flow", True, "rate"),
    ObjectiveSpec(
        "A8",
        "shuffled rate-target negative control",
        "flow",
        True,
        "rate",
        shuffled_targets=True,
    ),
    ObjectiveSpec("A9", "exact finite-graph rate oracle", "flow", True, "rate", exact_only=True),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule(rows: Sequence[LegalEditBridgeRowV1]) -> torch.Tensor:
    """Inference-visible progress; never consume gold distance or sampled time."""
    return torch.tensor(
        [min(1.0, row.step_index / FIXED_EDIT_BUDGET) for row in rows],
        dtype=torch.float32,
    )


def _row_weights(
    spec: ObjectiveSpec, rows: Sequence[LegalEditBridgeRowV1]
) -> torch.Tensor:
    if spec.row_weighting == "uniform":
        values = [1.0 for _ in rows]
    elif spec.row_weighting == "bridge":
        values = [1.0 + row.step_index / FIXED_EDIT_BUDGET for row in rows]
    elif spec.row_weighting == "inverse_time":
        values = [1.0 / max(0.25, (row.step_index + 1) / FIXED_EDIT_BUDGET) for row in rows]
    else:
        values = [1.0 / max(1, len(row.positive_candidate_ids)) for row in rows]
    weights = torch.tensor(values, dtype=torch.float32)
    return weights / weights.mean()


def _weighted_set_loss(
    logits: torch.Tensor, batch: LegalEditBatch, weights: torch.Tensor
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row in range(len(batch.row_ids)):
        start, end = int(batch.row_offsets[row]), int(batch.row_offsets[row + 1])
        positives = batch.positive_mask[start:end]
        row_logits = logits[start:end]
        losses.append(
            (
                torch.logsumexp(row_logits, dim=0)
                - torch.logsumexp(row_logits[positives], dim=0)
            )
            * weights[row]
        )
    return torch.stack(losses).mean()


def _shuffle_targets(
    targets: tuple[LegalEditRateTargetV1, ...],
) -> tuple[LegalEditRateTargetV1, ...]:
    """Deterministically rotate rates within each exact live candidate set."""
    shuffled: list[LegalEditRateTargetV1] = []
    for target in targets:
        rates = list(target.edge_rates)
        supervised_indices = [
            index
            for index, candidate in enumerate(target.candidate_ids)
            if candidate in target.supervised_candidate_ids
        ]
        supervised_rates = [rates[index] for index in supervised_indices]
        rotated = supervised_rates[1:] + supervised_rates[:1]
        for index, rate in zip(supervised_indices, rotated, strict=True):
            rates[index] = rate
        rates_tuple = tuple(rates)
        positives = tuple(
            candidate
            for candidate, rate in zip(
                target.candidate_ids, rates_tuple, strict=True
            )
            if rate > 0.0
        )
        shuffled.append(
            LegalEditRateTargetV1(
                row_id=target.row_id,
                candidate_ids=target.candidate_ids,
                edge_rates=rates_tuple,
                total_hazard=target.total_hazard,
                positive_candidate_ids=positives,
                supervised_candidate_ids=tuple(
                    candidate
                    for candidate in target.candidate_ids
                    if candidate in target.supervised_candidate_ids
                ),
                hazard_supervised=target.hazard_supervised,
                terminal_probability=target.terminal_probability,
                time=target.time,
                fidelity="surrogate_rate_weight",
            )
        )
    return tuple(shuffled)


def _model(seed: int) -> LegalEditFlow:
    # Every production arm carries the same scorer, time projection, and
    # terminal/hazard capacity. Simpler arms leave unused heads dormant.
    return LegalEditFlow(
        LegalEditFlowConfig(
            enabled=True,
            scorer=LegalEditScorerConfig(time_encoding="linear", seed=seed),
        )
    )


def _parameter_digest(model: LegalEditFlow) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _train(
    spec: ObjectiveSpec,
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: dict[str, Any],
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    deadline: float,
) -> tuple[LegalEditFlow, dict[str, Any]]:
    torch.manual_seed(seed)
    model = _model(seed)
    initial_parameter_digest = _parameter_digest(model)
    batch = LegalEditBatch.pack(rows, candidate_sets)
    progress = _schedule(rows)
    weights = _row_weights(spec, rows)
    targets = from_bridge_rows(rows)
    if spec.shuffled_targets:
        targets = _shuffle_targets(targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for _ in range(steps):
        if time.monotonic() > deadline:
            raise TimeoutError("SLM-200 matrix exceeded max_wall_minutes")
        optimizer.zero_grad(set_to_none=True)
        if spec.kind == "flow":
            prediction = model(batch, schedule_progress=progress)
            loss, components = legal_edit_flow_losses(prediction, batch, targets)
        else:
            logits = model.scorer(
                batch,
                schedule_progress=progress,
                time_encoding="linear" if spec.time_conditioned else "no_time",
            )
            ce = _weighted_set_loss(logits, batch, weights)
            components = {"multi_positive_mass": ce}
            loss = ce
            if spec.kind == "ce_hazard":
                prediction = model(batch, schedule_progress=progress)
                hazard = F.mse_loss(
                    prediction.row_hazards,
                    prediction.row_hazards.new_ones(len(rows)),
                )
                components["total_hazard"] = hazard
                loss = loss + hazard
        loss.backward()
        optimizer.step()
        history.append(
            {"total": float(loss.detach())}
            | {name: float(value.detach()) for name, value in components.items()}
        )
    return model, {
        "initial": history[0],
        "final": history[-1],
        "steps": steps,
        "row_updates": steps * len(rows),
        "candidate_scores": steps * len(batch.candidate_ids),
        "initial_parameter_digest": initial_parameter_digest,
        "active_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
    }


def _teacher_forced(
    model: LegalEditFlow,
    spec: ObjectiveSpec,
    rows: Sequence[LegalEditBridgeRowV1],
    candidate_sets: dict[str, Any],
) -> dict[str, Any]:
    batch = LegalEditBatch.pack(rows, candidate_sets)
    with torch.no_grad():
        logits = model.scorer(
            batch,
            schedule_progress=_schedule(rows),
            time_encoding="linear" if spec.time_conditioned else "no_time",
        )
    positive_mass: list[float] = []
    top1: list[bool] = []
    for row in range(len(rows)):
        start, end = int(batch.row_offsets[row]), int(batch.row_offsets[row + 1])
        probabilities = F.softmax(logits[start:end], dim=0)
        positives = batch.positive_mask[start:end]
        positive_mass.append(float(probabilities[positives].sum()))
        top1.append(bool(positives[int(probabilities.argmax())]))
    return {
        "rows": len(rows),
        "candidate_scores": len(batch.candidate_ids),
        "mean_positive_mass": sum(positive_mass) / max(1, len(positive_mass)),
        "top1_positive_rate": sum(top1) / max(1, len(top1)),
        "candidate_set_digests": list(batch.candidate_set_digests),
        "row_order": list(batch.row_ids),
    }


def _records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _free_running(
    model: LegalEditFlow,
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    # The same one-edit-per-NFE greedy exact-candidate decoder is used for
    # every objective. Softplus rate ranking is monotone in scorer logits.
    policy = DirectLegalEditPolicy(model.scorer)
    started = time.monotonic()
    outcomes: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        trace = policy.decode_exact(
            record["source_program"],
            RequestEditContractV1.from_dict(record["request_contract"]),
            termination=FixedKPolicy(k=2, max_steps=2),
            max_steps=2,
            seed=seed + index,
        )
        outcomes.append(
            {
                "target_cluster_id": record["id"],
                "target_exact": trace.final_fingerprint
                == canonical_fingerprint(record["target_program"]),
                "all_actions_live": all(
                    decision["selected_candidate_id"] in decision["candidate_ids"]
                    for decision in trace.decisions
                ),
                "edit_count": len(trace.decisions),
                "model_calls": trace.model_calls,
                "stop_reason": trace.stop_reason,
                "trajectory": list(trace.decisions),
            }
        )
    return {
        "outcomes": outcomes,
        "target_exact_rate": sum(item["target_exact"] for item in outcomes)
        / max(1, len(outcomes)),
        "all_actions_live_rate": sum(item["all_actions_live"] for item in outcomes)
        / max(1, len(outcomes)),
        "model_calls": sum(item["model_calls"] for item in outcomes),
        "elapsed_seconds": time.monotonic() - started,
    }


def validate_primary_parity(arms: dict[str, Any]) -> dict[str, Any]:
    measured = [arm for arm in arms.values() if arm.get("status") == "measured_fixture"]
    param_counts = {arm["parameter_count"] for arm in measured}
    row_digests = {tuple(arm["row_order_digest"]) for arm in measured}
    decoders = {arm["decoder_manifest"] for arm in measured}
    seed_digests: dict[int, set[str]] = {}
    for arm in measured:
        for run in arm["runs"]:
            seed_digests.setdefault(run["seed"], set()).add(
                run["training"]["initial_parameter_digest"]
            )
    return {
        "parameter_count_within_one_percent": len(param_counts) == 1,
        "parameter_counts": sorted(param_counts),
        "identical_row_order": len(row_digests) == 1,
        "identical_decoder": len(decoders) == 1,
        "paired_initialization": all(
            len(digests) == 1 for digests in seed_digests.values()
        ),
        "initialization_digest_counts": {
            str(seed): len(digests) for seed, digests in seed_digests.items()
        },
        "declared_differences_only": (
            len(param_counts) == len(row_digests) == len(decoders) == 1
            and all(len(digests) == 1 for digests in seed_digests.values())
        ),
    }


def run_matrix(
    *,
    corpus_dir: Path = DEFAULT_CORPUS,
    records_path: Path = DEFAULT_RECORDS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    steps: int = 8,
    learning_rate: float = 0.03,
    max_wall_minutes: float = float(MAX_RUN_MINUTES),
) -> dict[str, Any]:
    if not 0 < max_wall_minutes <= MAX_RUN_MINUTES:
        raise ValueError("max_wall_minutes must be in (0, 3]")
    if not seeds or steps <= 0:
        raise ValueError("seeds and steps must be nonempty/positive")
    started = time.monotonic()
    deadline = started + max_wall_minutes * 60
    rows, candidate_sets, corpus_manifest = load_corpus(corpus_dir)
    train_rows = tuple(row for row in rows if row.split == "train")
    dev_rows = tuple(row for row in rows if row.split == "dev")
    records = _records(records_path)
    row_order = [row.row_id for row in train_rows]
    row_order_digest = hashlib.sha256(
        json.dumps(row_order, separators=(",", ":")).encode()
    ).hexdigest()
    arms: dict[str, Any] = {
        "A0": {
            "status": "unavailable",
            "reason": "X22/local-corruption control has no hash-pinned bridge-state corpus",
        }
    }
    for spec in OBJECTIVES:
        if spec.exact_only:
            arms[spec.arm_id] = {
                "status": "measured_exact_fixture",
                "spec": asdict(spec),
                "oracle": _exact_oracle(
                    seeds[0], 128, fit_steps=100, deadline=deadline
                ),
            }
            continue
        runs: list[dict[str, Any]] = []
        for seed in seeds:
            model, training = _train(
                spec,
                train_rows,
                candidate_sets,
                seed=seed,
                steps=steps,
                learning_rate=learning_rate,
                deadline=deadline,
            )
            runs.append(
                {
                    "seed": seed,
                    "training": training,
                    "teacher_forced": _teacher_forced(
                        model, spec, dev_rows, candidate_sets
                    ),
                    "free_running": _free_running(model, records, seed=seed),
                }
            )
        arms[spec.arm_id] = {
            "status": "measured_fixture",
            "spec": asdict(spec),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "active_parameter_policy": (
                "all arms carry matched capacity; dormant heads are reported"
            ),
            "active_parameter_counts": sorted(
                {
                    run["training"]["active_parameter_count"]
                    for run in runs
                }
            ),
            "row_order_digest": [row_order_digest],
            "decoder_manifest": "exact-live-greedy-one-edit-fixed-k2/v1",
            "runs": runs,
        }
    parity = validate_primary_parity(arms)
    production = {
        key: value
        for key, value in arms.items()
        if value.get("status") == "measured_fixture"
    }
    development_scores = {
        key: sum(run["free_running"]["target_exact_rate"] for run in value["runs"])
        / len(value["runs"])
        for key, value in production.items()
        if key not in {"A7", "A8"}
    }
    strongest = max(development_scores, key=development_scores.get)
    flow_score = sum(
        run["free_running"]["target_exact_rate"] for run in production["A7"]["runs"]
    ) / len(production["A7"]["runs"])
    control_score = development_scores[strongest]
    shuffled_score = sum(
        run["free_running"]["target_exact_rate"] for run in production["A8"]["runs"]
    ) / len(production["A8"]["runs"])
    power = _load_json(POWER_PROTOCOL)
    resolution = _load_json(RESOLUTION_PROTOCOL)
    utility = _load_json(UTILITY_PROTOCOL)
    independent_targets = int(
        corpus_manifest["diagnostics"]["independent_targets"]
    )
    confirmation_reasons = [
        "SLM-196 corpus manifest is non-publishable fixture evidence",
        f"only {independent_targets} independent targets are available",
        "A0 lacks a hash-pinned identical-state corpus",
        "SLM-183 fixture measured only 0.11 power at the preregistered 0.08 MDE",
        "no frozen production confirmation suite/checkpoints are available",
    ]
    elapsed = time.monotonic() - started
    if elapsed > max_wall_minutes * 60:
        raise TimeoutError("SLM-200 matrix exceeded max_wall_minutes")
    return {
        "schema": "FlowObjectiveAttributionReportV1",
        "issue": "SLM-200",
        "matrix_set": MATRIX_SET,
        "status": "measured_fixture_screen",
        "claim_class": "wiring",
        "arms": arms,
        "primary_parity": parity,
        "analysis": {
            "selection_set": "development_fixture_only",
            "strongest_simpler_control": strongest,
            "a7_target_exact_rate": flow_score,
            "control_target_exact_rate": control_score,
            "paired_delta": flow_score - control_score,
            "a8_shuffled_target_exact_rate": shuffled_score,
            "negative_control_degraded": shuffled_score < flow_score,
            "negative_control_investigation_required": shuffled_score >= flow_score,
            "minimum_resolvable_delta": resolution["global_floor"],
            "preregistered_mde": power["manifest"]["mde"],
            "classification": "no_conclusion_underpowered_fixture",
            "flow_win": False,
        },
        "protocol_pins": {
            "power_protocol_sha256": _sha(POWER_PROTOCOL),
            "power": power["manifest"]["power"],
            "alpha": power["manifest"]["alpha"],
            "mde": power["manifest"]["mde"],
            "multiplicity_family": power["manifest"]["multiplicity_family"],
            "resolution_protocol_sha256": _sha(RESOLUTION_PROTOCOL),
            "semantic_resolution_floor": resolution["global_floor"],
            "utility_protocol_sha256": _sha(UTILITY_PROTOCOL),
            "utility_version": utility["weight_manifest"]["version"],
        },
        "inputs": {
            "corpus": str(corpus_dir),
            "corpus_manifest_sha256": _sha(corpus_dir / "manifest.json"),
            "corpus_content_fingerprint": corpus_manifest["content_fingerprint"],
            "corpus_publishable": corpus_manifest["publishable"],
            "records": str(records_path),
            "records_sha256": _sha(records_path),
            "train_row_order": row_order,
            "train_row_order_digest": row_order_digest,
        },
        "recipe": {
            "device": "cpu",
            "backend": "torch exact-live fixture",
            "steps": steps,
            "learning_rate": learning_rate,
            "seeds": list(seeds),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "independent_targets": independent_targets,
            "target_decision_exposure_view": "fixed rows and candidate scores",
            "wall_view": "single cumulative bounded fixture screen",
            "max_wall_minutes": max_wall_minutes,
        },
        "confirmation": {
            "status": "not_touched",
            "touch_ledger": [],
            "reasons": confirmation_reasons,
        },
        "checkpoint": {
            "written": False,
            "reason": "underpowered non-publishable fixture screen",
        },
        "honest_verdict": "no_conclusion_underpowered_fixture",
        "elapsed_seconds": elapsed,
    }
