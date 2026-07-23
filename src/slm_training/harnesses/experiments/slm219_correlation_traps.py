"""SLM-219 correlation-trap trajectory diagnostics.

This module owns deterministic, repository-native spectral trap metrics,
independent collapse labeling, and warning-rule evaluation.  It deliberately
does not alter a training loop or authorize early stopping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    spectral_trap_statistics,
)
from slm_training.versioning import build_version_stamp, git_commit

__all__ = [
    "CollapseRuleV1",
    "TrapMetricsV1",
    "TrajectoryPointV1",
    "WarningRuleV1",
    "build_trajectory_inventory",
    "collapse_onset",
    "evaluate_warning_rule",
    "native_trap_metrics",
    "run_retrospective",
]

MATRIX_SET = "slm219_correlation_traps"
MATRIX_VERSION = "ncs1-03-v2"
ACTUAL_EVIDENCE = "docs/design/iter-slm219-correlation-trap-evidence-20260723.json"
HISTORICAL_SOURCES = (
    "docs/design/iter-e501-e396-e500-warm-start-20260719.json",
    "docs/design/iter-e502-initialization-prior-retention-20260719.json",
    "docs/design/iter-e503-initialized-weight-retention-20260719.json",
    "docs/design/iter-e504-parent-corpus-replay-20260719.json",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _without_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_volatile(child)
            for key, child in value.items()
            if key
            not in {
                "stamped_at",
                "timestamp",
                "code_commit",
                "code_dirty",
                "source_commit",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile(child) for child in value]
    return value


@dataclass(frozen=True)
class TrapMetricsV1:
    """Scale-invariant native spectral trap observables."""

    top_gap_ratio: float
    outlier_energy_fraction: float
    stable_rank: float
    effective_rank: float
    spectral_entropy: float
    trap_z: float
    null_draws: int
    null_mean_outlier_energy: float
    null_sd_outlier_energy: float
    schema: str = "TrapMetricsV1"
    backend: str = "pytorch-native-svd-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryPointV1:
    trajectory_id: str
    family: str
    seed: int
    step: int
    tokens: int
    role: str
    trap: TrapMetricsV1
    heldout_nll: float | None
    gradient_norm: float | None
    rms_drift: float
    update_norm: float
    structural_similarity: float
    repetition_rate: float
    recall: float
    fidelity: float
    debt_rate: float
    elapsed_seconds: float
    train_loss_proxy: float | None = None
    schema: str = "TrajectoryPointV1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trap"] = self.trap.to_dict()
        return payload


@dataclass(frozen=True)
class CollapseRuleV1:
    """Frozen outcome-only collapse definition."""

    max_structural_similarity: float = 0.15
    min_repetition_rate: float = 1 / 3
    consecutive_snapshots: int = 1
    schema: str = "CollapseRuleV1"
    spectral_fields_used: tuple[str, ...] = ()
    preregistration_sources: tuple[str, ...] = (
        "docs/design/iter-e501-e396-e500-warm-start-20260719.json",
        "docs/design/iter-e502-initialization-prior-retention-20260719.json",
    )


@dataclass(frozen=True)
class WarningRuleV1:
    min_trap_z: float = 2.0
    min_outlier_delta: float = 0.10
    consecutive_snapshots: int = 2
    schema: str = "WarningRuleV1"
    calibration_scope: str = "preregistered synthetic controls only"


def native_trap_metrics(
    matrix: torch.Tensor,
    *,
    null_draws: int = 24,
    seed: int = 0,
) -> TrapMetricsV1:
    """Project the canonical SLM-214 spectral owner into trap observables."""
    return TrapMetricsV1(
        **spectral_trap_statistics(matrix, null_draws=null_draws, seed=seed)
    )


def collapse_onset(
    points: Iterable[TrajectoryPointV1],
    rule: CollapseRuleV1 = CollapseRuleV1(),
) -> int | None:
    """Return the first step of the first consecutive outcome-only collapse span."""
    ordered = sorted(points, key=lambda point: point.step)
    run: list[TrajectoryPointV1] = []
    for point in ordered:
        collapsed = (
            point.structural_similarity <= rule.max_structural_similarity
            and point.repetition_rate >= rule.min_repetition_rate
        )
        run = [*run, point] if collapsed else []
        if len(run) >= rule.consecutive_snapshots:
            return run[-rule.consecutive_snapshots].step
    return None


def warning_onset(
    points: Iterable[TrajectoryPointV1],
    rule: WarningRuleV1 = WarningRuleV1(),
    *,
    trap_values: Iterable[float] | None = None,
    outlier_values: Iterable[float] | None = None,
) -> int | None:
    ordered = sorted(points, key=lambda point: point.step)
    values = (
        list(trap_values)
        if trap_values is not None
        else [point.trap.trap_z for point in ordered]
    )
    outliers = (
        list(outlier_values)
        if outlier_values is not None
        else [point.trap.outlier_energy_fraction for point in ordered]
    )
    if len(values) != len(ordered) or len(outliers) != len(ordered):
        raise ValueError("spectral values must align one-to-one with trajectory points")
    parent_outlier = outliers[0] if outliers else 0.0
    run: list[int] = []
    for point, trap_z, outlier in zip(ordered, values, outliers, strict=True):
        qualifies = (
            trap_z >= rule.min_trap_z
            or outlier - parent_outlier >= rule.min_outlier_delta
        )
        run = [*run, point.step] if qualifies else []
        if len(run) >= rule.consecutive_snapshots:
            # A consecutive rule becomes observable only at confirmation time.
            return run[-1]
    return None


def _baseline_onset(
    points: list[TrajectoryPointV1],
    field: str,
    multiplier: float,
) -> int | None:
    if not points:
        return None
    baseline_value = getattr(points[0], field)
    if baseline_value is None:
        return None
    baseline = float(baseline_value)
    if baseline <= 0:
        return None
    for point in points[1:]:
        value = getattr(point, field)
        if value is not None and float(value) >= baseline * multiplier:
            return point.step
    return None


def evaluate_warning_rule(
    trajectories: Iterable[Iterable[TrajectoryPointV1]],
    *,
    collapse_rule: CollapseRuleV1 = CollapseRuleV1(),
    warning_rule: WarningRuleV1 = WarningRuleV1(),
    time_shuffle: bool = False,
) -> dict[str, Any]:
    """Evaluate trajectory-level precision/recall/FPR and warning lead time."""
    rows = []
    excluded_empty = 0
    tp = fp = fn = tn = 0
    lead_times = []
    for trajectory in trajectories:
        points = sorted(trajectory, key=lambda point: point.step)
        if not points:
            excluded_empty += 1
            rows.append(
                {
                    "trajectory_id": "",
                    "excluded": True,
                    "exclusion_reason": "empty trajectory",
                }
            )
            continue
        collapse = collapse_onset(points, collapse_rule)
        values = [point.trap.trap_z for point in points]
        outliers = [point.trap.outlier_energy_fraction for point in points]
        if time_shuffle:
            values = list(reversed(values))
            outliers = list(reversed(outliers))
        warning = warning_onset(
            points,
            warning_rule,
            trap_values=values,
            outlier_values=outliers,
        )
        valid_warning = (
            warning is not None and collapse is not None and warning < collapse
        )
        if collapse is not None and valid_warning:
            tp += 1
            lead_times.append(collapse - warning)
        elif collapse is not None:
            fn += 1
        elif warning is not None:
            fp += 1
        else:
            tn += 1
        rows.append(
            {
                "trajectory_id": points[0].trajectory_id if points else "",
                "collapse_onset_step": collapse,
                "warning_onset_step": warning,
                "valid_pre_collapse_warning": valid_warning,
                "lead_steps": collapse - warning
                if collapse is not None and valid_warning
                else None,
                "baseline_onsets": {
                    "heldout_nll_x1.25": _baseline_onset(points, "heldout_nll", 1.25),
                    "gradient_norm_x1.5": _baseline_onset(points, "gradient_norm", 1.5),
                    "rms_drift_x2": _baseline_onset(points, "rms_drift", 2.0),
                    "train_loss_proxy_x1.25": _baseline_onset(
                        points,
                        "train_loss_proxy",
                        1.25,
                    ),
                },
            }
        )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {
        "rows": rows,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "mean_lead_steps": sum(lead_times) / len(lead_times) if lead_times else None,
        "time_shuffled": time_shuffle,
        "excluded_empty_trajectories": excluded_empty,
    }


def _load_actual_evidence(repo_root: Path) -> dict[str, Any]:
    evidence_path = repo_root / ACTUAL_EVIDENCE
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected_hash = evidence.get("evidence_hash")
    unsigned = {
        key: value
        for key, value in evidence.items()
        if key not in {"evidence_hash", "version_stamp"}
    }
    if expected_hash != _sha(unsigned):
        raise ValueError(f"SLM-219 evidence hash mismatch: {evidence_path}")
    for point in evidence["points"]:
        for artifact in point["evaluation"]["agentv_artifacts"].values():
            artifact_path = repo_root / artifact["path"]
            if not artifact_path.is_file():
                raise ValueError(f"missing SLM-219 AgentV artifact: {artifact_path}")
            actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual_hash != artifact["sha256"]:
                raise ValueError(
                    f"SLM-219 AgentV artifact hash mismatch: {artifact_path}"
                )
    return evidence


def build_trajectory_inventory(repo_root: Path) -> dict[str, Any]:
    """Inventory historical endpoint reports without inventing intermediate states."""
    rows = []
    for source in HISTORICAL_SOURCES:
        path = repo_root / source
        payload = json.loads(path.read_text(encoding="utf-8"))
        endpoints = [
            row for row in payload.get("matched_runs", []) if isinstance(row, dict)
        ]
        declared_paths = [
            value
            for row in endpoints
            for key in ("checkpoint", "checkpoint_path")
            if isinstance((value := row.get(key)), str)
        ]
        resolved = [value for value in declared_paths if (repo_root / value).is_file()]
        rows.append(
            {
                "source": source,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "family_run_id": payload.get("run_id"),
                "endpoint_count": len(endpoints),
                "declared_step_checkpoint_paths": declared_paths,
                "resolved_step_checkpoint_paths": resolved,
                "eligible_pre_collapse_trajectory": len(resolved) >= 3,
                "missing_checkpoint_intervals": (
                    []
                    if len(resolved) >= 3
                    else ["all pre-final intervals; reports retain endpoints only"]
                ),
            }
        )
    actual = _load_actual_evidence(repo_root)
    actual_rows = [
        {
            "run_id": point["run_id"],
            "step": point["step"],
            "tokens": point["tokens"],
            "checkpoint_sha256": point["checkpoint_sha256"],
            "source_locator": point["source_locator"],
            "source_resolution": point["source_resolution"],
        }
        for point in actual["points"]
    ]
    return {
        "schema": "CorrelationTrapTrajectoryInventoryV1",
        "sources": rows,
        "eligible_historical_trajectories": sum(
            row["eligible_pre_collapse_trajectory"] for row in rows
        ),
        "actual_reproduction": {
            "source": ACTUAL_EVIDENCE,
            "evidence_hash": actual["evidence_hash"],
            "trajectory_count": 1,
            "checkpoint_count": len(actual_rows),
            "roles": actual["roles"],
            "points": actual_rows,
            "eligible_pre_collapse_trajectory": len(actual_rows) >= 3,
        },
        "inventory_hash": _sha([*rows, actual_rows]),
    }


def _actual_trajectories(repo_root: Path) -> tuple[tuple[TrajectoryPointV1, ...], ...]:
    evidence = _load_actual_evidence(repo_root)
    trajectories = []
    for role in evidence["roles"]:
        points = []
        for row in evidence["points"]:
            role_row = row["roles"][role]
            evaluation = row["evaluation"]
            points.append(
                TrajectoryPointV1(
                    trajectory_id=f"{evidence['run_id']}:{role}",
                    family=evidence["family"],
                    seed=evidence["seed"],
                    step=row["step"],
                    tokens=row["tokens"],
                    role=role,
                    trap=TrapMetricsV1(**role_row["trap"]),
                    heldout_nll=row["heldout_nll"],
                    gradient_norm=row["gradient_norm"],
                    rms_drift=role_row["rms_drift_from_parent"],
                    update_norm=role_row["update_norm_from_previous"],
                    structural_similarity=evaluation["structural_similarity"],
                    repetition_rate=evaluation["duplicate_subtree_rate"],
                    recall=evaluation["component_type_recall"],
                    fidelity=evaluation["placeholder_fidelity"],
                    debt_rate=1.0 - evaluation["reward_score"],
                    elapsed_seconds=row["elapsed_wall_seconds"],
                    train_loss_proxy=row["train_loss_proxy"],
                )
            )
        trajectories.append(tuple(points))
    return tuple(trajectories)


def _weightwatcher_comparison(repo_root: Path) -> dict[str, Any]:
    evidence = _load_actual_evidence(repo_root)
    roles = {}
    stable_rank_errors = []
    for role in evidence["roles"]:
        rows = []
        for point in evidence["points"]:
            native = point["roles"][role]["trap"]
            watcher = point["roles"][role]["weightwatcher"]
            error = abs(native["stable_rank"] - watcher["stable_rank"])
            stable_rank_errors.append(error)
            rows.append(
                {
                    "label": point["label"],
                    "native_trap_z": native["trap_z"],
                    "native_stable_rank": native["stable_rank"],
                    "weightwatcher_alpha": watcher["alpha"],
                    "weightwatcher_stable_rank": watcher["stable_rank"],
                    "stable_rank_abs_error": error,
                }
            )
        roles[role] = {
            "rows": rows,
            "parent_to_final_alpha_delta": (
                rows[-1]["weightwatcher_alpha"] - rows[0]["weightwatcher_alpha"]
            ),
        }
    return {
        "status": "completed",
        "backend": "weightwatcher-0.7.5",
        "install_scope": "ephemeral analysis environment; not a runtime dependency",
        "matrix_rows": sum(len(value["rows"]) for value in roles.values()),
        "roles": roles,
        "max_stable_rank_abs_error": max(stable_rank_errors),
        "interpretation": (
            "native and WeightWatcher stable rank agree numerically; "
            "WeightWatcher alpha changes are descriptive because this is one "
            "seed/family with a transient collapse"
        ),
        "claim_authorized": False,
    }


def _fixture_trajectory(
    trajectory_id: str,
    *,
    seed: int,
    collapse_step: int | None,
    mode: str,
) -> list[TrajectoryPointV1]:
    """Produce a bounded synthetic matrix trajectory for contract tests."""
    if mode not in {"bulk", "bulk_plus_spike", "rank_collapse"}:
        raise ValueError(f"unknown synthetic trajectory mode: {mode}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn((16, 16), generator=generator, dtype=torch.float64)
    u = torch.randn((16,), generator=generator, dtype=torch.float64)
    v = torch.randn((16,), generator=generator, dtype=torch.float64)
    rank_one = torch.outer(u / u.norm(), v / v.norm())
    points = []
    parent = base.clone()
    for index, step in enumerate(range(0, 80, 10)):
        strength = max(0.0, index - 1) * 3.0 if mode != "bulk" else 0.0
        noise = torch.randn((16, 16), generator=generator, dtype=torch.float64) * 0.015
        if mode == "rank_collapse":
            fraction = min(0.90, max(0.0, index - 1) * 0.15)
            matrix = (1.0 - fraction) * base + strength * rank_one + noise
        else:
            matrix = base + noise + strength * rank_one
        trap = native_trap_metrics(matrix, null_draws=16, seed=seed * 100 + index)
        collapsed = collapse_step is not None and step >= collapse_step
        points.append(
            TrajectoryPointV1(
                trajectory_id=trajectory_id,
                family=f"synthetic_{mode}",
                seed=seed,
                step=step,
                tokens=step * 32,
                role="mlp_out",
                trap=trap,
                heldout_nll=1.0 + 0.01 * index + (0.4 if collapsed else 0.0),
                gradient_norm=0.8 + 0.02 * index + (0.6 if collapsed else 0.0),
                rms_drift=float(torch.sqrt(torch.mean((matrix - parent).square()))),
                update_norm=float(torch.linalg.matrix_norm(matrix - parent)),
                structural_similarity=0.62 if not collapsed else 0.12,
                repetition_rate=0.08 if not collapsed else 0.74,
                recall=0.55 if not collapsed else 0.10,
                fidelity=0.58 if not collapsed else 0.16,
                debt_rate=0.08 if not collapsed else 0.62,
                elapsed_seconds=index * 0.02,
            )
        )
    return points


def _same_shape_null_fpr() -> dict[str, Any]:
    values = []
    for seed in range(32):
        generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        matrix = torch.randn((16, 16), generator=generator, dtype=torch.float64)
        values.append(
            native_trap_metrics(matrix, null_draws=24, seed=20_000 + seed).trap_z
        )
    return {
        "draws": len(values),
        "threshold": 2.0,
        "false_positive_rate": sum(value >= 2.0 for value in values) / len(values),
        "max_trap_z": max(values),
    }


@dataclass(frozen=True)
class CorrelationTrapReportV1:
    run_id: str
    trajectory_inventory: dict[str, Any]
    bounded_continuation_preflight: dict[str, Any]
    collapse_rule: CollapseRuleV1
    warning_rule: WarningRuleV1
    actual_trajectories: tuple[tuple[TrajectoryPointV1, ...], ...]
    synthetic_trajectories: tuple[tuple[TrajectoryPointV1, ...], ...]
    warning_evaluation: dict[str, Any]
    time_shuffle_control: dict[str, Any]
    synthetic_contract_evaluation: dict[str, Any]
    same_shape_null_control: dict[str, Any]
    weightwatcher_comparison: dict[str, Any]
    verdict: str
    rationale: tuple[str, ...]
    recommendation: None
    source_commit: str
    version_stamp: dict[str, Any]
    schema: str = "CorrelationTrapReportV1"
    claim_class: str = "diagnostic"
    honesty_mode: str = "historical_deterministic_prefix_reproduction_no_design_context"

    @property
    def report_hash(self) -> str:
        return _sha(_without_volatile(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["actual_trajectories"] = [
            [point.to_dict() for point in trajectory]
            for trajectory in self.actual_trajectories
        ]
        payload["synthetic_trajectories"] = [
            [point.to_dict() for point in trajectory]
            for trajectory in self.synthetic_trajectories
        ]
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def run_retrospective(repo_root: Path) -> CorrelationTrapReportV1:
    inventory = build_trajectory_inventory(repo_root)
    actual_trajectories = _actual_trajectories(repo_root)
    synthetic_trajectories = (
        tuple(
            _fixture_trajectory("bulk-seed-0", seed=0, collapse_step=None, mode="bulk")
        ),
        tuple(
            _fixture_trajectory("bulk-seed-1", seed=1, collapse_step=None, mode="bulk")
        ),
        tuple(
            _fixture_trajectory(
                "spike-collapse-2",
                seed=2,
                collapse_step=50,
                mode="bulk_plus_spike",
            )
        ),
        tuple(
            _fixture_trajectory(
                "spike-collapse-3",
                seed=3,
                collapse_step=50,
                mode="bulk_plus_spike",
            )
        ),
        tuple(
            _fixture_trajectory(
                "rank-collapse-4",
                seed=4,
                collapse_step=50,
                mode="rank_collapse",
            )
        ),
    )
    return CorrelationTrapReportV1(
        run_id="slm219-correlation-trap-retrospective-20260723",
        trajectory_inventory=inventory,
        bounded_continuation_preflight={
            "schema": "BoundedContinuationExecutionV1",
            "parent_run_id": "e396-balanced-type-head-continuation-r1",
            "parent_remote_uri": "hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/",
            "parent_checkpoint_sha256": "feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0",
            "parent_resolved_and_hash_verified": True,
            "corpus_version": "e500_documentized_expression_candidate_r2_20260718",
            "corpus_committed": True,
            "current_contract_compatible": False,
            "current_main_preflight": (
                "failed before optimizer step zero because the historical corpus "
                "predates symbol_only/v2"
            ),
            "historical_code_commit": ("f2ab01f8ae6af6be49db3f294cd166fe034b67a5"),
            "execution": (
                "deterministic independent prefix reruns from one hash-verified "
                "parent, seed, corpus, and uniform sample order"
            ),
            "checkpoint_tokens": [0, 1039, 2047, 3008, 4007, 5019],
            "checkpoint_steps": [0, 22, 42, 61, 80, 99],
            "seed": 0,
            "device": "cpu",
            "context_backend": "hf",
            "suite": "smoke",
            "suite_n": 3,
            "max_wall_minutes_per_stage": 3,
            "checkpoint_sync": "disabled_scratch_rejected_continuation",
            "decision": (
                "use the historical code revision that originally admitted the "
                "committed corpus; do not weaken current symbol_only/v2"
            ),
        },
        collapse_rule=CollapseRuleV1(),
        warning_rule=WarningRuleV1(),
        actual_trajectories=actual_trajectories,
        synthetic_trajectories=synthetic_trajectories,
        warning_evaluation=evaluate_warning_rule(actual_trajectories),
        time_shuffle_control=evaluate_warning_rule(
            actual_trajectories,
            time_shuffle=True,
        ),
        synthetic_contract_evaluation=evaluate_warning_rule(synthetic_trajectories),
        same_shape_null_control=_same_shape_null_fpr(),
        weightwatcher_comparison=_weightwatcher_comparison(repo_root),
        verdict="inconclusive",
        rationale=(
            "E501-E504 preserve final endpoints but no resolvable step-indexed pre-collapse checkpoints",
            "a deterministic six-point prefix reproduction on the historical code revision produced one independently labeled transient collapse at 4k tokens",
            "the three matrix roles are dependent views of one seed/family, so their warning counts cannot establish held-out precision, recall, or false-positive rate",
            "time shuffling and the single seed/family do not establish temporal specificity or cross-family generalization",
            "WeightWatcher stable rank agrees with the native implementation, but its alpha trajectory remains descriptive rather than predictive",
            "historical telemetry omitted held-out NLL and gradient norms; train loss and RMS drift are retained as explicitly weaker baselines",
        ),
        recommendation=None,
        source_commit=git_commit() or "UNKNOWN",
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.slm219_correlation_traps",
        ),
    )


def render_markdown(report: CorrelationTrapReportV1) -> str:
    def rate(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.3f}"

    inventory = report.trajectory_inventory
    evaluation = report.warning_evaluation
    shuffled = report.time_shuffle_control
    synthetic = report.synthetic_contract_evaluation
    null = report.same_shape_null_control
    actual = inventory["actual_reproduction"]
    by_role = {
        trajectory[0].role: trajectory
        for trajectory in report.actual_trajectories
        if trajectory
    }
    watcher = report.weightwatcher_comparison
    lines = [
        "# SLM-219: correlation-trap early-warning retrospective",
        "",
        f"**Verdict:** `{report.verdict}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Inventory hash:** `{inventory['inventory_hash']}`",
        "",
        "## Outcome",
        "",
        "The deterministic historical-prefix reproduction produced one transient "
        "outcome-only collapse at the 4k-token checkpoint. The three spectral roles "
        "are dependent views of the same seed and family, so this run cannot establish "
        "that a correlation trap predicts collapse.",
        "",
        "Correlation-trap language is **not authorized as an early-stopping rationale**. "
        "No recommendation artifact was emitted and no production behavior changed.",
        "",
        "## Trajectory inventory",
        "",
        "| Source | Endpoints | Resolved step checkpoints | Eligible trajectory | Missing intervals |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in inventory["sources"]:
        lines.append(
            f"| `{row['source']}` | {row['endpoint_count']} | "
            f"{len(row['resolved_step_checkpoint_paths'])} | "
            f"`{str(row['eligible_pre_collapse_trajectory']).lower()}` | "
            f"{'; '.join(row['missing_checkpoint_intervals']) or '—'} |"
        )
    lines.extend(
        [
            "",
            "The original E501-E504 artifacts retain endpoints, not time series. SLM-219 "
            f"therefore reproduced one six-point trajectory with {actual['checkpoint_count']} "
            "hash-verified deterministic-prefix checkpoints and three preregistered matrix roles.",
            "",
            "| Point | Step | Tokens | Structure | Duplicate-subtree rate | MLP trap z | Cross-attn trap z | LM-head trap z |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    primary = by_role["mlp_out"]
    cross = by_role["cross_attn_out"]
    head = by_role["lm_head"]
    for index, point in enumerate(primary):
        lines.append(
            f"| t{point.tokens:04d} | {point.step} | {point.tokens} | "
            f"{point.structural_similarity:.4f} | {point.repetition_rate:.3f} | "
            f"{point.trap.trap_z:.3f} | {cross[index].trap.trap_z:.3f} | "
            f"{head[index].trap.trap_z:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen rules and controls",
            "",
            f"- Collapse: structure ≤ {report.collapse_rule.max_structural_similarity:.2f} "
            f"and repetition ≥ {report.collapse_rule.min_repetition_rate:.2f} for "
            f"{report.collapse_rule.consecutive_snapshots} snapshot; spectral fields used: none. "
            "The threshold was frozen from published E501/E502 parent and collapsed-endpoint "
            "evidence before inspecting the reproduced spectra.",
            f"- Warning: trap z ≥ {report.warning_rule.min_trap_z:.1f} or parent-relative "
            f"outlier-energy increase ≥ {report.warning_rule.min_outlier_delta:.2f} for "
            f"{report.warning_rule.consecutive_snapshots} snapshots; onset is the confirmation "
            "step, never the backdated first qualifying step.",
            f"- Actual dependent-role evaluation: TP `{evaluation['true_positive']}`, "
            f"FP `{evaluation['false_positive']}`, FN `{evaluation['false_negative']}`, "
            f"TN `{evaluation['true_negative']}`, FPR "
            f"`{rate(evaluation['false_positive_rate'])}` (no independent "
            "non-collapse trajectory means no FPR denominator).",
            f"- Time-shuffled control: precision `{rate(shuffled['precision'])}`, recall "
            f"`{rate(shuffled['recall'])}`, FPR "
            f"`{rate(shuffled['false_positive_rate'])}`.",
            f"- Synthetic contract control: precision `{rate(synthetic['precision'])}`, "
            f"recall `{rate(synthetic['recall'])}`, FPR "
            f"`{rate(synthetic['false_positive_rate'])}`. Synthetic results are not model evidence.",
            f"- Independent same-shape null: {null['draws']} draws, FPR "
            f"`{null['false_positive_rate']:.3f}` at z ≥ {null['threshold']:.1f}.",
            "",
            "## Native versus WeightWatcher",
            "",
            f"Pinned `{watcher['backend']}` completed {watcher['matrix_rows']} matrix "
            "comparisons in the analysis environment. Native and WeightWatcher stable "
            f"rank agree to maximum absolute error `{watcher['max_stable_rank_abs_error']:.3e}`. "
            "WeightWatcher alpha changed only descriptively because the evidence is "
            "limited to one seed/family with a transient collapse:",
            "",
            "| Role | Parent alpha | Final alpha | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for role, comparison in watcher["roles"].items():
        rows = comparison["rows"]
        lines.append(
            f"| `{role}` | {rows[0]['weightwatcher_alpha']:.4f} | "
            f"{rows[-1]['weightwatcher_alpha']:.4f} | "
            f"{comparison['parent_to_final_alpha_delta']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recipe and durable evaluation evidence",
            "",
            "- CPU; historical code `f2ab01f8ae6af6be49db3f294cd166fe034b67a5`; "
            "HF context backend; seed 0; batch 2; LR 3e-4; 22/42/61/80/99 "
            "optimizer steps; 1,039/2,047/3,008/4,007/5,019 target tokens. Each "
            "checkpoint is an independent deterministic prefix rerun from the same "
            "parent and uniform sample order, not a resumed-stage claim.",
            "- Matrix set `slm219_correlation_traps`, version `ncs1-03-v2`; three roles; "
            "24 same-shape null draws per checkpoint/role; honesty mode "
            "`no-design-md-context`.",
            "- Smoke suite n=3 at every checkpoint. Each evaluation emitted an AgentEvals "
            "JSONL spec and pinned AgentV result bundle; all six were honest non-ship "
            "0/1 results with zero execution errors.",
            f"- Evidence manifest: `{ACTUAL_EVIDENCE}`. Normalized AgentV artifacts: "
            "`docs/design/iter-slm219-correlation-trap-agentv-20260723/`.",
            "- Held-out NLL and gradient norms are explicitly unavailable in the historical "
            "telemetry. Train loss is retained as a proxy and RMS drift/update norm are "
            "computed from the hash-verified checkpoints; neither is relabeled as held-out evidence.",
            "- Scratch continuation checkpoints were rejected, not promoted, and not synced. "
            "They do not alter the serving roster or model card.",
            "",
            "Current `main` correctly rejects the old E500 corpus under `symbol_only/v2`. "
            "The reproduction used the historical code revision that originally admitted "
            "that committed corpus and deterministic prefix reruns; no current gate was weakened.",
            "",
            "## Decision",
            "",
            *[f"- {reason}" for reason in report.rationale],
            "",
            "Verdict: **inconclusive and not supported for use**. The actual trajectory "
            "contains one independently labeled collapse, but one seed/family and three "
            "dependent matrix roles cannot identify generalizable precision, recall, or "
            "false-positive rate. A new multi-seed/family study requires independently "
            "labeled trajectories and persisted held-out NLL/gradient telemetry.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python "
            "-m scripts.run_correlation_trap_retrospective --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
