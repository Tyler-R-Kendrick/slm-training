"""Decision-conditioned functional spectra over exact DecisionStateV2 inputs.

This module is diagnostic-only. It captures inputs to explicitly named
``nn.Linear`` modules in evaluation mode and analyzes ``W Sigma^(1/2)`` using
split- and state-bound activation evidence. It never changes model behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH as SEMANTIC_FLOOR_GATE_PATH,
    load_semantic_floor_gate,
)
from slm_training.harnesses.preference.local_decisions import DecisionStateV2
from slm_training.versioning import build_version_stamp, git_commit

__all__ = [
    "FunctionalObservationV1",
    "FunctionalSpectralSnapshotV1",
    "StreamingCovariance",
    "analyze_functional_spectrum",
    "capture_linear_inputs",
    "run_fixture_study",
]

MATRIX_SET = "slm217_functional_spectra"
MATRIX_VERSION = "ncs1-01-v1"
ORIENTATION = (
    "PyTorch nn.Linear weight is [out_features,in_features]; row activations "
    "[n,in_features] map as X @ W.T; functional operator is W @ Sigma^(1/2)"
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
            if key not in {"stamped_at", "timestamp"}
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile(child) for child in value]
    return value


def _normalized_esd_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    size = max(left.numel(), right.numel())
    if not size:
        return 0.0
    grid = torch.linspace(0, 1, size, dtype=torch.float64)

    def quantiles(values: torch.Tensor) -> torch.Tensor:
        values = values.double().sort().values
        if values.numel() == 1:
            return values.repeat(size)
        positions = grid * (values.numel() - 1)
        low = positions.floor().long()
        high = positions.ceil().long()
        fraction = positions - low
        return values[low] * (1 - fraction) + values[high] * fraction

    left_q = quantiles(left)
    right_q = quantiles(right)
    scale = max(float(left_q.abs().mean()), float(right_q.abs().mean()), 1e-12)
    return float(torch.sqrt(torch.mean((left_q - right_q) ** 2)) / scale)


def _stable_rank(values: torch.Tensor) -> float:
    maximum = float(values.square().max()) if values.numel() else 0.0
    return float(values.square().sum()) / maximum if maximum else 0.0


def _effective_rank(values: torch.Tensor) -> float:
    energy = values.square()
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probabilities = energy / total
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-30)))
    return float(torch.exp(entropy))


@dataclass(frozen=True)
class FunctionalObservationV1:
    """One exact-state input row to a named linear map."""

    state_id: str
    group_id: str
    decision_kind: str
    abstract_state_role: str
    split: str
    module_path: str
    values: tuple[float, ...]
    observation_weighting: str = "one_selected_input_row_per_state"

    @classmethod
    def from_state(
        cls,
        state: DecisionStateV2,
        *,
        module_path: str,
        values: torch.Tensor,
    ) -> "FunctionalObservationV1":
        if values.ndim != 1:
            raise ValueError("one observation must be a one-dimensional input row")
        return cls(
            state_id=state.state_id,
            group_id=state.group_id,
            decision_kind=state.decision_kind,
            abstract_state_role=state.abstract_state_role,
            split=state.split,
            module_path=module_path,
            values=tuple(float(value) for value in values.detach().cpu().double()),
        )

    def identity_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("values")
        return payload


class StreamingCovariance:
    """Numerically stable float64 CPU covariance using parallel Welford updates."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.m2 = torch.zeros((dimension, dimension), dtype=torch.float64)

    def update(self, rows: torch.Tensor) -> None:
        rows = rows.detach().cpu().double()
        if rows.ndim == 1:
            rows = rows.unsqueeze(0)
        if rows.ndim != 2 or rows.shape[1] != self.dimension:
            raise ValueError("activation dimension does not match covariance")
        if not rows.shape[0]:
            return
        batch_count = rows.shape[0]
        batch_mean = rows.mean(dim=0)
        centered = rows - batch_mean
        batch_m2 = centered.T @ centered
        if not self.count:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + torch.outer(delta, delta) * (
            self.count * batch_count / total
        )
        self.mean += delta * (batch_count / total)
        self.count = total

    def covariance(self, *, ridge: float = 0.0) -> torch.Tensor:
        if self.count < 2:
            covariance = torch.zeros_like(self.m2)
        else:
            covariance = self.m2 / (self.count - 1)
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        return covariance + torch.eye(self.dimension, dtype=torch.float64) * ridge


@dataclass(frozen=True)
class FunctionalSpectralSnapshotV1:
    checkpoint_sha: str
    module_path: str
    semantic_role: str
    decision_kind: str
    abstract_state_role: str
    split: str
    state_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    activation_artifact_hash: str
    state_manifest_hash: str
    base_spectral_reference: str
    orientation: str
    observation_weighting: str
    support_count: int
    group_count: int
    input_dimension: int
    covariance_rank: int
    covariance_eigenvalues: tuple[float, ...]
    covariance_condition: float | None
    ridge: float
    eligibility: str
    warnings: tuple[str, ...]
    raw_singular_values: tuple[float, ...]
    functional_singular_values: tuple[float, ...]
    raw_stable_rank: float
    functional_stable_rank: float
    raw_effective_rank: float
    functional_effective_rank: float
    raw_functional_esd_distance: float
    isotropic_null_esd_distance: float
    init_null_esd_distance: float | None
    permutation_null_mean: float | None
    permutation_null_interval: tuple[float, float] | None
    bootstrap_interval: tuple[float, float] | None
    constraint_debt_join: dict[str, float | None]
    schema: str = "FunctionalSpectralSnapshotV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _covariance(rows: torch.Tensor, ridge: float) -> tuple[torch.Tensor, StreamingCovariance]:
    accumulator = StreamingCovariance(rows.shape[1])
    accumulator.update(rows)
    return accumulator.covariance(ridge=ridge), accumulator


def _functional_values(weight: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    sqrt_covariance = eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    return torch.linalg.svdvals(weight.detach().cpu().double() @ sqrt_covariance)


def _interval(values: Sequence[float]) -> tuple[float, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return (
        ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))],
        ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))],
    )


def analyze_functional_spectrum(
    weight: torch.Tensor,
    observations: Sequence[FunctionalObservationV1],
    *,
    checkpoint_sha: str,
    semantic_role: str,
    base_spectral_reference: str,
    init_weight: torch.Tensor | None = None,
    null_observations: Sequence[FunctionalObservationV1] = (),
    ridge: float = 1e-6,
    min_support: int = 4,
    null_draws: int = 16,
    bootstrap_draws: int = 32,
    seed: int = 0,
    constraint_debt_join: dict[str, float | None] | None = None,
) -> FunctionalSpectralSnapshotV1:
    """Analyze one checkpoint/module/kind/role/split evidence unit."""
    if weight.ndim != 2:
        raise ValueError("functional spectra require a two-dimensional weight")
    if not observations:
        raise ValueError("at least one exact-state observation is required")
    strata = {
        (row.module_path, row.decision_kind, row.abstract_state_role, row.split)
        for row in observations
    }
    if len(strata) != 1:
        raise ValueError("decision kind, abstract role, module, and split cannot mix")
    dimension = len(observations[0].values)
    if dimension != weight.shape[1] or any(len(row.values) != dimension for row in observations):
        raise ValueError("nn.Linear input rows must match weight in_features")
    rows = torch.tensor([row.values for row in observations], dtype=torch.float64)
    covariance, accumulator = _covariance(rows, ridge)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    rank = int(torch.linalg.matrix_rank(covariance - torch.eye(dimension) * ridge))
    positive = covariance_eigenvalues[covariance_eigenvalues > max(ridge, 1e-12)]
    condition = (
        float(positive.max() / positive.min()) if positive.numel() > 1 else None
    )
    raw = torch.linalg.svdvals(weight.detach().cpu().double())
    functional = _functional_values(weight, covariance)
    trace_scale = float(torch.trace(covariance) / dimension)
    isotropic = torch.eye(dimension, dtype=torch.float64) * trace_scale
    isotropic_values = _functional_values(weight, isotropic)
    init_distance = (
        _normalized_esd_distance(functional, _functional_values(init_weight, covariance))
        if init_weight is not None
        else None
    )

    generator = torch.Generator().manual_seed(seed)
    permutation_distances: list[float] = []
    if null_observations:
        null_rows = torch.tensor(
            [row.values for row in null_observations], dtype=torch.float64
        )
        if null_rows.shape[1] != dimension:
            raise ValueError("permutation-null activations have the wrong dimension")
        for _ in range(null_draws):
            indices = torch.randint(
                len(null_rows), (len(rows),), generator=generator
            )
            null_covariance, _ = _covariance(null_rows[indices], ridge)
            permutation_distances.append(
                _normalized_esd_distance(
                    functional, _functional_values(weight, null_covariance)
                )
            )

    by_group: dict[str, list[FunctionalObservationV1]] = {}
    for row in observations:
        by_group.setdefault(row.group_id, []).append(row)
    bootstrap_distances: list[float] = []
    group_ids = sorted(by_group)
    if group_ids:
        for _ in range(bootstrap_draws):
            sampled = torch.randint(
                len(group_ids), (len(group_ids),), generator=generator
            )
            sample_rows = [
                item.values
                for index in sampled
                for item in by_group[group_ids[int(index)]]
            ]
            sample = torch.tensor(sample_rows, dtype=torch.float64)
            sample_covariance, _ = _covariance(sample, ridge)
            bootstrap_distances.append(
                _normalized_esd_distance(
                    raw, _functional_values(weight, sample_covariance)
                )
            )

    warnings: list[str] = []
    eligibility = "eligible"
    if accumulator.count < min_support:
        eligibility = "ineligible_low_support"
        warnings.append(f"support {accumulator.count} is below min_support {min_support}")
    if rank < dimension:
        warnings.append(f"empirical covariance rank {rank} is below dimension {dimension}")
        if not ridge:
            eligibility = "ineligible_rank_deficient"
    module_path, decision_kind, abstract_state_role, split = next(iter(strata))
    identities = [row.identity_dict() for row in observations]
    return FunctionalSpectralSnapshotV1(
        checkpoint_sha=checkpoint_sha,
        module_path=module_path,
        semantic_role=semantic_role,
        decision_kind=decision_kind,
        abstract_state_role=abstract_state_role,
        split=split,
        state_ids=tuple(sorted({row.state_id for row in observations})),
        group_ids=tuple(group_ids),
        activation_artifact_hash=_sha(
            [{"identity": identity, "values": row.values} for identity, row in zip(identities, observations)]
        ),
        state_manifest_hash=_sha(identities),
        base_spectral_reference=base_spectral_reference,
        orientation=ORIENTATION,
        observation_weighting=observations[0].observation_weighting,
        support_count=accumulator.count,
        group_count=len(group_ids),
        input_dimension=dimension,
        covariance_rank=rank,
        covariance_eigenvalues=tuple(float(value) for value in covariance_eigenvalues),
        covariance_condition=condition,
        ridge=ridge,
        eligibility=eligibility,
        warnings=tuple(warnings),
        raw_singular_values=tuple(float(value) for value in raw),
        functional_singular_values=tuple(float(value) for value in functional),
        raw_stable_rank=_stable_rank(raw),
        functional_stable_rank=_stable_rank(functional),
        raw_effective_rank=_effective_rank(raw),
        functional_effective_rank=_effective_rank(functional),
        raw_functional_esd_distance=_normalized_esd_distance(raw, functional),
        isotropic_null_esd_distance=_normalized_esd_distance(functional, isotropic_values),
        init_null_esd_distance=init_distance,
        permutation_null_mean=(
            sum(permutation_distances) / len(permutation_distances)
            if permutation_distances
            else None
        ),
        permutation_null_interval=_interval(permutation_distances),
        bootstrap_interval=_interval(bootstrap_distances),
        constraint_debt_join=dict(constraint_debt_join or {}),
    )


def capture_linear_inputs(
    model: nn.Module,
    *,
    module_path: str,
    states: Sequence[DecisionStateV2],
    run_state: Callable[[DecisionStateV2], Any],
    select_input: Callable[[DecisionStateV2, torch.Tensor], torch.Tensor],
) -> tuple[list[FunctionalObservationV1], list[Any]]:
    """Capture selected input rows without changing outputs or model mode."""
    modules = dict(model.named_modules())
    module = modules.get(module_path)
    if not isinstance(module, nn.Linear):
        raise ValueError(f"{module_path!r} is not an nn.Linear module")
    active_state: DecisionStateV2 | None = None
    observations: list[FunctionalObservationV1] = []

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if active_state is None or not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("linear input capture received no active tensor state")
        selected = select_input(active_state, args[0].detach())
        if selected.ndim == 1:
            selected = selected.unsqueeze(0)
        for row in selected:
            observations.append(
                FunctionalObservationV1.from_state(
                    active_state, module_path=module_path, values=row
                )
            )

    was_training = model.training
    model.eval()
    handle = module.register_forward_pre_hook(hook)
    outputs: list[Any] = []
    try:
        with torch.inference_mode():
            for state in states:
                active_state = state
                outputs.append(run_state(state))
    finally:
        active_state = None
        handle.remove()
        model.train(was_training)
    return observations, outputs


@dataclass(frozen=True)
class FunctionalSpectralReportV1:
    run_id: str
    snapshots: tuple[FunctionalSpectralSnapshotV1, ...]
    verdict: str
    rationale: tuple[str, ...]
    semantic_floor_hash: str
    semantic_floor_verdict: str
    checkpoint_references: tuple[str, ...]
    source_commit: str
    version_stamp: dict[str, Any]
    schema: str = "FunctionalSpectralReportV1"
    claim_class: str = "diagnostic"
    honesty_mode: str = "fixture_cpu"

    @property
    def report_hash(self) -> str:
        return _sha(_without_volatile(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "run_id": self.run_id,
            "claim_class": self.claim_class,
            "honesty_mode": self.honesty_mode,
            "snapshots": [row.to_dict() for row in self.snapshots],
            "verdict": self.verdict,
            "rationale": list(self.rationale),
            "semantic_floor_hash": self.semantic_floor_hash,
            "semantic_floor_verdict": self.semantic_floor_verdict,
            "checkpoint_references": list(self.checkpoint_references),
            "source_commit": self.source_commit,
            "version_stamp": self.version_stamp,
        }
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def run_fixture_study(repo_root: Path) -> FunctionalSpectralReportV1:
    """Run a deterministic analytical fixture; do not represent it as checkpoint evidence."""
    floor = load_semantic_floor_gate(repo_root / SEMANTIC_FLOOR_GATE_PATH)
    weight = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.4, 0.0], [0.0, 0.0, 0.0, 0.2]]
    )
    init_weight = torch.eye(4)

    def rows(kind: str, role: str, scale: tuple[float, ...]) -> list[FunctionalObservationV1]:
        output: list[FunctionalObservationV1] = []
        for group_index in range(4):
            for row_index in range(2):
                values = tuple(
                    value * (1.0 + 0.1 * row_index) + (0.01 * group_index)
                    for value in scale
                )
                output.append(
                    FunctionalObservationV1(
                        state_id=f"{kind}-state-{group_index}-{row_index}",
                        group_id=f"{kind}-group-{group_index}",
                        decision_kind=kind,
                        abstract_state_role=role,
                        split="held_out",
                        module_path="probe",
                        values=values,
                    )
                )
        return output

    component = rows("component", "component_slot", (3.0, 0.4, 0.2, 0.1))
    binding = rows("binding", "binding_slot", (0.2, 2.0, 0.3, 0.1))
    snapshots = (
        analyze_functional_spectrum(
            weight,
            component,
            checkpoint_sha="fixture:no-current-checkpoint",
            semantic_role="probe",
            base_spectral_reference="slm214:fixture",
            init_weight=init_weight,
            null_observations=binding,
            seed=217,
            constraint_debt_join={"d_good": None, "protected_margin": None},
        ),
        analyze_functional_spectrum(
            weight,
            binding,
            checkpoint_sha="fixture:no-current-checkpoint",
            semantic_role="probe",
            base_spectral_reference="slm214:fixture",
            init_weight=init_weight,
            null_observations=component,
            seed=218,
            constraint_debt_join={"d_good": None, "protected_margin": None},
        ),
    )
    return FunctionalSpectralReportV1(
        run_id="slm217-functional-spectra-20260723",
        snapshots=snapshots,
        verdict="inconclusive",
        rationale=(
            "analytical fixture confirms functional spectra diverge across exact-state activation strata",
            "no compatible durable current-contract checkpoint and DecisionEvent manifest pair is committed",
            f"SemanticFloorGateV1 is {floor.verdict}; semantic interpretation is blocked",
        ),
        semantic_floor_hash=floor.gate_hash,
        semantic_floor_verdict=floor.verdict,
        checkpoint_references=(),
        source_commit=git_commit() or "UNKNOWN",
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.semantic_floor_gate",
            "harness.experiments.slm217_functional_spectra",
        ),
    )


def render_markdown(report: FunctionalSpectralReportV1) -> str:
    lines = [
        "# SLM-217: decision-conditioned functional spectra",
        "",
        f"**Verdict:** `{report.verdict}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Semantic floor:** `{report.semantic_floor_hash}` (`{report.semantic_floor_verdict}`)",
        "",
        f"**Orientation:** {ORIENTATION}.",
        "",
        "| Kind / role / split | Support / groups | Covariance rank | Raw→functional ESD | Isotropic-null ESD | Permutation-null mean | Eligibility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.snapshots:
        permutation = (
            f"{row.permutation_null_mean:.6f}"
            if row.permutation_null_mean is not None
            else "unavailable"
        )
        lines.append(
            f"| `{row.decision_kind}` / `{row.abstract_state_role}` / `{row.split}` "
            f"| {row.support_count} / {row.group_count} | {row.covariance_rank} "
            f"| {row.raw_functional_esd_distance:.6f} "
            f"| {row.isotropic_null_esd_distance:.6f} | {permutation} "
            f"| `{row.eligibility}` |"
        )
    lines.extend(
        [
            "",
            "## Verdict rationale",
            "",
            *[f"- {reason}" for reason in report.rationale],
            "",
            "This run used deterministic analytical activations and no model checkpoint. "
            "It validates the contract and null plumbing only; it does not establish "
            "out-of-family predictive value, causal relevance, semantic capability, "
            "promotion eligibility, or ship readiness.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python "
            "-m scripts.run_functional_spectral_fixture --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
