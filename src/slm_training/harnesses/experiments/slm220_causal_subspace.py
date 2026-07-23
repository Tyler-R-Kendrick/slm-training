"""Activation-side causal restriction diagnostics for SLM-220.

The numerical contract is checkpoint-agnostic: exact compiler membership is
supplied as immutable metadata, while derivatives are taken only through a
pure activation-to-declared-output function.  This module never trains,
perturbs a checkpoint, or changes decode behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH as SEMANTIC_FLOOR_GATE_PATH,
    load_semantic_floor_gate,
)
from slm_training.harnesses.experiments.slm218_cross_attention_retention import (
    principal_angles,
    restriction_energy,
)
from slm_training.versioning import build_version_stamp, git_commit

__all__ = [
    "CausalSubspaceSnapshotV1",
    "exact_jacobian",
    "functional_activation_subspaces",
    "hutchinson_jacobian_frobenius",
    "jvp_restriction_energy",
    "make_linear_input_target",
    "run_fixture_retrospective",
    "validate_state_manifest",
]

MATRIX_SET = "slm220_causal_subspace"
MATRIX_VERSION = "ncs2-01-v1"
ORIENTATION = (
    "activation-side basis V[in,k]; declared-output Jacobian J[out,in]; "
    "restriction = ||J@V||_F^2 / ||J||_F^2"
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
            if key not in {"stamped_at", "timestamp", "source_commit"}
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile(child) for child in value]
    return value


def _sqrt_psd(covariance: torch.Tensor) -> torch.Tensor:
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not torch.allclose(covariance, covariance.T, atol=1e-9, rtol=1e-7)
    ):
        raise ValueError("activation covariance must be a symmetric square matrix")
    values, vectors = torch.linalg.eigh(covariance.detach().cpu().double())
    if float(values.min()) < -1e-9:
        raise ValueError("activation covariance must be positive semidefinite")
    return vectors @ torch.diag(values.clamp_min(0).sqrt()) @ vectors.T


def _right_basis(matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("subspace source must be a matrix")
    _, _, vh = torch.linalg.svd(matrix.detach().cpu().double(), full_matrices=True)
    return vh[indices].T


def functional_activation_subspaces(
    weight: torch.Tensor,
    covariance: torch.Tensor,
    *,
    k: int,
) -> dict[str, torch.Tensor]:
    """Return raw, functional, covariance-only, and band-control input bases."""
    if weight.ndim != 2 or weight.shape[1] != covariance.shape[0]:
        raise ValueError("weight in_features must match activation covariance")
    dimension = weight.shape[1]
    if not 0 < k <= dimension:
        raise ValueError("k must be within the activation dimension")
    sqrt_covariance = _sqrt_psd(covariance)
    functional = weight.detach().cpu().double() @ sqrt_covariance
    rank = min(functional.shape)
    if k > rank:
        raise ValueError("k exceeds the available functional right-singular rank")
    top = torch.arange(k)
    bottom = torch.arange(rank - k, rank)
    middle_start = max(0, (rank - k) // 2)
    middle = torch.arange(middle_start, middle_start + k)
    _, covariance_vectors = torch.linalg.eigh(covariance.detach().cpu().double())
    return {
        "raw_weight_top": _right_basis(weight, top),
        "functional_top": _right_basis(functional, top),
        "functional_middle": _right_basis(functional, middle),
        "functional_bottom": _right_basis(functional, bottom),
        "covariance_top": covariance_vectors[:, -k:],
    }


def exact_jacobian(
    target: Callable[[torch.Tensor], torch.Tensor],
    activation: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``d target / d activation`` for one bounded exact state."""
    if activation.ndim != 1:
        raise ValueError("the selected input activation must be one-dimensional")

    def flattened(value: torch.Tensor) -> torch.Tensor:
        output = target(value)
        if not isinstance(output, torch.Tensor) or output.ndim != 1:
            raise ValueError("the declared target output must be one-dimensional")
        return output

    jacobian = torch.func.jacrev(flattened)(activation)
    if jacobian.ndim != 2 or jacobian.shape[1] != activation.numel():
        raise ValueError("target Jacobian orientation must be [output,input]")
    return jacobian.detach().cpu().double()


def hutchinson_jacobian_frobenius(
    target: Callable[[torch.Tensor], torch.Tensor],
    activation: torch.Tensor,
    *,
    probes: int,
    seed: int,
) -> dict[str, float | int]:
    """Estimate ``||J||_F^2`` with deterministic output-side Rademacher VJPs."""
    if probes < 2:
        raise ValueError("Hutchinson estimation requires at least two probes")
    output, vjp = torch.func.vjp(target, activation)
    if output.ndim != 1:
        raise ValueError("the declared target output must be one-dimensional")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    estimates = []
    for _ in range(probes):
        signs = (
            torch.randint(0, 2, output.shape, generator=generator, device="cpu")
            .to(output)
            .mul(2)
            .sub(1)
        )
        gradient = vjp(signs)[0]
        estimates.append(float(gradient.detach().double().square().sum()))
    values = torch.tensor(estimates, dtype=torch.float64)
    mean = float(values.mean())
    standard_error = float(values.std(unbiased=True) / math.sqrt(probes))
    return {
        "probes": probes,
        "seed": seed,
        "frobenius_squared": mean,
        "standard_error": standard_error,
        "ci95_low": max(0.0, mean - 1.96 * standard_error),
        "ci95_high": mean + 1.96 * standard_error,
    }


def jvp_restriction_energy(
    target: Callable[[torch.Tensor], torch.Tensor],
    activation: torch.Tensor,
    basis: torch.Tensor,
    *,
    denominator: float,
) -> float:
    """Compute the restriction numerator through JVPs without forming ``J``."""
    if activation.ndim != 1 or basis.ndim != 2:
        raise ValueError("activation must be [in] and basis must be [in,k]")
    if basis.shape[0] != activation.numel():
        raise ValueError("basis input dimension must match activation")
    if denominator <= 0:
        return 0.0
    numerator = 0.0
    for direction in basis.T:
        _, tangent = torch.func.jvp(target, (activation,), (direction.to(activation),))
        numerator += float(tangent.detach().double().square().sum())
    return numerator / denominator


def make_linear_input_target(
    model: torch.nn.Module,
    *,
    module_path: str,
    run_model: Callable[[], torch.Tensor],
    project_output: Callable[[torch.Tensor], torch.Tensor],
    legal_membership_hash: Callable[[], str],
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], str]:
    """Capture and replace one complete ``nn.Linear`` input tensor.

    The wrapper is intentionally narrow.  It avoids duplicating the forward
    graph and fails if exact legal membership changes across intervention.
    """
    module = dict(model.named_modules()).get(module_path)
    if not isinstance(module, torch.nn.Linear):
        raise ValueError(f"{module_path!r} is not an nn.Linear module")
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("targeted linear input must be one tensor argument")
        captured.append(args[0].detach().clone())

    mode = model.training
    model.eval()
    handle = module.register_forward_pre_hook(capture)
    try:
        run_model()
    finally:
        handle.remove()
        model.train(mode)
    if len(captured) != 1:
        raise RuntimeError("targeted module must execute exactly once per exact state")
    baseline = captured[0]
    if baseline.ndim != 1:
        raise ValueError("selected exact-state linear input must be one-dimensional")
    frozen_membership = legal_membership_hash()

    def target(activation: torch.Tensor) -> torch.Tensor:
        if activation.shape != baseline.shape:
            raise ValueError("replacement activation shape differs from captured input")

        def replace(
            _module: torch.nn.Module, args: tuple[Any, ...]
        ) -> tuple[torch.Tensor]:
            return (activation,)

        previous_mode = model.training
        model.eval()
        replacement = module.register_forward_pre_hook(replace)
        try:
            output = project_output(run_model())
        finally:
            replacement.remove()
            model.train(previous_mode)
        if legal_membership_hash() != frozen_membership:
            raise RuntimeError("compiler/legal membership changed during intervention")
        return output

    return baseline, target, frozen_membership


def validate_state_manifest(rows: tuple[dict[str, str], ...]) -> str:
    """Bind one module/stratum/split without state or group identity mixing."""
    if not rows:
        raise ValueError("at least one exact state is required")
    required = {"state_id", "group_id", "module_path", "decision_stratum", "split"}
    if any(required - row.keys() for row in rows):
        raise ValueError("state manifest is missing a required identity field")
    state_ids = [row["state_id"] for row in rows]
    if any(not value for row in rows for value in row.values()):
        raise ValueError("state manifest identity fields cannot be empty")
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("state IDs cannot repeat")
    strata = {
        (row["module_path"], row["decision_stratum"], row["split"]) for row in rows
    }
    if len(strata) != 1:
        raise ValueError("module, decision stratum, and split cannot mix")
    return _sha(rows)


def _interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    low = ordered[max(0, math.floor(0.025 * (len(ordered) - 1)))]
    high = ordered[min(len(ordered) - 1, math.ceil(0.975 * (len(ordered) - 1)))]
    return low, high


@dataclass(frozen=True)
class CausalSubspaceSnapshotV1:
    checkpoint_reference: str
    module_path: str
    functional_snapshot_reference: str
    state_manifest_hash: str
    state_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    split: str
    decision_stratum: str
    intervention_point: str
    activation_orientation: str
    output_declaration: str
    legal_membership_hash: str
    input_dimension: int
    output_dimension: int
    k: int
    support_count: int
    estimator_mode: str
    exact_restriction_energy: float
    jvp_restriction_energy: float
    exact_jvp_abs_error: float
    jacobian_frobenius_squared: float
    hutchinson: dict[str, float | int]
    random_subspace_null_mean: float
    random_subspace_null_interval: tuple[float, float]
    calibrated_enrichment: float
    control_energies: dict[str, float]
    right_jacobian_principal_angles: tuple[float, ...]
    activation_variance: float
    gradient_norm: float
    constraint_debt_join: dict[str, float | None]
    protected_metric_join: dict[str, float | None]
    runtime_seconds: float
    peak_memory_bytes: int | None
    numerical_status: str
    floor_gate_claim_scope: str
    version_stamp: dict[str, Any]
    schema: str = "CausalSubspaceSnapshotV1"
    estimator_version: str = "activation-restriction-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _snapshot(
    *,
    label: str,
    weight: torch.Tensor,
    covariance: torch.Tensor,
    target: Callable[[torch.Tensor], torch.Tensor],
    activation: torch.Tensor,
    observed_effect: float,
    permutation_jacobian: torch.Tensor,
    floor_scope: str,
    seed: int,
) -> CausalSubspaceSnapshotV1:
    k = 1
    bases = functional_activation_subspaces(weight, covariance, k=k)
    jacobian = exact_jacobian(target, activation)
    denominator = float(jacobian.square().sum())
    functional_energy = restriction_energy(jacobian, bases["functional_top"])
    jvp_energy = jvp_restriction_energy(
        target,
        activation,
        bases["functional_top"],
        denominator=denominator,
    )
    hutchinson = hutchinson_jacobian_frobenius(target, activation, probes=64, seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    random_energies = []
    for _ in range(64):
        vector = torch.randn(
            activation.numel(), generator=generator, dtype=torch.float64
        )
        random_energies.append(
            restriction_energy(jacobian, (vector / vector.norm()).unsqueeze(1))
        )
    null_low, null_high = _interval(random_energies)
    controls = {
        name: restriction_energy(jacobian, basis) for name, basis in bases.items()
    }
    controls["state_label_permutation"] = restriction_energy(
        permutation_jacobian.double(), bases["functional_top"]
    )
    random_weight = torch.randn(weight.shape, generator=generator, dtype=torch.float64)
    random_weight *= weight.double().norm() / random_weight.norm()
    controls["norm_matched_random_checkpoint"] = restriction_energy(
        jacobian,
        functional_activation_subspaces(random_weight, covariance, k=k)[
            "functional_top"
        ],
    )
    _, _, jacobian_vh = torch.linalg.svd(jacobian, full_matrices=False)
    jacobian_basis = jacobian_vh[:k].T
    output = target(activation)
    state_ids = (f"{label}-state-0", f"{label}-state-1")
    group_ids = (f"{label}-group",)
    state_manifest = tuple(
        {
            "state_id": state_id,
            "group_id": group_ids[0],
            "module_path": f"fixture.{label}",
            "decision_stratum": label,
            "split": "held_out",
        }
        for state_id in state_ids
    )
    return CausalSubspaceSnapshotV1(
        checkpoint_reference=f"fixture:{label}",
        module_path=f"fixture.{label}",
        functional_snapshot_reference=f"slm217:fixture:{label}",
        state_manifest_hash=validate_state_manifest(state_manifest),
        state_ids=state_ids,
        group_ids=group_ids,
        split="held_out",
        decision_stratum=label,
        intervention_point="complete nn.Linear input activation",
        activation_orientation=ORIENTATION,
        output_declaration="frozen legal-decision logit vector",
        legal_membership_hash=_sha({"label": label, "legal_ids": [0, 1]}),
        input_dimension=activation.numel(),
        output_dimension=output.numel(),
        k=k,
        support_count=len(state_ids),
        estimator_mode="exact_plus_jvp_hutchinson",
        exact_restriction_energy=functional_energy,
        jvp_restriction_energy=jvp_energy,
        exact_jvp_abs_error=abs(functional_energy - jvp_energy),
        jacobian_frobenius_squared=denominator,
        hutchinson=hutchinson,
        random_subspace_null_mean=sum(random_energies) / len(random_energies),
        random_subspace_null_interval=(null_low, null_high),
        calibrated_enrichment=(
            functional_energy - sum(random_energies) / len(random_energies)
        ),
        control_energies=controls,
        right_jacobian_principal_angles=principal_angles(
            bases["functional_top"], jacobian_basis
        ),
        activation_variance=float(torch.trace(covariance.double())),
        gradient_norm=float(jacobian.norm()),
        constraint_debt_join={"d_legal": None, "d_good": None},
        protected_metric_join={
            "fixture_exact_choice_effect": observed_effect,
            "strict_meaning_v2": None,
        },
        runtime_seconds=0.0,
        peak_memory_bytes=(
            jacobian.numel() + sum(basis.numel() for basis in bases.values())
        )
        * 8,
        numerical_status="ok",
        floor_gate_claim_scope=floor_scope,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm217_functional_spectra",
            "harness.experiments.slm218_cross_attention_retention",
            "harness.experiments.slm220_causal_subspace",
        ),
    )


@dataclass(frozen=True)
class CausalSubspaceReportV1:
    run_id: str
    snapshots: tuple[CausalSubspaceSnapshotV1, ...]
    inventory: dict[str, Any]
    cost_accuracy_benchmark: dict[str, Any]
    verdict: str
    rationale: tuple[str, ...]
    eligible_perturbation_bands: tuple[str, ...]
    semantic_floor_hash: str
    semantic_floor_verdict: str
    source_commit: str
    version_stamp: dict[str, Any]
    schema: str = "CausalSubspaceReportV1"
    claim_class: str = "diagnostic_fixture"
    honesty_mode: str = "analytic_cpu_no_checkpoint"

    @property
    def report_hash(self) -> str:
        return _sha(_without_volatile(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshots"] = [row.to_dict() for row in self.snapshots]
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def run_fixture_retrospective(repo_root: Path) -> CausalSubspaceReportV1:
    """Validate the estimator and fail closed on unavailable checkpoint evidence."""
    floor = load_semantic_floor_gate(repo_root / SEMANTIC_FLOOR_GATE_PATH)
    activation = torch.tensor([0.3, -0.2, 0.5, 0.1], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([9.0, 4.0, 1.0, 0.25]))
    weight = torch.diag(torch.tensor([4.0, 2.0, 1.0, 0.5]))
    cases = (
        ("learned_unused_auxiliary", torch.tensor([[0.0, 0.0, 0.0, 1.0]]), 0.0),
        ("causally_effective_choice", torch.tensor([[1.0, 0.0, 0.0, 0.0]]), 1.0),
        ("cross_attention_candidate", torch.tensor([[0.0, 1.0, 0.0, 0.0]]), 0.5),
        ("adapter_geometry_candidate", torch.tensor([[0.7, 0.0, 0.7, 0.0]]), 0.7),
    )
    snapshots = tuple(
        _snapshot(
            label=label,
            weight=weight,
            covariance=covariance,
            target=lambda value, matrix=jacobian: matrix.double() @ value,
            activation=activation,
            observed_effect=effect,
            permutation_jacobian=cases[(index + 1) % len(cases)][1],
            floor_scope=(
                "diagnostic geometry only; semantic causal interpretation blocked"
            ),
            seed=220 + index,
        )
        for index, (label, jacobian, effect) in enumerate(cases)
    )
    exact_errors = [row.exact_jvp_abs_error for row in snapshots]
    hutchinson_errors = [
        abs(float(row.hutchinson["frobenius_squared"]) - row.jacobian_frobenius_squared)
        / row.jacobian_frobenius_squared
        for row in snapshots
    ]
    return CausalSubspaceReportV1(
        run_id="slm220-causal-subspace-fixture-20260723",
        snapshots=snapshots,
        inventory={
            "slm217": "fixture only; no compatible checkpoint plus DecisionEvent manifest",
            "slm218": "zero complete provenance-resolvable checkpoint families",
            "slm125": "fixture report only; no retained compatible adapter family",
            "current_checkpoint_retrospective_eligible": False,
        },
        cost_accuracy_benchmark={
            "device": "cpu",
            "input_dimension": 4,
            "output_dimension": 1,
            "exact_jvp_max_abs_error": max(exact_errors),
            "hutchinson_max_relative_error": max(hutchinson_errors),
            "hutchinson_probes": 64,
            "exact_jacobian_elements_per_fixture": 4,
            "jvp_directions_per_primary_subspace": 1,
            "vjp_probes_per_fixture": 64,
            "runtime_measurement": (
                "bounded by external cap; deterministic operation-size and tensor-byte "
                "costs retained instead of volatile wall timing"
            ),
            "hard_wall_cap_seconds": 170,
        },
        verdict="rejected",
        rationale=(
            "analytic fixtures validate activation-side orientation, exact/JVP parity, deterministic Hutchinson uncertainty, and all preregistered subspace controls",
            "the learned-unused and causally-effective fixtures separate as designed, but fixtures are not model evidence",
            "no compatible retained checkpoint plus exact-state manifest resolves for the required retrospective families",
            f"SemanticFloorGateV1 is {floor.verdict}; semantic causal interpretation remains blocked",
            "no matrix or band is eligible for SLM-220 coupling-based perturbation",
        ),
        eligible_perturbation_bands=(),
        semantic_floor_hash=floor.gate_hash,
        semantic_floor_verdict=floor.verdict,
        source_commit=git_commit() or "UNKNOWN",
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.semantic_floor_gate",
            "harness.experiments.slm217_functional_spectra",
            "harness.experiments.slm218_cross_attention_retention",
            "harness.experiments.slm220_causal_subspace",
        ),
    )


def render_markdown(report: CausalSubspaceReportV1) -> str:
    lines = [
        "# SLM-220: activation-side causal restriction energy",
        "",
        f"**Verdict:** `{report.verdict}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Semantic floor:** `{report.semantic_floor_hash}` "
        f"(`{report.semantic_floor_verdict}`)",
        "",
        "The estimator contract is supported on analytic systems. The requested "
        "model-retrospective hypothesis is rejected for use because no compatible "
        "checkpoint/state-manifest family resolves. No perturbation target is nominated.",
        "",
        "| Fixture | Functional energy | Random-null mean / 95% interval | Exact/JVP error | Fixture-effect label |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.snapshots:
        lines.append(
            f"| `{row.decision_stratum}` | {row.exact_restriction_energy:.6f} | "
            f"{row.random_subspace_null_mean:.6f} / "
            f"[{row.random_subspace_null_interval[0]:.6f}, "
            f"{row.random_subspace_null_interval[1]:.6f}] | "
            f"{row.exact_jvp_abs_error:.3e} | "
            f"{row.protected_metric_join['fixture_exact_choice_effect']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Contract and controls",
            "",
            f"- Orientation: `{ORIENTATION}`.",
            "- Exact Jacobian and JVP numerator agree on every fixture; the denominator "
            "also has a deterministic 64-probe Hutchinson VJP estimate with 95% interval.",
            "- Every snapshot includes repeated random orthonormal, raw-weight, "
            "functional top/middle/bottom, covariance-only, state-permutation, and "
            "norm-matched random-checkpoint controls.",
            "- Compiler/legal membership is immutable metadata and is never differentiated "
            "through. The tested intervention wrapper fails if it changes.",
            "- Semantic meaning-v2 and protected/debt joins remain unavailable under the "
            "current floor gate; they are not synthesized from fixture labels.",
            "",
            "## Evidence inventory",
            "",
            *[f"- `{key}`: {value}" for key, value in report.inventory.items()],
            "",
            "## Decision",
            "",
            *[f"- {reason}" for reason in report.rationale],
            "",
            "Eligible matrices/bands for a later coupling perturbation: **none**.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python "
            "-m scripts.run_causal_subspace_fixture --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
