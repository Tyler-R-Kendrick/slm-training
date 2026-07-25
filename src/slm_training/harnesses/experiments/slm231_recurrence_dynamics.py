"""Residual-correct recurrence dynamics diagnostics for SLM-231."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower

DYNAMICS_SCHEMA = "RecurrenceDynamicsSnapshotV1"
ESTIMATOR_VERSION = "rsc1-02-v1"


def _math_attention_target(
    target: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Force forward-AD-compatible SDPA without changing attention math."""

    def wrapped(value: torch.Tensor) -> torch.Tensor:
        with sdpa_kernel(SDPBackend.MATH):
            return target(value)

    return wrapped


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


class DynamicsVerdict(str, Enum):
    STRUCTURED_STABLE_REFINEMENT = "structured_stable_refinement"
    DEAD_INCREMENT = "dead_increment"
    OVERCONTRACTIVE = "overcontractive"
    EXPANSIVE_UNSTABLE = "expansive_unstable"
    ROTATING_OSCILLATORY = "rotating_oscillatory"
    IDENTITY_NEAR_ONE_ONLY = "identity_near_one_only"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class StateProjectionV1:
    """A fixed active-position projection for tuple ``(y,z)`` state."""

    active_positions: tuple[int, ...]
    sequence_length: int
    hidden_size: int
    includes_z: bool
    policy: str = "active_token_subset"
    schema: str = "RecurrenceStateProjectionV1"

    @property
    def dimension(self) -> int:
        multiplier = 2 if self.includes_z else 1
        return len(self.active_positions) * self.hidden_size * multiplier

    def validate(self) -> None:
        if not self.active_positions:
            raise ValueError("state projection requires at least one active position")
        if tuple(sorted(set(self.active_positions))) != self.active_positions:
            raise ValueError("active positions must be sorted and unique")
        if self.active_positions[-1] >= self.sequence_length:
            raise ValueError("active position exceeds sequence length")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")


@dataclass(frozen=True)
class SingularEstimateV1:
    singular_values: tuple[float, ...]
    iterations: int
    seed: int
    residual: float
    method: str = "block_jvp_vjp"


@dataclass(frozen=True)
class RecurrenceDynamicsSnapshotV1:
    checkpoint_sha256: str
    transition_hash: str
    request_id: str
    group_id: str
    split: str
    suite: str
    trained_depth: int
    evaluated_depth: int
    projection: dict[str, Any]
    increment_by_depth: tuple[dict[str, Any], ...]
    composite_by_depth: tuple[dict[str, Any], ...]
    product_singular_values: tuple[float, ...]
    finite_time_lyapunov: tuple[float, ...]
    alignment_by_depth: tuple[dict[str, Any], ...]
    outcome_join: dict[str, Any]
    nulls: dict[str, Any]
    estimator: dict[str, Any]
    numerical_flags: tuple[str, ...]
    floor_gate_claim_scope: str
    verdict: str
    version_stamp: dict[str, Any]
    schema: str = DYNAMICS_SCHEMA

    def validate(self) -> None:
        if self.schema != DYNAMICS_SCHEMA:
            raise ValueError(f"unsupported dynamics schema: {self.schema}")
        if (
            self.trained_depth < 1
            or not 1 <= self.evaluated_depth <= self.trained_depth
        ):
            raise ValueError("evaluated depth must be inside trained depth")
        if len(self.increment_by_depth) != self.evaluated_depth:
            raise ValueError("increment summaries must cover every evaluated depth")
        if len(self.composite_by_depth) != self.evaluated_depth:
            raise ValueError("composite summaries must cover every evaluated depth")
        if len(self.alignment_by_depth) != self.evaluated_depth:
            raise ValueError("alignment summaries must cover every evaluated depth")
        if len(self.product_singular_values) != len(self.finite_time_lyapunov):
            raise ValueError("product singular values and FTLE lengths differ")
        if self.verdict not in {item.value for item in DynamicsVerdict}:
            raise ValueError(f"unsupported dynamics verdict: {self.verdict}")
        if not self.version_stamp:
            raise ValueError("version_stamp is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def exact_jacobian(
    target: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> torch.Tensor:
    if state.ndim != 1:
        raise ValueError("state must be one-dimensional")
    target = _math_attention_target(target)
    jacobian = torch.func.jacrev(target)(state)
    if jacobian.ndim != 2 or jacobian.shape[1] != state.numel():
        raise ValueError("Jacobian orientation must be [output,input]")
    return jacobian.detach().cpu().double()


def residual_jacobians(
    transition: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(J_delta, J_T)`` with the residual identity kept explicit."""
    composite = exact_jacobian(transition, state)
    if composite.shape[0] != composite.shape[1]:
        raise ValueError("residual transition must preserve state dimension")
    increment = composite - torch.eye(composite.shape[0], dtype=composite.dtype)
    return increment, composite


def _orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("block iterate must be a matrix")
    return torch.linalg.qr(matrix, mode="reduced").Q


def block_singular_estimate(
    target: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    *,
    k: int,
    iterations: int,
    seed: int,
) -> SingularEstimateV1:
    """Estimate leading singular values with alternating JVP/VJP block power."""
    if state.ndim != 1 or k < 1 or k > state.numel() or iterations < 1:
        raise ValueError("invalid block singular estimator settings")
    target = _math_attention_target(target)
    output, vjp = torch.func.vjp(target, state)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    right = torch.randn(
        state.numel(), k, generator=generator, dtype=state.dtype, device="cpu"
    ).to(state.device)
    right = _orthonormalize(right)
    left = torch.empty(output.numel(), k, dtype=state.dtype, device=state.device)
    for _ in range(iterations):
        left = torch.stack(
            [
                torch.func.jvp(target, (state,), (direction,))[1]
                for direction in right.T
            ],
            dim=1,
        )
        left = _orthonormalize(left)
        right = torch.stack([vjp(direction)[0] for direction in left.T], dim=1)
        right = _orthonormalize(right)
    image = torch.stack(
        [torch.func.jvp(target, (state,), (direction,))[1] for direction in right.T],
        dim=1,
    )
    left, singular, rotation = torch.linalg.svd(image.double(), full_matrices=False)
    right_singular = right.double() @ rotation.T
    adjoint = torch.stack([vjp(direction.to(output))[0] for direction in left.T], dim=1)
    residual = torch.linalg.vector_norm(
        adjoint.double() - right_singular * singular.unsqueeze(0)
    ) / singular.sum().clamp_min(torch.finfo(torch.float64).eps)
    return SingularEstimateV1(
        singular_values=tuple(float(value) for value in singular),
        iterations=iterations,
        seed=seed,
        residual=float(residual),
    )


def exact_product(jacobians: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compose ``J_(R-1) ... J_0`` in trajectory order."""
    if not jacobians:
        raise ValueError("at least one Jacobian is required")
    dimension = jacobians[0].shape[0]
    product = torch.eye(dimension, dtype=jacobians[0].dtype)
    for jacobian in jacobians:
        if jacobian.shape != (dimension, dimension):
            raise ValueError("trajectory Jacobians must share a square shape")
        product = jacobian @ product
    return product


def finite_time_lyapunov(
    product_singular_values: Sequence[float],
    *,
    depth: int,
    epsilon: float = 1e-12,
) -> tuple[float, ...]:
    if depth < 1 or epsilon <= 0:
        raise ValueError("depth and epsilon must be positive")
    return tuple(
        math.log(max(float(value), 0.0) + epsilon) / depth
        for value in product_singular_values
    )


def trajectory_product_estimate(
    transitions: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    states: Sequence[torch.Tensor],
    *,
    k: int,
    iterations: int,
    seed: int,
) -> SingularEstimateV1:
    """Estimate ``P_R`` through sequential trajectory JVPs and reverse VJPs."""
    if not transitions or len(transitions) != len(states):
        raise ValueError(
            "transitions and trajectory states must be non-empty and aligned"
        )
    dimension = states[0].numel()
    if any(state.ndim != 1 or state.numel() != dimension for state in states):
        raise ValueError("trajectory states must share one flat dimension")
    transitions = tuple(_math_attention_target(target) for target in transitions)
    vjps = [
        torch.func.vjp(target, state)[1] for target, state in zip(transitions, states)
    ]

    def product_jvp(direction: torch.Tensor) -> torch.Tensor:
        tangent = direction
        for target, state in zip(transitions, states):
            tangent = torch.func.jvp(target, (state,), (tangent,))[1]
        return tangent

    def product_vjp(direction: torch.Tensor) -> torch.Tensor:
        tangent = direction
        for vjp in reversed(vjps):
            tangent = vjp(tangent)[0]
        return tangent

    generator = torch.Generator(device="cpu").manual_seed(seed)
    right = _orthonormalize(
        torch.randn(dimension, k, generator=generator, dtype=states[0].dtype).to(
            states[0].device
        )
    )
    for _ in range(iterations):
        left = _orthonormalize(
            torch.stack([product_jvp(direction) for direction in right.T], dim=1)
        )
        right = _orthonormalize(
            torch.stack([product_vjp(direction) for direction in left.T], dim=1)
        )
    image = torch.stack([product_jvp(direction) for direction in right.T], dim=1)
    left, singular, rotation = torch.linalg.svd(image.double(), full_matrices=False)
    right_singular = right.double() @ rotation.T
    adjoint = torch.stack(
        [product_vjp(direction.to(states[-1])) for direction in left.T], dim=1
    )
    residual = torch.linalg.vector_norm(
        adjoint.double() - right_singular * singular.unsqueeze(0)
    ) / singular.sum().clamp_min(torch.finfo(torch.float64).eps)
    return SingularEstimateV1(
        singular_values=tuple(float(value) for value in singular),
        iterations=iterations,
        seed=seed,
        residual=float(residual),
        method="trajectory_block_jvp_vjp_qr",
    )


def state_projection(
    y: torch.Tensor,
    z: torch.Tensor | None,
    active_positions: Sequence[int],
) -> StateProjectionV1:
    if y.ndim != 3 or y.shape[0] != 1:
        raise ValueError("bounded dynamics projection requires y shape [1,T,D]")
    positions = tuple(sorted(set(int(value) for value in active_positions)))
    projection = StateProjectionV1(
        active_positions=positions,
        sequence_length=y.shape[1],
        hidden_size=y.shape[2],
        includes_z=z is not None,
    )
    projection.validate()
    if z is not None and z.shape != y.shape:
        raise ValueError("z must match y shape")
    return projection


def flatten_projected_state(
    y: torch.Tensor,
    z: torch.Tensor | None,
    projection: StateProjectionV1,
) -> torch.Tensor:
    indices = torch.tensor(projection.active_positions, device=y.device)
    pieces = [y.index_select(1, indices).reshape(-1)]
    if projection.includes_z:
        if z is None:
            raise ValueError("projection requires z state")
        pieces.append(z.index_select(1, indices).reshape(-1))
    return torch.cat(pieces)


def replace_projected_state(
    state: torch.Tensor,
    base_y: torch.Tensor,
    base_z: torch.Tensor | None,
    projection: StateProjectionV1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if state.ndim != 1 or state.numel() != projection.dimension:
        raise ValueError("flat state does not match projection dimension")
    indices = torch.tensor(projection.active_positions, device=base_y.device)
    width = len(projection.active_positions) * projection.hidden_size
    y = base_y.clone()
    y[:, indices] = state[:width].reshape(1, len(indices), projection.hidden_size)
    z = base_z
    if projection.includes_z:
        if base_z is None:
            raise ValueError("projection requires z state")
        z = base_z.clone()
        z[:, indices] = state[width:].reshape(1, len(indices), projection.hidden_size)
    return y, z


def projected_transition(
    tower: SharedRecursiveDenoiserTower,
    *,
    base_y: torch.Tensor,
    base_z: torch.Tensor | None,
    context: torch.Tensor,
    self_pad_mask: torch.Tensor,
    ctx_pad_mask: torch.Tensor | None,
    runtime_symbol_features: torch.Tensor | None,
    projection: StateProjectionV1,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Bind fixed conditioning and expose a pure flat-state transition."""

    def target(state: torch.Tensor) -> torch.Tensor:
        y, z = replace_projected_state(state, base_y, base_z, projection)
        result = tower.transition_step(
            y,
            z,
            context,
            self_pad_mask,
            ctx_pad_mask,
            runtime_symbol_features=runtime_symbol_features,
        )
        next_y = result["y"]
        next_z = result["z"]
        assert isinstance(next_y, torch.Tensor)
        assert next_z is None or isinstance(next_z, torch.Tensor)
        return flatten_projected_state(next_y, next_z, projection)

    return target


def matrix_summary(matrix: torch.Tensor) -> dict[str, Any]:
    singular = torch.linalg.svdvals(matrix.detach().cpu().double())
    frobenius = float(torch.linalg.vector_norm(matrix.detach().cpu().double()))
    spectral = float(singular[0]) if singular.numel() else 0.0
    stable_rank = 0.0 if spectral == 0 else frobenius**2 / spectral**2
    return {
        "singular_values": [float(value) for value in singular],
        "spectral_norm": spectral,
        "frobenius_norm": frobenius,
        "stable_rank": stable_rank,
        "contracting_fraction": float((singular < 0.95).double().mean()),
        "neutral_fraction": float(
            ((singular >= 0.95) & (singular <= 1.05)).double().mean()
        ),
        "expanding_fraction": float((singular > 1.05).double().mean()),
    }


def principal_angles(
    left: torch.Tensor, right: torch.Tensor, *, rank: int
) -> tuple[float, ...]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(
            "subspace samples must be [samples,features] with shared features"
        )
    k = min(rank, left.shape[0], right.shape[0], left.shape[1])
    if k < 1:
        return ()
    left_basis = torch.linalg.svd(left.double(), full_matrices=False).Vh[:k].T
    right_basis = torch.linalg.svd(right.double(), full_matrices=False).Vh[:k].T
    cosine = torch.linalg.svdvals(left_basis.T @ right_basis).clamp(0, 1)
    return tuple(float(value) for value in torch.acos(cosine))


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("CKA inputs must share [samples,features] shape")
    left = left.double() - left.double().mean(dim=0, keepdim=True)
    right = right.double() - right.double().mean(dim=0, keepdim=True)
    numerator = torch.linalg.vector_norm(left.T @ right).square()
    denominator = torch.linalg.vector_norm(left.T @ left) * torch.linalg.vector_norm(
        right.T @ right
    )
    if float(denominator) == 0:
        return 0.0
    return float(numerator / denominator)


def classify_dynamics(
    *,
    increment_spectral_norm: float,
    product_spectral_norm: float,
    maximum_ftle: float,
    update_alignment_cosine: float,
    outcome_verdict: str,
) -> DynamicsVerdict:
    """Apply frozen diagnostic thresholds; semantic outcomes retain authority."""
    if increment_spectral_norm <= 1e-5:
        return DynamicsVerdict.DEAD_INCREMENT
    if product_spectral_norm >= 4.0 or maximum_ftle >= 0.35:
        return DynamicsVerdict.EXPANSIVE_UNSTABLE
    if product_spectral_norm <= 0.1:
        return DynamicsVerdict.OVERCONTRACTIVE
    if update_alignment_cosine <= -0.25:
        return DynamicsVerdict.ROTATING_OSCILLATORY
    if increment_spectral_norm <= 0.05 and 0.95 <= product_spectral_norm <= 1.05:
        return DynamicsVerdict.IDENTITY_NEAR_ONE_ONLY
    if outcome_verdict == "refining" and maximum_ftle <= 0.1:
        return DynamicsVerdict.STRUCTURED_STABLE_REFINEMENT
    return DynamicsVerdict.INCONCLUSIVE
