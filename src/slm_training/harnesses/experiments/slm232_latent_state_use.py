"""Evaluation-only latent-state rank and causal-use contracts for SLM-232."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

GATE_SCHEMA = "LatentStateUseGateV1"
ABLATION_SCHEMA = "RecursiveStateAblationV1"
RESULT_SCHEMA = "LatentAblationResultV1"
INITIAL_ABLATIONS = (
    "none",
    "zero_z0",
    "mean_z0",
    "shuffle_z_across_examples",
    "swap_z_matched",
    "zero_ctx_proj",
    "zero_z_latent",
    "remove_z_position",
    "random_norm_matched",
)
PATH_ABLATIONS = ("detach_z_to_y", "detach_y_to_z")
CONTROL_ABLATIONS = ("y_only_repeated_control",)
NON_APPLICABLE_ABLATIONS = ("gold_oracle_z",)


class LatentStateVerdict(str, Enum):
    CAUSALLY_USED = "causally_used"
    CONTEXT_ONLY = "context_only"
    BROADCAST_LOW_RANK = "broadcast_low_rank"
    Y_ONLY_EQUIVALENT = "y_only_equivalent"
    DECORATIVE = "decorative"
    UNSTABLE = "unstable"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class RecursiveStateAblationV1:
    ablation_id: str
    seed: int = 232
    evaluation_only: bool = True
    schema: str = ABLATION_SCHEMA

    def validate(self) -> None:
        known = (
            INITIAL_ABLATIONS
            + PATH_ABLATIONS
            + CONTROL_ABLATIONS
            + NON_APPLICABLE_ABLATIONS
        )
        if self.schema != ABLATION_SCHEMA or self.ablation_id not in known:
            raise ValueError(
                f"unsupported recursive-state ablation: {self.ablation_id}"
            )
        if not self.evaluation_only:
            raise ValueError("recursive-state ablations are evaluation-only")


@dataclass(frozen=True)
class LatentAblationResultV1:
    ablation_id: str
    applicability: str
    support: int
    seed: int | None
    pair_manifest_sha256: str | None
    achieved_norm_max_abs_delta: float | None
    teacher_forced_full_vocab_kl: float | None
    teacher_forced_top1_change_rate: float | None
    exact_candidate_status: str
    protected_outcome_status: str
    free_running_status: str
    numerical_status: str
    reason: str | None = None
    schema: str = RESULT_SCHEMA

    def validate(self) -> None:
        RecursiveStateAblationV1(self.ablation_id).validate()
        if self.schema != RESULT_SCHEMA or self.applicability not in {
            "applicable",
            "not_applicable",
        }:
            raise ValueError("invalid latent ablation result identity")
        if self.support < 0:
            raise ValueError("ablation support cannot be negative")
        if self.applicability == "not_applicable" and not self.reason:
            raise ValueError("not-applicable ablations require a reason")
        if self.applicability == "applicable" and self.support < 1:
            raise ValueError("applicable ablations require positive support")
        if self.ablation_id == "swap_z_matched" and self.applicability == "applicable":
            if not self.pair_manifest_sha256:
                raise ValueError("matched swap requires a pair manifest")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class LatentStateUseGateV1:
    checkpoint_sha256: str
    checkpoint_config_sha256: str
    state_projection: str
    trained_depth: int
    evaluated_depths: tuple[int, ...]
    support: int
    group_support: int
    representation: dict[str, Any]
    ablations: tuple[LatentAblationResultV1, ...]
    exact_state_evidence: dict[str, Any]
    protected_outcome_evidence: dict[str, Any]
    free_running_evidence: dict[str, Any]
    control_comparison: dict[str, Any]
    uncertainty: dict[str, Any]
    slm230_join: dict[str, Any]
    slm231_join: dict[str, Any]
    floor_gate_scope: str
    verdict: str
    allowed_downstream_work: tuple[str, ...]
    blocking_evidence: tuple[str, ...]
    version_stamp: dict[str, Any]
    schema: str = GATE_SCHEMA

    def validate(self) -> None:
        if (
            self.schema != GATE_SCHEMA
            or self.support < 1
            or self.group_support < 1
            or self.trained_depth < 1
            or not self.evaluated_depths
        ):
            raise ValueError("invalid latent-state gate identity/support")
        if len(self.checkpoint_sha256) != 64 or len(self.checkpoint_config_sha256) != 64:
            raise ValueError("checkpoint and config SHA-256 identities are required")
        for row in self.ablations:
            row.validate()
        if self.verdict not in {item.value for item in LatentStateVerdict}:
            raise ValueError(f"unsupported latent-state verdict: {self.verdict}")
        if not self.version_stamp:
            raise ValueError("version_stamp is required")
        if (
            self.verdict != LatentStateVerdict.CAUSALLY_USED.value
            and not self.blocking_evidence
        ):
            raise ValueError(
                "non-positive latent-state gates require blocking evidence"
            )
        if self.verdict == LatentStateVerdict.CAUSALLY_USED.value:
            required = (
                self.exact_state_evidence.get("reproducible_actual_legal_effect"),
                self.protected_outcome_evidence.get("reproducible_protected_effect"),
                self.control_comparison.get("targeted_exceeds_nuisance"),
                self.control_comparison.get("matched_y_only_available"),
                self.uncertainty.get("positive_effect_excludes_zero"),
                self.free_running_evidence.get("nonvacuous"),
            )
            if any(value is not True for value in required):
                raise ValueError(
                    "causally_used requires reproduced legal/protected, control, "
                    "uncertainty, y-only, and nonvacuous evidence"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def compose_z0(
    components: Mapping[str, torch.Tensor | bool | None],
) -> torch.Tensor | None:
    terms = [
        components.get("z_latent_component"),
        components.get("z_context_component"),
        components.get("z_position_component"),
    ]
    tensors = [term for term in terms if isinstance(term, torch.Tensor)]
    return sum(tensors[1:], tensors[0]) if tensors else None


def apply_initial_ablation(
    components: Mapping[str, torch.Tensor | bool | None],
    ablation: RecursiveStateAblationV1,
    *,
    mean_z0: torch.Tensor | None = None,
    matched_z0: torch.Tensor | None = None,
    permutation: torch.Tensor | None = None,
    pair_manifest_sha256: str | None = None,
) -> torch.Tensor | None:
    """Return an ablated z0 without changing component tensors or parameters."""
    ablation.validate()
    z0 = compose_z0(components)
    mode = ablation.ablation_id
    if mode in PATH_ABLATIONS:
        return z0
    if mode in NON_APPLICABLE_ABLATIONS:
        return None
    if z0 is None:
        raise ValueError("initial z ablation requires an explicit z state")
    if mode == "none":
        return z0
    if mode == "zero_z0":
        return torch.zeros_like(z0)
    if mode == "mean_z0":
        if mean_z0 is None:
            raise ValueError("mean_z0 requires a frozen calibration mean")
        try:
            return torch.broadcast_to(mean_z0, z0.shape)
        except RuntimeError as exc:
            raise ValueError(
                "mean_z0 must broadcast to the evaluation state"
            ) from exc
    if mode == "shuffle_z_across_examples":
        if permutation is None:
            raise ValueError("shuffle requires a preregistered permutation")
        if permutation.shape != (z0.shape[0],):
            raise ValueError("permutation must have one index per example")
        expected = torch.arange(z0.shape[0], device=permutation.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("permutation must be a bijection")
        return z0[permutation.to(z0.device)]
    if mode == "swap_z_matched":
        if matched_z0 is None or matched_z0.shape != z0.shape:
            raise ValueError("matched swap requires a shape-matched z0")
        if not pair_manifest_sha256 or len(pair_manifest_sha256) != 64:
            raise ValueError("matched swap requires a SHA-256 pair manifest")
        return matched_z0
    if mode == "zero_ctx_proj":
        if components.get("context_projection_applied") is not True:
            raise ValueError("zero_ctx_proj requires an applied learned projection")
        return z0 - _required_component(components, "z_context_component")
    if mode == "zero_z_latent":
        return z0 - _required_component(components, "z_latent_component")
    if mode == "remove_z_position":
        return z0 - _required_component(components, "z_position_component")
    if mode == "random_norm_matched":
        generator = torch.Generator(device="cpu").manual_seed(ablation.seed)
        random = torch.randn(z0.shape, generator=generator, dtype=z0.dtype).to(z0)
        target_norm = torch.linalg.vector_norm(z0, dim=-1, keepdim=True)
        random_norm = torch.linalg.vector_norm(random, dim=-1, keepdim=True)
        return random * (target_norm / random_norm.clamp_min(torch.finfo(z0.dtype).eps))
    raise AssertionError(f"unhandled ablation {mode}")


def _required_component(
    components: Mapping[str, torch.Tensor | bool | None], key: str
) -> torch.Tensor:
    value = components.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} is unavailable for this z-state mode")
    return value


def within_group_permutation(
    group_ids: Sequence[str], *, seed: int = 232
) -> tuple[torch.Tensor, str]:
    """Build a deterministic, leakage-safe within-group permutation + manifest."""
    if len(group_ids) < 2:
        raise ValueError("shuffle requires at least two examples")
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(index)
    if any(len(indices) < 2 for indices in groups.values()):
        raise ValueError("every shuffle group requires at least two examples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.arange(len(group_ids))
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator)
        if torch.equal(order, torch.arange(len(indices))):
            order = order.roll(1)
        source = torch.tensor(indices)
        permutation[source] = source[order]
    manifest = {
        "schema": "LatentPairManifestV1",
        "seed": seed,
        "group_ids": list(group_ids),
        "permutation": permutation.tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return permutation, digest


def representation_summary(rows: torch.Tensor) -> dict[str, Any]:
    """Summarize centered variation without treating token rows as groups."""
    if rows.ndim != 2 or rows.shape[0] < 2:
        raise ValueError("representation rows must be [samples,features], samples >= 2")
    centered = rows.double() - rows.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if float(total) == 0.0:
        return {
            "singular_values": [float(value) for value in singular],
            "matrix_rank": 0,
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "total_centered_energy": 0.0,
        }
    probabilities = energy / total.clamp_min(torch.finfo(torch.float64).eps)
    effective_rank = float(
        torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
    )
    participation = float(total.square() / energy.square().sum().clamp_min(1e-30))
    return {
        "singular_values": [float(value) for value in singular],
        "matrix_rank": int(torch.linalg.matrix_rank(centered)),
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "total_centered_energy": float(total),
    }


def classify_latent_state_use(
    *,
    rank_qualified: bool | None,
    context_only: bool | None,
    targeted_effect_reproduced: bool | None,
    targeted_exceeds_nuisance: bool | None,
    matched_y_only_equivalent: bool | None,
    powered_no_effect: bool | None,
    actual_legal_effect: bool | None,
    protected_outcome_effect: bool | None,
    uncertainty_excludes_zero: bool | None,
    unstable_dynamics: bool,
    nonvacuous_outcome: bool,
) -> LatentStateVerdict:
    if unstable_dynamics:
        return LatentStateVerdict.UNSTABLE
    required = (
        rank_qualified,
        context_only,
        targeted_effect_reproduced,
        targeted_exceeds_nuisance,
        matched_y_only_equivalent,
        powered_no_effect,
        actual_legal_effect,
        protected_outcome_effect,
        uncertainty_excludes_zero,
    )
    if any(value is None for value in required):
        return LatentStateVerdict.INCONCLUSIVE
    if rank_qualified is False:
        return LatentStateVerdict.BROADCAST_LOW_RANK
    if matched_y_only_equivalent is True:
        return LatentStateVerdict.Y_ONLY_EQUIVALENT
    if powered_no_effect is True:
        return LatentStateVerdict.DECORATIVE
    if context_only is True and targeted_effect_reproduced is True:
        return LatentStateVerdict.CONTEXT_ONLY
    if (
        targeted_effect_reproduced is True
        and targeted_exceeds_nuisance is True
        and matched_y_only_equivalent is False
        and actual_legal_effect is True
        and protected_outcome_effect is True
        and uncertainty_excludes_zero is True
        and nonvacuous_outcome
    ):
        return LatentStateVerdict.CAUSALLY_USED
    return LatentStateVerdict.INCONCLUSIVE
