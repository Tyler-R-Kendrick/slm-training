"""SDE4-04 (SLM-182) pretrained-denoiser activation manifest.

Plan-only wiring slice: frozen, versioned activation/budget/candidate manifest
for deciding whether to spend on a pretrained denoising-language-model backbone
versus staying with the current small controller.  No model download, training,
or eval happens here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

MANIFEST_SCHEMA = "pretrained_denoiser_activation/v1"
HYPOTHESIS_ID = "H18"

ACTIVATION_VERDICTS = frozenset(
    {
        "activation_blocked",
        "ready_to_spend",
        "no_eligible_candidate",
        "license_incompatible",
        "budget_or_yield_blocked",
        "unrun",
    }
)

CAMPAIGN_VERDICTS = frozenset(
    {
        "pretraining_scale_justified",
        "quality_gain_but_deployment_rejected",
        "small_model_pareto_dominant",
        "no_eligible_candidate",
        "infrastructure_or_budget_blocked",
        "inconclusive",
        "unrun",
    }
)

ARM_KINDS = frozenset(
    {
        "current_small_controller_baseline",
        "b4_pilot_reference",
        "pretrained_denoiser_plus_adapters",
        "frozen_backbone_connector_only",
        "adapters_disabled_diagnostic",
        "equal_compute_small_controller_control",
        "random_init_short_budget_diagnostic",
    }
)


def _stable_hash(parts: Mapping[str, Any]) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LicenseTerms:
    """License terms for a pretrained denoiser candidate."""

    spdx_id: str
    commercial_use_allowed: bool
    redistribution_allowed: bool
    modification_allowed: bool
    attribution_required: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spdx_id": self.spdx_id,
            "commercial_use_allowed": self.commercial_use_allowed,
            "redistribution_allowed": self.redistribution_allowed,
            "modification_allowed": self.modification_allowed,
            "attribution_required": self.attribution_required,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LicenseTerms":
        return cls(
            spdx_id=str(data.get("spdx_id", "")),
            commercial_use_allowed=bool(data.get("commercial_use_allowed", False)),
            redistribution_allowed=bool(data.get("redistribution_allowed", False)),
            modification_allowed=bool(data.get("modification_allowed", False)),
            attribution_required=bool(data.get("attribution_required", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class PretrainedDenoiserCandidate:
    """One concrete pretrained denoiser candidate under evaluation."""

    candidate_id: str
    provider: str
    repository: str
    model: str
    revision: str
    file_hashes: dict[str, str]
    license: LicenseTerms
    architecture: str
    pretraining_objective: str
    parameter_count: int
    hidden_width: int
    num_layers: int
    context_length: int
    tokenizer_id: str
    conversion_method: str
    supported_formats: tuple[str, ...]
    estimated_train_memory_bytes: int
    estimated_inference_memory_bytes: int
    estimated_flops_per_forward: int
    expected_serialized_bytes: int
    expected_deployed_bytes: int
    local_offline_available: bool
    unsupported_operations: tuple[str, ...]
    hardware_requirements: tuple[str, ...]
    selection_evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "provider": self.provider,
            "repository": self.repository,
            "model": self.model,
            "revision": self.revision,
            "file_hashes": dict(self.file_hashes),
            "license": self.license.to_dict(),
            "architecture": self.architecture,
            "pretraining_objective": self.pretraining_objective,
            "parameter_count": self.parameter_count,
            "hidden_width": self.hidden_width,
            "num_layers": self.num_layers,
            "context_length": self.context_length,
            "tokenizer_id": self.tokenizer_id,
            "conversion_method": self.conversion_method,
            "supported_formats": list(self.supported_formats),
            "estimated_train_memory_bytes": self.estimated_train_memory_bytes,
            "estimated_inference_memory_bytes": self.estimated_inference_memory_bytes,
            "estimated_flops_per_forward": self.estimated_flops_per_forward,
            "expected_serialized_bytes": self.expected_serialized_bytes,
            "expected_deployed_bytes": self.expected_deployed_bytes,
            "local_offline_available": self.local_offline_available,
            "unsupported_operations": list(self.unsupported_operations),
            "hardware_requirements": list(self.hardware_requirements),
            "selection_evidence": self.selection_evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainedDenoiserCandidate":
        license_data = data.get("license") if isinstance(data.get("license"), dict) else {}
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            provider=str(data.get("provider", "")),
            repository=str(data.get("repository", "")),
            model=str(data.get("model", "")),
            revision=str(data.get("revision", "")),
            file_hashes=dict(data.get("file_hashes", {})),
            license=LicenseTerms.from_dict(license_data),
            architecture=str(data.get("architecture", "")),
            pretraining_objective=str(data.get("pretraining_objective", "")),
            parameter_count=int(data.get("parameter_count", 0)),
            hidden_width=int(data.get("hidden_width", 0)),
            num_layers=int(data.get("num_layers", 0)),
            context_length=int(data.get("context_length", 0)),
            tokenizer_id=str(data.get("tokenizer_id", "")),
            conversion_method=str(data.get("conversion_method", "")),
            supported_formats=tuple(data.get("supported_formats", [])),
            estimated_train_memory_bytes=int(data.get("estimated_train_memory_bytes", 0)),
            estimated_inference_memory_bytes=int(
                data.get("estimated_inference_memory_bytes", 0)
            ),
            estimated_flops_per_forward=int(data.get("estimated_flops_per_forward", 0)),
            expected_serialized_bytes=int(data.get("expected_serialized_bytes", 0)),
            expected_deployed_bytes=int(data.get("expected_deployed_bytes", 0)),
            local_offline_available=bool(data.get("local_offline_available", False)),
            unsupported_operations=tuple(data.get("unsupported_operations", [])),
            hardware_requirements=tuple(data.get("hardware_requirements", [])),
            selection_evidence=str(data.get("selection_evidence", "")),
        )


@dataclass(frozen=True)
class ActivationGate:
    """An external issue gate that must be available before activation."""

    gate_id: str
    depends_on_issue_id: str
    required_status: str
    available: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "depends_on_issue_id": self.depends_on_issue_id,
            "required_status": self.required_status,
            "available": self.available,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActivationGate":
        return cls(
            gate_id=str(data.get("gate_id", "")),
            depends_on_issue_id=str(data.get("depends_on_issue_id", "")),
            required_status=str(data.get("required_status", "")),
            available=bool(data.get("available", False)),
            evidence=str(data.get("evidence", "")),
        )


@dataclass(frozen=True)
class BudgetCap:
    """Spending caps for the pretrained-denoiser activation decision."""

    model_acquisition_dollars: float | None = None
    gpu_hours: float | None = None
    storage_dollars: float | None = None
    conversion_dollars: float | None = None
    eval_dollars: float | None = None
    total_dollars: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_acquisition_dollars": self.model_acquisition_dollars,
            "gpu_hours": self.gpu_hours,
            "storage_dollars": self.storage_dollars,
            "conversion_dollars": self.conversion_dollars,
            "eval_dollars": self.eval_dollars,
            "total_dollars": self.total_dollars,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetCap":
        def _float(key: str) -> float | None:
            value = data.get(key)
            return None if value is None else float(value)

        return cls(
            model_acquisition_dollars=_float("model_acquisition_dollars"),
            gpu_hours=_float("gpu_hours"),
            storage_dollars=_float("storage_dollars"),
            conversion_dollars=_float("conversion_dollars"),
            eval_dollars=_float("eval_dollars"),
            total_dollars=_float("total_dollars"),
        )


@dataclass(frozen=True)
class PretrainedDenoiserArm:
    """One preregistered experimental arm for SDE4-04."""

    arm_id: str
    arm_kind: Literal[
        "current_small_controller_baseline",
        "b4_pilot_reference",
        "pretrained_denoiser_plus_adapters",
        "frozen_backbone_connector_only",
        "adapters_disabled_diagnostic",
        "equal_compute_small_controller_control",
        "random_init_short_budget_diagnostic",
    ]
    eligible: bool
    omission_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_kind": self.arm_kind,
            "eligible": self.eligible,
            "omission_reason": self.omission_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainedDenoiserArm":
        return cls(
            arm_id=str(data.get("arm_id", "")),
            arm_kind=data["arm_kind"],  # type: ignore[arg-type]
            eligible=bool(data.get("eligible", False)),
            omission_reason=data.get("omission_reason"),
        )


DEFAULT_ACTIVATION_GATES: tuple[ActivationGate, ...] = (
    ActivationGate(
        gate_id="slm161_data_contract_closed",
        depends_on_issue_id="SLM-161",
        required_status="closed",
        available=False,
        evidence="",
    ),
    ActivationGate(
        gate_id="slm24_evaluation_ready",
        depends_on_issue_id="SLM-24",
        required_status="closed",
        available=False,
        evidence="",
    ),
    ActivationGate(
        gate_id="slm175_connector_spec_closed",
        depends_on_issue_id="SLM-175",
        required_status="closed",
        available=False,
        evidence="",
    ),
    ActivationGate(
        gate_id="small_baseline_stable",
        depends_on_issue_id="SDE4-04",
        required_status="passed",
        available=False,
        evidence="",
    ),
    ActivationGate(
        gate_id="budget_approved",
        depends_on_issue_id="SDE4-04",
        required_status="approved",
        available=False,
        evidence="",
    ),
    ActivationGate(
        gate_id="license_compatible",
        depends_on_issue_id="SDE4-04",
        required_status="approved",
        available=False,
        evidence="",
    ),
)

DEFAULT_ARMS: tuple[PretrainedDenoiserArm, ...] = (
    PretrainedDenoiserArm(
        arm_id="current_small_controller_baseline",
        arm_kind="current_small_controller_baseline",
        eligible=True,
    ),
    PretrainedDenoiserArm(
        arm_id="b4_pilot_reference",
        arm_kind="b4_pilot_reference",
        eligible=True,
    ),
    PretrainedDenoiserArm(
        arm_id="pretrained_denoiser_plus_adapters",
        arm_kind="pretrained_denoiser_plus_adapters",
        eligible=True,
    ),
    PretrainedDenoiserArm(
        arm_id="frozen_backbone_connector_only",
        arm_kind="frozen_backbone_connector_only",
        eligible=True,
    ),
    PretrainedDenoiserArm(
        arm_id="adapters_disabled_diagnostic",
        arm_kind="adapters_disabled_diagnostic",
        eligible=False,
        omission_reason="zero-training diagnostic only",
    ),
    PretrainedDenoiserArm(
        arm_id="equal_compute_small_controller_control",
        arm_kind="equal_compute_small_controller_control",
        eligible=True,
    ),
    PretrainedDenoiserArm(
        arm_id="random_init_short_budget_diagnostic",
        arm_kind="random_init_short_budget_diagnostic",
        eligible=False,
        omission_reason="optional short-budget diagnostic only when financially justified",
    ),
)


def _default_candidate() -> PretrainedDenoiserCandidate:
    return PretrainedDenoiserCandidate(
        candidate_id="unset",
        provider="unset",
        repository="",
        model="unset",
        revision="main",
        file_hashes={},
        license=LicenseTerms(
            spdx_id="unset",
            commercial_use_allowed=False,
            redistribution_allowed=False,
            modification_allowed=False,
            attribution_required=False,
            notes="",
        ),
        architecture="unset",
        pretraining_objective="denoising_lm",
        parameter_count=0,
        hidden_width=0,
        num_layers=0,
        context_length=0,
        tokenizer_id="unset",
        conversion_method="unset",
        supported_formats=("safetensors",),
        estimated_train_memory_bytes=0,
        estimated_inference_memory_bytes=0,
        estimated_flops_per_forward=0,
        expected_serialized_bytes=0,
        expected_deployed_bytes=0,
        local_offline_available=False,
        unsupported_operations=(),
        hardware_requirements=(),
        selection_evidence="candidate not yet selected",
    )


def _default_budget() -> BudgetCap:
    return BudgetCap(total_dollars=0.0)


def _license_incompatible(candidate: PretrainedDenoiserCandidate) -> bool:
    """Return True when the candidate license is not usable for the project."""
    license_ = candidate.license
    return not (
        license_.commercial_use_allowed
        and license_.redistribution_allowed
        and license_.modification_allowed
    )


def _activation_status(verdict: str) -> str:
    return {
        "unrun": "unrun",
        "ready_to_spend": "ready",
        "no_eligible_candidate": "closed",
        "activation_blocked": "blocked",
        "license_incompatible": "blocked",
        "budget_or_yield_blocked": "blocked",
    }.get(verdict, "blocked")


@dataclass(frozen=True)
class PretrainedDenoiserActivationManifest:
    """Frozen activation/candidate/budget manifest for SDE4-04."""

    manifest_id: str
    schema_version: str
    hypothesis_id: str
    activation_status: str
    activation_verdict: str
    campaign_verdict: str
    activation_gates: tuple[ActivationGate, ...]
    candidate: PretrainedDenoiserCandidate
    budget: BudgetCap
    arms: tuple[PretrainedDenoiserArm, ...]
    primary_metric: str
    seeds: tuple[int, ...]
    max_deployed_bytes: int
    manifest_hash: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "activation_status": self.activation_status,
            "activation_verdict": self.activation_verdict,
            "campaign_verdict": self.campaign_verdict,
            "activation_gates": [g.to_dict() for g in self.activation_gates],
            "candidate": self.candidate.to_dict(),
            "budget": self.budget.to_dict(),
            "arms": [a.to_dict() for a in self.arms],
            "primary_metric": self.primary_metric,
            "seeds": list(self.seeds),
            "max_deployed_bytes": self.max_deployed_bytes,
            "manifest_hash": self.manifest_hash,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PretrainedDenoiserActivationManifest":
        return cls(
            manifest_id=str(data.get("manifest_id", "")),
            schema_version=str(data.get("schema_version", "")),
            hypothesis_id=str(data.get("hypothesis_id", HYPOTHESIS_ID)),
            activation_status=str(data.get("activation_status", "")),
            activation_verdict=str(data.get("activation_verdict", "")),
            campaign_verdict=str(data.get("campaign_verdict", "unrun")),
            activation_gates=tuple(
                ActivationGate.from_dict(g) for g in data.get("activation_gates", [])
            ),
            candidate=PretrainedDenoiserCandidate.from_dict(data.get("candidate", {})),
            budget=BudgetCap.from_dict(data.get("budget", {})),
            arms=tuple(PretrainedDenoiserArm.from_dict(a) for a in data.get("arms", [])),
            primary_metric=str(data.get("primary_metric", "")),
            seeds=tuple(int(s) for s in data.get("seeds", [])),
            max_deployed_bytes=int(data.get("max_deployed_bytes", 0)),
            manifest_hash=str(data.get("manifest_hash", "")),
            note=str(data.get("note", "")),
        )


def build_pretrained_denoiser_activation_manifest(
    candidate: PretrainedDenoiserCandidate | None = None,
    budget: BudgetCap | None = None,
    arms: Iterable[PretrainedDenoiserArm] | None = None,
    *,
    manifest_id: str = "pretrained_denoiser_activation/v1",
    candidate_selection: Literal[
        "unknown", "candidate_selected", "no_candidate_meets_constraints"
    ] = "unknown",
    activation_gates: Iterable[ActivationGate] | None = None,
    primary_metric: str = "binding_aware_meaningful_program_rate",
    seeds: Iterable[int] = (0, 1, 2),
    max_deployed_bytes: int = 1_000_000_000,
    note: str = "",
) -> PretrainedDenoiserActivationManifest:
    """Build a deterministic SDE4-04 activation manifest.

    The manifest is plan-only: it records the candidate, gates, budget, and arms
    without launching training or evaluation.
    """
    candidate = candidate if candidate is not None else _default_candidate()
    budget = budget if budget is not None else _default_budget()
    arm_list = tuple(arms if arms is not None else DEFAULT_ARMS)
    gate_list = tuple(activation_gates if activation_gates is not None else DEFAULT_ACTIVATION_GATES)

    if candidate_selection == "no_candidate_meets_constraints":
        activation_verdict: str = "no_eligible_candidate"
    elif any(not gate.available for gate in gate_list):
        activation_verdict = "activation_blocked"
    elif _license_incompatible(candidate):
        activation_verdict = "license_incompatible"
    elif budget.total_dollars is None or budget.total_dollars == 0:
        activation_verdict = "budget_or_yield_blocked"
    else:
        activation_verdict = "ready_to_spend"

    activation_status = _activation_status(activation_verdict)
    campaign_verdict = "unrun"

    hash_payload = {
        "manifest_id": manifest_id,
        "schema_version": MANIFEST_SCHEMA,
        "hypothesis_id": HYPOTHESIS_ID,
        "activation_status": activation_status,
        "activation_verdict": activation_verdict,
        "campaign_verdict": campaign_verdict,
        "activation_gates": [g.to_dict() for g in gate_list],
        "candidate": candidate.to_dict(),
        "budget": budget.to_dict(),
        "arms": [a.to_dict() for a in arm_list],
        "primary_metric": primary_metric,
        "seeds": list(seeds),
        "max_deployed_bytes": max_deployed_bytes,
        "note": note,
    }
    manifest_hash = _stable_hash(hash_payload)

    return PretrainedDenoiserActivationManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA,
        hypothesis_id=HYPOTHESIS_ID,
        activation_status=activation_status,
        activation_verdict=activation_verdict,
        campaign_verdict=campaign_verdict,
        activation_gates=gate_list,
        candidate=candidate,
        budget=budget,
        arms=arm_list,
        primary_metric=primary_metric,
        seeds=tuple(seeds),
        max_deployed_bytes=max_deployed_bytes,
        manifest_hash=manifest_hash,
        note=note,
    )


def validate_pretrained_denoiser_activation_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return validation errors; empty means valid."""
    errors: list[str] = []

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")

    for key in (
        "manifest_id",
        "hypothesis_id",
        "activation_status",
        "activation_verdict",
        "campaign_verdict",
        "primary_metric",
        "manifest_hash",
    ):
        if not isinstance(manifest.get(key), str) or not manifest.get(key):
            errors.append(f"{key} must be a non-empty string")

    if manifest.get("hypothesis_id") != HYPOTHESIS_ID:
        errors.append(f"hypothesis_id must be {HYPOTHESIS_ID!r}")

    if manifest.get("activation_verdict") not in ACTIVATION_VERDICTS:
        errors.append(
            f"activation_verdict must be one of {sorted(ACTIVATION_VERDICTS)}"
        )
    if manifest.get("campaign_verdict") not in CAMPAIGN_VERDICTS:
        errors.append(
            f"campaign_verdict must be one of {sorted(CAMPAIGN_VERDICTS)}"
        )

    if not isinstance(manifest.get("max_deployed_bytes"), int):
        errors.append("max_deployed_bytes must be an integer")

    if not isinstance(manifest.get("seeds"), list) or not manifest.get("seeds"):
        errors.append("seeds must be a non-empty list")

    gates = manifest.get("activation_gates")
    if not isinstance(gates, list) or not gates:
        errors.append("activation_gates must be a non-empty list")
    else:
        for idx, gate in enumerate(gates):
            prefix = f"activation_gates[{idx}]"
            if not isinstance(gate, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for key in ("gate_id", "depends_on_issue_id", "required_status"):
                if not isinstance(gate.get(key), str) or not gate.get(key):
                    errors.append(f"{prefix}.{key} must be a non-empty string")
            if "available" not in gate:
                errors.append(f"{prefix}.available is required")

    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict):
        errors.append("candidate must be an object")
    else:
        for key in (
            "candidate_id",
            "provider",
            "model",
            "revision",
            "architecture",
            "pretraining_objective",
            "tokenizer_id",
            "conversion_method",
            "selection_evidence",
        ):
            if not isinstance(candidate.get(key), str) or not candidate.get(key):
                errors.append(f"candidate.{key} must be a non-empty string")
        for key in (
            "parameter_count",
            "hidden_width",
            "num_layers",
            "context_length",
            "estimated_train_memory_bytes",
            "estimated_inference_memory_bytes",
            "estimated_flops_per_forward",
            "expected_serialized_bytes",
            "expected_deployed_bytes",
        ):
            if not isinstance(candidate.get(key), int):
                errors.append(f"candidate.{key} must be an integer")
        if not isinstance(candidate.get("supported_formats"), list) or not candidate.get(
            "supported_formats"
        ):
            errors.append("candidate.supported_formats must be a non-empty list")
        if not isinstance(candidate.get("file_hashes"), dict):
            errors.append("candidate.file_hashes must be a dict")
        if not isinstance(candidate.get("local_offline_available"), bool):
            errors.append("candidate.local_offline_available must be a boolean")
        license_data = candidate.get("license")
        if not isinstance(license_data, dict):
            errors.append("candidate.license must be an object")
        else:
            for key in ("spdx_id",):
                if not isinstance(license_data.get(key), str) or not license_data.get(key):
                    errors.append(f"candidate.license.{key} must be a non-empty string")
            for key in (
                "commercial_use_allowed",
                "redistribution_allowed",
                "modification_allowed",
                "attribution_required",
            ):
                if not isinstance(license_data.get(key), bool):
                    errors.append(f"candidate.license.{key} must be a boolean")

    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        budget_values = [
            budget.get("model_acquisition_dollars"),
            budget.get("gpu_hours"),
            budget.get("storage_dollars"),
            budget.get("conversion_dollars"),
            budget.get("eval_dollars"),
            budget.get("total_dollars"),
        ]
        if not any(v is not None for v in budget_values):
            errors.append("budget must have at least one cap set")
        for key in (
            "model_acquisition_dollars",
            "gpu_hours",
            "storage_dollars",
            "conversion_dollars",
            "eval_dollars",
            "total_dollars",
        ):
            value = budget.get(key)
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"budget.{key} must be a number when set")

    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms must be a non-empty list")
    else:
        for idx, arm in enumerate(arms):
            prefix = f"arms[{idx}]"
            if not isinstance(arm, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for key in ("arm_id", "arm_kind"):
                if not isinstance(arm.get(key), str) or not arm.get(key):
                    errors.append(f"{prefix}.{key} must be a non-empty string")
            if arm.get("arm_kind") not in ARM_KINDS:
                errors.append(
                    f"{prefix}.arm_kind must be one of {sorted(ARM_KINDS)}"
                )
            if "eligible" not in arm:
                errors.append(f"{prefix}.eligible is required")
            elif arm.get("eligible") is False and not arm.get("omission_reason"):
                errors.append(f"{prefix} omitted arm must have omission_reason")

    return errors
