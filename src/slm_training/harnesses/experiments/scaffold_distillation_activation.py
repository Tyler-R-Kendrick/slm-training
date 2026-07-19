"""SDE4-01 (SLM-179) scaffold-distillation activation/budget/manifest contract.

This module defines a frozen, versioned activation manifest for the scaffold-
distillation campaign. It is intentionally plan-only: it records the activation
gates, trace contract, budget cap, and preregistered arms without launching
teacher trace collection, student training, or eval runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

MANIFEST_SCHEMA = "scaffold_distillation_activation/v1"
HYPOTHESIS_ID = "H13"

ScaffoldDistillationArmKind = Literal[
    "scaffolded_teacher_selected",
    "teacher_first_attempt_only",
    "lever_off_gold_sft",
    "selected_trajectory_distillation",
    "sft_plus_legal_set_kl",
    "sft_kl_plus_preference",
    "permuted_teacher_specificity_control",
    "impossible_information_inventory_control",
]

ScaffoldDecomposition = Literal[
    "unknown",
    "value_demonstrated",
    "no_value",
    "inventory_required",
]

ACTIVATION_VERDICTS = frozenset(
    {
        "activation_blocked",
        "ready_to_spend",
        "no_scaffold_value",
        "inventory_information_blocked",
        "budget_or_yield_blocked",
        "unrun",
    }
)
CAMPAIGN_VERDICTS = frozenset(
    {
        "distill_scaffolding",
        "distill_only_content_or_retry",
        "public_contract_required",
        "teacher_signal_not_learnable",
        "no_distillation_needed",
        "inconclusive",
        "unrun",
    }
)

_ARM_KINDS: frozenset[str] = frozenset(
    {
        "scaffolded_teacher_selected",
        "teacher_first_attempt_only",
        "lever_off_gold_sft",
        "selected_trajectory_distillation",
        "sft_plus_legal_set_kl",
        "sft_kl_plus_preference",
        "permuted_teacher_specificity_control",
        "impossible_information_inventory_control",
    }
)
_OBJECTIVES: frozenset[str] = frozenset(
    {"sft", "kl", "preference", "scaffold_intervention", "selector_aux"}
)


def _stable_hash(parts: Mapping[str, Any]) -> str:
    """Return a stable 16-hex hash of a JSON-sortable mapping."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ActivationGate:
    """A prerequisite gate that must be available before campaign activation."""

    gate_id: str
    depends_on_issue_id: str | None = None
    required_status: str | None = None
    available: bool = False
    evidence: str | None = None

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
            gate_id=str(data["gate_id"]),
            depends_on_issue_id=data.get("depends_on_issue_id"),
            required_status=data.get("required_status"),
            available=bool(data.get("available", False)),
            evidence=data.get("evidence"),
        )


@dataclass(frozen=True)
class BudgetCap:
    """Budget caps for the scaffold-distillation campaign.

    At least one cap must be set (non-``None``). A cap of ``0`` is valid but
    signals a budget/yield block.
    """

    teacher_trace_compute_dollars: float | None = None
    student_training_dollars: float | None = None
    student_training_gpu_hours: float | None = None
    eval_dollars: float | None = None
    total_dollars: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_trace_compute_dollars": self.teacher_trace_compute_dollars,
            "student_training_dollars": self.student_training_dollars,
            "student_training_gpu_hours": self.student_training_gpu_hours,
            "eval_dollars": self.eval_dollars,
            "total_dollars": self.total_dollars,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetCap":
        def _float_or_none(value: Any) -> float | None:
            if value is None:
                return None
            return float(value)

        return cls(
            teacher_trace_compute_dollars=_float_or_none(
                data.get("teacher_trace_compute_dollars")
            ),
            student_training_dollars=_float_or_none(data.get("student_training_dollars")),
            student_training_gpu_hours=_float_or_none(
                data.get("student_training_gpu_hours")
            ),
            eval_dollars=_float_or_none(data.get("eval_dollars")),
            total_dollars=_float_or_none(data.get("total_dollars")),
        )


@dataclass(frozen=True)
class TeacherTraceContract:
    """Plan-only contract describing the teacher traces the campaign may consume."""

    teacher_checkpoint_id: str
    teacher_run_id: str
    trace_store_uri: str
    trace_schema_version: str
    min_traces: int
    max_traces: int
    scaffold_config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_checkpoint_id": self.teacher_checkpoint_id,
            "teacher_run_id": self.teacher_run_id,
            "trace_store_uri": self.trace_store_uri,
            "trace_schema_version": self.trace_schema_version,
            "min_traces": self.min_traces,
            "max_traces": self.max_traces,
            "scaffold_config_hash": self.scaffold_config_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TeacherTraceContract":
        return cls(
            teacher_checkpoint_id=str(data["teacher_checkpoint_id"]),
            teacher_run_id=str(data["teacher_run_id"]),
            trace_store_uri=str(data["trace_store_uri"]),
            trace_schema_version=str(data.get("trace_schema_version", "v1")),
            min_traces=int(data["min_traces"]),
            max_traces=int(data["max_traces"]),
            scaffold_config_hash=str(data["scaffold_config_hash"]),
        )


@dataclass(frozen=True)
class ScaffoldDistillationArm:
    """One preregistered experimental arm in the scaffold-distillation campaign."""

    arm_id: str
    arm_kind: ScaffoldDistillationArmKind
    eligible: bool
    omission_reason: str | None = None
    objectives: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "arm_kind": self.arm_kind,
            "eligible": self.eligible,
            "omission_reason": self.omission_reason,
            "objectives": list(self.objectives),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScaffoldDistillationArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_kind=data["arm_kind"],  # type: ignore[arg-type]
            eligible=bool(data["eligible"]),
            omission_reason=data.get("omission_reason"),
            objectives=tuple(data.get("objectives", [])),
        )


@dataclass(frozen=True)
class ScaffoldDistillationActivationManifest:
    """Frozen preregistered activation manifest for SDE4-01.

    The manifest must be committed before expensive execution. Any change
    requires a new manifest hash and explanation.
    """

    manifest_id: str
    schema_version: str
    hypothesis_id: str
    activation_status: str
    activation_verdict: str
    campaign_verdict: str
    activation_gates: tuple[ActivationGate, ...]
    trace_contract: TeacherTraceContract
    budget: BudgetCap
    arms: tuple[ScaffoldDistillationArm, ...]
    primary_metric: str
    seeds: tuple[int, ...]
    max_attempts_for_teacher: int
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
            "trace_contract": self.trace_contract.to_dict(),
            "budget": self.budget.to_dict(),
            "arms": [a.to_dict() for a in self.arms],
            "primary_metric": self.primary_metric,
            "seeds": list(self.seeds),
            "max_attempts_for_teacher": self.max_attempts_for_teacher,
            "manifest_hash": self.manifest_hash,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScaffoldDistillationActivationManifest":
        return cls(
            manifest_id=str(data["manifest_id"]),
            schema_version=str(data["schema_version"]),
            hypothesis_id=str(data["hypothesis_id"]),
            activation_status=str(data["activation_status"]),
            activation_verdict=str(data["activation_verdict"]),
            campaign_verdict=str(data["campaign_verdict"]),
            activation_gates=tuple(
                ActivationGate.from_dict(g) for g in data["activation_gates"]
            ),
            trace_contract=TeacherTraceContract.from_dict(data["trace_contract"]),
            budget=BudgetCap.from_dict(data["budget"]),
            arms=tuple(ScaffoldDistillationArm.from_dict(a) for a in data["arms"]),
            primary_metric=str(data["primary_metric"]),
            seeds=tuple(int(s) for s in data["seeds"]),
            max_attempts_for_teacher=int(data["max_attempts_for_teacher"]),
            manifest_hash=str(data["manifest_hash"]),
            note=str(data.get("note", "")),
        )


DEFAULT_ACTIVATION_GATES: tuple[ActivationGate, ...] = (
    ActivationGate(
        gate_id="slm161_machine_readable_decomposition",
        depends_on_issue_id="SLM-161",
        required_status="Done",
        available=False,
        evidence="Machine-readable decomposition of the scaffold into levered operations.",
    ),
    ActivationGate(
        gate_id="slm162_metric_gaming_suite",
        depends_on_issue_id="SLM-162",
        required_status="Done",
        available=False,
        evidence="Metric-gaming stress suite demonstrates the scaffold is not gamed.",
    ),
    ActivationGate(
        gate_id="slm168_public_structured_contract_pointer",
        depends_on_issue_id="SLM-168",
        required_status="Done",
        available=False,
        evidence="Public structured contract pointer exposes the inventory gap.",
    ),
    ActivationGate(
        gate_id="scaffold_value_demonstrated",
        depends_on_issue_id="SLM-161",
        required_status="Done",
        available=False,
        evidence="The scaffold improves over no-scaffold baselines on a held slice.",
    ),
    ActivationGate(
        gate_id="latency_or_complexity_worth_amortizing",
        available=False,
        evidence="Teacher scaffolding cost is worth amortizing over student training.",
    ),
    ActivationGate(
        gate_id="budget_approved",
        available=False,
        evidence="Campaign budget approved for teacher traces and student runs.",
    ),
)

DEFAULT_ARMS: tuple[ScaffoldDistillationArm, ...] = (
    ScaffoldDistillationArm(
        arm_id="scaffolded_teacher_selected",
        arm_kind="scaffolded_teacher_selected",
        eligible=True,
    ),
    ScaffoldDistillationArm(
        arm_id="teacher_first_attempt_only",
        arm_kind="teacher_first_attempt_only",
        eligible=True,
    ),
    ScaffoldDistillationArm(
        arm_id="lever_off_gold_sft",
        arm_kind="lever_off_gold_sft",
        eligible=True,
    ),
    ScaffoldDistillationArm(
        arm_id="selected_trajectory_distillation",
        arm_kind="selected_trajectory_distillation",
        eligible=True,
        objectives=("sft",),
    ),
    ScaffoldDistillationArm(
        arm_id="sft_plus_legal_set_kl",
        arm_kind="sft_plus_legal_set_kl",
        eligible=True,
        objectives=("sft", "kl"),
    ),
    ScaffoldDistillationArm(
        arm_id="sft_kl_plus_preference",
        arm_kind="sft_kl_plus_preference",
        eligible=True,
        objectives=("sft", "kl", "preference"),
    ),
    ScaffoldDistillationArm(
        arm_id="permuted_teacher_specificity_control",
        arm_kind="permuted_teacher_specificity_control",
        eligible=True,
        objectives=("sft",),
    ),
    ScaffoldDistillationArm(
        arm_id="impossible_information_inventory_control",
        arm_kind="impossible_information_inventory_control",
        eligible=False,
        omission_reason=(
            "non-promotable control only; quantifies the information gap when the "
            "student cannot access the teacher inventory"
        ),
    ),
)


def _budget_all_zero_or_none(budget: BudgetCap) -> bool:
    values = [
        budget.teacher_trace_compute_dollars,
        budget.student_training_dollars,
        budget.student_training_gpu_hours,
        budget.eval_dollars,
        budget.total_dollars,
    ]
    return all((v is None) or (v == 0) for v in values)


def _gate_available(
    gates: Iterable[ActivationGate], gate_id: str
) -> bool:
    for gate in gates:
        if gate.gate_id == gate_id:
            return gate.available
    return False


def build_scaffold_distillation_activation_manifest(
    *,
    manifest_id: str,
    teacher_trace_contract: TeacherTraceContract,
    budget: BudgetCap,
    arms: Iterable[ScaffoldDistillationArm],
    activation_gates: Iterable[ActivationGate] | None = None,
    scaffold_decomposition: ScaffoldDecomposition = "unknown",
    primary_metric: str = "binding_aware_meaningful_program_rate",
    seeds: Iterable[int] = (0, 1, 2),
    max_attempts_for_teacher: int = 1,
    campaign_verdict: str = "unrun",
    note: str = "",
) -> ScaffoldDistillationActivationManifest:
    """Build a deterministic SDE4-01 activation manifest.

    ``scaffold_decomposition`` drives the activation verdict:
    - ``no_value`` closes the campaign.
    - ``value_demonstrated`` plus available gates and non-zero budget opens it.
    - ``inventory_required`` without the SLM-168 contract pointer blocks on the
      inventory information gap.
    """
    gate_list = tuple(activation_gates) if activation_gates is not None else DEFAULT_ACTIVATION_GATES
    arm_list = tuple(arms)

    if scaffold_decomposition == "no_value":
        activation_status = "closed"
        activation_verdict = "no_scaffold_value"
    elif any(not gate.available for gate in gate_list):
        activation_status = "blocked"
        activation_verdict = "activation_blocked"
    elif scaffold_decomposition not in ("value_demonstrated", "inventory_required"):
        activation_status = "blocked"
        activation_verdict = "activation_blocked"
    elif _budget_all_zero_or_none(budget):
        activation_status = "blocked"
        activation_verdict = "budget_or_yield_blocked"
    elif (
        scaffold_decomposition == "inventory_required"
        and not _gate_available(gate_list, "slm168_public_structured_contract_pointer")
    ):
        activation_status = "blocked"
        activation_verdict = "inventory_information_blocked"
    else:
        activation_status = "ready"
        activation_verdict = "ready_to_spend"

    hash_input = {
        "manifest_id": manifest_id,
        "hypothesis_id": HYPOTHESIS_ID,
        "activation_gates": [g.to_dict() for g in gate_list],
        "trace_contract": teacher_trace_contract.to_dict(),
        "budget": budget.to_dict(),
        "arms": [a.to_dict() for a in arm_list],
        "primary_metric": primary_metric,
        "seeds": list(seeds),
        "max_attempts_for_teacher": max_attempts_for_teacher,
        "note": note,
    }
    manifest_hash = _stable_hash(hash_input)

    return ScaffoldDistillationActivationManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA,
        hypothesis_id=HYPOTHESIS_ID,
        activation_status=activation_status,
        activation_verdict=activation_verdict,
        campaign_verdict=campaign_verdict,
        activation_gates=gate_list,
        trace_contract=teacher_trace_contract,
        budget=budget,
        arms=arm_list,
        primary_metric=primary_metric,
        seeds=tuple(seeds),
        max_attempts_for_teacher=max_attempts_for_teacher,
        manifest_hash=manifest_hash,
        note=note,
    )


def validate_scaffold_distillation_activation_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return validation errors; empty means valid."""
    errors: list[str] = []

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")

    for key in ("manifest_id", "hypothesis_id", "activation_status", "primary_metric"):
        if not isinstance(manifest.get(key), str) or not manifest.get(key):
            errors.append(f"{key} must be a non-empty string")

    activation_verdict = manifest.get("activation_verdict")
    if activation_verdict not in ACTIVATION_VERDICTS:
        errors.append(
            f"activation_verdict {activation_verdict!r} not in {sorted(ACTIVATION_VERDICTS)}"
        )

    campaign_verdict = manifest.get("campaign_verdict")
    if campaign_verdict not in CAMPAIGN_VERDICTS:
        errors.append(
            f"campaign_verdict {campaign_verdict!r} not in {sorted(CAMPAIGN_VERDICTS)}"
        )

    gates = manifest.get("activation_gates")
    if not isinstance(gates, list) or not gates:
        errors.append("activation_gates must be a non-empty list")
    else:
        for idx, gate in enumerate(gates):
            prefix = f"activation_gates[{idx}]"
            if not isinstance(gate, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not isinstance(gate.get("gate_id"), str) or not gate.get("gate_id"):
                errors.append(f"{prefix} missing or empty gate_id")
            if gate.get("depends_on_issue_id") and not gate.get("required_status"):
                errors.append(f"{prefix} depends_on_issue_id requires required_status")

    trace_contract = manifest.get("trace_contract")
    if not isinstance(trace_contract, dict):
        errors.append("trace_contract must be an object")
    else:
        for key in (
            "teacher_checkpoint_id",
            "teacher_run_id",
            "trace_store_uri",
            "scaffold_config_hash",
        ):
            if not isinstance(trace_contract.get(key), str) or not trace_contract.get(key):
                errors.append(f"trace_contract missing or empty {key!r}")
        for key in ("min_traces", "max_traces"):
            if not isinstance(trace_contract.get(key), int):
                errors.append(f"trace_contract {key!r} must be an integer")

    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        caps = [
            budget.get("teacher_trace_compute_dollars"),
            budget.get("student_training_dollars"),
            budget.get("student_training_gpu_hours"),
            budget.get("eval_dollars"),
            budget.get("total_dollars"),
        ]
        if all(c is None for c in caps):
            errors.append("budget must have at least one cap set")

    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms must be a non-empty list")
        return errors

    for idx, arm in enumerate(arms):
        prefix = f"arms[{idx}]"
        if not isinstance(arm, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if not isinstance(arm.get("arm_id"), str) or not arm.get("arm_id"):
            errors.append(f"{prefix} missing or empty arm_id")
        arm_kind = arm.get("arm_kind")
        if arm_kind not in _ARM_KINDS:
            errors.append(
                f"{prefix} arm_kind {arm_kind!r} not in {sorted(_ARM_KINDS)}"
            )
        objectives = arm.get("objectives", [])
        if not isinstance(objectives, (list, tuple)):
            errors.append(f"{prefix} objectives must be a list")
        else:
            for obj in objectives:
                if obj not in _OBJECTIVES:
                    errors.append(
                        f"{prefix} objective {obj!r} not in {sorted(_OBJECTIVES)}"
                    )
        if arm.get("eligible") is False and not arm.get("omission_reason"):
            errors.append(f"{prefix} omitted arm must have omission_reason")

    return errors
