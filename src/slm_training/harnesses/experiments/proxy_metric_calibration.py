"""SDE3-03 (SLM-177) proxy-metric calibration activation/budget manifest.

Plan-only wiring slice: a frozen, versioned manifest that must exist before any
cheap proxy is used to triage experiment rows. It records the preregistered
feature contract, activation gates, budget caps, and calibration arms without
training a proxy, running evaluation, or changing promotion authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

MANIFEST_SCHEMA = "proxy_metric_calibration/v1"
HYPOTHESIS_ID = "H23"

ACTIVATION_VERDICTS = frozenset(
    {
        "activation_blocked",
        "ready_to_spend",
        "feature_contract_unsafe",
        "budget_or_yield_blocked",
        "unrun",
    }
)

CAMPAIGN_VERDICTS = frozenset(
    {
        "proxy_safe_for_triage",
        "proxy_diagnostic_only",
        "proxy_rejected",
        "inconclusive",
        "unrun",
    }
)

ARM_KINDS = frozenset(
    {
        "rule_baseline",
        "regularized_linear",
        "bounded_tree",
        "shadow_only",
    }
)

PROXY_EVAL_MODES = frozenset({"off", "shadow", "triage"})

ProxyEvalMode = Literal["off", "shadow", "triage"]


def _stable_hash(parts: Mapping[str, Any]) -> str:
    """Return a stable 16-hex hash of a JSON-sortable mapping."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ActivationGate:
    """A prerequisite gate that must be available before proxy calibration."""

    gate_id: str
    depends_on_issue_id: str | None = None
    required_status: str | None = None
    available: bool = False
    evidence: str = ""

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
            depends_on_issue_id=data.get("depends_on_issue_id"),
            required_status=data.get("required_status"),
            available=bool(data.get("available", False)),
            evidence=str(data.get("evidence", "")),
        )


@dataclass(frozen=True)
class BudgetCap:
    """Spending caps for the proxy calibration campaign."""

    max_historical_rows: int | None = None
    max_dollars: float | None = None
    gpu_hours: float | None = None
    eval_dollars: float | None = None
    total_dollars: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_historical_rows": self.max_historical_rows,
            "max_dollars": self.max_dollars,
            "gpu_hours": self.gpu_hours,
            "eval_dollars": self.eval_dollars,
            "total_dollars": self.total_dollars,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetCap":
        def _num(key: str) -> float | int | None:
            return data.get(key)

        return cls(
            max_historical_rows=_num("max_historical_rows"),
            max_dollars=_num("max_dollars"),
            gpu_hours=_num("gpu_hours"),
            eval_dollars=_num("eval_dollars"),
            total_dollars=_num("total_dollars"),
        )


@dataclass(frozen=True)
class ProxyFeatureSet:
    """Preregistered cheap-feature contract and forbidden-feature list."""

    feature_schema_version: str
    feature_names: tuple[str, ...]
    target_primary: str
    target_gate: str
    allowed_sources: tuple[str, ...]
    forbidden_features: tuple[str, ...]
    requires_lineage_disjoint_split: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
            "target_primary": self.target_primary,
            "target_gate": self.target_gate,
            "allowed_sources": list(self.allowed_sources),
            "forbidden_features": list(self.forbidden_features),
            "requires_lineage_disjoint_split": self.requires_lineage_disjoint_split,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProxyFeatureSet":
        return cls(
            feature_schema_version=str(data.get("feature_schema_version", "")),
            feature_names=tuple(str(f) for f in data.get("feature_names", [])),
            target_primary=str(data.get("target_primary", "")),
            target_gate=str(data.get("target_gate", "")),
            allowed_sources=tuple(str(s) for s in data.get("allowed_sources", [])),
            forbidden_features=tuple(str(f) for f in data.get("forbidden_features", [])),
            requires_lineage_disjoint_split=bool(
                data.get("requires_lineage_disjoint_split", True)
            ),
        )


@dataclass(frozen=True)
class CalibrationArm:
    """One preregistered proxy model/policy arm."""

    arm_id: str
    arm_kind: Literal[
        "rule_baseline",
        "regularized_linear",
        "bounded_tree",
        "shadow_only",
    ]
    eligible: bool
    omission_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "arm_id": self.arm_id,
            "arm_kind": self.arm_kind,
            "eligible": self.eligible,
        }
        if self.omission_reason is not None:
            data["omission_reason"] = self.omission_reason
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationArm":
        return cls(
            arm_id=str(data.get("arm_id", "")),
            arm_kind=data["arm_kind"],  # type: ignore[arg-type]
            eligible=bool(data.get("eligible", False)),
            omission_reason=data.get("omission_reason"),
        )


@dataclass(frozen=True)
class ProxyMetricCalibrationManifest:
    """Frozen activation/budget/feature contract for the H23 proxy experiment."""

    manifest_id: str
    schema_version: str
    hypothesis_id: str
    activation_status: str
    activation_verdict: str
    campaign_verdict: str
    activation_gates: tuple[ActivationGate, ...]
    feature_set: ProxyFeatureSet
    budget: BudgetCap
    arms: tuple[CalibrationArm, ...]
    primary_metric: str
    seeds: tuple[int, ...]
    conservative_floor: float
    risk_budget: float
    proxy_eval_mode: ProxyEvalMode
    audit_rate: float
    force_full_every_n: int
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
            "feature_set": self.feature_set.to_dict(),
            "budget": self.budget.to_dict(),
            "arms": [a.to_dict() for a in self.arms],
            "primary_metric": self.primary_metric,
            "seeds": list(self.seeds),
            "conservative_floor": self.conservative_floor,
            "risk_budget": self.risk_budget,
            "proxy_eval_mode": self.proxy_eval_mode,
            "audit_rate": self.audit_rate,
            "force_full_every_n": self.force_full_every_n,
            "manifest_hash": self.manifest_hash,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProxyMetricCalibrationManifest":
        return cls(
            manifest_id=str(data.get("manifest_id", "")),
            schema_version=str(data.get("schema_version", MANIFEST_SCHEMA)),
            hypothesis_id=str(data.get("hypothesis_id", HYPOTHESIS_ID)),
            activation_status=str(data.get("activation_status", "")),
            activation_verdict=str(data.get("activation_verdict", "")),
            campaign_verdict=str(data.get("campaign_verdict", "unrun")),
            activation_gates=tuple(
                ActivationGate.from_dict(g) for g in data.get("activation_gates", [])
            ),
            feature_set=ProxyFeatureSet.from_dict(data.get("feature_set", {})),
            budget=BudgetCap.from_dict(data.get("budget", {})),
            arms=tuple(CalibrationArm.from_dict(a) for a in data.get("arms", [])),
            primary_metric=str(data.get("primary_metric", "")),
            seeds=tuple(int(s) for s in data.get("seeds", [])),
            conservative_floor=float(data.get("conservative_floor", 0.0)),
            risk_budget=float(data.get("risk_budget", 0.0)),
            proxy_eval_mode=data.get("proxy_eval_mode", "off"),  # type: ignore[arg-type]
            audit_rate=float(data.get("audit_rate", 0.0)),
            force_full_every_n=int(data.get("force_full_every_n", 0)),
            manifest_hash=str(data.get("manifest_hash", "")),
            note=str(data.get("note", "")),
        )


DEFAULT_ACTIVATION_GATES: tuple[ActivationGate, ...] = (
    ActivationGate(
        gate_id="slm105_binding_aware_metrics",
        depends_on_issue_id="SLM-105",
        required_status="Done",
        available=False,
        evidence="Binding-aware deterministic metrics stable and versioned.",
    ),
    ActivationGate(
        gate_id="slm169_canonical_ast_binding",
        depends_on_issue_id="SLM-169",
        required_status="Done",
        available=False,
        evidence="Canonical AST, codec round-trip, binding integrity gates.",
    ),
    ActivationGate(
        gate_id="slm175_eval_cache",
        depends_on_issue_id="SLM-175",
        required_status="Done",
        available=False,
        evidence="Content-addressed evaluation cache artifacts available.",
    ),
    ActivationGate(
        gate_id="proxy_feature_contract_reviewed",
        depends_on_issue_id="SLM-177",
        required_status="Done",
        available=False,
        evidence="Feature contract reviewed: no forbidden features included.",
    ),
    ActivationGate(
        gate_id="budget_approved",
        depends_on_issue_id="SLM-177",
        required_status="approved",
        available=False,
        evidence="Calibration budget approved.",
    ),
)

DEFAULT_ARMS: tuple[CalibrationArm, ...] = (
    CalibrationArm(
        arm_id="rule_baseline",
        arm_kind="rule_baseline",
        eligible=True,
    ),
    CalibrationArm(
        arm_id="regularized_linear",
        arm_kind="regularized_linear",
        eligible=True,
    ),
    CalibrationArm(
        arm_id="bounded_tree",
        arm_kind="bounded_tree",
        eligible=False,
        omission_reason="reserved for ablation if linear models are insufficient",
    ),
    CalibrationArm(
        arm_id="shadow_only",
        arm_kind="shadow_only",
        eligible=True,
    ),
)


def _default_feature_set() -> ProxyFeatureSet:
    return ProxyFeatureSet(
        feature_schema_version="proxy_features/v1",
        feature_names=(
            "parser_valid",
            "schema_valid",
            "binding_aware_meaningful_rate",
            "component_recall",
            "role_recall",
            "minimality_flag",
            "empty_output_flag",
            "first_attempt_action_count",
            "legal_action_margin",
            "entropy",
            "termination_confidence",
            "ast_node_count",
            "binding_graph_edges",
            "latency_ms",
            "output_length",
            "tree_depth",
            "component_count",
        ),
        target_primary="binding_aware_meaningful_program_rate",
        target_gate="full_gate_pass",
        allowed_sources=(
            "parser",
            "binding_aware_metric",
            "semantic_contract",
            "first_attempt_stats",
            "legal_action_stats",
            "canonical_ast",
            "timing",
            "suite_metadata",
        ),
        forbidden_features=(
            "agentv_score",
            "external_judge_score",
            "full_gate_result",
            "gold_action_trace",
            "checkpoint_id",
            "experiment_name",
            "source_commit",
        ),
    )


def _default_budget() -> BudgetCap:
    return BudgetCap(max_historical_rows=0)


def _budget_is_empty(budget: BudgetCap) -> bool:
    return all(
        cap in (None, 0, 0.0)
        for cap in (
            budget.max_historical_rows,
            budget.max_dollars,
            budget.gpu_hours,
            budget.eval_dollars,
            budget.total_dollars,
        )
    )


def _activation_status(verdict: str) -> str:
    return {
        "unrun": "unrun",
        "ready_to_spend": "ready",
        "feature_contract_unsafe": "blocked",
        "activation_blocked": "blocked",
        "budget_or_yield_blocked": "blocked",
    }.get(verdict, "blocked")


def _feature_contract_safe(feature_set: ProxyFeatureSet) -> bool:
    """Return True when no forbidden feature name appears in the contract."""
    lower_features = {f.lower() for f in feature_set.feature_names}
    lower_allowed = {s.lower() for s in feature_set.allowed_sources}
    forbidden_present = lower_features & {f.lower() for f in feature_set.forbidden_features}
    if forbidden_present:
        return False
    if "agentv" in lower_allowed or "judge" in lower_allowed:
        return False
    return True


def build_proxy_metric_calibration_manifest(
    *,
    manifest_id: str,
    feature_set: ProxyFeatureSet | None = None,
    budget: BudgetCap | None = None,
    arms: Iterable[CalibrationArm] | None = None,
    activation_gates: Iterable[ActivationGate] | None = None,
    primary_metric: str = "binding_aware_meaningful_program_rate",
    seeds: Iterable[int] = (0, 1, 2),
    conservative_floor: float = 0.70,
    risk_budget: float = 0.05,
    proxy_eval_mode: ProxyEvalMode = "off",
    audit_rate: float = 0.10,
    force_full_every_n: int = 20,
    campaign_verdict: str = "unrun",
    note: str = "SDE3-03 proxy-metric calibration activation/budget manifest (wiring slice).",
) -> ProxyMetricCalibrationManifest:
    """Build a deterministic H23 proxy calibration manifest.

    The manifest is plan-only: it records the feature contract, gates, budget,
    and arms without training a proxy or changing full-suite authority.
    """
    feature_set = feature_set if feature_set is not None else _default_feature_set()
    budget = budget if budget is not None else _default_budget()
    arm_tuple = tuple(arms) if arms is not None else DEFAULT_ARMS
    gate_tuple = tuple(activation_gates) if activation_gates is not None else DEFAULT_ACTIVATION_GATES

    if not _feature_contract_safe(feature_set):
        activation_verdict: str = "feature_contract_unsafe"
    elif any(not gate.available for gate in gate_tuple):
        activation_verdict = "activation_blocked"
    elif _budget_is_empty(budget):
        activation_verdict = "budget_or_yield_blocked"
    elif not any(arm.eligible for arm in arm_tuple):
        activation_verdict = "activation_blocked"
    else:
        activation_verdict = "ready_to_spend"

    activation_status = _activation_status(activation_verdict)

    hash_payload = {
        "manifest_id": manifest_id,
        "schema_version": MANIFEST_SCHEMA,
        "hypothesis_id": HYPOTHESIS_ID,
        "activation_status": activation_status,
        "activation_verdict": activation_verdict,
        "campaign_verdict": campaign_verdict,
        "activation_gates": [g.to_dict() for g in gate_tuple],
        "feature_set": feature_set.to_dict(),
        "budget": budget.to_dict(),
        "arms": [a.to_dict() for a in arm_tuple],
        "primary_metric": primary_metric,
        "seeds": list(seeds),
        "conservative_floor": conservative_floor,
        "risk_budget": risk_budget,
        "proxy_eval_mode": proxy_eval_mode,
        "audit_rate": audit_rate,
        "force_full_every_n": force_full_every_n,
        "note": note,
    }
    manifest_hash = _stable_hash(hash_payload)

    return ProxyMetricCalibrationManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA,
        hypothesis_id=HYPOTHESIS_ID,
        activation_status=activation_status,
        activation_verdict=activation_verdict,
        campaign_verdict=campaign_verdict,
        activation_gates=gate_tuple,
        feature_set=feature_set,
        budget=budget,
        arms=arm_tuple,
        primary_metric=primary_metric,
        seeds=tuple(seeds),
        conservative_floor=conservative_floor,
        risk_budget=risk_budget,
        proxy_eval_mode=proxy_eval_mode,
        audit_rate=audit_rate,
        force_full_every_n=force_full_every_n,
        manifest_hash=manifest_hash,
        note=note,
    )


def validate_proxy_metric_calibration_manifest(
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

    activation_verdict = manifest.get("activation_verdict")
    if activation_verdict not in ACTIVATION_VERDICTS:
        errors.append(
            f"activation_verdict must be one of {sorted(ACTIVATION_VERDICTS)}"
        )

    campaign_verdict = manifest.get("campaign_verdict")
    if campaign_verdict not in CAMPAIGN_VERDICTS:
        errors.append(
            f"campaign_verdict must be one of {sorted(CAMPAIGN_VERDICTS)}"
        )

    proxy_eval_mode = manifest.get("proxy_eval_mode")
    if proxy_eval_mode not in PROXY_EVAL_MODES:
        errors.append(
            f"proxy_eval_mode must be one of {sorted(PROXY_EVAL_MODES)}"
        )

    for key in ("conservative_floor", "risk_budget", "audit_rate"):
        value = manifest.get(key)
        if not isinstance(value, (int, float)):
            errors.append(f"{key} must be a number")
        elif value < 0 or value > 1:
            errors.append(f"{key} must be between 0 and 1")

    if not isinstance(manifest.get("force_full_every_n"), int) or manifest.get("force_full_every_n", 0) <= 0:
        errors.append("force_full_every_n must be a positive integer")

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
            if not isinstance(gate.get("gate_id"), str) or not gate.get("gate_id"):
                errors.append(f"{prefix} missing or empty gate_id")

    feature_set = manifest.get("feature_set")
    if not isinstance(feature_set, dict):
        errors.append("feature_set must be an object")
    else:
        for key in ("feature_schema_version", "target_primary", "target_gate"):
            if not isinstance(feature_set.get(key), str) or not feature_set.get(key):
                errors.append(f"feature_set.{key} must be a non-empty string")
        if not isinstance(feature_set.get("feature_names"), list) or not feature_set.get("feature_names"):
            errors.append("feature_set.feature_names must be a non-empty list")
        if not isinstance(feature_set.get("forbidden_features"), list):
            errors.append("feature_set.forbidden_features must be a list")
        else:
            lower_forbidden = {f.lower() for f in feature_set.get("forbidden_features", [])}
            lower_features = {f.lower() for f in feature_set.get("feature_names", [])}
            if lower_features & lower_forbidden:
                errors.append("feature_set contains forbidden features")

    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        caps = [
            budget.get("max_historical_rows"),
            budget.get("max_dollars"),
            budget.get("gpu_hours"),
            budget.get("eval_dollars"),
            budget.get("total_dollars"),
        ]
        if not any(c is not None for c in caps):
            errors.append("budget must have at least one cap set")

    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms must be a non-empty list")
    else:
        for idx, arm in enumerate(arms):
            prefix = f"arms[{idx}]"
            if not isinstance(arm, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not isinstance(arm.get("arm_id"), str) or not arm.get("arm_id"):
                errors.append(f"{prefix} missing or empty arm_id")
            arm_kind = arm.get("arm_kind")
            if arm_kind not in ARM_KINDS:
                errors.append(
                    f"{prefix} arm_kind must be one of {sorted(ARM_KINDS)}"
                )
            if "eligible" not in arm:
                errors.append(f"{prefix} missing 'eligible'")
            elif arm.get("eligible") is False and not arm.get("omission_reason"):
                errors.append(f"{prefix} omitted arm must have omission_reason")

    return errors
