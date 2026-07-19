"""SDE3-04 (SLM-178) constraint-backend benchmark manifest contract.

This module defines a frozen, versioned activation/budget/capability manifest
for comparing grammar-backed decoders (SynCode, DOMINO, XGrammar) against the
current OpenUI decoder and an unconstrained timing control. It is intentionally
plan-only: it records which backends are wired, which arms are preregistered,
and what gates must clear before any benchmark budget is spent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Mapping

MANIFEST_SCHEMA = "constraint_backend_benchmark/v1"
HYPOTHESIS_ID = "H17"

BackendId = Literal[
    "current",
    "syncode",
    "domino",
    "xgrammar",
    "unconstrained",
]

BenchmarkLayer = Literal[
    "static_micro",
    "language_equivalence",
    "end_to_end_surface",
]

ACTIVATION_VERDICTS = frozenset(
    {
        "activation_blocked",
        "ready_to_spend",
        "no_eligible_backend",
        "budget_or_yield_blocked",
        "unrun",
    }
)

CAMPAIGN_VERDICTS = frozenset(
    {
        "retain_current",
        "adopt_backend_for_surface_controls",
        "adopt_backend_broadly",
        "backend_not_equivalent",
        "inconclusive",
        "unrun",
    }
)

ALLOWED_BACKEND_IDS = {"current", "syncode", "domino", "xgrammar", "unconstrained"}
ALLOWED_BENCHMARK_LAYERS = {"static_micro", "language_equivalence", "end_to_end_surface"}


def _stable_hash(parts: Mapping[str, Any]) -> str:
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class BackendAdapter:
    """A grammar-backend package or control that may be benchmarked."""

    backend_id: BackendId
    package_name: str
    package_version: str
    commit_hash: str = ""
    license_spdx: str = ""
    local_offline_available: bool = True
    supported_grammar_kinds: tuple[str, ...] = ()
    unsupported_reason: str | None = None
    provenance_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "commit_hash": self.commit_hash,
            "license_spdx": self.license_spdx,
            "local_offline_available": self.local_offline_available,
            "supported_grammar_kinds": list(self.supported_grammar_kinds),
            "unsupported_reason": self.unsupported_reason,
            "provenance_url": self.provenance_url,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackendAdapter":
        return cls(
            backend_id=data["backend_id"],
            package_name=data["package_name"],
            package_version=data["package_version"],
            commit_hash=data.get("commit_hash", ""),
            license_spdx=data.get("license_spdx", ""),
            local_offline_available=data.get("local_offline_available", True),
            supported_grammar_kinds=tuple(data.get("supported_grammar_kinds", [])),
            unsupported_reason=data.get("unsupported_reason"),
            provenance_url=data.get("provenance_url", ""),
        )


@dataclass(frozen=True)
class ActivationGate:
    """A prerequisite that must be satisfied before budget can be released."""

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
            gate_id=data["gate_id"],
            depends_on_issue_id=data.get("depends_on_issue_id"),
            required_status=data.get("required_status"),
            available=data.get("available", False),
            evidence=data.get("evidence", ""),
        )


@dataclass(frozen=True)
class BudgetCap:
    """Spending limits for the benchmark campaign.

    At least one cap must be set (even if zero) for the manifest to be valid.
    """

    microbenchmark_repetitions: int | None = None
    end_to_end_repetitions: int | None = None
    max_dollars: float | None = None
    gpu_hours: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "microbenchmark_repetitions": self.microbenchmark_repetitions,
            "end_to_end_repetitions": self.end_to_end_repetitions,
            "max_dollars": self.max_dollars,
            "gpu_hours": self.gpu_hours,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetCap":
        return cls(
            microbenchmark_repetitions=data.get("microbenchmark_repetitions"),
            end_to_end_repetitions=data.get("end_to_end_repetitions"),
            max_dollars=data.get("max_dollars"),
            gpu_hours=data.get("gpu_hours"),
        )


@dataclass(frozen=True)
class BenchmarkArm:
    """One backend × benchmark-layer comparison arm."""

    arm_id: str
    backend_id: BackendId
    benchmark_layer: BenchmarkLayer
    eligible: bool
    omission_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "backend_id": self.backend_id,
            "benchmark_layer": self.benchmark_layer,
            "eligible": self.eligible,
            "omission_reason": self.omission_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BenchmarkArm":
        return cls(
            arm_id=data["arm_id"],
            backend_id=data["backend_id"],
            benchmark_layer=data["benchmark_layer"],
            eligible=data["eligible"],
            omission_reason=data.get("omission_reason"),
        )


@dataclass(frozen=True)
class ConstraintBackendBenchmarkManifest:
    """Frozen preregistered plan for the SDE3-04 constraint-backend benchmark."""

    manifest_id: str
    schema_version: str
    hypothesis_id: str
    activation_status: str
    activation_verdict: str
    campaign_verdict: str
    activation_gates: tuple[ActivationGate, ...]
    backends: tuple[BackendAdapter, ...]
    budget: BudgetCap
    arms: tuple[BenchmarkArm, ...]
    primary_metric: str
    seeds: tuple[int, ...]
    null_threshold_percent: float
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
            "backends": [b.to_dict() for b in self.backends],
            "budget": self.budget.to_dict(),
            "arms": [a.to_dict() for a in self.arms],
            "primary_metric": self.primary_metric,
            "seeds": list(self.seeds),
            "null_threshold_percent": self.null_threshold_percent,
            "manifest_hash": self.manifest_hash,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConstraintBackendBenchmarkManifest":
        return cls(
            manifest_id=data["manifest_id"],
            schema_version=data["schema_version"],
            hypothesis_id=data["hypothesis_id"],
            activation_status=data["activation_status"],
            activation_verdict=data["activation_verdict"],
            campaign_verdict=data["campaign_verdict"],
            activation_gates=tuple(
                ActivationGate.from_dict(g) for g in data.get("activation_gates", [])
            ),
            backends=tuple(
                BackendAdapter.from_dict(b) for b in data.get("backends", [])
            ),
            budget=BudgetCap.from_dict(data["budget"]),
            arms=tuple(BenchmarkArm.from_dict(a) for a in data.get("arms", [])),
            primary_metric=data["primary_metric"],
            seeds=tuple(data.get("seeds", [])),
            null_threshold_percent=data["null_threshold_percent"],
            manifest_hash=data["manifest_hash"],
            note=data.get("note", ""),
        )


def _default_activation_gates() -> tuple[ActivationGate, ...]:
    return (
        ActivationGate(
            gate_id="eval_cache_or_cost_approved",
            depends_on_issue_id="SLM-175",
            required_status="Done",
            available=False,
            evidence="",
        ),
        ActivationGate(
            gate_id="budget_approved",
            available=False,
            evidence="",
        ),
    )


def _default_backends() -> tuple[BackendAdapter, ...]:
    return (
        BackendAdapter(
            backend_id="current",
            package_name="openui_current",
            package_version="repo",
            supported_grammar_kinds=("openui",),
        ),
        BackendAdapter(
            backend_id="syncode",
            package_name="syncode",
            package_version="unset",
        ),
        BackendAdapter(
            backend_id="domino",
            package_name="domino",
            package_version="unset",
        ),
        BackendAdapter(
            backend_id="xgrammar",
            package_name="xgrammar",
            package_version="unset",
        ),
        BackendAdapter(
            backend_id="unconstrained",
            package_name="unconstrained",
            package_version="repo",
            supported_grammar_kinds=(),
        ),
    )


def _default_arms(
    backends: tuple[BackendAdapter, ...],
) -> tuple[BenchmarkArm, ...]:
    layers: tuple[BenchmarkLayer, ...] = (
        "static_micro",
        "language_equivalence",
        "end_to_end_surface",
    )
    arms: list[BenchmarkArm] = []
    for backend, layer in product(backends, layers):
        arms.append(
            BenchmarkArm(
                arm_id=f"{backend.backend_id}_{layer}",
                backend_id=backend.backend_id,
                benchmark_layer=layer,
                eligible=True,
            )
        )
    return tuple(arms)


def _budget_is_empty(budget: BudgetCap) -> bool:
    return all(
        cap in (None, 0, 0.0)
        for cap in (
            budget.microbenchmark_repetitions,
            budget.end_to_end_repetitions,
            budget.max_dollars,
            budget.gpu_hours,
        )
    )


def build_constraint_backend_benchmark_manifest(
    *,
    manifest_id: str,
    activation_gates: tuple[ActivationGate, ...] | None = None,
    backends: tuple[BackendAdapter, ...] | None = None,
    budget: BudgetCap | None = None,
    arms: tuple[BenchmarkArm, ...] | None = None,
    primary_metric: str = "binding_aware_meaningful_program_rate",
    seeds: tuple[int, ...] = (0, 1, 2),
    null_threshold_percent: float = 5.0,
    note: str = "",
) -> ConstraintBackendBenchmarkManifest:
    """Build a deterministic SDE3-04 constraint-backend benchmark manifest."""
    gate_tuple = activation_gates if activation_gates is not None else _default_activation_gates()
    backend_tuple = backends if backends is not None else _default_backends()
    budget_obj = budget if budget is not None else BudgetCap(
        microbenchmark_repetitions=0,
        end_to_end_repetitions=0,
        max_dollars=0.0,
        gpu_hours=0.0,
    )
    arm_tuple = arms if arms is not None else _default_arms(backend_tuple)

    if not all(gate.available for gate in gate_tuple):
        activation_verdict = "activation_blocked"
    elif _budget_is_empty(budget_obj):
        activation_verdict = "budget_or_yield_blocked"
    elif not any(arm.eligible for arm in arm_tuple):
        activation_verdict = "no_eligible_backend"
    else:
        activation_verdict = "ready_to_spend"

    campaign_verdict = "unrun"
    activation_status = activation_verdict

    hash_payload = {
        "manifest_id": manifest_id,
        "hypothesis_id": HYPOTHESIS_ID,
        "activation_gates": [g.to_dict() for g in gate_tuple],
        "backends": [b.to_dict() for b in backend_tuple],
        "budget": budget_obj.to_dict(),
        "arms": [a.to_dict() for a in arm_tuple],
        "primary_metric": primary_metric,
        "seeds": list(seeds),
        "null_threshold_percent": null_threshold_percent,
        "note": note,
    }
    manifest_hash = _stable_hash(hash_payload)

    return ConstraintBackendBenchmarkManifest(
        manifest_id=manifest_id,
        schema_version=MANIFEST_SCHEMA,
        hypothesis_id=HYPOTHESIS_ID,
        activation_status=activation_status,
        activation_verdict=activation_verdict,
        campaign_verdict=campaign_verdict,
        activation_gates=gate_tuple,
        backends=backend_tuple,
        budget=budget_obj,
        arms=arm_tuple,
        primary_metric=primary_metric,
        seeds=seeds,
        null_threshold_percent=null_threshold_percent,
        manifest_hash=manifest_hash,
        note=note,
    )


def validate_constraint_backend_benchmark_manifest(
    manifest: Mapping[str, Any],
) -> list[str]:
    """Return validation errors; empty means valid."""
    errors: list[str] = []

    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")

    required_fields = [
        "manifest_id",
        "hypothesis_id",
        "activation_status",
        "activation_verdict",
        "campaign_verdict",
        "activation_gates",
        "backends",
        "budget",
        "arms",
        "primary_metric",
        "seeds",
        "null_threshold_percent",
        "manifest_hash",
        "note",
    ]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"missing required field {field!r}")

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

    gates = manifest.get("activation_gates")
    if not isinstance(gates, list) or not gates:
        errors.append("activation_gates must be a non-empty list")
    else:
        for idx, gate in enumerate(gates):
            prefix = f"activation_gates[{idx}]"
            if not isinstance(gate, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not gate.get("gate_id"):
                errors.append(f"{prefix} missing or empty gate_id")

    backends = manifest.get("backends")
    if not isinstance(backends, list) or not backends:
        errors.append("backends must be a non-empty list")
        backend_ids: set[str] = set()
    else:
        backend_ids = set()
        for idx, backend in enumerate(backends):
            prefix = f"backends[{idx}]"
            if not isinstance(backend, dict):
                errors.append(f"{prefix} must be an object")
                continue
            backend_id = backend.get("backend_id")
            if backend_id not in ALLOWED_BACKEND_IDS:
                errors.append(
                    f"{prefix} backend_id must be one of {sorted(ALLOWED_BACKEND_IDS)}"
                )
            else:
                backend_ids.add(backend_id)
            if not backend.get("package_name"):
                errors.append(f"{prefix} missing or empty package_name")

    arms = manifest.get("arms")
    if not isinstance(arms, list) or not arms:
        errors.append("arms must be a non-empty list")
    else:
        for idx, arm in enumerate(arms):
            prefix = f"arms[{idx}]"
            if not isinstance(arm, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not arm.get("arm_id"):
                errors.append(f"{prefix} missing or empty arm_id")
            backend_id = arm.get("backend_id")
            if backend_id not in ALLOWED_BACKEND_IDS:
                errors.append(
                    f"{prefix} backend_id must be one of {sorted(ALLOWED_BACKEND_IDS)}"
                )
            elif backend_id not in backend_ids:
                errors.append(
                    f"{prefix} backend_id {backend_id!r} not declared in backends"
                )
            benchmark_layer = arm.get("benchmark_layer")
            if benchmark_layer not in ALLOWED_BENCHMARK_LAYERS:
                errors.append(
                    f"{prefix} benchmark_layer must be one of "
                    f"{sorted(ALLOWED_BENCHMARK_LAYERS)}"
                )
            if arm.get("eligible") is False and not arm.get("omission_reason"):
                errors.append(f"{prefix} omitted arm must have omission_reason")

    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        caps = (
            budget.get("microbenchmark_repetitions"),
            budget.get("end_to_end_repetitions"),
            budget.get("max_dollars"),
            budget.get("gpu_hours"),
        )
        if all(cap is None for cap in caps):
            errors.append("budget must have at least one cap")

    return errors
