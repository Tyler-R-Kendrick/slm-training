"""Checked-in synthesis plans and fail-closed capability transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from slm_training.dsl.language_contract import (
    SYMBOLIC_SURFACE_POLICY_VERSION,
    SymbolicSurfacePolicyV1,
)
from slm_training.dsl.pack import DslPack, PackSlotUnavailable, get_pack
from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harness_core.versioning import component_version, load_registry
from slm_training.harnesses.staged import (
    Capability,
    Difficulty,
    EvaluationSource,
    SupervisionSource,
)

SCHEMA_VERSION = "synthesis_plan/v1"


class PlanAction(str, Enum):
    SYNTHESIZE = "synthesize"
    DISTILL = "distill"
    TRACE_PROMOTE = "trace_promote"


@dataclass(frozen=True)
class ComponentRefV1:
    component_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.version:
            raise ValueError("component_id and version are required")

    def to_dict(self) -> dict[str, str]:
        return {"component_id": self.component_id, "version": self.version}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentRefV1:
        _require_keys(value, {"component_id", "version"}, "component reference")
        return cls(
            component_id=_strict_str(value["component_id"], "component_id"),
            version=_strict_str(value["version"], "version"),
        )


@dataclass(frozen=True)
class CertificateRefV1:
    capability: Capability
    certificate_id: str
    sha256: str
    verified: bool

    def __post_init__(self) -> None:
        if not self.certificate_id:
            raise ValueError("certificate_id is required")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("certificate sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value,
            "certificate_id": self.certificate_id,
            "sha256": self.sha256,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CertificateRefV1:
        _require_keys(
            value,
            {"capability", "certificate_id", "sha256", "verified"},
            "certificate reference",
        )
        verified = value["verified"]
        if not isinstance(verified, bool):
            raise ValueError("certificate verified must be a boolean")
        return cls(
            capability=Capability(
                _strict_str(value["capability"], "certificate capability")
            ),
            certificate_id=_strict_str(value["certificate_id"], "certificate_id"),
            sha256=_strict_str(value["sha256"], "certificate sha256"),
            verified=verified,
        )


@dataclass(frozen=True)
class SplitPolicyV1:
    policy_id: str
    group_key: str
    holdout_modulus: int
    holdout_buckets: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.policy_id or not self.group_key:
            raise ValueError("split policy id and group key are required")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least 2")
        if not self.holdout_buckets:
            raise ValueError("at least one holdout bucket is required")
        if len(set(self.holdout_buckets)) != len(self.holdout_buckets):
            raise ValueError("holdout buckets must be unique")
        if any(
            bucket < 0 or bucket >= self.holdout_modulus
            for bucket in self.holdout_buckets
        ):
            raise ValueError("holdout bucket is outside the modulus")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "group_key": self.group_key,
            "holdout_modulus": self.holdout_modulus,
            "holdout_buckets": list(self.holdout_buckets),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitPolicyV1:
        _require_keys(
            value,
            {"policy_id", "group_key", "holdout_modulus", "holdout_buckets"},
            "split policy",
        )
        return cls(
            policy_id=_strict_str(value["policy_id"], "split policy_id"),
            group_key=_strict_str(value["group_key"], "split group_key"),
            holdout_modulus=_strict_int(value["holdout_modulus"], "holdout_modulus"),
            holdout_buckets=tuple(
                _strict_int(item, "holdout bucket")
                for item in _sequence(value["holdout_buckets"], "holdout_buckets")
            ),
        )


_COMPONENTS: dict[str, tuple[str, str]] = {
    "pack.corpus_generator": ("harness.train_data", "corpus_generator"),
    "pack.oracle": ("harness.train_data", "oracle"),
    "symbolic_surface": ("dsl.symbolic_surface", "placeholder_policy"),
}


@dataclass(frozen=True)
class SynthesisPlanV1:
    plan_id: str
    action: PlanAction
    capability: Capability
    supervision_source: SupervisionSource
    evaluation_source: EvaluationSource
    difficulty_families: tuple[Difficulty, ...]
    dsl_pack_id: str
    dsl_pack_version: str
    surface_policy_version: str
    generators: tuple[ComponentRefV1, ...]
    validators: tuple[ComponentRefV1, ...]
    split_policy: SplitPolicyV1
    gate_spec: ComponentRefV1
    seeds: tuple[int, ...]
    destinations: tuple[str, ...]
    prerequisite: CertificateRefV1 | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        if not self.plan_id:
            raise ValueError("plan_id is required")
        for name, values in (
            ("difficulty_families", self.difficulty_families),
            ("generators", self.generators),
            ("validators", self.validators),
            ("seeds", self.seeds),
            ("destinations", self.destinations),
        ):
            if not values:
                raise ValueError(f"{name} must not be empty")
        if len(set(self.difficulty_families)) != len(self.difficulty_families):
            raise ValueError("difficulty families must be unique")
        if len(set(self.seeds)) != len(self.seeds) or any(
            seed < 0 for seed in self.seeds
        ):
            raise ValueError("seeds must be unique non-negative integers")
        for destination in self.destinations:
            path = PurePosixPath(destination)
            if not destination or path.is_absolute() or ".." in path.parts:
                raise ValueError("destinations must be non-empty relative paths")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "action": self.action.value,
            "capability": self.capability.value,
            "supervision_source": self.supervision_source.value,
            "evaluation_source": self.evaluation_source.value,
            "difficulty_families": [
                difficulty.value for difficulty in self.difficulty_families
            ],
            "dsl_pack_id": self.dsl_pack_id,
            "dsl_pack_version": self.dsl_pack_version,
            "surface_policy_version": self.surface_policy_version,
            "generators": [item.to_dict() for item in self.generators],
            "validators": [item.to_dict() for item in self.validators],
            "split_policy": self.split_policy.to_dict(),
            "gate_spec": self.gate_spec.to_dict(),
            "seeds": list(self.seeds),
            "destinations": list(self.destinations),
            "prerequisite": (
                None if self.prerequisite is None else self.prerequisite.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SynthesisPlanV1:
        _require_keys(
            value,
            {
                "schema_version",
                "plan_id",
                "action",
                "capability",
                "supervision_source",
                "evaluation_source",
                "difficulty_families",
                "dsl_pack_id",
                "dsl_pack_version",
                "surface_policy_version",
                "generators",
                "validators",
                "split_policy",
                "gate_spec",
                "seeds",
                "destinations",
                "prerequisite",
            },
            "synthesis plan",
        )
        prerequisite = value["prerequisite"]
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            plan_id=_strict_str(value["plan_id"], "plan_id"),
            action=PlanAction(_strict_str(value["action"], "action")),
            capability=Capability(_strict_str(value["capability"], "capability")),
            supervision_source=SupervisionSource(
                _strict_str(value["supervision_source"], "supervision_source")
            ),
            evaluation_source=EvaluationSource(
                _strict_str(value["evaluation_source"], "evaluation_source")
            ),
            difficulty_families=tuple(
                Difficulty(_strict_str(item, "difficulty family"))
                for item in _sequence(
                    value["difficulty_families"], "difficulty_families"
                )
            ),
            dsl_pack_id=_strict_str(value["dsl_pack_id"], "dsl_pack_id"),
            dsl_pack_version=_strict_str(value["dsl_pack_version"], "dsl_pack_version"),
            surface_policy_version=_strict_str(
                value["surface_policy_version"], "surface_policy_version"
            ),
            generators=tuple(
                ComponentRefV1.from_dict(_mapping(item, "generator"))
                for item in _sequence(value["generators"], "generators")
            ),
            validators=tuple(
                ComponentRefV1.from_dict(_mapping(item, "validator"))
                for item in _sequence(value["validators"], "validators")
            ),
            split_policy=SplitPolicyV1.from_dict(
                _mapping(value["split_policy"], "split_policy")
            ),
            gate_spec=ComponentRefV1.from_dict(
                _mapping(value["gate_spec"], "gate_spec")
            ),
            seeds=tuple(
                _strict_int(item, "seed") for item in _sequence(value["seeds"], "seeds")
            ),
            destinations=tuple(
                _strict_str(item, "destination")
                for item in _sequence(value["destinations"], "destinations")
            ),
            prerequisite=(
                None
                if prerequisite is None
                else CertificateRefV1.from_dict(_mapping(prerequisite, "prerequisite"))
            ),
        )

    @classmethod
    def load(cls, path: Path) -> SynthesisPlanV1:
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            value = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        else:
            raise ValueError(f"unsupported synthesis plan format: {path.suffix}")
        return cls.from_dict(_mapping(value, "synthesis plan"))

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())

    def require_executable(self) -> None:
        pack = get_pack(self.dsl_pack_id)
        actual_pack_version = (
            SymbolicSurfacePolicyV1(pack_id=pack.pack_id).evaluate("").pack_version
        )
        if self.dsl_pack_version != actual_pack_version:
            raise ValueError(
                f"DSL pack version mismatch: plan={self.dsl_pack_version!r}, "
                f"active={actual_pack_version!r}"
            )
        if self.surface_policy_version != SYMBOLIC_SURFACE_POLICY_VERSION:
            raise ValueError(
                "surface policy version mismatch: "
                f"plan={self.surface_policy_version!r}, "
                f"active={SYMBOLIC_SURFACE_POLICY_VERSION!r}"
            )
        self._require_components(pack, self.generators, "generator")
        self._require_components(pack, self.validators, "validator")
        if "symbolic_surface" not in {
            validator.component_id for validator in self.validators
        }:
            raise ValueError("validators must include symbolic_surface")
        _require_current_component(self.gate_spec, "gate")
        gate_entry = load_registry()["components"][self.gate_spec.component_id]
        if gate_entry.get("kind") != "gate":
            raise ValueError(f"component {self.gate_spec.component_id!r} is not a gate")
        CapabilityStateMachineV1.require_allowed(self)

    @staticmethod
    def _require_components(
        pack: DslPack, refs: tuple[ComponentRefV1, ...], kind: str
    ) -> None:
        ids = [ref.component_id for ref in refs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{kind} component ids must be unique")
        for ref in refs:
            try:
                version_component, pack_slot = _COMPONENTS[ref.component_id]
            except KeyError as error:
                raise ValueError(f"unknown {kind} {ref.component_id!r}") from error
            if (
                kind == "generator" and ref.component_id != "pack.corpus_generator"
            ) or (kind == "validator" and ref.component_id == "pack.corpus_generator"):
                raise ValueError(f"component {ref.component_id!r} is not a {kind}")
            _require_current_component(ref, kind, version_component=version_component)
            try:
                pack.require(pack_slot)
            except PackSlotUnavailable as error:
                raise ValueError(str(error)) from error


class CapabilityStateMachineV1:
    """Executable CAP0 -> CAP1 -> CAP2 -> promotion boundary."""

    @staticmethod
    def require_allowed(plan: SynthesisPlanV1) -> None:
        if plan.action is PlanAction.DISTILL:
            if plan.capability is not Capability.CAP2_TRANSFORM:
                raise ValueError("distillation eligibility belongs to CAP2_TRANSFORM")
            _require_certificate(plan, Capability.CAP2_TRANSFORM)
            if plan.supervision_source is not SupervisionSource.SUP_DISTILL:
                raise ValueError("distillation requires SUP_DISTILL")
            return
        if plan.action is PlanAction.TRACE_PROMOTE:
            if plan.capability is not Capability.CAP2_TRANSFORM:
                raise ValueError(
                    "trace-promotion eligibility belongs to CAP2_TRANSFORM"
                )
            _require_certificate(plan, Capability.CAP2_TRANSFORM)
            if plan.evaluation_source is not EvaluationSource.EVAL_TRACE:
                raise ValueError("trace promotion requires EVAL_TRACE")
            return
        if plan.capability is Capability.CAP0_GRAMMAR:
            if plan.prerequisite is not None:
                raise ValueError(
                    "CAP0 synthesis must not cite a capability certificate"
                )
            if plan.supervision_source is not SupervisionSource.SUP_COMPILER:
                raise ValueError("CAP0 synthesis requires SUP_COMPILER")
        elif plan.capability is Capability.CAP1_SEMANTICS:
            _require_certificate(plan, Capability.CAP0_GRAMMAR)
        elif plan.capability is Capability.CAP2_TRANSFORM:
            _require_certificate(plan, Capability.CAP1_SEMANTICS)
            if plan.supervision_source is not SupervisionSource.SUP_PARAPHRASE:
                raise ValueError("CAP2 synthesis requires SUP_PARAPHRASE")


class SynthesisPlanRegistry:
    """Plan registry that resolves, but never duplicates, the DslPack registry."""

    def __init__(self) -> None:
        self._plans: dict[str, SynthesisPlanV1] = {}

    def register(self, plan: SynthesisPlanV1) -> SynthesisPlanV1:
        if plan.plan_id in self._plans:
            raise ValueError(f"duplicate synthesis plan {plan.plan_id!r}")
        plan.require_executable()
        self._plans[plan.plan_id] = plan
        return plan

    def load_directory(self, root: Path) -> None:
        for path in sorted(
            item
            for pattern in ("*.json", "*.yaml", "*.yml")
            for item in root.glob(pattern)
        ):
            self.register(SynthesisPlanV1.load(path))

    def get(self, plan_id: str) -> SynthesisPlanV1:
        try:
            return self._plans[plan_id]
        except KeyError as error:
            raise KeyError(
                f"unknown synthesis plan {plan_id!r}; known={sorted(self._plans)}"
            ) from error

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._plans))


def _require_certificate(plan: SynthesisPlanV1, capability: Capability) -> None:
    certificate = plan.prerequisite
    if certificate is None:
        raise ValueError(
            f"{plan.action.value} requires a {capability.value} certificate"
        )
    if certificate.capability is not capability:
        raise ValueError(
            f"expected {capability.value} certificate, got "
            f"{certificate.capability.value}"
        )
    if not certificate.verified:
        raise ValueError(f"{capability.value} certificate is not verified")


def _require_current_component(
    ref: ComponentRefV1,
    kind: str,
    *,
    version_component: str | None = None,
) -> None:
    component_id = version_component or ref.component_id
    try:
        actual = component_version(component_id)
    except KeyError as error:
        raise ValueError(f"unknown {kind} component {component_id!r}") from error
    if ref.version != actual:
        raise ValueError(
            f"{kind} version mismatch for {ref.component_id!r}: "
            f"plan={ref.version!r}, active={actual!r}"
        )


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _strict_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value
