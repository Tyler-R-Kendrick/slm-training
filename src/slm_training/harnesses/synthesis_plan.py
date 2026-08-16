"""Checked-in synthesis plans and fail-closed capability transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

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
SCHEMA_VERSION_V2 = "synthesis_plan/v2"
CORPUS_GENERATION_SCHEMA = "corpus_generation/v1"

#: Prompt registers. ``complex_nl`` is reserved and fails closed until a
#: capability gate authorizes it; adding the enum value does not open the rung.
class PromptSurface(str, Enum):
    GRAMMAR_SCHEMA = "grammar_schema"
    SIMPLIFIED_NL = "simplified_nl"
    TRANSFORMATION_NL = "transformation_nl"
    COMPLEX_NL = "complex_nl"


class GenerationMode(str, Enum):
    COUNT = "count"
    UNTIL_COVERAGE = "until_coverage"
    SCALING_LADDER = "scaling_ladder"


class ExhaustionBehavior(str, Enum):
    FAIL = "fail"
    REPORT = "report"


class GeneratorStrategy(str, Enum):
    COVERAGE_GUIDED_TYPED_AST = "coverage_guided_typed_ast"


class RendererFamily(str, Enum):
    CONCISE_INTENT = "concise_intent"
    TASK_CONTEXT = "task_context"
    COMPOSITIONAL_REQUIREMENT = "compositional_requirement"
    PARAPHRASE = "paraphrase"
    GRAMMAR_SCHEMA = "grammar_schema"
    TRANSFORMATION = "transformation"


class CoverageAxis(str, Enum):
    ROOT_FAMILY = "root_family"
    COMPONENT = "component"
    COMPONENT_PAIR = "component_pair"
    SELECTED_COMPONENT_TRIPLE = "selected_component_triple"
    PARENT_CHILD_EDGE = "parent_child_edge"
    CARDINALITY_FAMILY = "cardinality_family"
    ORDER_SENSITIVE_SIBLING = "order_sensitive_sibling"
    PROPERTY_VALUE_CLASS = "property_value_class"
    REFERENCE_TOPOLOGY = "reference_topology"
    REQUIRED_SLOT_OWNERSHIP = "required_slot_ownership"
    LENGTH = "length"
    AMBIGUITY_FAMILY = "ambiguity_family"
    PROMPT_SEMANTIC_FACT = "prompt_semantic_fact"


class CounterfactualDimension(str, Enum):
    ROLE = "role"
    ORDER = "order"
    CARDINALITY = "cardinality"
    CLOSED_VALUE = "closed_value"
    REQUIRED_CHILD = "required_child"
    RELATION = "relation"


_CAPABILITY_SURFACES: dict[Capability, frozenset[PromptSurface]] = {
    Capability.CAP0_GRAMMAR: frozenset({PromptSurface.GRAMMAR_SCHEMA}),
    Capability.CAP1_SEMANTICS: frozenset({PromptSurface.SIMPLIFIED_NL}),
    Capability.CAP2_TRANSFORM: frozenset({PromptSurface.TRANSFORMATION_NL}),
}


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


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _strict_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _strict_ratio(value: Any, label: str) -> float:
    number = _strict_float(value, label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _unique_strs(values: Any, label: str) -> tuple[str, ...]:
    items = tuple(_strict_str(item, label) for item in _sequence(values, label))
    if len(set(items)) != len(items):
        raise ValueError(f"{label} must be unique")
    return items


@dataclass(frozen=True)
class ProgramGenerationPolicy:
    strategy: GeneratorStrategy
    max_depth: int
    max_width: int
    components: tuple[str, ...]
    required_component_groups: tuple[tuple[str, ...], ...]
    property_value_classes: tuple[str, ...]
    forward_reference_families: tuple[str, ...]
    required_coverage_axes: tuple[CoverageAxis, ...]
    coverage_minimums: tuple[tuple[str, int], ...]
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(
                f"generator schema mismatch: {self.schema_version!r}"
            )
        if self.max_depth < 1 or self.max_width < 1:
            raise ValueError("max_depth and max_width must be positive")
        if not self.required_coverage_axes:
            raise ValueError("required_coverage_axes must not be empty")
        if len(set(self.required_coverage_axes)) != len(self.required_coverage_axes):
            raise ValueError("required_coverage_axes must be unique")
        known = {axis.value for axis in self.required_coverage_axes}
        for axis, minimum in self.coverage_minimums:
            if axis not in known:
                raise ValueError(f"coverage minimum for undeclared axis {axis!r}")
            if minimum < 1:
                raise ValueError("coverage minimums must be positive")
        for group in self.required_component_groups:
            if not group:
                raise ValueError("required component groups must be non-empty")
            if len(group) > self.max_width:
                raise ValueError("required component group exceeds max_width")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy.value,
            "max_depth": self.max_depth,
            "max_width": self.max_width,
            "components": list(self.components),
            "required_component_groups": [list(group) for group in self.required_component_groups],
            "property_value_classes": list(self.property_value_classes),
            "forward_reference_families": list(self.forward_reference_families),
            "required_coverage_axes": [axis.value for axis in self.required_coverage_axes],
            "coverage_minimums": {
                axis: minimum for axis, minimum in self.coverage_minimums
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProgramGenerationPolicy:
        _require_keys(
            value,
            {
                "schema_version",
                "strategy",
                "max_depth",
                "max_width",
                "components",
                "required_component_groups",
                "property_value_classes",
                "forward_reference_families",
                "required_coverage_axes",
                "coverage_minimums",
            },
            "program generation policy",
        )
        minima = _mapping(value["coverage_minimums"], "coverage_minimums")
        groups = tuple(
            tuple(_strict_str(item, "component") for item in _sequence(group, "group"))
            for group in _sequence(
                value["required_component_groups"], "required_component_groups"
            )
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            strategy=GeneratorStrategy(_strict_str(value["strategy"], "strategy")),
            max_depth=_strict_int(value["max_depth"], "max_depth"),
            max_width=_strict_int(value["max_width"], "max_width"),
            components=_unique_strs(value["components"], "component"),
            required_component_groups=groups,
            property_value_classes=_unique_strs(
                value["property_value_classes"], "property value class"
            ),
            forward_reference_families=_unique_strs(
                value["forward_reference_families"], "forward reference family"
            ),
            required_coverage_axes=tuple(
                CoverageAxis(_strict_str(item, "coverage axis"))
                for item in _sequence(
                    value["required_coverage_axes"], "required_coverage_axes"
                )
            ),
            coverage_minimums=tuple(
                (
                    _strict_str(axis, "coverage axis"),
                    _strict_int(minimum, "coverage minimum"),
                )
                for axis, minimum in sorted(minima.items())
            ),
        )


@dataclass(frozen=True)
class PromptSurfacePolicy:
    surface: PromptSurface
    prompts_per_root: int
    renderer_families: tuple[RendererFamily, ...]
    hard_forbidden_cues: tuple[str, ...]
    balanced_generic_cues: tuple[str, ...]
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(f"prompt schema mismatch: {self.schema_version!r}")
        if self.prompts_per_root < 1:
            raise ValueError("prompts_per_root must be positive")
        if not self.renderer_families:
            raise ValueError("renderer_families must not be empty")
        if len(set(self.renderer_families)) != len(self.renderer_families):
            raise ValueError("renderer_families must be unique")
        if self.surface is PromptSurface.COMPLEX_NL:
            raise ValueError(
                "complex_nl is unavailable until the existing capability gate "
                "authorizes it"
            )
        overlap = set(self.hard_forbidden_cues) & set(self.balanced_generic_cues)
        if overlap:
            raise ValueError(
                "hard-forbidden and balanced-generic cues overlap: "
                f"{sorted(overlap)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "surface": self.surface.value,
            "prompts_per_root": self.prompts_per_root,
            "renderer_families": [item.value for item in self.renderer_families],
            "hard_forbidden_cues": list(self.hard_forbidden_cues),
            "balanced_generic_cues": list(self.balanced_generic_cues),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromptSurfacePolicy:
        _require_keys(
            value,
            {
                "schema_version",
                "surface",
                "prompts_per_root",
                "renderer_families",
                "hard_forbidden_cues",
                "balanced_generic_cues",
            },
            "prompt surface policy",
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            surface=PromptSurface(_strict_str(value["surface"], "surface")),
            prompts_per_root=_strict_int(value["prompts_per_root"], "prompts_per_root"),
            renderer_families=tuple(
                RendererFamily(_strict_str(item, "renderer family"))
                for item in _sequence(value["renderer_families"], "renderer_families")
            ),
            hard_forbidden_cues=_unique_strs(
                value["hard_forbidden_cues"], "hard forbidden cue"
            ),
            balanced_generic_cues=_unique_strs(
                value["balanced_generic_cues"], "balanced generic cue"
            ),
        )


@dataclass(frozen=True)
class DerivativePolicy:
    semantic_counterfactuals_per_eligible_root: int
    counterfactual_dimensions: tuple[CounterfactualDimension, ...]
    repairs_per_root: int
    edits_per_root: int
    scope_per_root: int
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(f"derivative schema mismatch: {self.schema_version!r}")
        for name, value in (
            ("semantic_counterfactuals_per_eligible_root", self.semantic_counterfactuals_per_eligible_root),
            ("repairs_per_root", self.repairs_per_root),
            ("edits_per_root", self.edits_per_root),
            ("scope_per_root", self.scope_per_root),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.semantic_counterfactuals_per_eligible_root and not self.counterfactual_dimensions:
            raise ValueError("counterfactual dimensions required when count > 0")
        if len(set(self.counterfactual_dimensions)) != len(self.counterfactual_dimensions):
            raise ValueError("counterfactual dimensions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_counterfactuals_per_eligible_root": (
                self.semantic_counterfactuals_per_eligible_root
            ),
            "counterfactual_dimensions": [
                item.value for item in self.counterfactual_dimensions
            ],
            "repairs_per_root": self.repairs_per_root,
            "edits_per_root": self.edits_per_root,
            "scope_per_root": self.scope_per_root,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DerivativePolicy:
        _require_keys(
            value,
            {
                "schema_version",
                "semantic_counterfactuals_per_eligible_root",
                "counterfactual_dimensions",
                "repairs_per_root",
                "edits_per_root",
                "scope_per_root",
            },
            "derivative policy",
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            semantic_counterfactuals_per_eligible_root=_strict_int(
                value["semantic_counterfactuals_per_eligible_root"],
                "semantic_counterfactuals_per_eligible_root",
            ),
            counterfactual_dimensions=tuple(
                CounterfactualDimension(_strict_str(item, "counterfactual dimension"))
                for item in _sequence(
                    value["counterfactual_dimensions"], "counterfactual_dimensions"
                )
            ),
            repairs_per_root=_strict_int(value["repairs_per_root"], "repairs_per_root"),
            edits_per_root=_strict_int(value["edits_per_root"], "edits_per_root"),
            scope_per_root=_strict_int(value["scope_per_root"], "scope_per_root"),
        )


@dataclass(frozen=True)
class SourceAnchorPolicy:
    retain_available_trusted_roots: bool
    minimum_unique_root_fraction: float
    trusted_source_families: tuple[str, ...]
    promotion_requires_anchors: bool
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(f"anchor schema mismatch: {self.schema_version!r}")
        if self.minimum_unique_root_fraction < 0.0 or self.minimum_unique_root_fraction > 1.0:
            raise ValueError("minimum_unique_root_fraction must be in [0, 1]")
        if self.promotion_requires_anchors and not self.trusted_source_families:
            raise ValueError("promotion_requires_anchors needs trusted_source_families")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "retain_available_trusted_roots": self.retain_available_trusted_roots,
            "minimum_unique_root_fraction": self.minimum_unique_root_fraction,
            "trusted_source_families": list(self.trusted_source_families),
            "promotion_requires_anchors": self.promotion_requires_anchors,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceAnchorPolicy:
        _require_keys(
            value,
            {
                "schema_version",
                "retain_available_trusted_roots",
                "minimum_unique_root_fraction",
                "trusted_source_families",
                "promotion_requires_anchors",
            },
            "source anchor policy",
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            retain_available_trusted_roots=_strict_bool(
                value["retain_available_trusted_roots"],
                "retain_available_trusted_roots",
            ),
            minimum_unique_root_fraction=_strict_ratio(
                value["minimum_unique_root_fraction"],
                "minimum_unique_root_fraction",
            ),
            trusted_source_families=_unique_strs(
                value["trusted_source_families"], "trusted source family"
            ),
            promotion_requires_anchors=_strict_bool(
                value["promotion_requires_anchors"], "promotion_requires_anchors"
            ),
        )


@dataclass(frozen=True)
class PairQualityPolicy:
    reject_target_surface_leaks: bool
    max_exact_template_share: float
    max_normalized_prefix_share: float
    max_cue_only_accuracy_lift: float
    max_source_only_accuracy_lift: float
    min_unique_structure_fraction: float
    max_renderer_share: float
    min_no_generic_cue_fraction: float
    per_root_exposure_cap: int
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(f"quality schema mismatch: {self.schema_version!r}")
        if self.per_root_exposure_cap < 1:
            raise ValueError("per_root_exposure_cap must be positive")
        for name in (
            "max_exact_template_share",
            "max_normalized_prefix_share",
            "max_cue_only_accuracy_lift",
            "max_source_only_accuracy_lift",
            "min_unique_structure_fraction",
            "max_renderer_share",
            "min_no_generic_cue_fraction",
        ):
            value = getattr(self, name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reject_target_surface_leaks": self.reject_target_surface_leaks,
            "max_exact_template_share": self.max_exact_template_share,
            "max_normalized_prefix_share": self.max_normalized_prefix_share,
            "max_cue_only_accuracy_lift": self.max_cue_only_accuracy_lift,
            "max_source_only_accuracy_lift": self.max_source_only_accuracy_lift,
            "min_unique_structure_fraction": self.min_unique_structure_fraction,
            "max_renderer_share": self.max_renderer_share,
            "min_no_generic_cue_fraction": self.min_no_generic_cue_fraction,
            "per_root_exposure_cap": self.per_root_exposure_cap,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairQualityPolicy:
        _require_keys(
            value,
            {
                "schema_version",
                "reject_target_surface_leaks",
                "max_exact_template_share",
                "max_normalized_prefix_share",
                "max_cue_only_accuracy_lift",
                "max_source_only_accuracy_lift",
                "min_unique_structure_fraction",
                "max_renderer_share",
                "min_no_generic_cue_fraction",
                "per_root_exposure_cap",
            },
            "pair quality policy",
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            reject_target_surface_leaks=_strict_bool(
                value["reject_target_surface_leaks"], "reject_target_surface_leaks"
            ),
            max_exact_template_share=_strict_ratio(
                value["max_exact_template_share"], "max_exact_template_share"
            ),
            max_normalized_prefix_share=_strict_ratio(
                value["max_normalized_prefix_share"], "max_normalized_prefix_share"
            ),
            max_cue_only_accuracy_lift=_strict_ratio(
                value["max_cue_only_accuracy_lift"], "max_cue_only_accuracy_lift"
            ),
            max_source_only_accuracy_lift=_strict_ratio(
                value["max_source_only_accuracy_lift"], "max_source_only_accuracy_lift"
            ),
            min_unique_structure_fraction=_strict_ratio(
                value["min_unique_structure_fraction"],
                "min_unique_structure_fraction",
            ),
            max_renderer_share=_strict_ratio(
                value["max_renderer_share"], "max_renderer_share"
            ),
            min_no_generic_cue_fraction=_strict_ratio(
                value["min_no_generic_cue_fraction"], "min_no_generic_cue_fraction"
            ),
            per_root_exposure_cap=_strict_int(
                value["per_root_exposure_cap"], "per_root_exposure_cap"
            ),
        )


@dataclass(frozen=True)
class NestedLadderPolicy:
    require_nested_prefix: bool
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(f"ladder schema mismatch: {self.schema_version!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "require_nested_prefix": self.require_nested_prefix,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NestedLadderPolicy:
        _require_keys(
            value,
            {"schema_version", "require_nested_prefix"},
            "nested ladder policy",
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            require_nested_prefix=_strict_bool(
                value["require_nested_prefix"], "require_nested_prefix"
            ),
        )


@dataclass(frozen=True)
class CorpusGenerationPolicy:
    """Capability-aware unique-root corpus policy. Companion to synthesis_plan/v2."""

    mode: GenerationMode
    unique_root_targets: tuple[int, ...]
    seed: int
    shard_count: int
    shard_index: int
    max_attempts: int
    exhaustion: ExhaustionBehavior
    generator: ProgramGenerationPolicy
    prompts: PromptSurfacePolicy
    derivatives: DerivativePolicy
    source_anchors: SourceAnchorPolicy
    quality: PairQualityPolicy
    ladder: NestedLadderPolicy
    schema_version: str = CORPUS_GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CORPUS_GENERATION_SCHEMA:
            raise ValueError(f"corpus schema mismatch: {self.schema_version!r}")
        if not self.unique_root_targets:
            raise ValueError("unique_root_targets must not be empty")
        if any(target < 1 for target in self.unique_root_targets):
            raise ValueError("unique_root_targets must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.shard_count < 1:
            raise ValueError("shard_count must be positive")
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            raise ValueError("shard_index is outside shard_count")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.mode is GenerationMode.SCALING_LADDER:
            if len(self.unique_root_targets) < 2:
                raise ValueError("scaling_ladder requires at least two unique-root targets")
            if list(self.unique_root_targets) != sorted(self.unique_root_targets):
                raise ValueError("scaling_ladder unique_root_targets must be strictly increasing")
            if len(set(self.unique_root_targets)) != len(self.unique_root_targets):
                raise ValueError("scaling_ladder unique_root_targets must be unique")
        elif len(self.unique_root_targets) != 1:
            raise ValueError(f"{self.mode.value} requires exactly one unique-root target")
        if self.quality.per_root_exposure_cap < self.prompts.prompts_per_root:
            raise ValueError("per_root_exposure_cap must cover prompts_per_root")

    @property
    def requested_unique_roots(self) -> int:
        return self.unique_root_targets[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "unique_root_targets": list(self.unique_root_targets),
            "seed": self.seed,
            "shard_count": self.shard_count,
            "shard_index": self.shard_index,
            "max_attempts": self.max_attempts,
            "exhaustion": self.exhaustion.value,
            "generator": self.generator.to_dict(),
            "prompts": self.prompts.to_dict(),
            "derivatives": self.derivatives.to_dict(),
            "source_anchors": self.source_anchors.to_dict(),
            "quality": self.quality.to_dict(),
            "ladder": self.ladder.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CorpusGenerationPolicy:
        _require_keys(
            value,
            {
                "schema_version",
                "mode",
                "unique_root_targets",
                "seed",
                "shard_count",
                "shard_index",
                "max_attempts",
                "exhaustion",
                "generator",
                "prompts",
                "derivatives",
                "source_anchors",
                "quality",
                "ladder",
            },
            "corpus generation policy",
        )
        return cls(
            schema_version=_strict_str(value["schema_version"], "schema_version"),
            mode=GenerationMode(_strict_str(value["mode"], "mode")),
            unique_root_targets=tuple(
                _strict_int(item, "unique root target")
                for item in _sequence(value["unique_root_targets"], "unique_root_targets")
            ),
            seed=_strict_int(value["seed"], "seed"),
            shard_count=_strict_int(value["shard_count"], "shard_count"),
            shard_index=_strict_int(value["shard_index"], "shard_index"),
            max_attempts=_strict_int(value["max_attempts"], "max_attempts"),
            exhaustion=ExhaustionBehavior(_strict_str(value["exhaustion"], "exhaustion")),
            generator=ProgramGenerationPolicy.from_dict(
                _mapping(value["generator"], "generator")
            ),
            prompts=PromptSurfacePolicy.from_dict(_mapping(value["prompts"], "prompts")),
            derivatives=DerivativePolicy.from_dict(
                _mapping(value["derivatives"], "derivatives")
            ),
            source_anchors=SourceAnchorPolicy.from_dict(
                _mapping(value["source_anchors"], "source_anchors")
            ),
            quality=PairQualityPolicy.from_dict(_mapping(value["quality"], "quality")),
            ladder=NestedLadderPolicy.from_dict(_mapping(value["ladder"], "ladder")),
        )

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())


def tiny_corpus_generation_policy(
    *,
    capability: Capability = Capability.CAP1_SEMANTICS,
    mode: GenerationMode = GenerationMode.COUNT,
    unique_root_targets: tuple[int, ...] = (4,),
    seed: int = 17,
) -> CorpusGenerationPolicy:
    """Deterministic CI-scale policy. Never used as a silent v1 default."""

    surface = {
        Capability.CAP0_GRAMMAR: PromptSurface.GRAMMAR_SCHEMA,
        Capability.CAP1_SEMANTICS: PromptSurface.SIMPLIFIED_NL,
        Capability.CAP2_TRANSFORM: PromptSurface.TRANSFORMATION_NL,
    }[capability]
    families = {
        PromptSurface.GRAMMAR_SCHEMA: (RendererFamily.GRAMMAR_SCHEMA,),
        PromptSurface.SIMPLIFIED_NL: (
            RendererFamily.CONCISE_INTENT,
            RendererFamily.TASK_CONTEXT,
            RendererFamily.COMPOSITIONAL_REQUIREMENT,
            RendererFamily.PARAPHRASE,
        ),
        PromptSurface.TRANSFORMATION_NL: (RendererFamily.TRANSFORMATION,),
    }[surface]
    return CorpusGenerationPolicy(
        mode=mode,
        unique_root_targets=unique_root_targets,
        seed=seed,
        shard_count=1,
        shard_index=0,
        max_attempts=max(64, unique_root_targets[-1] * 16),
        exhaustion=ExhaustionBehavior.FAIL,
        generator=ProgramGenerationPolicy(
            strategy=GeneratorStrategy.COVERAGE_GUIDED_TYPED_AST,
            max_depth=2,
            max_width=3,
            components=(),
            required_component_groups=(),
            property_value_classes=(),
            forward_reference_families=(),
            required_coverage_axes=(
                CoverageAxis.COMPONENT,
                CoverageAxis.COMPONENT_PAIR,
                CoverageAxis.REFERENCE_TOPOLOGY,
                CoverageAxis.LENGTH,
            ),
            coverage_minimums=(("component", 2),),
        ),
        prompts=PromptSurfacePolicy(
            surface=surface,
            prompts_per_root=2,
            renderer_families=families,
            hard_forbidden_cues=(
                "openui",
                "dsl",
                "ast",
                "component inventory",
                "declarations",
                "reference graph",
                "placeholder",
            ),
            balanced_generic_cues=(
                "ui",
                "screen",
                "layout",
                "page",
                "interface",
                "build",
                "create",
                "generate",
                "design",
            ),
        ),
        derivatives=DerivativePolicy(
            semantic_counterfactuals_per_eligible_root=1,
            counterfactual_dimensions=(
                CounterfactualDimension.ROLE,
                CounterfactualDimension.ORDER,
                CounterfactualDimension.CARDINALITY,
                CounterfactualDimension.CLOSED_VALUE,
                CounterfactualDimension.REQUIRED_CHILD,
                CounterfactualDimension.RELATION,
            ),
            repairs_per_root=1,
            edits_per_root=1,
            scope_per_root=0,
        ),
        source_anchors=SourceAnchorPolicy(
            retain_available_trusted_roots=True,
            minimum_unique_root_fraction=0.10,
            trusted_source_families=("human_curated", "rico_real", "frontier_described"),
            promotion_requires_anchors=True,
        ),
        quality=PairQualityPolicy(
            reject_target_surface_leaks=True,
            max_exact_template_share=0.05,
            max_normalized_prefix_share=0.10,
            max_cue_only_accuracy_lift=0.05,
            max_source_only_accuracy_lift=0.05,
            min_unique_structure_fraction=0.90,
            max_renderer_share=0.50,
            min_no_generic_cue_fraction=0.15,
            per_root_exposure_cap=8,
        ),
        ladder=NestedLadderPolicy(require_nested_prefix=mode is GenerationMode.SCALING_LADDER),
    )


def validate_capability_prompt_surface(
    capability: Capability, surface: PromptSurface
) -> None:
    if surface is PromptSurface.COMPLEX_NL:
        raise ValueError(
            "complex_nl is unavailable until the existing capability gate authorizes it"
        )
    allowed = _CAPABILITY_SURFACES[capability]
    if surface not in allowed:
        raise ValueError(
            f"{capability.value} requires prompt surface in "
            f"{sorted(item.value for item in allowed)}, got {surface.value}"
        )


@dataclass(frozen=True)
class SynthesisPlanV2:
    """v2 plan: exact v1 fields plus a required corpus-generation policy."""

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
    corpus_generation: CorpusGenerationPolicy
    prerequisite: CertificateRefV1 | None = None
    schema_version: str = SCHEMA_VERSION_V2

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION_V2:
            raise ValueError(
                f"schema mismatch: expected {SCHEMA_VERSION_V2!r}, "
                f"got {self.schema_version!r}"
            )
        self.to_v1()
        validate_capability_prompt_surface(
            self.capability, self.corpus_generation.prompts.surface
        )

    def to_v1(self) -> SynthesisPlanV1:
        return SynthesisPlanV1(
            plan_id=self.plan_id,
            action=self.action,
            capability=self.capability,
            supervision_source=self.supervision_source,
            evaluation_source=self.evaluation_source,
            difficulty_families=self.difficulty_families,
            dsl_pack_id=self.dsl_pack_id,
            dsl_pack_version=self.dsl_pack_version,
            surface_policy_version=self.surface_policy_version,
            generators=self.generators,
            validators=self.validators,
            split_policy=self.split_policy,
            gate_spec=self.gate_spec,
            seeds=self.seeds,
            destinations=self.destinations,
            prerequisite=self.prerequisite,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_v1().to_dict()
        payload["schema_version"] = self.schema_version
        payload["corpus_generation"] = self.corpus_generation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SynthesisPlanV2:
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
                "corpus_generation",
            },
            "synthesis plan v2",
        )
        v1_payload = {
            key: value[key]
            for key in value
            if key not in {"schema_version", "corpus_generation"}
        }
        v1_payload["schema_version"] = SCHEMA_VERSION
        v1 = SynthesisPlanV1.from_dict(v1_payload)
        return cls(
            plan_id=v1.plan_id,
            action=v1.action,
            capability=v1.capability,
            supervision_source=v1.supervision_source,
            evaluation_source=v1.evaluation_source,
            difficulty_families=v1.difficulty_families,
            dsl_pack_id=v1.dsl_pack_id,
            dsl_pack_version=v1.dsl_pack_version,
            surface_policy_version=v1.surface_policy_version,
            generators=v1.generators,
            validators=v1.validators,
            split_policy=v1.split_policy,
            gate_spec=v1.gate_spec,
            seeds=v1.seeds,
            destinations=v1.destinations,
            prerequisite=v1.prerequisite,
            corpus_generation=CorpusGenerationPolicy.from_dict(
                _mapping(value["corpus_generation"], "corpus_generation")
            ),
            schema_version=_strict_str(value["schema_version"], "schema_version"),
        )

    @classmethod
    def from_v1(
        cls, plan: SynthesisPlanV1, corpus_generation: CorpusGenerationPolicy
    ) -> SynthesisPlanV2:
        return cls(
            plan_id=plan.plan_id,
            action=plan.action,
            capability=plan.capability,
            supervision_source=plan.supervision_source,
            evaluation_source=plan.evaluation_source,
            difficulty_families=plan.difficulty_families,
            dsl_pack_id=plan.dsl_pack_id,
            dsl_pack_version=plan.dsl_pack_version,
            surface_policy_version=plan.surface_policy_version,
            generators=plan.generators,
            validators=plan.validators,
            split_policy=plan.split_policy,
            gate_spec=plan.gate_spec,
            seeds=plan.seeds,
            destinations=plan.destinations,
            prerequisite=plan.prerequisite,
            corpus_generation=corpus_generation,
        )

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())

    def require_executable(self) -> None:
        self.to_v1().require_executable()
        validate_capability_prompt_surface(
            self.capability, self.corpus_generation.prompts.surface
        )


def load_synthesis_plan(path: Path) -> SynthesisPlanV1 | SynthesisPlanV2:
    """Shared loader. v1 bytes stay on SynthesisPlanV1; v2 is a new schema."""

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported synthesis plan format: {path.suffix}")
    payload = _mapping(value, "synthesis plan")
    version = payload.get("schema_version")
    if version == SCHEMA_VERSION:
        return SynthesisPlanV1.from_dict(payload)
    if version == SCHEMA_VERSION_V2:
        return SynthesisPlanV2.from_dict(payload)
    raise ValueError(f"unsupported synthesis plan schema: {version!r}")


def migrate_plan_v1_to_v2(
    plan: SynthesisPlanV1, corpus_generation: CorpusGenerationPolicy
) -> SynthesisPlanV2:
    """Explicit content-addressed migration. Does not rewrite v1 files."""

    return SynthesisPlanV2.from_v1(plan, corpus_generation)


def apply_corpus_generation_overrides(
    policy: CorpusGenerationPolicy,
    *,
    generation_mode: str | None = None,
    unique_root_target: int | None = None,
    generator_seed: int | None = None,
    generator_max_depth: int | None = None,
    generator_max_width: int | None = None,
    prompt_surface: str | None = None,
    prompts_per_root: int | None = None,
    renderer_families: Sequence[str] | None = None,
    counterfactuals_per_root: int | None = None,
    retain_trusted_anchors: bool | None = None,
    component_coverage_minimum: int | None = None,
) -> CorpusGenerationPolicy:
    """CLI/knob overlays. Does not add or rename corpus_generation/v1 keys."""

    updates: dict[str, Any] = {}
    if generation_mode is not None:
        updates["mode"] = GenerationMode(generation_mode)
    if unique_root_target is not None:
        mode = updates.get("mode", policy.mode)
        if mode is GenerationMode.SCALING_LADDER:
            prior = policy.unique_root_targets[:-1]
            if prior and unique_root_target <= prior[-1]:
                raise ValueError(
                    "unique_root_target must exceed the previous scaling_ladder rung"
                )
            updates["unique_root_targets"] = (*prior, unique_root_target)
        else:
            updates["unique_root_targets"] = (unique_root_target,)
        updates["max_attempts"] = max(policy.max_attempts, unique_root_target * 16)
    if generator_seed is not None:
        updates["seed"] = generator_seed
    generator = policy.generator
    if generator_max_depth is not None:
        generator = replace(generator, max_depth=generator_max_depth)
    if generator_max_width is not None:
        generator = replace(generator, max_width=generator_max_width)
    if component_coverage_minimum is not None:
        # Overlay the existing component-axis minimum; other axes keep theirs.
        # Raising it and building with exhaustion=fail is the targeted answer
        # to under-witnessed components (sample-adequacy coverage floor).
        merged = dict(generator.coverage_minimums)
        merged["component"] = component_coverage_minimum
        generator = replace(
            generator, coverage_minimums=tuple(sorted(merged.items()))
        )
    if generator is not policy.generator:
        updates["generator"] = generator
    prompts = policy.prompts
    if prompt_surface is not None:
        prompts = replace(prompts, surface=PromptSurface(prompt_surface))
    if prompts_per_root is not None:
        prompts = replace(prompts, prompts_per_root=prompts_per_root)
        if prompts_per_root > policy.quality.per_root_exposure_cap:
            updates["quality"] = replace(
                policy.quality, per_root_exposure_cap=prompts_per_root
            )
    if renderer_families is not None:
        prompts = replace(
            prompts,
            renderer_families=tuple(
                RendererFamily(item) for item in renderer_families
            ),
        )
    if prompts is not policy.prompts:
        updates["prompts"] = prompts
    if counterfactuals_per_root is not None:
        updates["derivatives"] = replace(
            policy.derivatives,
            semantic_counterfactuals_per_eligible_root=counterfactuals_per_root,
        )
    if retain_trusted_anchors is not None:
        updates["source_anchors"] = replace(
            policy.source_anchors,
            retain_available_trusted_roots=retain_trusted_anchors,
        )
    return replace(policy, **updates) if updates else policy
