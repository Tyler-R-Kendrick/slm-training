"""Versioned, provenance-aware semantic plan skeleton for SPV0."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PlanProvenance = Literal["gold", "predicted", "retrieved", "merged", "oracle_override"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanIdentity(_StrictModel):
    pack_id: str = Field(min_length=1)
    contract_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    source_program_fingerprint: str | None = None
    prompt_context_hash: str | None = None
    provenance: PlanProvenance


class PlanArchetype(_StrictModel):
    id: str | None = None
    distribution: dict[str, float] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RoleSlot(_StrictModel):
    role_id: str = Field(min_length=1)
    component_family: str | None = None
    candidate_distribution: dict[str, float] | None = None
    min_cardinality: int | None = Field(default=None, ge=0)
    max_cardinality: int | None = Field(default=None, ge=0)
    required: bool | None = None
    evidence_spans: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def check_cardinality_bounds(self) -> "RoleSlot":
        if (
            self.min_cardinality is not None
            and self.max_cardinality is not None
            and self.min_cardinality > self.max_cardinality
        ):
            raise ValueError("min_cardinality cannot exceed max_cardinality")
        return self


class PlanTopology(_StrictModel):
    parent_relation_candidates: tuple[dict[str, Any], ...] | None = None
    sibling_order_groups: tuple[tuple[str, ...], ...] | None = None
    depth_bounds: tuple[int, int] | None = None
    cardinality_bounds: dict[str, tuple[int, int]] | None = None
    partial_order_constraints: tuple[dict[str, Any], ...] | None = None

    @field_validator("depth_bounds")
    @classmethod
    def check_depth_bounds(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and value[0] > value[1]:
            raise ValueError("depth lower bound cannot exceed upper bound")
        return value


class PlanSymbol(_StrictModel):
    symbol_id: str = Field(min_length=1)
    semantic_role: str | None = None
    allowed_pointer_targets: tuple[str, ...] | None = None


class PlanBinding(_StrictModel):
    role_slot_id: str = Field(min_length=1)
    candidate_symbols: tuple[str, ...] | None = None
    placeholder_fallback: bool | None = None


class PlanCoverage(_StrictModel):
    named_requirements_accounted_for: tuple[str, ...] | None = None
    unresolved_requirements: tuple[str, ...] | None = None


class PlanConfidenceCalibration(_StrictModel):
    per_factor_confidence: dict[str, float] | None = None
    abstention_reason: str | None = None

    @field_validator("per_factor_confidence")
    @classmethod
    def check_confidence_range(
        cls, value: dict[str, float] | None
    ) -> dict[str, float] | None:
        if value is None:
            return value
        for key, conf in value.items():
            if not 0.0 <= conf <= 1.0:
                raise ValueError(f"confidence for {key!r} must be in [0, 1]")
        return value


class SemanticPlanV1(_StrictModel):
    """Pack-neutral semantic plan hypothesis with fail-closed provenance."""

    plan_version: str = Field(default="1", pattern=r"^1$")
    identity: PlanIdentity
    archetype: PlanArchetype = Field(default_factory=PlanArchetype)
    role_slots: tuple[RoleSlot, ...] = ()
    topology: PlanTopology = Field(default_factory=PlanTopology)
    symbols: tuple[PlanSymbol, ...] = ()
    bindings: tuple[PlanBinding, ...] = ()
    coverage: PlanCoverage = Field(default_factory=PlanCoverage)
    confidence_calibration: PlanConfidenceCalibration = Field(
        default_factory=PlanConfidenceCalibration
    )

    @model_validator(mode="after")
    def check_identity_bijections(self) -> "SemanticPlanV1":
        symbol_ids = [symbol.symbol_id for symbol in self.symbols]
        if len(symbol_ids) != len(set(symbol_ids)):
            raise ValueError("symbol_id values must be unique")
        role_ids = [slot.role_id for slot in self.role_slots]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("role_id values must be unique")
        binding_roles = [binding.role_slot_id for binding in self.bindings]
        if len(binding_roles) != len(set(binding_roles)):
            raise ValueError("role_slot_id bindings must be unique")
        known_symbols = set(symbol_ids)
        for binding in self.bindings:
            unknown = set(binding.candidate_symbols or ()) - known_symbols
            if unknown:
                raise ValueError(
                    f"binding references unknown symbol ids: {sorted(unknown)}"
                )
        return self

    @property
    def is_oracle_only(self) -> bool:
        return self.identity.provenance in {"gold", "oracle_override"}

    @property
    def is_abstained(self) -> bool:
        return bool(self.confidence_calibration.abstention_reason)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticPlanV1":
        if str(value.get("plan_version")) != "1":
            raise ValueError("unsupported SemanticPlanV1 version")
        return cls.model_validate(value)

    def to_production_dict(
        self, *, honesty_mode: str = "production"
    ) -> dict[str, Any]:
        """Return a plan safe for production manifests.

        Oracle-only provenance is rejected unless honesty_mode is
        "oracle_diagnostic". Gold/oracle fields are stripped.
        """
        if honesty_mode not in {"production", "oracle_diagnostic"}:
            raise ValueError("honesty_mode must be production or oracle_diagnostic")
        if self.is_oracle_only and honesty_mode != "oracle_diagnostic":
            raise ValueError(
                "gold/oracle_override plan cannot enter a production manifest"
            )
        payload = self.to_dict()
        # Strip oracle-only fields from production output regardless of mode.
        identity = dict(payload.get("identity", {}))
        identity.pop("source_program_fingerprint", None)
        payload["identity"] = identity
        return payload

    def compile_to_baseline(self) -> bool:
        """True when this plan must compile to unchanged baseline behavior.

        A plan compiles to baseline when it is abstained, oracle-only in
        production mode, or carries no actionable predicted/retrieved/merged
        structure.
        """
        if self.is_abstained:
            return True
        if self.is_oracle_only:
            return True
        has_actionable_structure = (
            self.role_slots
            or self.topology.parent_relation_candidates is not None
            or self.topology.sibling_order_groups is not None
            or self.symbols
            or self.bindings
            or self.coverage.named_requirements_accounted_for
        )
        return not has_actionable_structure
