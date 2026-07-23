"""Preregistered, content-addressed governance for autoresearch experiments."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from slm_training.autoresearch.schemas import CampaignBudget, StrictModel, utc_now
from slm_training.lineage.records import canonical_json

ClaimClass = Literal[
    "wiring",
    "fixture",
    "diagnostic",
    "screening",
    "promotion_candidate",
    "ship_gate",
]
_PROMOTION_ARTIFACT_KINDS = frozenset(
    {
        "version_stamp",
        "seed_result",
        "paired_examples",
        "endpoint_result",
        "holm_family",
        "agentevals",
        "agentv",
    }
)


def _unique(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} identifiers must be unique")


class CampaignEndpointV1(StrictModel):
    endpoint_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    role: Literal["primary", "secondary"]
    direction: Literal["increase", "decrease"]
    minimum_effect: float

    @model_validator(mode="after")
    def finite_effect(self) -> CampaignEndpointV1:
        if not math.isfinite(self.minimum_effect):
            raise ValueError("minimum_effect must be finite")
        return self


class CampaignArmV1(StrictModel):
    arm_id: str = Field(min_length=1)
    role: Literal["control", "candidate"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CampaignControlV1(StrictModel):
    control_id: str = Field(min_length=1)
    description: str = Field(min_length=8)
    kind: Literal["positive", "negative", "quality"]


class CampaignGateV1(StrictModel):
    gate_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    operator: Literal["ge", "gt", "le", "lt", "eq"]
    threshold: float

    @model_validator(mode="after")
    def finite_threshold(self) -> CampaignGateV1:
        if not math.isfinite(self.threshold):
            raise ValueError("gate threshold must be finite")
        return self


class MultiplicityFamilyV1(StrictModel):
    family_id: str = Field(min_length=1)
    hypothesis_ids: tuple[str, ...] = Field(min_length=1)
    alpha: float = Field(gt=0, lt=1)
    method: Literal["holm"] = "holm"

    @model_validator(mode="after")
    def unique_members(self) -> MultiplicityFamilyV1:
        _unique(self.hypothesis_ids, "multiplicity family")
        return self


class ArtifactRequirementV1(StrictModel):
    kind: Literal[
        "version_stamp",
        "seed_result",
        "paired_examples",
        "endpoint_result",
        "holm_family",
        "agentevals",
        "agentv",
        "ship_gates",
    ]
    minimum_count: int = Field(default=1, ge=1)


class AP001CertificationV1(StrictModel):
    disposition: Literal["certified", "revise", "blocked"]
    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExperimentCampaignV1(StrictModel):
    """Decision-bearing plan locked before any outcome is observed."""

    schema_version: Literal["ExperimentCampaignV1"] = "ExperimentCampaignV1"
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    hypothesis: str = Field(min_length=12)
    decision: str = Field(min_length=8)
    endpoints: tuple[CampaignEndpointV1, ...] = Field(min_length=1)
    arms: tuple[CampaignArmV1, ...] = Field(min_length=2)
    seeds: tuple[int, ...] = Field(min_length=1)
    budget: CampaignBudget
    stopping_rules: tuple[str, ...] = Field(min_length=1)
    controls: tuple[CampaignControlV1, ...] = Field(min_length=1)
    negative_controls: tuple[str, ...] = Field(min_length=1)
    multiplicity_families: tuple[MultiplicityFamilyV1, ...] = Field(min_length=1)
    promotion_gates: tuple[CampaignGateV1, ...] = Field(min_length=1)
    rollback_gates: tuple[CampaignGateV1, ...] = Field(min_length=1)
    artifact_requirements: tuple[ArtifactRequirementV1, ...] = Field(min_length=1)
    claim_class: ClaimClass
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    author: str = Field(min_length=1)
    created_at: str = Field(default_factory=utc_now)
    requires_rl: bool = False
    rl_readiness_report_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    rl_evaluation_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @field_validator("seeds", mode="before")
    @classmethod
    def strict_seed_identifiers(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in value
        ):
            raise TypeError("seeds must contain only integer identifiers")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> ExperimentCampaignV1:
        endpoint_ids = tuple(item.endpoint_id for item in self.endpoints)
        arm_ids = tuple(item.arm_id for item in self.arms)
        control_ids = tuple(item.control_id for item in self.controls)
        gate_ids = tuple(
            item.gate_id for item in (*self.promotion_gates, *self.rollback_gates)
        )
        requirement_kinds = tuple(item.kind for item in self.artifact_requirements)
        family_ids = tuple(item.family_id for item in self.multiplicity_families)
        for values, label in (
            (endpoint_ids, "endpoint"),
            (arm_ids, "arm"),
            (control_ids, "control"),
            (gate_ids, "gate"),
            (requirement_kinds, "artifact requirement"),
            (family_ids, "multiplicity family"),
        ):
            _unique(values, label)
        if sum(item.role == "primary" for item in self.endpoints) != 1:
            raise ValueError("exactly one primary endpoint is required")
        if not any(item.role == "control" for item in self.arms):
            raise ValueError("at least one control arm is required")
        if not any(item.role == "candidate" for item in self.arms):
            raise ValueError("at least one candidate arm is required")
        negative_control_ids = {
            item.control_id for item in self.controls if item.kind == "negative"
        }
        if set(self.negative_controls) != negative_control_ids:
            raise ValueError(
                "negative_controls must name every and only negative control"
            )
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise TypeError("seeds must contain only integer identifiers")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be unique")
        unknown_gate_endpoints = {
            gate.endpoint_id
            for gate in (*self.promotion_gates, *self.rollback_gates)
            if gate.endpoint_id not in endpoint_ids
        }
        if unknown_gate_endpoints:
            raise ValueError(
                f"gates reference unknown endpoints: {sorted(unknown_gate_endpoints)}"
            )
        declared_hypotheses = {
            hypothesis
            for family in self.multiplicity_families
            for hypothesis in family.hypothesis_ids
        }
        if not declared_hypotheses:
            raise ValueError("at least one multiplicity hypothesis is required")
        if sum(
            len(family.hypothesis_ids) for family in self.multiplicity_families
        ) != len(declared_hypotheses):
            raise ValueError("multiplicity hypotheses may belong to only one family")
        if not declared_hypotheses.issubset(set(endpoint_ids)):
            raise ValueError("multiplicity hypotheses must reference declared endpoints")
        if self.claim_class in {"promotion_candidate", "ship_gate"}:
            required_kinds = set(_PROMOTION_ARTIFACT_KINDS)
            if self.claim_class == "ship_gate":
                required_kinds.add("ship_gates")
            missing_kinds = required_kinds - set(requirement_kinds)
            if missing_kinds:
                raise ValueError(
                    f"promotion artifact requirements missing: {sorted(missing_kinds)}"
                )
        if self.requires_rl and (
            not self.rl_readiness_report_sha256 or not self.rl_evaluation_sha256
        ):
            raise ValueError(
                "RL campaigns require readiness-report and evaluation digests"
            )
        return self


class CampaignLockV1(StrictModel):
    schema_version: Literal["CampaignLockV1"] = "CampaignLockV1"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: ExperimentCampaignV1
    locked_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verify_digest(self) -> CampaignLockV1:
        validated = ExperimentCampaignV1.model_validate(
            self.manifest.model_dump(mode="json")
        )
        actual = campaign_manifest_sha256(validated)
        if actual != self.manifest_sha256:
            raise ValueError("campaign manifest digest mismatch")
        return self


class CampaignDeviationV1(StrictModel):
    schema_version: Literal["CampaignDeviationV1"] = "CampaignDeviationV1"
    campaign_id: str
    experiment_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_field: str = Field(min_length=1)
    old_value: Any
    new_value: Any
    reason: str = Field(min_length=8)
    author: str = Field(min_length=1)
    outcome_accessed: bool
    classification: Literal["exploratory"] = "exploratory"
    created_at: str = Field(default_factory=utc_now)


class CampaignArtifactV1(StrictModel):
    kind: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CampaignResultV1(StrictModel):
    campaign_id: str
    experiment_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_class: ClaimClass
    arm_seed_results: tuple[tuple[str, StrictInt], ...] = ()
    paired_example_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    endpoint_values: dict[str, float] = Field(default_factory=dict)
    holm_results: tuple[HolmResultV1, ...] = ()
    artifacts: tuple[CampaignArtifactV1, ...] = ()
    exploratory: bool = False
    ship_gates_passed: bool = False


class HolmResultV1(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    raw_p_value: float = Field(ge=0, le=1)
    rank: int = Field(ge=1)
    threshold: float = Field(gt=0, lt=1)
    adjusted_p_value: float = Field(ge=0, le=1)
    rejected: bool

    @model_validator(mode="after")
    def finite_statistics(self) -> HolmResultV1:
        values = (
            self.raw_p_value,
            self.threshold,
            self.adjusted_p_value,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Holm statistics must be finite")
        return self


def campaign_manifest_sha256(manifest: ExperimentCampaignV1) -> str:
    payload = manifest.model_dump(mode="json")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_ap001_certification(path: Path | None) -> AP001CertificationV1 | None:
    if path is None or not path.is_file():
        return None
    certification = AP001CertificationV1.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    artifact_path = (path.parent / certification.artifact_path).resolve()
    if not artifact_path.is_file():
        return None
    raw = artifact_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != certification.artifact_sha256:
        return None
    try:
        artifact = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if artifact.get("disposition") != certification.disposition:
        return None
    return certification


def select_primary_endpoint(
    certification: AP001CertificationV1 | None,
) -> Literal["binder_reference_f1", "binding_aware_meaningful_v2"]:
    if certification is not None and certification.disposition == "certified":
        return "binding_aware_meaningful_v2"
    return "binder_reference_f1"


def validate_result_claim(
    manifest: ExperimentCampaignV1,
    result: CampaignResultV1,
    *,
    artifact_root: Path | None = None,
) -> tuple[str, ...]:
    """Return fail-closed governance failures for a claimed result."""
    failures: list[str] = []
    expected_sha = campaign_manifest_sha256(manifest)
    if result.campaign_id != manifest.campaign_id:
        failures.append("campaign_id_mismatch")
    if result.experiment_id != manifest.experiment_id:
        failures.append("experiment_id_mismatch")
    if result.manifest_sha256 != expected_sha:
        failures.append("manifest_sha256_mismatch")
    if result.claim_class != manifest.claim_class:
        failures.append("claim_class_mismatch")
    if result.claim_class not in {"promotion_candidate", "ship_gate"}:
        return tuple(failures)
    if result.exploratory:
        failures.append("exploratory_result")

    expected_arm_seeds = {
        (arm.arm_id, seed) for arm in manifest.arms for seed in manifest.seeds
    }
    if (
        len(result.arm_seed_results) != len(set(result.arm_seed_results))
        or set(result.arm_seed_results) != expected_arm_seeds
    ):
        failures.append("incomplete_arm_seed_results")
    control_ids = {arm.arm_id for arm in manifest.arms if arm.role == "control"}
    candidate_ids = {arm.arm_id for arm in manifest.arms if arm.role == "candidate"}
    paired = result.paired_example_ids
    if (
        set(paired) != control_ids | candidate_ids
        or any(not ids or len(ids) != len(set(ids)) for ids in paired.values())
        or (paired and len({ids for ids in paired.values()}) != 1)
    ):
        failures.append("incomplete_paired_examples")
    if set(result.endpoint_values) != {
        item.endpoint_id for item in manifest.endpoints
    } or any(not math.isfinite(value) for value in result.endpoint_values.values()):
        failures.append("incomplete_endpoints")
    expected_holm = {
        item
        for family in manifest.multiplicity_families
        for item in family.hypothesis_ids
    }
    holm_ids = tuple(item.hypothesis_id for item in result.holm_results)
    ranks = tuple(item.rank for item in result.holm_results)
    if (
        len(holm_ids) != len(set(holm_ids))
        or set(holm_ids) != expected_holm
        or set(ranks) != set(range(1, len(expected_holm) + 1))
    ):
        failures.append("incomplete_holm_family")
    if any(
        not _gate_matches(gate, result.endpoint_values.get(gate.endpoint_id))
        for gate in manifest.promotion_gates
    ):
        failures.append("promotion_gates_not_passed")
    if any(
        _gate_matches(gate, result.endpoint_values.get(gate.endpoint_id))
        for gate in manifest.rollback_gates
    ):
        failures.append("rollback_gates_not_passed")
    artifact_counts: dict[str, int] = {}
    artifact_keys: set[tuple[str, str, str]] = set()
    for artifact in result.artifacts:
        key = (artifact.kind, artifact.uri, artifact.sha256)
        if key in artifact_keys:
            continue
        artifact_keys.add(key)
        artifact_counts[artifact.kind] = artifact_counts.get(artifact.kind, 0) + 1
    for requirement in manifest.artifact_requirements:
        if artifact_counts.get(requirement.kind, 0) < requirement.minimum_count:
            failures.append(f"missing_artifact:{requirement.kind}")
    if artifact_root is None:
        failures.append("artifact_root_missing")
    else:
        root = artifact_root.resolve()
        for artifact in result.artifacts:
            path = (root / artifact.uri).resolve()
            if root not in path.parents or not path.is_file():
                failures.append(f"artifact_unverified:{artifact.kind}")
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.sha256:
                failures.append(f"artifact_digest_mismatch:{artifact.kind}")
    if result.claim_class == "ship_gate" and not result.ship_gates_passed:
        failures.append("ship_gates_not_passed")
    return tuple(failures)


def _gate_matches(gate: CampaignGateV1, value: float | None) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return {
        "ge": value >= gate.threshold,
        "gt": value > gate.threshold,
        "le": value <= gate.threshold,
        "lt": value < gate.threshold,
        "eq": value == gate.threshold,
    }[gate.operator]
