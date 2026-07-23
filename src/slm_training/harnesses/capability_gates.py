"""Fail-closed capability progression gates and immutable certificates."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from slm_training.harness_core.checkpoint_reference import CheckpointReferenceV1
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.staged import Capability

SCHEMA_VERSION = "capability_gate/v1"
_CAPABILITY_ORDER = tuple(Capability)
_CERTIFYING_RUN_CLASSES = frozenset({"ship_eval"})


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_artifact_type(value: Mapping[str, Any], expected: str) -> None:
    if value.get("artifact_type") != expected:
        raise ValueError(f"expected artifact_type {expected!r}")


class GateRunStatus(str, Enum):
    COMPLETED = "completed"
    DIAGNOSTIC = "diagnostic"
    INTERRUPTED = "interrupted"
    INVALID = "invalid"
    TIMEOUT = "timeout"


class PromotionAuthority(str, Enum):
    HUMAN = "human"
    CI = "ci"


@dataclass(frozen=True)
class ConfidenceThresholdV1:
    metric: str
    minimum_lower_bound: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("threshold metric is required")
        if not math.isfinite(self.minimum_lower_bound):
            raise ValueError("minimum_lower_bound must be finite")


@dataclass(frozen=True)
class ConfidenceBoundV1:
    metric: str
    lower_bound: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("confidence-bound metric is required")
        if not math.isfinite(self.lower_bound):
            raise ValueError("lower_bound must be finite")


@dataclass(frozen=True)
class RetentionResultV1:
    suite_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        _require_digest(self.suite_sha256, "retention suite sha256")
        if not isinstance(self.passed, bool):
            raise ValueError("retention passed must be a boolean")


@dataclass(frozen=True)
class CapabilityGateSpecV1:
    capability: Capability
    thresholds: tuple[ConfidenceThresholdV1, ...]
    retention_suite_hashes: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}")
        metrics = [item.metric for item in self.thresholds]
        if not metrics or len(metrics) != len(set(metrics)):
            raise ValueError("threshold metrics must be non-empty and unique")
        if len(self.retention_suite_hashes) != len(set(self.retention_suite_hashes)):
            raise ValueError("retention suite hashes must be unique")
        if (
            self.capability is not Capability.CAP0_GRAMMAR
            and not self.retention_suite_hashes
        ):
            raise ValueError("higher capability gates require retention suites")
        for value in self.retention_suite_hashes:
            _require_digest(value, "retention suite sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": "capability_gate_spec",
            "capability": self.capability.value,
            "thresholds": [asdict(value) for value in self.thresholds],
            "retention_suite_hashes": sorted(self.retention_suite_hashes),
        }

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityGateSpecV1:
        _require_artifact_type(value, "capability_gate_spec")
        return cls(
            schema_version=str(value["schema_version"]),
            capability=Capability(str(value["capability"])),
            thresholds=tuple(
                ConfidenceThresholdV1(
                    metric=str(item["metric"]),
                    minimum_lower_bound=float(item["minimum_lower_bound"]),
                )
                for item in value["thresholds"]
            ),
            retention_suite_hashes=tuple(
                str(item) for item in value.get("retention_suite_hashes", ())
            ),
        )

    @classmethod
    def load(cls, path: Path) -> CapabilityGateSpecV1:
        return cls.from_dict(_load_json(path))


@dataclass(frozen=True)
class CapabilityGateResultV1:
    capability: Capability
    gate_spec_sha256: str
    gate_implementation_sha256: str
    checkpoint_reference_sha256: str
    checkpoint_sha256: str
    dataset_sha256: str
    eval_suite_hashes: tuple[str, ...]
    code_sha256: str
    config_sha256: str
    run_class: str
    status: GateRunStatus
    confidence_bounds: tuple[ConfidenceBoundV1, ...]
    retention_results: tuple[RetentionResultV1, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}")
        for name in (
            "gate_spec_sha256",
            "gate_implementation_sha256",
            "checkpoint_reference_sha256",
            "checkpoint_sha256",
            "dataset_sha256",
            "code_sha256",
            "config_sha256",
        ):
            _require_digest(getattr(self, name), name)
        if not self.eval_suite_hashes:
            raise ValueError("eval_suite_hashes must not be empty")
        if len(self.eval_suite_hashes) != len(set(self.eval_suite_hashes)):
            raise ValueError("eval suite hashes must be unique")
        for value in self.eval_suite_hashes:
            _require_digest(value, "eval suite sha256")
        metrics = [item.metric for item in self.confidence_bounds]
        if len(metrics) != len(set(metrics)):
            raise ValueError("confidence-bound metrics must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_type": "capability_gate_result",
            "capability": self.capability.value,
            "gate_spec_sha256": self.gate_spec_sha256,
            "gate_implementation_sha256": self.gate_implementation_sha256,
            "checkpoint_reference_sha256": self.checkpoint_reference_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_sha256": self.dataset_sha256,
            "eval_suite_hashes": sorted(self.eval_suite_hashes),
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "run_class": self.run_class,
            "status": self.status.value,
            "confidence_bounds": [asdict(value) for value in self.confidence_bounds],
            "retention_results": [asdict(value) for value in self.retention_results],
        }

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityGateResultV1:
        _require_artifact_type(value, "capability_gate_result")
        return cls(
            schema_version=str(value["schema_version"]),
            capability=Capability(str(value["capability"])),
            gate_spec_sha256=str(value["gate_spec_sha256"]),
            gate_implementation_sha256=str(value["gate_implementation_sha256"]),
            checkpoint_reference_sha256=str(value["checkpoint_reference_sha256"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
            dataset_sha256=str(value["dataset_sha256"]),
            eval_suite_hashes=tuple(str(item) for item in value["eval_suite_hashes"]),
            code_sha256=str(value["code_sha256"]),
            config_sha256=str(value["config_sha256"]),
            run_class=str(value["run_class"]),
            status=GateRunStatus(str(value["status"])),
            confidence_bounds=tuple(
                ConfidenceBoundV1(
                    metric=str(item["metric"]), lower_bound=float(item["lower_bound"])
                )
                for item in value["confidence_bounds"]
            ),
            retention_results=tuple(
                RetentionResultV1(
                    suite_sha256=str(item["suite_sha256"]),
                    passed=_strict_bool(item["passed"], "retention passed"),
                )
                for item in value.get("retention_results", ())
            ),
        )

    @classmethod
    def load(cls, path: Path) -> CapabilityGateResultV1:
        return cls.from_dict(_load_json(path))


@dataclass(frozen=True)
class CapabilityCertificateV1:
    capability: Capability
    gate_spec_sha256: str
    gate_result_sha256: str
    gate_implementation_sha256: str
    checkpoint_reference_sha256: str
    checkpoint_sha256: str
    dataset_sha256: str
    eval_suite_hashes: tuple[str, ...]
    code_sha256: str
    config_sha256: str
    prior_certificate_ids: tuple[str, ...]
    distillation_allowed: bool
    promotion_authority: PromotionAuthority
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}")
        for name in (
            "gate_spec_sha256",
            "gate_result_sha256",
            "gate_implementation_sha256",
            "checkpoint_reference_sha256",
            "checkpoint_sha256",
            "dataset_sha256",
            "code_sha256",
            "config_sha256",
        ):
            _require_digest(getattr(self, name), name)
        for value in (*self.eval_suite_hashes, *self.prior_certificate_ids):
            _require_digest(value, "bound identity")
        if len(self.eval_suite_hashes) != len(set(self.eval_suite_hashes)):
            raise ValueError("eval suite hashes must be unique")
        if len(self.prior_certificate_ids) != len(set(self.prior_certificate_ids)):
            raise ValueError("prior certificate IDs must be unique")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "artifact_type": "capability_progression_certificate",
            "capability": self.capability.value,
            "gate_spec_sha256": self.gate_spec_sha256,
            "gate_result_sha256": self.gate_result_sha256,
            "gate_implementation_sha256": self.gate_implementation_sha256,
            "checkpoint_reference_sha256": self.checkpoint_reference_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_sha256": self.dataset_sha256,
            "eval_suite_hashes": sorted(self.eval_suite_hashes),
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "prior_certificate_ids": list(self.prior_certificate_ids),
            "distillation_allowed": self.distillation_allowed,
            "promotion_authority": self.promotion_authority.value,
        }
        if include_id:
            value["certificate_id"] = self.certificate_id
        return value

    @property
    def certificate_id(self) -> str:
        return content_sha(self.to_dict(include_id=False))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CapabilityCertificateV1:
        _require_artifact_type(value, "capability_progression_certificate")
        certificate = cls(
            schema_version=str(value["schema_version"]),
            capability=Capability(str(value["capability"])),
            gate_spec_sha256=str(value["gate_spec_sha256"]),
            gate_result_sha256=str(value["gate_result_sha256"]),
            gate_implementation_sha256=str(value["gate_implementation_sha256"]),
            checkpoint_reference_sha256=str(value["checkpoint_reference_sha256"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
            dataset_sha256=str(value["dataset_sha256"]),
            eval_suite_hashes=tuple(str(item) for item in value["eval_suite_hashes"]),
            code_sha256=str(value["code_sha256"]),
            config_sha256=str(value["config_sha256"]),
            prior_certificate_ids=tuple(
                str(item) for item in value.get("prior_certificate_ids", ())
            ),
            distillation_allowed=_strict_bool(
                value["distillation_allowed"], "distillation_allowed"
            ),
            promotion_authority=PromotionAuthority(str(value["promotion_authority"])),
        )
        claimed_id = value.get("certificate_id")
        if claimed_id is not None and str(claimed_id) != certificate.certificate_id:
            raise ValueError("certificate_id does not match immutable certificate body")
        return certificate

    @classmethod
    def load(cls, path: Path) -> CapabilityCertificateV1:
        return cls.from_dict(_load_json(path))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def required_prior_capabilities(capability: Capability) -> tuple[Capability, ...]:
    return _CAPABILITY_ORDER[: _CAPABILITY_ORDER.index(capability)]


def _ordered_priors(
    capability: Capability,
    priors: tuple[CapabilityCertificateV1, ...],
) -> tuple[CapabilityCertificateV1, ...]:
    by_capability = {certificate.capability: certificate for certificate in priors}
    if len(by_capability) != len(priors):
        raise ValueError("prior certificates must have unique capabilities")
    expected = required_prior_capabilities(capability)
    if set(by_capability) != set(expected):
        raise ValueError(
            "prior certificate capabilities must be exactly "
            + ", ".join(value.value for value in expected)
        )
    ordered = tuple(by_capability[value] for value in expected)
    for index, certificate in enumerate(ordered):
        expected_ids = tuple(item.certificate_id for item in ordered[:index])
        if certificate.prior_certificate_ids != expected_ids:
            raise ValueError(f"{certificate.capability.value} certificate chain is invalid")
    return ordered


def issue_certificate(
    spec: CapabilityGateSpecV1,
    result: CapabilityGateResultV1,
    checkpoint_reference: CheckpointReferenceV1,
    *,
    priors: tuple[CapabilityCertificateV1, ...] = (),
    authority: PromotionAuthority,
    distillation_allowed: bool = False,
) -> CapabilityCertificateV1:
    """Validate all evidence and return a content-addressed certificate."""

    if result.capability is not spec.capability:
        raise ValueError("gate result capability does not match gate spec")
    if result.gate_spec_sha256 != spec.sha:
        raise ValueError("gate result does not bind the supplied gate spec")
    if result.status is not GateRunStatus.COMPLETED:
        raise ValueError(f"{result.status.value} gate results cannot certify")
    if result.run_class not in _CERTIFYING_RUN_CLASSES:
        raise ValueError(f"{result.run_class!r} evidence cannot certify")
    if checkpoint_reference.claim_class in {"fixture", "diagnostic"}:
        raise ValueError("fixture or diagnostic checkpoints cannot certify")
    checkpoint_reference.require_publishable()
    if result.checkpoint_reference_sha256 != checkpoint_reference.sha:
        raise ValueError("gate result does not bind the supplied checkpoint reference")
    if result.checkpoint_sha256 != checkpoint_reference.sha256:
        raise ValueError("gate result checkpoint hash does not match its reference")
    if result.dataset_sha256 != checkpoint_reference.corpus_manifest_hash:
        raise ValueError("gate result dataset hash does not match its checkpoint")

    bounds = {item.metric: item.lower_bound for item in result.confidence_bounds}
    for threshold in spec.thresholds:
        if bounds.get(threshold.metric, float("-inf")) < threshold.minimum_lower_bound:
            raise ValueError(
                f"{threshold.metric} lower confidence bound misses threshold"
            )
    retention = {
        item.suite_sha256: item.passed for item in result.retention_results
    }
    if set(retention) != set(spec.retention_suite_hashes):
        raise ValueError("retention results do not exactly match the gate spec")
    if not all(retention.values()):
        raise ValueError("retention regression prevents certification")

    ordered_priors = _ordered_priors(spec.capability, priors)
    return CapabilityCertificateV1(
        capability=spec.capability,
        gate_spec_sha256=spec.sha,
        gate_result_sha256=result.sha,
        gate_implementation_sha256=result.gate_implementation_sha256,
        checkpoint_reference_sha256=result.checkpoint_reference_sha256,
        checkpoint_sha256=result.checkpoint_sha256,
        dataset_sha256=result.dataset_sha256,
        eval_suite_hashes=result.eval_suite_hashes,
        code_sha256=result.code_sha256,
        config_sha256=result.config_sha256,
        prior_certificate_ids=tuple(item.certificate_id for item in ordered_priors),
        distillation_allowed=distillation_allowed,
        promotion_authority=authority,
    )


def require_training_authorized(config: Any) -> None:
    """Fail before corpus or weight loading when a requested stage is unauthorized."""

    requested = getattr(config, "requested_capability", None)
    if requested is None:
        return
    capability = Capability(str(requested))
    paths = tuple(Path(path) for path in getattr(config, "capability_certificates", ()))
    certificates = tuple(CapabilityCertificateV1.load(path) for path in paths)
    ordered = _ordered_priors(capability, certificates)

    from slm_training.levers import require_capability_lever_profile

    require_capability_lever_profile(config, capability)
    if bool(getattr(config, "capability_distillation", False)):
        if not ordered or not ordered[-1].distillation_allowed:
            raise ValueError("distillation requires an explicit prior-certificate permission")
    _require_dataset_capability(config, capability)


def _require_dataset_capability(
    config: Any, requested: Capability
) -> None:
    plan_path = getattr(config, "capability_plan", None)
    if plan_path is None:
        raise ValueError("staged training requires an explicit capability plan")

    from slm_training.harnesses.synthesis_plan import SynthesisPlanV1

    plan = SynthesisPlanV1.load(Path(plan_path))
    manifest_path = Path(config.train_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("staged training requires a dataset manifest")
    manifest = _load_json(manifest_path)
    plan_value = manifest.get("synthesis_plan")
    if not isinstance(plan_value, dict):
        raise ValueError("staged training requires synthesis-plan manifest metadata")
    if plan_value.get("plan_id") != plan.plan_id or plan_value.get("sha256") != plan.sha:
        raise ValueError("dataset does not bind the supplied capability plan")
    if plan.capability is not requested:
        raise ValueError(
            f"dataset capability {plan.capability.value} does not match "
            f"requested {requested.value}"
        )
