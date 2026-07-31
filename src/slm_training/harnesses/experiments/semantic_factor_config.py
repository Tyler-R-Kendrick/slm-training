"""Load versioned external SFF metrics / scorer-params / campaign schemas.

Harness code dispatches by formula ``type`` and arm id; parameter values and
metric inventories live under
``src/slm_training/resources/experiments/semantic_factor_frontier/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "SFF_RESOURCE_DIR",
    "SFFMetricsConfig",
    "SFFScorerParams",
    "SFFCampaignConfig",
    "load_sff_metrics",
    "load_sff_scorer_params",
    "load_sff_campaign",
    "SFFConfigError",
]

# Path(__file__) = .../harnesses/experiments/this.py → parents[2] = package root
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SFF_RESOURCE_DIR = (
    _PACKAGE_ROOT / "resources" / "experiments" / "semantic_factor_frontier"
)
# Repo root is two levels above the package (src/slm_training → repo).
_REPO_ROOT = _PACKAGE_ROOT.parents[1]

_METRICS_SCHEMA = "sff_metrics/v1"
_SCORER_SCHEMA = "sff_scorer_params/v1"
_CAMPAIGN_SCHEMA = "sff_campaign/v1"


class SFFConfigError(ValueError):
    """External SFF schema/resource is invalid or unsupported."""


def _repo_relative(path: Path) -> str:
    """Stable identity path for durable scoreboards (never absolute)."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resource_identity_fields(path: Path, schema: str, version: str) -> dict[str, str]:
    return {
        "path": _repo_relative(path),
        "schema": schema,
        "version": version,
        "sha256": _file_sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SFFConfigError(f"missing SFF resource: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SFFConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SFFConfigError(f"{path} root must be an object")
    return payload


def _require_schema(payload: Mapping[str, Any], expected: str, path: Path) -> None:
    schema = str(payload.get("schema") or "")
    version = str(payload.get("version") or "")
    if schema != expected:
        raise SFFConfigError(
            f"{path}: schema {schema!r} != required {expected!r}"
        )
    if not version:
        raise SFFConfigError(f"{path}: missing version field")


@dataclass(frozen=True)
class SFFMetricsConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def scoreboard_kind(self) -> str:
        return str(self.payload["scoreboard_kind"])

    @property
    def required_arm_fields(self) -> frozenset[str]:
        return frozenset(str(x) for x in self.payload["required_arm_fields"])

    @property
    def required_runtime_block(self) -> frozenset[str]:
        return frozenset(str(x) for x in self.payload["required_runtime_block"])

    @property
    def required_campaign_runtime_endpoints(self) -> frozenset[str]:
        return frozenset(
            str(x) for x in self.payload["required_campaign_runtime_endpoints"]
        )

    @property
    def endpoints(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(x) for x in self.payload["endpoints"])

    @property
    def aggregates(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(x) for x in self.payload["aggregates"])

    @property
    def derived(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(x) for x in self.payload["derived"])

    @property
    def wilson_z(self) -> float:
        return float(self.payload.get("wilson_z", 1.96))

    @property
    def timer(self) -> str:
        return str(self.payload.get("timer", "time.perf_counter"))

    @property
    def decision_path(self) -> str:
        return str(self.payload.get("decision_path", ""))

    @property
    def efficiency_gate(self) -> dict[str, Any]:
        return dict(self.payload.get("efficiency_gate") or {})

    def resource_identity(self) -> dict[str, str]:
        return _resource_identity_fields(self.path, self.schema, self.version)


@dataclass(frozen=True)
class SFFScorerParams:
    path: Path
    payload: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def defaults(self) -> dict[str, Any]:
        return dict(self.payload.get("defaults") or {})

    @property
    def arms(self) -> dict[str, dict[str, Any]]:
        raw = self.payload.get("arms") or {}
        return {str(k): dict(v) for k, v in raw.items()}

    @property
    def control_arm_id(self) -> str:
        return str(self.payload.get("control_arm_id") or "control_none")

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(self.arms.keys())

    def arm(self, arm_id: str) -> dict[str, Any]:
        if arm_id not in self.arms:
            raise SFFConfigError(f"unknown arm_id in scorer params: {arm_id}")
        return dict(self.arms[arm_id])

    def is_decode_on(self, arm_id: str) -> bool:
        prefixes = tuple(
            str(p) for p in self.payload.get("decode_on_arm_prefixes") or ()
        )
        ids = set(str(x) for x in self.payload.get("decode_on_arm_ids") or ())
        if arm_id in ids:
            return True
        return any(arm_id.startswith(p) for p in prefixes)

    def resource_identity(self) -> dict[str, str]:
        return _resource_identity_fields(self.path, self.schema, self.version)


@dataclass(frozen=True)
class SFFCampaignConfig:
    path: Path
    payload: dict[str, Any]
    metrics: SFFMetricsConfig
    scorer: SFFScorerParams

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    def resource_identity(self) -> dict[str, Any]:
        return {
            "campaign": _resource_identity_fields(
                self.path, self.schema, self.version
            ),
            "metrics": self.metrics.resource_identity(),
            "scorer_params": self.scorer.resource_identity(),
        }


@lru_cache(maxsize=4)
def load_sff_metrics(name: str = "metrics.v1.json") -> SFFMetricsConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _METRICS_SCHEMA, path)
    for key in (
        "scoreboard_kind",
        "required_arm_fields",
        "required_runtime_block",
        "aggregates",
        "derived",
        "endpoints",
        "required_campaign_runtime_endpoints",
    ):
        if key not in payload:
            raise SFFConfigError(f"{path}: missing {key}")
    return SFFMetricsConfig(path=path, payload=payload)


@lru_cache(maxsize=4)
def load_sff_scorer_params(name: str = "scorer_params.v1.json") -> SFFScorerParams:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _SCORER_SCHEMA, path)
    if "arms" not in payload or not payload["arms"]:
        raise SFFConfigError(f"{path}: arms must be non-empty")
    for arm_id, arm in payload["arms"].items():
        for req in ("representation", "decode_apply", "alpha"):
            if req not in arm:
                raise SFFConfigError(f"{path}: arm {arm_id} missing {req}")
    return SFFScorerParams(path=path, payload=payload)


@lru_cache(maxsize=4)
def load_sff_campaign(name: str = "campaign.v1.json") -> SFFCampaignConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _CAMPAIGN_SCHEMA, path)
    metrics_name = str(payload.get("metrics_resource") or "metrics.v1.json")
    scorer_name = str(payload.get("scorer_params_resource") or "scorer_params.v1.json")
    metrics = load_sff_metrics(metrics_name)
    scorer = load_sff_scorer_params(scorer_name)
    return SFFCampaignConfig(
        path=path, payload=payload, metrics=metrics, scorer=scorer
    )
