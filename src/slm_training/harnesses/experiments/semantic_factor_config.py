"""Load versioned external SFF metrics / params / campaign / claims / suite schemas.

Harness code dispatches by formula ``type``, kill ``type``, and claim ``rule``;
parameter values and inventories live under
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
    "SFFClaimsConfig",
    "SFFSuiteConfig",
    "SFFProjectionConfig",
    "SFFMathProbesConfig",
    "SFFCampaignConfig",
    "load_sff_metrics",
    "load_sff_scorer_params",
    "load_sff_claims",
    "load_sff_suite",
    "load_sff_projection",
    "load_sff_math_probes",
    "load_sff_campaign",
    "SFFConfigError",
    "build_suite_from_config",
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SFF_RESOURCE_DIR = (
    _PACKAGE_ROOT / "resources" / "experiments" / "semantic_factor_frontier"
)
_REPO_ROOT = _PACKAGE_ROOT.parents[1]

_METRICS_SCHEMA = "sff_metrics/v1"
_SCORER_SCHEMA = "sff_scorer_params/v1"
_CAMPAIGN_SCHEMA = "sff_campaign/v1"
_CLAIMS_SCHEMA = "sff_claims/v1"
_SUITE_SCHEMA = "sff_suite/v1"
_PROJECTION_SCHEMA = "sff_projection/v1"
_MATH_SCHEMA = "sff_math_probes/v1"


class SFFConfigError(ValueError):
    """External SFF schema/resource is invalid or unsupported."""


def _repo_relative(path: Path) -> str:
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
        raise SFFConfigError(f"{path}: schema {schema!r} != required {expected!r}")
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

    @property
    def populations(self) -> dict[str, Any]:
        return dict(self.payload.get("populations") or {})

    @property
    def numerics(self) -> dict[str, float]:
        raw = self.payload.get("numerics") or {}
        return {str(k): float(v) for k, v in raw.items()}

    @property
    def promotion(self) -> dict[str, Any]:
        return dict(self.payload.get("promotion") or {})

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
class SFFClaimsConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def thresholds(self) -> dict[str, Any]:
        return dict(self.payload.get("thresholds") or {})

    @property
    def claims(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(c) for c in self.payload.get("claims") or ())

    def resource_identity(self) -> dict[str, str]:
        return _resource_identity_fields(self.path, self.schema, self.version)


@dataclass(frozen=True)
class SFFSuiteConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    def resource_identity(self) -> dict[str, str]:
        return _resource_identity_fields(self.path, self.schema, self.version)


@dataclass(frozen=True)
class SFFProjectionConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    def resource_identity(self) -> dict[str, str]:
        return _resource_identity_fields(self.path, self.schema, self.version)


@dataclass(frozen=True)
class SFFMathProbesConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    def resource_identity(self) -> dict[str, str]:
        return _resource_identity_fields(self.path, self.schema, self.version)


@dataclass(frozen=True)
class SFFCampaignConfig:
    path: Path
    payload: dict[str, Any]
    metrics: SFFMetricsConfig
    scorer: SFFScorerParams
    claims: SFFClaimsConfig
    suite: SFFSuiteConfig
    projection: SFFProjectionConfig
    math_probes: SFFMathProbesConfig

    @property
    def schema(self) -> str:
        return str(self.payload["schema"])

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.payload.get("seeds") or (0, 1, 2))

    @property
    def kill_rules(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(r) for r in self.payload.get("kill_rules") or ())

    @property
    def paired_comparisons(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(r) for r in self.payload.get("paired_comparisons") or ())

    def resource_identity(self) -> dict[str, Any]:
        return {
            "campaign": _resource_identity_fields(
                self.path, self.schema, self.version
            ),
            "metrics": self.metrics.resource_identity(),
            "scorer_params": self.scorer.resource_identity(),
            "claims": self.claims.resource_identity(),
            "suite": self.suite.resource_identity(),
            "projection": self.projection.resource_identity(),
            "math_probes": self.math_probes.resource_identity(),
        }


def build_suite_from_config(suite_cfg: SFFSuiteConfig) -> tuple[dict[str, Any], ...]:
    """Materialize the fixture suite from external suite.v1.json."""

    payload = suite_cfg.payload
    examples: list[dict[str, Any]] = []

    def _normalize(ex: Mapping[str, Any]) -> dict[str, Any]:
        comps = tuple(tuple(c) for c in ex.get("components") or ())
        edges = tuple(tuple(e) for e in ex.get("binder_edges") or ())
        legal = tuple(ex.get("legal") or ())
        return {
            "example_id": str(ex["example_id"]),
            "components": comps,
            "binder_edges": edges,
            "legal": legal,
            "coverage": str(ex.get("coverage") or "complete"),
            "gold": ex.get("gold"),
            "baseline": dict(ex.get("baseline") or {}),
            "family": str(ex.get("family") or "ranked"),
        }

    for ex in payload.get("static_examples") or []:
        examples.append(_normalize(ex))

    gen = dict(payload.get("ranked_generator") or {})
    n = int(gen.get("n") or 0)
    kinds = tuple(str(k) for k in gen.get("kinds") or ())
    if n > 0 and not kinds:
        raise SFFConfigError("ranked_generator.kinds required when n>0")
    baseline = dict(gen.get("baseline") or {})
    root_kind = str(gen.get("root_kind") or "Card")
    noise_kind = str(gen.get("noise_kind") or "List")
    binder_role = str(gen.get("binder_role") or "USE")
    every = int(gen.get("distractor_on_root_every") or 5)
    gold_b = float(baseline.get("gold", 0.10))
    dist_base = float(baseline.get("distractor_base", 0.40))
    dist_step = float(baseline.get("distractor_step", 0.01))
    dist_mod = int(baseline.get("distractor_mod", 3))
    noise_b = float(baseline.get("noise", 0.05))
    for i in range(n):
        gold = f"g{i}"
        distractor = f"d{i}"
        noise = f"n{i}"
        root = f"r{i}"
        kind = kinds[i % len(kinds)]
        bl = {
            gold: gold_b,
            distractor: dist_base + dist_step * (i % dist_mod),
            noise: noise_b,
        }
        edge2 = (
            (root, distractor, binder_role)
            if every > 0 and i % every == 0
            else (noise, distractor, binder_role)
        )
        examples.append(
            {
                "example_id": f"ranked_{i:02d}",
                "components": (
                    (root, root_kind),
                    (gold, kind),
                    (distractor, kinds[(i + 1) % len(kinds)]),
                    (noise, noise_kind),
                ),
                "binder_edges": ((root, gold, binder_role), edge2),
                "legal": (gold, distractor, noise),
                "coverage": "complete",
                "gold": gold,
                "baseline": bl,
                "family": "ranked",
            }
        )

    min_ranked = int(payload.get("min_ranked_design") or 0)
    ranked_fams = set(str(x) for x in payload.get("ranked_families") or ("ranked", "adversarial"))
    n_ranked = sum(1 for e in examples if e["family"] in ranked_fams)
    if min_ranked and n_ranked < min_ranked:
        raise SFFConfigError(
            f"suite materializes {n_ranked} ranked/adversarial examples < min_ranked_design {min_ranked}"
        )
    return tuple(examples)


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
def load_sff_claims(name: str = "claims.v1.json") -> SFFClaimsConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _CLAIMS_SCHEMA, path)
    if not payload.get("claims"):
        raise SFFConfigError(f"{path}: claims must be non-empty")
    for c in payload["claims"]:
        for req in ("claim_id", "statement", "kind", "rule"):
            if req not in c:
                raise SFFConfigError(f"{path}: claim missing {req}")
    return SFFClaimsConfig(path=path, payload=payload)


@lru_cache(maxsize=4)
def load_sff_suite(name: str = "suite.v1.json") -> SFFSuiteConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _SUITE_SCHEMA, path)
    return SFFSuiteConfig(path=path, payload=payload)


@lru_cache(maxsize=4)
def load_sff_projection(name: str = "projection.v1.json") -> SFFProjectionConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _PROJECTION_SCHEMA, path)
    return SFFProjectionConfig(path=path, payload=payload)


@lru_cache(maxsize=4)
def load_sff_math_probes(name: str = "math_probes.v1.json") -> SFFMathProbesConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _MATH_SCHEMA, path)
    return SFFMathProbesConfig(path=path, payload=payload)


@lru_cache(maxsize=4)
def load_sff_campaign(name: str = "campaign.v1.json") -> SFFCampaignConfig:
    path = SFF_RESOURCE_DIR / name
    payload = _read_json(path)
    _require_schema(payload, _CAMPAIGN_SCHEMA, path)
    metrics = load_sff_metrics(str(payload.get("metrics_resource") or "metrics.v1.json"))
    scorer = load_sff_scorer_params(
        str(payload.get("scorer_params_resource") or "scorer_params.v1.json")
    )
    claims = load_sff_claims(str(payload.get("claims_resource") or "claims.v1.json"))
    suite = load_sff_suite(str(payload.get("suite_resource") or "suite.v1.json"))
    projection = load_sff_projection(
        str(payload.get("projection_resource") or "projection.v1.json")
    )
    math_probes = load_sff_math_probes(
        str(payload.get("math_probes_resource") or "math_probes.v1.json")
    )
    return SFFCampaignConfig(
        path=path,
        payload=payload,
        metrics=metrics,
        scorer=scorer,
        claims=claims,
        suite=suite,
        projection=projection,
        math_probes=math_probes,
    )
