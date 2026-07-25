"""Canonical machine-readable result object for abstract-planning experiments (AP-037 / SLM-338).

One ``AbstractPlanningResultV1`` is the single source of truth for a planning
experiment: provenance, the locked campaign manifest digest, raw / constrained /
repaired metric paths, plan controls, causal interventions, latency, verifier
gates, audit references, and the claim class. Publication is fail-closed:
``publication_blockers`` names every missing provenance, control, or latency
field before any JSON / Markdown / model-card artifact is emitted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from slm_training.autoresearch.experiment_campaign import ClaimClass
from slm_training.harness_core.evidence_bundle import ArtifactRef
from slm_training.lineage.records import canonical_json
from slm_training.versioning import build_version_stamp

SCHEMA_ID = "AbstractPlanningResultV1"
COMPONENT_ID = "harness.experiments.slm338_evidence_publisher"
UNKNOWN = "UNKNOWN"
PROMOTION_CLASSES: frozenset[str] = frozenset({"promotion_candidate", "ship_gate"})
CLAIM_CLASSES: frozenset[str] = frozenset(
    ("wiring", "fixture", "diagnostic", "screening", "promotion_candidate", "ship_gate")
)
METRIC_PATHS: tuple[str, ...] = ("raw", "constrained", "repaired")
CONTROL_KINDS: tuple[str, ...] = ("oracle", "random", "empty", "shuffled")
NEGATIVE_CONTROL_KINDS: frozenset[str] = frozenset({"random", "empty", "shuffled"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _is_known(value: str | None) -> bool:
    return bool(value) and value != UNKNOWN


@dataclass(frozen=True)
class MetricPathV1:
    """Metrics for one decode path (raw / constrained / repaired)."""

    path: str
    n: int
    meaning_v2: float | None = None
    binder_reference_f1: float | None = None
    parse_rate: float | None = None
    artifact: ArtifactRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n": self.n,
            "meaning_v2": self.meaning_v2,
            "binder_reference_f1": self.binder_reference_f1,
            "parse_rate": self.parse_rate,
            "artifact": self.artifact.to_dict() if self.artifact else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricPathV1":
        artifact = data.get("artifact")
        return cls(
            path=str(data["path"]),
            n=int(data["n"]),
            meaning_v2=data.get("meaning_v2"),
            binder_reference_f1=data.get("binder_reference_f1"),
            parse_rate=data.get("parse_rate"),
            artifact=ArtifactRef.from_dict(artifact) if artifact else None,
        )


@dataclass(frozen=True)
class PlanControlV1:
    """One plan control arm (oracle / random / empty / shuffled)."""

    kind: str
    n: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n": self.n,
            "metrics": dict(self.metrics),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanControlV1":
        return cls(
            kind=str(data["kind"]),
            n=int(data["n"]),
            metrics={
                str(k): float(v) for k, v in dict(data.get("metrics", {})).items()
            },
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class CausalInterventionV1:
    """One causal intervention and its measured effect."""

    intervention_id: str
    description: str
    metric: str
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CausalInterventionV1":
        return cls(
            intervention_id=str(data["intervention_id"]),
            description=str(data["description"]),
            metric=str(data["metric"]),
            delta=float(data["delta"]),
        )


@dataclass(frozen=True)
class LatencyBreakdownV1:
    """Latency / compute breakdown in seconds."""

    plan_seconds: float | None = None
    generation_seconds: float | None = None
    verification_seconds: float | None = None
    total_seconds: float | None = None
    p95_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LatencyBreakdownV1":
        return cls(
            plan_seconds=data.get("plan_seconds"),
            generation_seconds=data.get("generation_seconds"),
            verification_seconds=data.get("verification_seconds"),
            total_seconds=data.get("total_seconds"),
            p95_seconds=data.get("p95_seconds"),
        )


@dataclass(frozen=True)
class VerifierGateV1:
    """One verifier gate outcome."""

    gate_id: str
    passed: bool
    observed: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VerifierGateV1":
        return cls(
            gate_id=str(data["gate_id"]),
            passed=bool(data["passed"]),
            observed=data.get("observed"),
            threshold=data.get("threshold"),
        )


@dataclass(frozen=True)
class AuditReferenceV1:
    """Reference to an AgentV or human-audit bundle."""

    path: str
    n: int
    calibration_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditReferenceV1":
        return cls(
            path=str(data["path"]),
            n=int(data["n"]),
            calibration_version=str(data["calibration_version"]),
        )


@dataclass(frozen=True)
class AbstractPlanningResultV1:
    """Single source of truth for one abstract-planning experiment result."""

    campaign_id: str
    campaign_manifest_sha256: str
    claim_class: ClaimClass
    source_commit: str
    source_dirty: bool
    data_snapshot_sha256: str
    checkpoint: ArtifactRef | None = None
    metrics: tuple[MetricPathV1, ...] = ()
    controls: tuple[PlanControlV1, ...] = ()
    interventions: tuple[CausalInterventionV1, ...] = ()
    latency: LatencyBreakdownV1 = field(default_factory=LatencyBreakdownV1)
    verifier_gates: tuple[VerifierGateV1, ...] = ()
    agentv: AuditReferenceV1 | None = None
    human_audit: AuditReferenceV1 | None = None
    version_stamp: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_id": self.campaign_id,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "claim_class": self.claim_class,
            "source_commit": self.source_commit,
            "source_dirty": self.source_dirty,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "metrics": [item.to_dict() for item in self.metrics],
            "controls": [item.to_dict() for item in self.controls],
            "interventions": [item.to_dict() for item in self.interventions],
            "latency": self.latency.to_dict(),
            "verifier_gates": [item.to_dict() for item in self.verifier_gates],
            "agentv": self.agentv.to_dict() if self.agentv else None,
            "human_audit": self.human_audit.to_dict() if self.human_audit else None,
            "version_stamp": dict(self.version_stamp),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AbstractPlanningResultV1":
        if data.get("schema") != SCHEMA_ID:
            raise ValueError(
                f"expected schema {SCHEMA_ID!r}, got {data.get('schema')!r}"
            )
        checkpoint = data.get("checkpoint")
        agentv = data.get("agentv")
        human_audit = data.get("human_audit")
        return cls(
            campaign_id=str(data["campaign_id"]),
            campaign_manifest_sha256=str(data["campaign_manifest_sha256"]),
            claim_class=str(data["claim_class"]),  # type: ignore[arg-type]
            source_commit=str(data["source_commit"]),
            source_dirty=bool(data["source_dirty"]),
            data_snapshot_sha256=str(data["data_snapshot_sha256"]),
            checkpoint=ArtifactRef.from_dict(checkpoint) if checkpoint else None,
            metrics=tuple(
                MetricPathV1.from_dict(item) for item in data.get("metrics", ())
            ),
            controls=tuple(
                PlanControlV1.from_dict(item) for item in data.get("controls", ())
            ),
            interventions=tuple(
                CausalInterventionV1.from_dict(item)
                for item in data.get("interventions", ())
            ),
            latency=LatencyBreakdownV1.from_dict(data.get("latency", {})),
            verifier_gates=tuple(
                VerifierGateV1.from_dict(item)
                for item in data.get("verifier_gates", ())
            ),
            agentv=AuditReferenceV1.from_dict(agentv) if agentv else None,
            human_audit=AuditReferenceV1.from_dict(human_audit)
            if human_audit
            else None,
            version_stamp=dict(data.get("version_stamp", {})),
        )

    def content_sha256(self) -> str:
        """Content digest over the canonical payload, excluding the version stamp."""
        payload = self.to_dict()
        payload["version_stamp"] = {}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def metric_path(self, path: str) -> MetricPathV1 | None:
        for item in self.metrics:
            if item.path == path:
                return item
        return None


def stamp_version(result: AbstractPlanningResultV1) -> AbstractPlanningResultV1:
    """Return a copy carrying a fresh normalized ``version_stamp`` envelope."""
    return replace(
        result, version_stamp=build_version_stamp(COMPONENT_ID, "harness.core")
    )


def publication_blockers(result: AbstractPlanningResultV1) -> list[str]:
    """Fail-closed reasons this result may not be published (empty = publishable)."""
    blockers: list[str] = []
    if result.schema != SCHEMA_ID:
        blockers.append(f"schema_mismatch:{result.schema}")
    if result.claim_class not in CLAIM_CLASSES:
        blockers.append(f"unknown_claim_class:{result.claim_class}")
    if not _SHA256_RE.match(result.campaign_manifest_sha256 or ""):
        blockers.append("missing_campaign_manifest_sha256")
    if not _is_known(result.source_commit) or not _COMMIT_RE.match(
        result.source_commit
    ):
        blockers.append("missing_provenance:source_commit")
    if not _is_known(result.data_snapshot_sha256) or not _SHA256_RE.match(
        result.data_snapshot_sha256
    ):
        blockers.append("missing_provenance:data_snapshot_sha256")
    if result.latency.total_seconds is None or result.latency.total_seconds <= 0:
        blockers.append("missing_total_latency")
    if result.claim_class in PROMOTION_CLASSES:
        missing_paths = [
            path for path in METRIC_PATHS if result.metric_path(path) is None
        ]
        if missing_paths:
            blockers.append(f"missing_metric_paths:{','.join(missing_paths)}")
        control_kinds = {item.kind for item in result.controls}
        if not control_kinds & NEGATIVE_CONTROL_KINDS:
            blockers.append("missing_plan_controls")
        if result.checkpoint is None:
            blockers.append("missing_provenance:checkpoint")
    return blockers


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the narrative Markdown strictly from the canonical JSON payload."""
    lines: list[str] = [
        f"# Abstract planning result — `{payload['campaign_id']}`",
        "",
        f"- Schema: `{payload['schema']}`",
        f"- Claim class: **{payload['claim_class']}**",
        f"- Campaign manifest sha256: `{payload['campaign_manifest_sha256']}`",
        f"- Source commit: `{payload['source_commit']}` (dirty: {payload['source_dirty']})",
        f"- Data snapshot sha256: `{payload['data_snapshot_sha256']}`",
    ]
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        lines.append(
            f"- Checkpoint: `{checkpoint['name']}` sha256 `{checkpoint['sha256']}`"
        )
    lines += [
        "",
        "## Decode-path metrics",
        "",
        "| Path | n | meaning-v2 | binder F1 | parse |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in payload.get("metrics", ()):
        lines.append(
            f"| {item['path']} | {item['n']} | {_fmt(item.get('meaning_v2'))} "
            f"| {_fmt(item.get('binder_reference_f1'))} | {_fmt(item.get('parse_rate'))} |"
        )
    controls = payload.get("controls", ())
    if controls:
        lines += ["", "## Plan controls", ""]
        for control in controls:
            metrics = ", ".join(
                f"{name}={_fmt(value)}"
                for name, value in sorted(control["metrics"].items())
            )
            lines.append(
                f"- **{control['kind']}** (n={control['n']}): {metrics or 'no metrics'}"
            )
    interventions = payload.get("interventions", ())
    if interventions:
        lines += ["", "## Causal interventions", ""]
        for item in interventions:
            lines.append(
                f"- `{item['intervention_id']}` — {item['description']} "
                f"({item['metric']} delta {item['delta']:+.4f})"
            )
    latency = payload.get("latency", {})
    lines += [
        "",
        "## Latency / compute (seconds)",
        "",
        "| plan | generation | verification | total | p95 |",
        "| --- | --- | --- | --- | --- |",
        "| {plan} | {gen} | {ver} | {total} | {p95} |".format(
            plan=_fmt(latency.get("plan_seconds")),
            gen=_fmt(latency.get("generation_seconds")),
            ver=_fmt(latency.get("verification_seconds")),
            total=_fmt(latency.get("total_seconds")),
            p95=_fmt(latency.get("p95_seconds")),
        ),
    ]
    gates = payload.get("verifier_gates", ())
    if gates:
        lines += [
            "",
            "## Verifier gates",
            "",
            "| Gate | Observed | Threshold | Passed |",
            "| --- | --- | --- | --- |",
        ]
        for gate in gates:
            lines.append(
                f"| {gate['gate_id']} | {_fmt(gate.get('observed'))} "
                f"| {_fmt(gate.get('threshold'))} | **{gate['passed']}** |"
            )
    for label, key in (("AgentV", "agentv"), ("Human audit", "human_audit")):
        ref = payload.get(key)
        if ref:
            lines.append(
                f"\n- {label}: `{ref['path']}` (n={ref['n']}, calibration `{ref['calibration_version']}`)"
            )
    return "\n".join(lines) + "\n"


def render_model_card_row(payload: Mapping[str, Any], *, evidence_path: str) -> str:
    """Render one MODEL_CARD roster-style row from the canonical JSON payload."""
    constrained = next(
        (item for item in payload.get("metrics", ()) if item["path"] == "constrained"),
        None,
    )
    summary = "no constrained metrics"
    if constrained is not None:
        summary = (
            f"constrained meaning-v2 {_fmt(constrained.get('meaning_v2'))}, "
            f"binder F1 {_fmt(constrained.get('binder_reference_f1'))}, "
            f"parse {_fmt(constrained.get('parse_rate'))}"
        )
    honesty = "not promotable or ship"
    if payload["claim_class"] in PROMOTION_CLASSES:
        honesty = "promotion-class evidence; see verifier gates"
    return (
        f"| {payload['campaign_id']} abstract-planning result | `{payload['campaign_id']}` "
        f"| plan experiment evidence ({payload['claim_class']}) "
        f"| `{evidence_path}` "
        f"| {summary} — **{honesty}** |"
    )
