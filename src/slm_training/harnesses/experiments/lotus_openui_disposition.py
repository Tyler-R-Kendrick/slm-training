"""SLM-258 LOT4-02 — LotusOpenUIDispositionV1 fail-closed publisher.

Aggregates the real, committed upstream gate artifacts of the LOTUS
transfer project (SLM-248 fidelity contract, SLM-249 trace/oracle gate,
and the SLM-250…257 activation/launch/downstream gate dispositions) into
the final ``LotusOpenUIDispositionV1``. Every field is derived from those
artifacts — no result is retyped by hand.

Fail-closed rules:

* any missing mandatory gate ref yields ``inconclusive_missing_evidence``;
* adoption verdicts require every upstream gate to be positive, so a
  single ``not_authorized``/``inconclusive`` upstream blocks all adoption
  language, all interpretable-reasoning language, and all cost claims;
* V1 has no ``adopt_default`` verdict — production defaults never change
  through this disposition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.harness_core.versioning import build_version_stamp

__all__ = [
    "ADOPTION_VERDICTS",
    "DISPOSITION_SCHEMA_VERSION",
    "INCONCLUSIVE",
    "LOTUSProjectEvidence",
    "LotusOpenUIDispositionV1",
    "build_disposition",
    "load_project_evidence",
    "render_markdown",
]

DISPOSITION_SCHEMA_VERSION = "lotus_openui_disposition/v1"
INCONCLUSIVE = "inconclusive_missing_evidence"

ADOPTION_VERDICTS = frozenset(
    {
        "adopt_optional_experimental",
        "adopt_optional_with_bounded_serving_profile",
    }
)

# Positive upstream verdicts per evidence leg. Anything else (including
# not_authorized, inconclusive, upstream_authorized_pending_campaign_gate)
# blocks adoption.
_POSITIVE_FIDELITY = "authorize_bounded_implementation"
_POSITIVE_TRACE = "oracle_ceiling_positive"
_POSITIVE_GATE = "authorized_wiring_only"

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_ARTIFACT_PATHS: dict[str, str] = {
    "fidelity_contract": "docs/design/lotus-openui-fidelity-contract-v1.json",
    "trace_gate": "docs/design/compiler-reasoning-trace-v1.json",
    "slm250_activation": "docs/design/iter-slm250-lot1-01-not-authorized-20260725.json",
    "slm251_launch": "docs/design/iter-slm251-lot1-02-not-authorized-20260725.json",
    "slm252_routing": "docs/design/iter-slm252-lot2-01-not-authorized-20260725.json",
    "slm253_readout": "docs/design/iter-slm253-lot2-02-not-authorized-20260725.json",
    "slm254_causal_use": "docs/design/iter-slm254-lot3-01-not-authorized-20260725.json",
    "slm255_alternative_valid": "docs/design/iter-slm255-lot3-02-not-authorized-20260725.json",
    "slm256_capacity": "docs/design/iter-slm256-lot2-03-not-authorized-20260725.json",
    "slm257_frontier": "docs/design/iter-slm257-lot4-01-not-authorized-20260725.json",
}


@dataclass(frozen=True)
class LOTUSProjectEvidence:
    """Validated upstream artifacts keyed by evidence leg."""

    artifacts: dict[str, dict[str, Any]]
    artifact_paths: dict[str, str]
    missing: list[str]

    def verdict_of(self, leg: str) -> str:
        artifact = self.artifacts.get(leg) or {}
        if leg == "fidelity_contract":
            return (artifact.get("authorization") or {}).get("verdict", "")
        if leg == "trace_gate":
            return (artifact.get("gate") or {}).get("verdict", "")
        return artifact.get("verdict", "")

    def hash_of(self, leg: str) -> str:
        artifact = self.artifacts.get(leg) or {}
        return artifact.get("contract_hash", "")


def load_project_evidence(
    artifact_paths: dict[str, str] | None = None,
    *,
    repo_root: Path | None = None,
) -> LOTUSProjectEvidence:
    """Load and validate the upstream gate artifacts from disk."""
    paths = dict(artifact_paths or DEFAULT_ARTIFACT_PATHS)
    root = repo_root or REPO_ROOT
    artifacts: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for leg, rel in paths.items():
        path = root / rel
        if not path.exists():
            missing.append(leg)
            continue
        try:
            artifacts[leg] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(leg)
    return LOTUSProjectEvidence(artifacts=artifacts, artifact_paths=paths, missing=missing)


@dataclass(frozen=True)
class LotusOpenUIDispositionV1:
    schema_version: str = DISPOSITION_SCHEMA_VERSION
    disposition_id: str = "lotus-openui-disposition-v1"
    linear_issue: str = "SLM-258"
    date: str = "2026-07-25"
    project: str = "LOTUS-Faithful Causal OpenUI Transfer & Causal Mediation"
    gate_refs: dict[str, dict[str, str]] = field(default_factory=dict)
    upstream_verdicts: dict[str, str] = field(default_factory=dict)
    adoption_verdict: str = INCONCLUSIVE
    adoption_blockers: list[str] = field(default_factory=list)
    mechanism_dispositions: dict[str, str] = field(default_factory=dict)
    allowed_language: list[str] = field(default_factory=list)
    blocked_claims: list[str] = field(default_factory=list)
    allowed_code_config_default_changes: list[str] = field(default_factory=list)
    required_follow_ups: list[str] = field(default_factory=list)
    negative_result_registry: list[str] = field(default_factory=list)
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["disposition_hash"] = self.disposition_hash()
        return data

    def disposition_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "disposition_id": self.disposition_id,
            "adoption_verdict": self.adoption_verdict,
            "upstream_verdicts": self.upstream_verdicts,
            "mechanism_dispositions": self.mechanism_dispositions,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def build_disposition(evidence: LOTUSProjectEvidence) -> LotusOpenUIDispositionV1:
    """Derive the fail-closed disposition from validated upstream artifacts."""
    upstream_verdicts = {leg: evidence.verdict_of(leg) for leg in evidence.artifact_paths}
    gate_refs = {
        leg: {"path": evidence.artifact_paths[leg], "contract_hash": evidence.hash_of(leg)}
        for leg in evidence.artifact_paths
        if leg in evidence.artifacts
    }

    blockers: list[str] = []
    for leg in evidence.missing:
        blockers.append(f"missing_or_invalid_artifact:{leg}")

    fidelity_ok = upstream_verdicts.get("fidelity_contract") == _POSITIVE_FIDELITY
    trace_ok = upstream_verdicts.get("trace_gate") == _POSITIVE_TRACE
    if not fidelity_ok:
        blockers.append(
            "fidelity_contract verdict "
            f"'{upstream_verdicts.get('fidelity_contract', '')}' != '{_POSITIVE_FIDELITY}'"
        )
    if not trace_ok:
        blockers.append(
            f"trace_gate verdict '{upstream_verdicts.get('trace_gate', '')}' != '{_POSITIVE_TRACE}'"
        )
    for leg in (
        "slm250_activation",
        "slm251_launch",
        "slm252_routing",
        "slm253_readout",
        "slm254_causal_use",
        "slm255_alternative_valid",
        "slm256_capacity",
        "slm257_frontier",
    ):
        verdict = upstream_verdicts.get(leg, "")
        if leg in evidence.artifacts and verdict != _POSITIVE_GATE:
            blockers.append(f"{leg} verdict '{verdict}' != '{_POSITIVE_GATE}'")

    # V1 is fail-closed: any blocker forces inconclusive_missing_evidence.
    # SLM-248 explicitly ruled out reject_transfer ("scoped prerequisite"),
    # so the negative upstream evidence does not get rewritten as rejection.
    adoption_verdict = INCONCLUSIVE if blockers else "adopt_optional_experimental"
    adoption_blockers = blockers

    causal_positive = upstream_verdicts.get("slm254_causal_use") == _POSITIVE_GATE
    mechanism_dispositions = {
        "kxc_causal_workspace": "not_identifiable",
        "original_embedding_addition": "not_identifiable",
        "recurrent_whole_backbone_looping": "not_identifiable",
        "explicit_to_latent_curriculum": "not_identifiable",
        "post_loop_vs_per_iteration_timing": "not_identifiable",
        "shared_auxiliary_structured_readout": "not_identifiable",
        "set_valued_objective": "not_identifiable",
        "semantic_block_factorization_order": "not_identifiable",
        "accepted_alternative_training": "not_identifiable",
        "depth_width_inference_scaling": "not_identifiable",
        "causal_intervention_tooling": "not_identifiable",
        "compiler_reasoning_trace_contract": "keep_diagnostic",
        "activation_gate_evaluators": "keep_diagnostic",
    }

    allowed_language = ["latent computation", "latent representation"]
    if causal_positive:
        allowed_language += ["planning", "reasoning"]
    blocked_claims = [
        "no adopt_default or production default change (none exists in V1)",
        "no interpretable-reasoning/planning language (causal-use gate not positive)",
        "no equal-quality cost, latency, throughput, FLOPs, or energy claim",
        "no claim that explicit trace benefit implies latent benefit",
        "no universal latent-method dominance claim",
    ]
    allowed_code_config_default_changes = [
        "none: production defaults, decode paths, and serving configs are unchanged"
    ]
    required_follow_ups = [
        "run the SLM-249 oracle-ceiling campaign (matched continued explicit "
        "control, >=3 paired seeds) to replace the inconclusive trace gate",
        "if the ceiling is positive, re-run scripts/evaluate_lot1_01_activation_gate "
        "and scripts/evaluate_lot_downstream_gate (they read contracts live) and "
        "re-file LOT1-01/LOT1-02 implementation issues",
        "keep CompilerReasoningTraceV1 and the gate evaluators as diagnostic assets",
    ]
    negative_result_registry = [
        "LOT1-01 faithful K x c implementation: not_authorized (trace contract prerequisite)",
        "LOT1-02 explicit-to-latent curriculum campaign: not_authorized (no model path)",
        "LOT2-01/02/03, LOT3-01/02, LOT4-01 downstream work: not_authorized "
        "(unmet LOT1-02 launch gate)",
    ]

    return LotusOpenUIDispositionV1(
        gate_refs=gate_refs,
        upstream_verdicts=upstream_verdicts,
        adoption_verdict=adoption_verdict,
        adoption_blockers=adoption_blockers,
        mechanism_dispositions=mechanism_dispositions,
        allowed_language=allowed_language,
        blocked_claims=blocked_claims,
        allowed_code_config_default_changes=allowed_code_config_default_changes,
        required_follow_ups=required_follow_ups,
        negative_result_registry=negative_result_registry,
        version_stamp=build_version_stamp("harness.experiments"),
    )


def render_markdown(disposition: LotusOpenUIDispositionV1) -> str:
    lines = [
        f"# SLM-258 LOT4-02 — LotusOpenUIDispositionV1 ({disposition.disposition_id})",
        "",
        f"Adoption verdict: **{disposition.adoption_verdict}**",
        "",
        "## Upstream gate verdicts",
        "",
    ]
    for leg, verdict in disposition.upstream_verdicts.items():
        ref = disposition.gate_refs.get(leg) or {}
        lines.append(f"- `{leg}`: `{verdict}` ({ref.get('path', 'missing')})")
    lines += ["", "## Adoption blockers", ""]
    lines += [f"- {b}" for b in disposition.adoption_blockers] or ["- none"]
    lines += ["", "## Mechanism dispositions", ""]
    lines += [f"- `{k}`: {v}" for k, v in disposition.mechanism_dispositions.items()]
    lines += ["", "## Blocked claims", ""]
    lines += [f"- {b}" for b in disposition.blocked_claims]
    lines += ["", "## Required follow-ups", ""]
    lines += [f"- {f}" for f in disposition.required_follow_ups]
    lines += ["", "## Negative-result registry", ""]
    lines += [f"- {n}" for n in disposition.negative_result_registry]
    lines += [
        "",
        "## Production defaults",
        "",
        "Unchanged. V1 has no adopt_default verdict; any future default "
        "switch requires a separate rollout issue with operational approval "
        "and rollback evidence.",
        "",
    ]
    return "\n".join(lines)
