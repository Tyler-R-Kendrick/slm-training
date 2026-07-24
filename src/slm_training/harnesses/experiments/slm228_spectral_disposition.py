"""SLM-228 machine-auditable closeout for null-calibrated spectral work."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp, git_commit

REPORT_SCHEMA = "SpectralDispositionV1"
MATRIX_SET = "slm228_spectral_disposition"
MATRIX_VERSION = "ncs4-03-v1"
DEFAULT_JSON = "docs/design/null-calibrated-spectral-learning-disposition.json"
DEFAULT_MARKDOWN = "docs/design/null-calibrated-spectral-learning-disposition.md"
SEMANTIC_FLOOR_PATH = "docs/design/semantic-floor-gate-v1.json"
ABSOLUTE_SPECTRAL_RECIPE_KEYS = frozenset(
    {
        "alpha_target",
        "spectral_alpha_target",
        "ww_pgd",
        "ww_pgd_enabled",
        "trace_log",
        "trace_log_enabled",
        "spectral_projection",
        "spectral_projection_mode",
    }
)


class Disposition(str, Enum):
    ADOPT_DIAGNOSTIC = "adopt_diagnostic"
    ADOPT_OPTIONAL = "adopt_optional"
    ADOPT_PRIMARY = "adopt_primary"
    RETAIN_RESEARCH = "retain_research"
    REJECT = "reject"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class EvidenceReferenceV1:
    path: str
    content_sha256: str
    artifact_identity: str
    schema: str
    status: str
    claim_class: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralMechanismDispositionV1:
    mechanism_id: str
    category: str
    issue_ids: tuple[str, ...]
    fidelity: str
    applicable_scope: str
    disposition: Disposition
    default_state: str
    hypothesis: str
    falsifier: str
    rationale: str
    known_confounds: tuple[str, ...]
    required_action: str
    evidence: tuple[EvidenceReferenceV1, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issue_ids"] = list(self.issue_ids)
        payload["disposition"] = self.disposition.value
        payload["known_confounds"] = list(self.known_confounds)
        payload["evidence"] = [row.to_dict() for row in self.evidence]
        return payload


@dataclass(frozen=True)
class SpectralDispositionReportV1:
    evidence_cutoff_commit: str
    generated_at: str
    semantic_floor_hash: str
    semantic_floor_verdict: str
    regime_gate_hash: str
    absolute_gate_hash: str
    entries: tuple[SpectralMechanismDispositionV1, ...]
    executive_finding: str
    canonical_policy: str
    status: str = "complete"
    claim_class: str = "governance_disposition"
    schema: str = REPORT_SCHEMA
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    version_stamp: dict[str, Any] = field(default_factory=dict)

    @property
    def report_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        payload.pop("generated_at", None)
        stamp = dict(payload["version_stamp"])
        stamp.pop("stamped_at", None)
        payload["version_stamp"] = stamp
        return _sha256_json(payload)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "status": self.status,
            "claim_class": self.claim_class,
            "evidence_cutoff_commit": self.evidence_cutoff_commit,
            "generated_at": self.generated_at,
            "semantic_floor_hash": self.semantic_floor_hash,
            "semantic_floor_verdict": self.semantic_floor_verdict,
            "regime_gate_hash": self.regime_gate_hash,
            "absolute_gate_hash": self.absolute_gate_hash,
            "entries": [entry.to_dict() for entry in self.entries],
            "summary": disposition_summary(self.entries),
            "executive_finding": self.executive_finding,
            "canonical_policy": self.canonical_policy,
            "version_stamp": self.version_stamp,
        }
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return _sha256_bytes(encoded)


def _load_json(repo_root: Path, path: str) -> dict[str, Any]:
    return json.loads((repo_root / path).read_text(encoding="utf-8"))


def _evidence_reference(repo_root: Path, path: str) -> EvidenceReferenceV1:
    target = repo_root / path
    raw = target.read_bytes()
    payload = json.loads(raw)
    identity = next(
        (
            str(payload[key])
            for key in ("report_hash", "gate_hash", "artifact_hash")
            if payload.get(key)
        ),
        _sha256_bytes(raw),
    )
    return EvidenceReferenceV1(
        path=path,
        content_sha256=_sha256_bytes(raw),
        artifact_identity=identity,
        schema=str(payload.get("schema", "unknown")),
        status=str(
            payload.get("status")
            or payload.get("verdict")
            or payload.get("disposition")
            or "unspecified"
        ),
        claim_class=str(payload.get("claim_class", "unspecified")),
    )


def _specs() -> tuple[dict[str, Any], ...]:
    snapshot = "docs/design/iter-slm214-spectral-snapshot-20260721.json"
    atlas = "docs/design/iter-slm215-spectral-atlas-20260721.json"
    regime = "docs/design/iter-slm216-spectral-regime-20260723.json"
    functional = "docs/design/iter-slm217-functional-spectra-20260723.json"
    retention = "docs/design/iter-slm218-cross-attention-retention-20260723.json"
    traps = "docs/design/iter-slm219-correlation-trap-20260723.json"
    causal = "docs/design/iter-slm220-causal-subspace-20260723.json"
    muon = "docs/design/iter-slm222-muon-baseline-20260721.json"
    absolute = "docs/design/iter-slm226-absolute-spectral-gate-20260723.json"
    return (
        {
            "mechanism_id": "native_spectral_snapshot_and_null_cache",
            "category": "measurement",
            "issue_ids": ("SLM-214",),
            "fidelity": "adapted",
            "applicable_scope": "data-free matrix diagnostics by exact role, shape, dtype, and initializer",
            "disposition": Disposition.ADOPT_DIAGNOSTIC,
            "default_state": "diagnostic_only",
            "hypothesis": "Native SVD statistics with same-shape nulls separate synthetic controls without raw-alpha folklore.",
            "falsifier": "Synthetic controls do not separate, aliases double count, or null identity is missing.",
            "rationale": "The canonical owner validates deterministic native statistics, synthetic controls, null keys, role classification, and tied-storage deduplication. It makes no quality or ship claim.",
            "known_confounds": ("fixture model only", "no WeightWatcher production dependency"),
            "required_action": "Retain as the canonical inspection API; require null evidence beside raw alpha.",
            "paths": (snapshot,),
        },
        {
            "mechanism_id": "raw_alpha_as_quality_or_criticality_signal",
            "category": "measurement",
            "issue_ids": ("SLM-214", "SLM-215", "SLM-226"),
            "fidelity": "surrogate",
            "applicable_scope": "all roles and shapes",
            "disposition": Disposition.REJECT,
            "default_state": "forbidden",
            "hypothesis": "Raw fitted alpha, especially alpha near 2, identifies model quality or criticality.",
            "falsifier": "Same-shape random nulls reproduce or shift the apparent target.",
            "rationale": "The 128x128, 256x128, and 512x128 Gaussian-null means are strongly shape-dependent; proximity to 2 is not positive evidence.",
            "known_confounds": ("finite size", "aspect ratio", "initializer", "tail support", "estimator choice"),
            "required_action": "Reject raw-alpha promotion features and require null-calibrated observables.",
            "paths": (snapshot, atlas, absolute),
        },
        {
            "mechanism_id": "checkpoint_atlas_outcome_prediction",
            "category": "measurement",
            "issue_ids": ("SLM-215",),
            "fidelity": "adapted",
            "applicable_scope": "committed checkpoint families with exact compatible provenance",
            "disposition": Disposition.INCONCLUSIVE,
            "default_state": "off",
            "hypothesis": "Null-calibrated checkpoint spectra predict non-floor protected outcomes.",
            "falsifier": "Coverage is incomplete or predictive relations vanish under role/shape controls.",
            "rationale": "The atlas wiring works, but committed compatible checkpoint/outcome coverage is insufficient for prediction.",
            "known_confounds": ("missing checkpoints", "non-independent snapshots", "semantic floor"),
            "required_action": "Revisit only after a provenance-complete checkpoint family exists.",
            "paths": (atlas,),
        },
        {
            "mechanism_id": "fixed_token_spectral_regime_diagnostics",
            "category": "measurement",
            "issue_ids": ("SLM-216",),
            "fidelity": "adapted",
            "applicable_scope": "bounded CPU scratch batch/data-scale diagnostics",
            "disposition": Disposition.RETAIN_RESEARCH,
            "default_state": "diagnostic_only",
            "hypothesis": "Batch and data scale create reproducible null-calibrated spectral departures.",
            "falsifier": "Effects are step-confounded, null-like, or unavailable on durable checkpoints.",
            "rationale": "The measured scratch matrix is useful negative-boundary evidence but is inconclusive and blocks optimizer, semantic, promotion, and ship claims.",
            "known_confounds": ("optimizer-step count", "scratch model", "no durable checkpoint"),
            "required_action": "Keep the report replayable; do not route it into training defaults.",
            "paths": (regime,),
        },
        {
            "mechanism_id": "decision_conditioned_functional_spectra",
            "category": "functional_causal",
            "issue_ids": ("SLM-217",),
            "fidelity": "adapted",
            "applicable_scope": "exact-state activation covariance with compatible checkpoint manifests",
            "disposition": Disposition.INCONCLUSIVE,
            "default_state": "off",
            "hypothesis": "W Sigma^(1/2) spectra identify decision-kind-specific functional geometry.",
            "falsifier": "No compatible checkpoint/state manifest or controls do not separate.",
            "rationale": "Orientation and streaming covariance plumbing are validated only on fixtures; no compatible durable checkpoint plus DecisionEvent manifest exists.",
            "known_confounds": ("fixture-only", "missing exact-state checkpoint family"),
            "required_action": "Retain the diagnostic owner but make no predictive or causal inference.",
            "paths": (functional,),
        },
        {
            "mechanism_id": "cross_attention_and_parent_child_retention_geometry",
            "category": "functional_causal",
            "issue_ids": ("SLM-218",),
            "fidelity": "adapted",
            "applicable_scope": "provenance-complete parent/child checkpoint families",
            "disposition": Disposition.INCONCLUSIVE,
            "default_state": "off",
            "hypothesis": "Cross-attention bottlenecks or retained parent subspaces support protected decisions.",
            "falsifier": "No complete checkpoint family or controls match the proposed geometry.",
            "rationale": "Zero complete provenance-resolvable checkpoint families were available; no cross-attention role or retention target was nominated.",
            "known_confounds": ("missing parents", "synthetic geometry"),
            "required_action": "Do not select retention targets from synthetic magnitudes.",
            "paths": (retention,),
        },
        {
            "mechanism_id": "weightwatcher_stable_rank_parity",
            "category": "measurement",
            "issue_ids": ("SLM-219",),
            "fidelity": "faithful",
            "applicable_scope": "optional pinned WeightWatcher stable-rank parity diagnostics",
            "disposition": Disposition.ADOPT_DIAGNOSTIC,
            "default_state": "optional_diagnostic",
            "hypothesis": "Pinned WeightWatcher stable rank agrees with the native canonical owner.",
            "falsifier": "The two implementations disagree beyond numerical tolerance.",
            "rationale": "WeightWatcher 0.7.5 completed 18 comparisons with maximum stable-rank absolute error 8.527e-14. Its fitted alpha remains descriptive only.",
            "known_confounds": ("optional dependency", "alpha parity is not quality evidence"),
            "required_action": "Keep WeightWatcher pinned and optional; use native metrics as the canonical owner.",
            "paths": (traps,),
        },
        {
            "mechanism_id": "correlation_trap_early_warning",
            "category": "functional_causal",
            "issue_ids": ("SLM-219",),
            "fidelity": "adapted",
            "applicable_scope": "diagnostic checkpoint trajectories",
            "disposition": Disposition.REJECT,
            "default_state": "not_supported",
            "hypothesis": "Spikes/traps precede continuation collapse under a preregistered warning rule.",
            "falsifier": "Warnings do not lead protected regressions or controls warn equally.",
            "rationale": "One seed/family and one transient collapse provide no independent non-collapse false-positive denominator; the report explicitly does not support an operational early-stop rule.",
            "known_confounds": ("small trajectory", "proxy collapse labels"),
            "required_action": "Retain the report for research replay; forbid operational early stopping or promotion decisions.",
            "paths": (traps,),
        },
        {
            "mechanism_id": "activation_side_causal_restriction_estimator",
            "category": "functional_causal",
            "issue_ids": ("SLM-220",),
            "fidelity": "adapted",
            "applicable_scope": "analytic fixtures and future exact-state checkpoint families",
            "disposition": Disposition.ADOPT_DIAGNOSTIC,
            "default_state": "diagnostic_only",
            "hypothesis": "Activation-side restriction energy measures causal support in the mathematically valid orientation.",
            "falsifier": "Exact/JVP estimators disagree or null controls fail.",
            "rationale": "Analytic exact/JVP/Hutchinson contracts pass, but the current-model retrospective is rejected and nominates no eligible bands.",
            "known_confounds": ("analytic labels are not model evidence", "semantic floor"),
            "required_action": "Keep the estimator; require a provenance-complete family before model use.",
            "paths": (causal,),
        },
        {
            "mechanism_id": "isospectral_and_band_causal_hypothesis",
            "category": "functional_causal",
            "issue_ids": ("SLM-221",),
            "fidelity": "adapted",
            "applicable_scope": "frozen checkpoint and preregistered eligible bands",
            "disposition": Disposition.BLOCKED,
            "default_state": "blocked",
            "hypothesis": "Singular values, vectors, or bands have reproducible local causal effects.",
            "falsifier": "Perturbations are inert after controls.",
            "rationale": "No eligible perturbation bands or provenance-resolvable frozen checkpoint/state family exists, so the battery could not start.",
            "known_confounds": ("no target artifact", "no source checkpoint"),
            "required_action": "Reopen only after the causal estimator nominates nonempty frozen targets.",
            "paths": (causal,),
        },
        {
            "mechanism_id": "muon_hybrid_optimizer",
            "category": "optimization_retention",
            "issue_ids": ("SLM-222",),
            "fidelity": "surrogate",
            "applicable_scope": "optimizer partition wiring fixture",
            "disposition": Disposition.RETAIN_RESEARCH,
            "default_state": "off",
            "hypothesis": "Muon improves convergence/protected outcomes over matched AdamW.",
            "falsifier": "Matched quality/cost outcomes do not improve.",
            "rationale": "Only a two-step fixture validates partitioning; no matched convergence, protected, downstream, or GPU evidence exists.",
            "known_confounds": ("single synthetic record", "unmatched downstream evidence"),
            "required_action": "Keep optional wiring default-off; do not call it the strongest baseline.",
            "paths": (muon,),
        },
        {
            "mechanism_id": "relative_spectral_optimizer_tournament",
            "category": "optimization_retention",
            "issue_ids": ("SLM-223",),
            "fidelity": "adapted",
            "applicable_scope": "roles authorized by SpectralRegimeGateV1",
            "disposition": Disposition.BLOCKED,
            "default_state": "blocked",
            "hypothesis": "Null-calibrated relative LR/decay ordering beats permuted and time-shifted controls.",
            "falsifier": "Ordering is unsupported or controls match.",
            "rationale": "The regime gate is inconclusive, no causal mechanism exists, and no qualified AdamW/Muon control exists; the tournament was not authorized.",
            "known_confounds": ("no durable checkpoint", "no matched baseline"),
            "required_action": "Do not add a controller until a future versioned gate authorizes exact roles.",
            "paths": (regime, causal, muon),
        },
        {
            "mechanism_id": "verifier_conditioned_spectral_lr",
            "category": "optimization_retention",
            "issue_ids": ("SLM-224",),
            "fidelity": "adapted",
            "applicable_scope": "winning relative spectral policy with protected geometry",
            "disposition": Disposition.REJECT,
            "default_state": "not_applicable",
            "hypothesis": "Verifier need and conflict gates make spectral LR allocation safer.",
            "falsifier": "No supported spectral ordering or mechanism remains.",
            "rationale": "SLM-223 supplied no winning spectral direction or controller state, so layering verifier policy was explicitly not applicable.",
            "known_confounds": ("missing base mechanism",),
            "required_action": "No implementation; reopen only after qualified spectral ordering.",
            "paths": (regime, causal),
        },
        {
            "mechanism_id": "causal_spectral_elastic_retention",
            "category": "optimization_retention",
            "issue_ids": ("SLM-225",),
            "fidelity": "adapted",
            "applicable_scope": "causally supported frozen singular-value/subspace targets",
            "disposition": Disposition.BLOCKED,
            "default_state": "not_supported",
            "hypothesis": "Causal spectral retention improves the plasticity/retention frontier.",
            "falsifier": "No reproducible singular-value, orientation, or band effect exists.",
            "rationale": "SLM-221 produced no eligible causal target or frozen weights; implementing a penalty would select from synthetic magnitude or outcomes.",
            "known_confounds": ("no target artifact",),
            "required_action": "Do not add retention code without a future causal result.",
            "paths": (retention, causal),
        },
        {
            "mechanism_id": "absolute_spectral_target_gate",
            "category": "absolute_targeting",
            "issue_ids": ("SLM-226",),
            "fidelity": "adapted",
            "applicable_scope": "exact role/shape/initializer finite-size authorization",
            "disposition": Disposition.ADOPT_DIAGNOSTIC,
            "default_state": "fail_closed",
            "hypothesis": "A versioned finite-size gate prevents null coincidences from authorizing correction.",
            "falsifier": "Raw alpha or one checkpoint can authorize a different role/shape.",
            "rationale": "The 200-draw width study is descriptive-only, authorizes no roles/shapes, and its guard blocks every absolute intervention.",
            "known_confounds": ("scratch linear probes", "no production width boundary"),
            "required_action": "Retain the gate and require it before any absolute-target manifest.",
            "paths": (absolute,),
        },
        {
            "mechanism_id": "ww_pgd_trace_log_projection",
            "category": "absolute_targeting",
            "issue_ids": ("SLM-227",),
            "fidelity": "adapted",
            "applicable_scope": "exact role/shape authorized by the absolute gate and causal target",
            "disposition": Disposition.BLOCKED,
            "default_state": "not_authorized",
            "hypothesis": "Damped alpha/trace-log projection improves protected quality beyond controls.",
            "falsifier": "The absolute gate or causal-shape prerequisite fails.",
            "rationale": "AbsoluteSpectralTargetGateV1 is descriptive-only with empty authorization and SLM-221 has no causal target; projection is not authorized.",
            "known_confounds": ("finite size", "norm changes", "SVD overhead", "guard selection"),
            "required_action": "No projection implementation; the absolute gate remains authoritative.",
            "paths": (absolute, causal, muon),
        },
    )


def build_entries(repo_root: Path) -> tuple[SpectralMechanismDispositionV1, ...]:
    entries: list[SpectralMechanismDispositionV1] = []
    for spec in _specs():
        evidence = tuple(_evidence_reference(repo_root, path) for path in spec["paths"])
        entries.append(
            SpectralMechanismDispositionV1(
                mechanism_id=spec["mechanism_id"],
                category=spec["category"],
                issue_ids=spec["issue_ids"],
                fidelity=spec["fidelity"],
                applicable_scope=spec["applicable_scope"],
                disposition=spec["disposition"],
                default_state=spec["default_state"],
                hypothesis=spec["hypothesis"],
                falsifier=spec["falsifier"],
                rationale=spec["rationale"],
                known_confounds=spec["known_confounds"],
                required_action=spec["required_action"],
                evidence=evidence,
            )
        )
    return tuple(entries)


def disposition_summary(
    entries: tuple[SpectralMechanismDispositionV1, ...],
) -> dict[str, int]:
    return {
        disposition.value: sum(entry.disposition == disposition for entry in entries)
        for disposition in Disposition
    }


def validate_report(report: SpectralDispositionReportV1, repo_root: Path) -> list[str]:
    errors: list[str] = []
    ids = [entry.mechanism_id for entry in report.entries]
    if len(ids) != len(set(ids)):
        errors.append("mechanism ids must be unique")
    required = {
        "native_spectral_snapshot_and_null_cache",
        "raw_alpha_as_quality_or_criticality_signal",
        "activation_side_causal_restriction_estimator",
        "relative_spectral_optimizer_tournament",
        "causal_spectral_elastic_retention",
        "absolute_spectral_target_gate",
        "ww_pgd_trace_log_projection",
    }
    missing = sorted(required - set(ids))
    if missing:
        errors.append(f"required mechanisms missing: {', '.join(missing)}")
    by_id = {entry.mechanism_id: entry for entry in report.entries}
    if by_id.get("raw_alpha_as_quality_or_criticality_signal", None) is None or (
        by_id["raw_alpha_as_quality_or_criticality_signal"].disposition
        != Disposition.REJECT
    ):
        errors.append("raw alpha without null evidence must be rejected")
    projection = by_id.get("ww_pgd_trace_log_projection")
    if projection is None or projection.disposition not in {
        Disposition.REJECT,
        Disposition.BLOCKED,
    }:
        errors.append("unauthorized absolute projection must fail closed")
    for entry in report.entries:
        if entry.fidelity not in {"faithful", "adapted", "surrogate", "adjacent"}:
            errors.append(f"{entry.mechanism_id}: unknown fidelity {entry.fidelity}")
        if entry.default_state in {"on", "production", "champion"} and (
            entry.disposition != Disposition.ADOPT_PRIMARY
        ):
            errors.append(f"{entry.mechanism_id}: non-primary mechanism cannot default on")
        for evidence in entry.evidence:
            path = repo_root / evidence.path
            if not path.is_file():
                errors.append(f"{entry.mechanism_id}: missing evidence {evidence.path}")
            elif _sha256_bytes(path.read_bytes()) != evidence.content_sha256:
                errors.append(f"{entry.mechanism_id}: stale evidence hash {evidence.path}")
    if report.semantic_floor_verdict != "floor_escaped":
        for entry in report.entries:
            if entry.disposition in {
                Disposition.ADOPT_OPTIONAL,
                Disposition.ADOPT_PRIMARY,
            }:
                errors.append(
                    f"{entry.mechanism_id}: adoption exceeds inconclusive semantic floor"
                )
    return errors


def _recipe_value_active(value: Any) -> bool:
    if value is None or value is False or value == 0:
        return False
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "off",
        "none",
        "false",
        "disabled",
    }:
        return False
    return True


def validate_spectral_recipe(
    recipe: dict[str, Any],
    *,
    requested_use: str,
) -> None:
    """Fail closed on rejected spectral recipe fields at lineage boundaries."""
    if requested_use not in {"manifest", "scratch", "ship_eval", "promotion"}:
        raise ValueError(f"unknown spectral recipe use: {requested_use}")
    active_absolute = sorted(
        key
        for key in ABSOLUTE_SPECTRAL_RECIPE_KEYS
        if key in recipe and _recipe_value_active(recipe[key])
    )
    if active_absolute:
        raise ValueError(
            "absolute spectral mechanisms are not authorized by "
            "SpectralDispositionV1: "
            + ", ".join(active_absolute)
        )
    if requested_use in {"ship_eval", "promotion"} and (
        str(recipe.get("optimizer_name", "adamw")) == "muon_hybrid"
    ):
        raise ValueError(
            "muon_hybrid is fixture/research-only under SpectralDispositionV1 "
            f"and cannot enter {requested_use}"
        )


def require_spectral_disposition(
    report: SpectralDispositionReportV1,
    *,
    mechanism_id: str,
    requested_use: str,
) -> None:
    """Authorize only uses allowed by the final disposition."""
    by_id = {entry.mechanism_id: entry for entry in report.entries}
    if mechanism_id not in by_id:
        raise KeyError(f"unknown spectral mechanism: {mechanism_id}")
    entry = by_id[mechanism_id]
    allowed = {
        "diagnostic": {
            Disposition.ADOPT_DIAGNOSTIC,
            Disposition.RETAIN_RESEARCH,
        },
        "training": {
            Disposition.ADOPT_OPTIONAL,
            Disposition.ADOPT_PRIMARY,
        },
        "promotion": {Disposition.ADOPT_PRIMARY},
        "production": {Disposition.ADOPT_PRIMARY},
    }
    if requested_use not in allowed:
        raise ValueError(f"unknown requested spectral use: {requested_use}")
    if entry.disposition not in allowed[requested_use]:
        raise RuntimeError(
            f"{mechanism_id} cannot be used for {requested_use}: "
            f"disposition is {entry.disposition.value}"
        )


def build_report(repo_root: Path) -> SpectralDispositionReportV1:
    semantic = _load_json(repo_root, SEMANTIC_FLOOR_PATH)
    regime = _load_json(
        repo_root, "docs/design/iter-slm216-spectral-regime-20260723.json"
    )
    absolute = _load_json(
        repo_root,
        "docs/design/iter-slm226-absolute-spectral-gate-20260723.json",
    )
    report = SpectralDispositionReportV1(
        evidence_cutoff_commit=git_commit(),
        generated_at=_now(),
        semantic_floor_hash=str(semantic["gate_hash"]),
        semantic_floor_verdict=str(semantic["verdict"]),
        regime_gate_hash=str(regime["report_hash"]),
        absolute_gate_hash=str(absolute["report_hash"]),
        entries=build_entries(repo_root),
        executive_finding=(
            "The program adopts four fail-closed diagnostics and no training or "
            "production mechanism. Raw alpha, verifier-conditioned spectral LR, "
            "causal spectral retention, and WW-PGD/trace-log correction are rejected "
            "or blocked in the measured scope; negative evidence remains replayable."
        ),
        canonical_policy=(
            "Keep SpectralSnapshotV1, the analytic activation-side estimator, and "
            "AbsoluteSpectralTargetGateV1 as diagnostic/governance owners. Spectral "
            "observables cannot influence champion selection, promotion, or production "
            "configuration unless a future versioned disposition reaches adopt_primary."
        ),
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm228_spectral_disposition",
        ),
    )
    errors = validate_report(report, repo_root)
    if errors:
        raise ValueError("; ".join(errors))
    return report


def render_markdown(report: SpectralDispositionReportV1) -> str:
    summary = disposition_summary(report.entries)
    lines = [
        "# Null-calibrated spectral learning disposition",
        "",
        f"**Schema:** `{report.schema}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Evidence cutoff:** `{report.evidence_cutoff_commit}`",
        "",
        f"**Semantic floor:** `{report.semantic_floor_verdict}` "
        f"(`{report.semantic_floor_hash}`)",
        "",
        "## Executive finding",
        "",
        report.executive_finding,
        "",
        "This is an evidence synthesis, not a new train/eval/profile run. It writes "
        "no checkpoint and makes no AgentV, promotion, or ship claim.",
        "",
        "## Primary sources and fidelity boundary",
        "",
        "- [Yang et al., Heavy-Tailed Self-Regularization in Deep Neural Networks "
        "(JMLR 2021)](https://jmlr.org/papers/v22/20-410.html)",
        "- [Martin and Mahoney, Implicit self-regularization in deep neural "
        "networks (Nature Communications 2021)]"
        "(https://www.nature.com/articles/s41467-021-24025-8)",
        "- [WeightWatcher spectral RG draft]"
        "(https://weightwatcher.ai/rg_theory_webpage/rg_theory.html)",
        "",
        "Repository rows label each use as faithful, adapted, or surrogate. The "
        "sources motivate measurements and hypotheses; they do not supply "
        "repository checkpoint, causal, protected-objective, or ship evidence.",
        "",
        "## Mechanism table",
        "",
        "| Category | Mechanism | Issues | Fidelity | Disposition | Default | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report.entries:
        evidence = ", ".join(f"`{row.path}`" for row in entry.evidence)
        lines.append(
            f"| {entry.category} | `{entry.mechanism_id}` | "
            f"{', '.join(entry.issue_ids)} | {entry.fidelity} | "
            f"**{entry.disposition.value}** | `{entry.default_state}` | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Disposition counts",
            "",
            *[
                f"- `{name}`: {count}"
                for name, count in summary.items()
                if count
            ],
            "",
            "## Finite-size and raw-alpha boundary",
            "",
            "The 200-draw Gaussian-null mean alpha changes from `2.273991` at "
            "`128x128` to `3.468501` at `256x128` and `4.973008` at `512x128`. "
            "Raw alpha and alpha near 2 are therefore rejected as quality, "
            "criticality, champion, or promotion evidence.",
            "",
            "## Functional and causal geometry",
            "",
            "Functional spectra, cross-attention/retention geometry, and correlation "
            "traps remain fixture or research diagnostics. The activation-side "
            "restriction estimator is adopted only as an analytic diagnostic. No "
            "eligible model band or frozen causal target exists.",
            "",
            "## Optimization and retention",
            "",
            "Muon is wiring-only; the relative-control tournament was blocked; "
            "verifier-conditioned allocation and causal spectral retention were "
            "rejected or blocked because their prerequisite mechanism/target did "
            "not exist. "
            "No spectral training default changes.",
            "",
            "## Quality, protected, and cost frontier",
            "",
            "No spectral training campaign cleared its activation gates, so there is "
            "no qualified quality/protected/cost Pareto frontier. Muon has only "
            "two-step wiring evidence; all SVD, guard, checkpoint, and target-hardware "
            "cost claims remain unavailable rather than inferred.",
            "",
            "## Absolute targeting",
            "",
            "AbsoluteSpectralTargetGateV1 is adopted as a fail-closed diagnostic "
            "gate with zero authorized roles/shapes. WW-PGD, trace-log projection, "
            "and absolute alpha targeting remain `not_authorized`.",
            "",
            "## Canonical policy",
            "",
            report.canonical_policy,
            "",
            "## Evidence identities and non-independence",
            "",
        ]
    )
    seen: set[str] = set()
    for entry in report.entries:
        for evidence in entry.evidence:
            if evidence.path in seen:
                continue
            seen.add(evidence.path)
            lines.append(
                f"- `{evidence.path}` — artifact `{evidence.artifact_identity}`, "
                f"file SHA-256 `{evidence.content_sha256}`"
            )
    lines.extend(
        [
            "",
            "SLM-214 is the shared spectral-statistics owner for SLM-215–220 and "
            "SLM-226; those rows are not independent replications. SLM-217 feeds "
            "SLM-218/220, while SLM-216, SLM-220, and SLM-222 jointly gate "
            "SLM-223–227. Downstream closures therefore preserve prerequisite "
            "missingness instead of counting it as repeated negative experiments.",
            "",
            "## Per-mechanism rationale and actions",
            "",
        ]
    )
    for entry in report.entries:
        lines.extend(
            [
                f"### `{entry.mechanism_id}`",
                "",
                entry.rationale,
                "",
                f"Required action: {entry.required_action}",
                "",
                "Known confounds: "
                + (
                    "; ".join(entry.known_confounds)
                    if entry.known_confounds
                    else "none recorded"
                )
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src "
            "/home/codex/repos/slm-training/.venv/bin/python "
            "-m scripts.publish_spectral_disposition --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
