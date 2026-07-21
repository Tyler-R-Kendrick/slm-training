"""SLM-186 (FFE0-04): verified-utility ladder and Goodhart canary wiring fixture.

Deterministic, CPU-only harness that exercises the multi-factor
``VerifiedUtilityV1`` schema, scalar/lexicographic ranking, Pareto dominance,
sensitivity analysis, and abstention economics on synthetic candidates.  No
model is trained, no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.metric_gaming import build_all_cases, evaluate_metric_gaming
from slm_training.evals.verified_utility import (
    FACTOR_NAMES,
    SCHEMA_VERSION,
    UtilityWeightManifestV1,
    VerifiedUtilityV1,
    abstention_economics,
    lexicographic_score,
    pareto_dominance,
    pareto_front,
    scalarized_score,
    sensitivity_rank_reversals,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "VerifiedUtilityAuditManifestV1",
    "VerifiedUtilityCandidateRecord",
    "VerifiedUtilityAuditReport",
    "build_default_weight_manifest",
    "build_fixture_candidates",
    "run_verified_utility_audit",
    "render_markdown",
]

MATRIX_VERSION = "verified-utility-v1"
MATRIX_SET = "slm186_verified_utility"
EXPERIMENT_ID = "slm186-verified-utility"

# Default lexicographic policy: hard validity first, then semantic fidelity,
# then human/judge signals, then cost.
_DEFAULT_LEXICOGRAPHIC_POLICY = [
    "binding_aware_meaningful_v2",
    "component_role_recall",
    "contract_coverage",
    "topology_node_f1",
    "reference_graph_exactness",
    "independent_judge_score",
    "human_pair_preference",
    "complexity_cost",
    "inference_cost",
]

# Default scalar weights.  Benefit factors have positive weight; cost factors
# have negative weight so lower cost increases utility.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "binding_aware_meaningful_v2": 0.25,
    "component_role_recall": 0.15,
    "contract_coverage": 0.10,
    "topology_node_f1": 0.10,
    "reference_graph_exactness": 0.10,
    "behavior_evidence": 0.05,
    "render_evidence": 0.05,
    "independent_judge_score": 0.10,
    "human_pair_preference": 0.05,
    "complexity_cost": -0.03,
    "inference_cost": -0.02,
}

_DEFAULT_PERMITTED_RANGES: dict[str, tuple[float, float]] = {
    "binding_aware_meaningful_v2": (0.15, 0.35),
    "component_role_recall": (0.10, 0.20),
    "contract_coverage": (0.05, 0.15),
    "topology_node_f1": (0.05, 0.15),
    "reference_graph_exactness": (0.05, 0.15),
    "behavior_evidence": (0.00, 0.10),
    "render_evidence": (0.00, 0.10),
    "independent_judge_score": (0.05, 0.15),
    "human_pair_preference": (0.00, 0.10),
    "complexity_cost": (-0.05, 0.00),
    "inference_cost": (-0.05, 0.00),
}


@dataclass(frozen=True)
class VerifiedUtilityAuditManifestV1:
    """Preregistered manifest for the SLM-186 verified-utility audit."""

    schema: str = "VerifiedUtilityAuditManifestV1"
    run_id: str = "slm186-verified-utility"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = (
        "A multi-factor verified-utility ladder can rank OpenUI candidates while "
        "keeping every factor explicit, availability-labeled, and auditable for "
        "Goodhart gaming."
    )
    falsifier: str = (
        "The scalarized ranking contradicts the lexicographic ranking on obviously "
        "dominated or abstained candidates, or small weight perturbations within "
        "the permitted ranges produce frequent rank reversals."
    )
    version_stamp: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "version_stamp": self.version_stamp,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedUtilityAuditManifestV1":
        return cls(
            schema=str(data.get("schema", "VerifiedUtilityAuditManifestV1")),
            run_id=str(data.get("run_id", "slm186-verified-utility")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", "")),
            falsifier=str(data.get("falsifier", "")),
            version_stamp=dict(data.get("version_stamp", {})),
            generated_at=str(data.get("generated_at", "")),
        )


@dataclass(frozen=True)
class VerifiedUtilityCandidateRecord:
    """One candidate plus its computed utility artifacts."""

    candidate_id: str
    scenario: str
    utility: VerifiedUtilityV1
    scalar_score: float
    lexicographic_tier: str
    rank_scalar: int = 0
    rank_lexicographic: int = 0
    pareto_optimal: bool = False
    dominated_by: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["utility"] = self.utility.to_dict()
        data["dominated_by"] = list(self.dominated_by)
        data["notes"] = list(self.notes)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedUtilityCandidateRecord":
        return cls(
            candidate_id=str(data["candidate_id"]),
            scenario=str(data.get("scenario", "")),
            utility=VerifiedUtilityV1.from_dict(data.get("utility", {})),
            scalar_score=float(data.get("scalar_score", 0.0)),
            lexicographic_tier=str(data.get("lexicographic_tier", "")),
            rank_scalar=int(data.get("rank_scalar", 0)),
            rank_lexicographic=int(data.get("rank_lexicographic", 0)),
            pareto_optimal=bool(data.get("pareto_optimal", False)),
            dominated_by=tuple(str(x) for x in data.get("dominated_by", [])),
            notes=tuple(str(x) for x in data.get("notes", [])),
        )


@dataclass(frozen=True)
class VerifiedUtilityAuditReport:
    """Full fixture report for SLM-186."""

    schema: str = "VerifiedUtilityAuditReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm186-verified-utility"
    status: str = "fixture"
    claim_class: str = "wiring"
    manifest: VerifiedUtilityAuditManifestV1 = field(
        default_factory=VerifiedUtilityAuditManifestV1
    )
    weight_manifest: UtilityWeightManifestV1 = field(
        default_factory=lambda: build_default_weight_manifest()
    )
    candidates: list[VerifiedUtilityCandidateRecord] = field(default_factory=list)
    scalar_ranking: list[str] = field(default_factory=list)
    lexicographic_ranking: list[str] = field(default_factory=list)
    pareto_front_ids: list[str] = field(default_factory=list)
    abstention_economics: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    canary_summary: dict[str, Any] = field(default_factory=dict)
    version_stamp: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "weight_manifest": self.weight_manifest.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "scalar_ranking": list(self.scalar_ranking),
            "lexicographic_ranking": list(self.lexicographic_ranking),
            "pareto_front_ids": list(self.pareto_front_ids),
            "abstention_economics": dict(self.abstention_economics),
            "sensitivity": dict(self.sensitivity),
            "canary_summary": dict(self.canary_summary),
            "version_stamp": self.version_stamp,
            "generated_at": self.generated_at,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerifiedUtilityAuditReport":
        return cls(
            schema=str(data.get("schema", "VerifiedUtilityAuditReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", "slm186-verified-utility")),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            manifest=VerifiedUtilityAuditManifestV1.from_dict(
                data.get("manifest", {})
            ),
            weight_manifest=UtilityWeightManifestV1.from_dict(
                data.get("weight_manifest", {})
            ),
            candidates=[
                VerifiedUtilityCandidateRecord.from_dict(c)
                for c in data.get("candidates", [])
            ],
            scalar_ranking=list(data.get("scalar_ranking", [])),
            lexicographic_ranking=list(data.get("lexicographic_ranking", [])),
            pareto_front_ids=list(data.get("pareto_front_ids", [])),
            abstention_economics=dict(data.get("abstention_economics", {})),
            sensitivity=dict(data.get("sensitivity", {})),
            canary_summary=dict(data.get("canary_summary", {})),
            version_stamp=dict(data.get("version_stamp", {})),
            generated_at=str(data.get("generated_at", "")),
        )


def build_default_weight_manifest() -> UtilityWeightManifestV1:
    """Return the default SLM-186 weight policy."""
    return UtilityWeightManifestV1(
        weights=dict(_DEFAULT_WEIGHTS),
        normalization="unit",
        primary_policy="scalarized",
        dev_fit_hash="fixture_dev_fit_hash_slm186",
        confirmation_hash="fixture_confirmation_hash_slm186",
        permitted_ranges=dict(_DEFAULT_PERMITTED_RANGES),
        version=SCHEMA_VERSION,
    )


def _available(**kwargs: Any) -> dict[str, str]:
    """Availability map marking the listed factors available."""
    availability = {name: "unavailable" for name in FACTOR_NAMES}
    for name in availability:
        if name in kwargs:
            availability[name] = "available"
    return availability


def _candidate(
    candidate_id: str,
    scenario: str,
    **kwargs: Any,
) -> VerifiedUtilityV1:
    """Build a VerifiedUtilityV1 with availability inferred from kwargs."""
    # Remove labeling fields; only VerifiedUtilityV1 fields remain in kwargs.
    kwargs.pop("candidate_id", None)
    kwargs.pop("scenario", None)
    availability = kwargs.pop("availability", None)
    if isinstance(availability, dict):
        pass
    elif availability == "available":
        availability = {name: "available" for name in FACTOR_NAMES}
    else:
        availability = _available(**kwargs)
    return VerifiedUtilityV1(availability=availability, **kwargs)


def build_fixture_candidates(seed: int = 0) -> list[tuple[str, str, VerifiedUtilityV1]]:
    """Return deterministic synthetic (candidate_id, scenario, utility) tuples."""
    rng = random.Random(seed)
    candidates: list[tuple[str, str, VerifiedUtilityV1]] = []

    # Pareto-dominant candidate: strong on every benefit, low cost.
    candidates.append(
        (
            "dominant",
            "pareto_dominant",
            _candidate(
                candidate_id="dominant",
                scenario="pareto_dominant",
                hard_valid=True,
                support_status="supported",
                contract_coverage=1.0,
                binding_aware_meaningful_v2=0.95,
                component_role_recall=0.95,
                topology_node_f1=0.95,
                topology_edge_f1=0.90,
                reference_graph_exactness=0.95,
                behavior_evidence=0.90,
                render_evidence=0.85,
                independent_judge_score=0.90,
                human_pair_preference=0.80,
                complexity_cost=0.10,
                inference_cost=0.10,
                availability="available",
            ),
        )
    )

    # Dominated candidate: lower on benefits, higher cost.
    candidates.append(
        (
            "dominated",
            "pareto_dominated",
            _candidate(
                candidate_id="dominated",
                scenario="pareto_dominated",
                hard_valid=True,
                support_status="supported",
                contract_coverage=0.60,
                binding_aware_meaningful_v2=0.55,
                component_role_recall=0.50,
                topology_node_f1=0.50,
                topology_edge_f1=0.45,
                reference_graph_exactness=0.55,
                behavior_evidence=0.40,
                render_evidence=0.35,
                independent_judge_score=0.45,
                human_pair_preference=0.30,
                complexity_cost=0.50,
                inference_cost=0.60,
                availability="available",
            ),
        )
    )

    # Abstained candidate: refuses to answer; low utility by construction.
    candidates.append(
        (
            "abstained",
            "abstained",
            _candidate(
                candidate_id="abstained",
                scenario="abstained",
                hard_valid=False,
                support_status="unsupported",
                contract_coverage=0.0,
                binding_aware_meaningful_v2=0.0,
                component_role_recall=0.0,
                topology_node_f1=0.0,
                topology_edge_f1=0.0,
                reference_graph_exactness=0.0,
                behavior_evidence=0.0,
                render_evidence=0.0,
                independent_judge_score=None,
                human_pair_preference=None,
                complexity_cost=0.0,
                inference_cost=0.05,
                abstained=True,
                failure_reason_codes=("abstained", "no_nontrivial_content"),
                availability="available",
            ),
        )
    )

    # Canary: high binding score but missing a key component (gaming recall).
    candidates.append(
        (
            "canary_missing_component",
            "canary",
            _candidate(
                candidate_id="canary_missing_component",
                scenario="canary",
                hard_valid=True,
                support_status="partial",
                contract_coverage=0.70,
                binding_aware_meaningful_v2=0.85,
                component_role_recall=0.35,
                topology_node_f1=0.80,
                topology_edge_f1=0.75,
                reference_graph_exactness=0.80,
                behavior_evidence=0.20,
                render_evidence=0.20,
                independent_judge_score=0.30,
                human_pair_preference=0.10,
                complexity_cost=0.20,
                inference_cost=0.15,
                failure_reason_codes=("prompt_component_missing",),
                availability="available",
            ),
        )
    )

    # Canary: high topology but wrong role binding.
    candidates.append(
        (
            "canary_wrong_binding",
            "canary",
            _candidate(
                candidate_id="canary_wrong_binding",
                scenario="canary",
                hard_valid=True,
                support_status="supported",
                contract_coverage=0.95,
                binding_aware_meaningful_v2=0.60,
                component_role_recall=0.90,
                topology_node_f1=0.85,
                topology_edge_f1=0.80,
                reference_graph_exactness=0.30,
                behavior_evidence=0.50,
                render_evidence=0.50,
                independent_judge_score=0.40,
                human_pair_preference=0.20,
                complexity_cost=0.25,
                inference_cost=0.20,
                failure_reason_codes=("placeholder_semantic_role_mismatch",),
                availability="available",
            ),
        )
    )

    # Economical near-gold candidate: lower judge score but very cheap.
    candidates.append(
        (
            "cheap_near_gold",
            "economy",
            _candidate(
                candidate_id="cheap_near_gold",
                scenario="economy",
                hard_valid=True,
                support_status="supported",
                contract_coverage=0.90,
                binding_aware_meaningful_v2=0.80,
                component_role_recall=0.80,
                topology_node_f1=0.80,
                topology_edge_f1=0.75,
                reference_graph_exactness=0.80,
                behavior_evidence=0.60,
                render_evidence=0.55,
                independent_judge_score=0.55,
                human_pair_preference=0.40,
                complexity_cost=0.05,
                inference_cost=0.05,
                availability="available",
            ),
        )
    )

    # Candidate with unavailable judge score (realistic wiring gap).
    candidates.append(
        (
            "no_judge",
            "partial_data",
            _candidate(
                candidate_id="no_judge",
                scenario="partial_data",
                hard_valid=True,
                support_status="supported",
                contract_coverage=0.80,
                binding_aware_meaningful_v2=0.75,
                component_role_recall=0.70,
                topology_node_f1=0.70,
                topology_edge_f1=0.65,
                reference_graph_exactness=0.70,
                behavior_evidence=0.60,
                render_evidence=0.60,
                independent_judge_score=None,
                human_pair_preference=None,
                complexity_cost=0.20,
                inference_cost=0.20,
                availability={
                    **_available(
                        hard_valid=True,
                        support_status="supported",
                        contract_coverage=0.80,
                        binding_aware_meaningful_v2=0.75,
                        component_role_recall=0.70,
                        topology_node_f1=0.70,
                        topology_edge_f1=0.65,
                        reference_graph_exactness=0.70,
                        behavior_evidence=0.60,
                        render_evidence=0.60,
                        complexity_cost=0.20,
                        inference_cost=0.20,
                    ),
                    "independent_judge_score": "unavailable",
                    "human_pair_preference": "unavailable",
                },
            ),
        )
    )

    # Shuffle with fixed seed so the fixture order is stable but not rank-order.
    rng.shuffle(candidates)
    return candidates


def _canary_summary() -> dict[str, Any]:
    """Summarize the SLM-186 Goodhart canary slices from metric_gaming."""
    try:
        report = evaluate_metric_gaming(build_all_cases(seed=0))
    except Exception as exc:  # noqa: BLE001 - fixture summary must not abort
        return {"error": str(exc), "slice_counts": {}}

    slice_counts: dict[str, int] = {}
    slice_strict_rates: dict[str, float] = {}
    for name, rep in report.slices.items():
        if name.startswith("canary_"):
            slice_counts[name] = rep.n
            slice_strict_rates[name] = rep.strict_rate

    return {
        "n_canary_cases": sum(slice_counts.values()),
        "slice_counts": slice_counts,
        "slice_strict_rates": slice_strict_rates,
        "overall_canary_strict_rate": (
            sum(slice_strict_rates.values()) / len(slice_strict_rates)
            if slice_strict_rates
            else 0.0
        ),
    }


def run_verified_utility_audit(
    *,
    candidates: list[tuple[str, str, VerifiedUtilityV1]] | None = None,
    weight_manifest: UtilityWeightManifestV1 | None = None,
    mode: str = "fixture",
    run_id: str = "slm186-verified-utility",
    output_dir: Path | None = None,
    seed: int = 0,
) -> tuple[VerifiedUtilityAuditManifestV1, VerifiedUtilityAuditReport]:
    """Run the SLM-186 verified-utility audit.

    In ``fixture`` mode, synthetic candidates are generated if none are
    supplied.  The report carries scalar and lexicographic rankings, Pareto
    analysis, abstention economics, and sensitivity rank-reversal statistics.
    """
    if mode not in {"fixture", "describe", "analyze-history", "sensitivity"}:
        raise ValueError(f"unknown mode {mode!r}")

    weight_manifest = weight_manifest or build_default_weight_manifest()
    errors = weight_manifest.validate()
    if errors:
        raise ValueError("weight manifest invalid: " + "; ".join(errors))

    if candidates is None:
        candidates = build_fixture_candidates(seed=seed)

    # Compute scalar and lexicographic scores.
    scored: list[tuple[str, str, VerifiedUtilityV1, float, str]] = []
    for cid, scenario, util in candidates:
        scalar = scalarized_score(util, weight_manifest)["score"]
        lex = lexicographic_score(util, _DEFAULT_LEXICOGRAPHIC_POLICY)["tier"]
        scored.append((cid, scenario, util, scalar, lex))

    # Rankings: higher scalar score is better.
    scalar_sorted = sorted(
        scored, key=lambda x: (-x[3], x[0])
    )
    scalar_ranking = [cid for cid, _, _, _, _ in scalar_sorted]

    # Lexicographic: higher tier string (rank vector) is better; Python tuple
    # comparison on the numeric vector works because unavailable maps to -inf.
    lex_values = {
        cid: lexicographic_score(util, _DEFAULT_LEXICOGRAPHIC_POLICY)["rank_vector"]
        for cid, _, util, _, _ in scored
    }
    lex_sorted = sorted(scored, key=lambda x: (lex_values[x[0]], x[0]), reverse=True)
    lexicographic_ranking = [cid for cid, _, _, _, _ in lex_sorted]

    # Pareto front and dominance.
    labeled = [(cid, util) for cid, _, util, _, _ in scored]
    front = pareto_front(labeled)
    front_ids = [cid for cid, _ in front]

    dominated_by: dict[str, list[str]] = {cid: [] for cid, _, _, _, _ in scored}
    for i, (left_id, _, left_util, _, _) in enumerate(scored):
        for j, (right_id, _, right_util, _, _) in enumerate(scored):
            if i == j:
                continue
            dom = pareto_dominance(right_util, left_util)
            if dom["left_dominates"]:
                dominated_by[left_id].append(right_id)

    # Build records.
    scalar_rank = {cid: idx + 1 for idx, cid in enumerate(scalar_ranking)}
    lex_rank = {cid: idx + 1 for idx, cid in enumerate(lexicographic_ranking)}
    records: list[VerifiedUtilityCandidateRecord] = []
    for cid, scenario, util, scalar, lex in scored:
        notes: list[str] = [scenario]
        if util.abstained:
            notes.append("abstained")
        if dominated_by[cid]:
            notes.append(f"dominated_by={','.join(dominated_by[cid])}")
        if cid in front_ids:
            notes.append("pareto_optimal")
        records.append(
            VerifiedUtilityCandidateRecord(
                candidate_id=cid,
                scenario=scenario,
                utility=util,
                scalar_score=scalar,
                lexicographic_tier=lex,
                rank_scalar=scalar_rank[cid],
                rank_lexicographic=lex_rank[cid],
                pareto_optimal=cid in front_ids,
                dominated_by=tuple(dominated_by[cid]),
                notes=tuple(notes),
            )
        )

    # Abstention economics.
    abstention = abstention_economics(
        [util for _, _, util, _, _ in scored], risk_threshold=0.3
    )

    # Sensitivity analysis.
    perturb_manifests = [
        weight_manifest,
        UtilityWeightManifestV1(
            weights={k: v * 0.8 for k, v in weight_manifest.weights.items()},
            normalization=weight_manifest.normalization,
            primary_policy=weight_manifest.primary_policy,
            permitted_ranges=weight_manifest.permitted_ranges,
            version=weight_manifest.version,
        ),
    ]
    sensitivity = sensitivity_rank_reversals(
        [(cid, util) for cid, _, util, _, _ in scored],
        perturb_manifests,
        perturbations_per_manifest=20,
        seed=seed,
    )

    manifest = VerifiedUtilityAuditManifestV1(
        run_id=run_id,
        status=mode if mode in {"fixture", "describe"} else "analyzed",
        version_stamp=build_version_stamp(
            "evals.verified_utility",
            "harness.experiments.slm186_verified_utility",
            "evals.scoring",
            "evals.meaningful_program",
        ),
    )

    report = VerifiedUtilityAuditReport(
        run_id=run_id,
        status=manifest.status,
        manifest=manifest,
        weight_manifest=weight_manifest,
        candidates=records,
        scalar_ranking=scalar_ranking,
        lexicographic_ranking=lexicographic_ranking,
        pareto_front_ids=front_ids,
        abstention_economics=abstention,
        sensitivity=sensitivity,
        canary_summary=_canary_summary(),
        version_stamp=manifest.version_stamp,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "verified_utility_report.json")

    return manifest, report


def render_markdown(report: VerifiedUtilityAuditReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-186 (FFE0-04): Verified-utility ladder fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable "
        "weights were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Fixture candidates",
        "",
        "| Candidate | Scenario | Scalar | Lex rank | Pareto | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in report.candidates:
        lines.append(
            f"| {rec.candidate_id} | {rec.scenario} | {rec.scalar_score:.3f} | "
            f"{rec.rank_lexicographic} | {rec.pareto_optimal} | "
            f"{', '.join(rec.notes)} |"
        )

    lines.extend(
        [
            "",
            "## Rankings",
            "",
            "**Scalar ranking (best first):** "
            + ", ".join(report.scalar_ranking),
            "",
            "**Lexicographic ranking (best first):** "
            + ", ".join(report.lexicographic_ranking),
            "",
            "**Pareto front:** " + ", ".join(report.pareto_front_ids),
            "",
            "## Abstention economics",
            "",
            f"Accepted: {report.abstention_economics.get('accepted', 0)}; "
            f"Abstained: {report.abstention_economics.get('abstained', 0)}; "
            f"mean utility accepted: {report.abstention_economics.get('mean_utility_accepted', 0.0):.3f}; "
            f"value of abstention: {report.abstention_economics.get('value_of_abstention', 0.0):.3f}.",
            "",
            "## Sensitivity",
            "",
            f"Perturbations: {report.sensitivity.get('total_perturbations', 0)}; "
            f"rank reversals: {report.sensitivity.get('reversal_count', 0)} "
            f"(rate: {report.sensitivity.get('reversal_rate', 0.0):.3f}).",
            "",
            "## Goodhart canary summary",
            "",
            f"Canary cases wired: {report.canary_summary.get('n_canary_cases', 0)}; "
            f"overall canary strict rate: "
            f"{report.canary_summary.get('overall_canary_strict_rate', 0.0):.3f}.",
            "",
        ]
    )

    slice_counts = report.canary_summary.get("slice_counts", {})
    if slice_counts:
        lines.extend(
            [
                "| Slice | Cases | Strict rate |",
                "| --- | --- | --- |",
            ]
        )
        for name, count in sorted(slice_counts.items()):
            rate = report.canary_summary.get("slice_strict_rates", {}).get(name, 0.0)
            lines.append(f"| {name} | {count} | {rate:.3f} |")

    lines.extend(
        [
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The verified-utility "
            "ladder, scalar/lexicographic rankings, Pareto frontier, abstention "
            "economics, and sensitivity analysis are wired and exercised on "
            "deterministic synthetic candidates.  Real eval records, independent "
            "judge scores, and human pair-preference data are required before "
            "claiming any floor-escape.  The mechanism remains "
            "``retain_diagnostic`` / ``blocked_pending_real_eval``.",
            "",
            "## Honest caveats",
            "",
            "- All utilities are synthetic fixtures; real eval records would change "
            "  numeric rankings and may expose rank reversals not seen here.",
            "- Independent judge score and human pair preference are marked "
            "  ``unavailable`` or synthesized; they must not be treated as real "
            "  human evidence.",
            "- The Goodhart canary slices are deterministic and scored by the "
            "  current ``binding_aware_meaningful_v2`` metric; new slices may be "
            "  added as additional gaming channels are identified.",
            "- No ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Next steps",
            "",
            "1. Replace synthetic utilities with real suite scores and judge "
            "   envelopes.",
            "2. Calibrate the weight manifest against held-out human preferences.",
            "3. Re-run sensitivity analysis after every metric or weight change.",
            "4. Close the loop with the SLM-186 Goodhart canary suite: any metric "
            "   change that flips a canary case must be documented and justified.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_verified_utility_audit --mode describe",
            "python -m scripts.run_verified_utility_audit --mode fixture",
            "python -m scripts.run_verified_utility_audit --mode analyze-history PATH",
            "python -m scripts.run_verified_utility_audit --mode sensitivity",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
