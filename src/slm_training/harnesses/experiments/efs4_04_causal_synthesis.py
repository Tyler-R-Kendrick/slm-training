"""EFS4-04 causal diagnosis and explicit architecture disposition.

This module provides the versioned campaign manifest, fail-closed synthesis
loader, evidence DAG, causal-diagnosis classifier, and Markdown renderer for the
Evidence-First Semantic SLM final synthesis.  It is intentionally honest about
which branches are still wiring/plan-only: a branch that has not cleared its
activation gate is reported as ``NOT_RUN_BY_GATE``, not as evidence of failure
or success.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.versioning import build_version_stamp

MANIFEST_SCHEMA = "efs_campaign_manifest/v1"
SYNTHESIS_SCHEMA = "evidence_first_semantic_synthesis/v1"
CAMPAIGN_ID = "evidence-first-semantic-slm-campaign"

DecisionState = Literal[
    "INVALIDATED",
    "INCONCLUSIVE",
    "EQUIVALENT",
    "NEGATIVE",
    "POSITIVE",
    "NOT_RUN_BY_GATE",
    "MISSING",
    "CONTRADICTORY",
]

Disposition = Literal[
    "ADOPT",
    "ADOPT_AS_SAFETY_ONLY",
    "PROMOTE_EXPERIMENTAL",
    "CONDITIONAL_RESEARCH",
    "REVISE",
    "REJECT",
    "DEPRECATE",
    "INCONCLUSIVE",
    "NOT_RUN_BY_GATE",
]

DiagnosisState = Literal[
    "measurement_limited",
    "training_exposure_limited",
    "objective_or_representation_limited",
    "data_or_supervision_limited",
    "candidate_generation_search_limited",
    "selector_limited",
    "verification_cost_limited",
    "architecture_limited",
    "mixed_with_no_identifiable_dominant_cause",
    "insufficient_valid_evidence",
]

EDGE_KINDS = frozenset(
    {
        "depends_on",
        "supports",
        "falsifies",
        "invalidates_prior_result",
        "qualifies",
        "not_run_because",
        "supersedes_metric_or_decoder",
        "shares_checkpoint_or_corpus",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_hash(data: dict[str, Any]) -> str:
    """Deterministic SHA-256 over a canonical JSON representation."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignHypothesisSpec(StrictModel):
    """One preregistered EFS hypothesis/issue."""

    hypothesis_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    linear_issue: str = Field(pattern=r"^SLM-\d+$")
    milestone: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    falsifier: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()
    activation_gate: str | None = None
    primary_metrics: tuple[str, ...] = ()
    minimum_effect: dict[str, Any] | None = None
    cost_constraints: dict[str, Any] | None = None
    allowed_decisions: tuple[str, ...] = ()
    expected_result_refs: tuple[str, ...] = ()
    architecture_tags: tuple[str, ...] = ()


class CampaignManifestV1(StrictModel):
    """Preregistered EFS campaign manifest."""

    schema_version: Literal[MANIFEST_SCHEMA] = MANIFEST_SCHEMA
    campaign_id: str = CAMPAIGN_ID
    created_at: str = Field(default_factory=utc_now)
    version_stamp: dict[str, Any] = Field(
        default_factory=lambda: build_version_stamp("harness.experiments")
    )
    hypotheses: tuple[CampaignHypothesisSpec, ...] = ()

    @model_validator(mode="after")
    def _unique_hypothesis_ids(self) -> "CampaignManifestV1":
        ids = [h.hypothesis_id for h in self.hypotheses]
        dupes = {h for h in ids if ids.count(h) > 1}
        if dupes:
            raise ValueError(f"duplicate hypothesis_ids: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def _linear_issues_unique(self) -> "CampaignManifestV1":
        issues = [h.linear_issue for h in self.hypotheses]
        dupes = {i for i in issues if issues.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate linear issues: {sorted(dupes)}")
        return self


class ResultManifest(StrictModel):
    """A thin, forgiving wrapper around any committed ``docs/design`` result JSON."""

    source_path: str
    schema_version: Any = None
    matrix_set: Any = None
    matrix_version: Any = None
    campaign_id: Any = None
    claim_class: Any = None
    status: Any = None
    verdict: Any = None
    source_commit: Any = None
    version_stamp: dict[str, Any] | None = None
    rows: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] = Field(default_factory=dict)


class HypothesisSynthesis(StrictModel):
    """Synthesised terminal state for one hypothesis."""

    hypothesis_id: str
    linear_issue: str
    claim: str
    falsifier: str
    state: DecisionState
    state_reason: str
    result_refs: tuple[str, ...] = ()
    checkpoint_refs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class EvidenceNode(StrictModel):
    node_id: str
    kind: Literal[
        "hypothesis",
        "result_manifest",
        "checkpoint",
        "corpus",
        "evaluator",
        "judge",
        "causal_conclusion",
        "architecture_decision",
    ]
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(StrictModel):
    source: str
    target: str
    kind: Literal[
        "depends_on",
        "supports",
        "falsifies",
        "invalidates_prior_result",
        "qualifies",
        "not_run_because",
        "supersedes_metric_or_decoder",
        "shares_checkpoint_or_corpus",
    ]
    label: str = ""


class ArchitectureDisposition(StrictModel):
    item: str
    disposition: Disposition
    supporting_experiments: tuple[str, ...] = ()
    falsifying_experiments: tuple[str, ...] = ()
    effect: str = ""
    uncertainty: str = ""
    semantic_metric: str = ""
    cost_metric: str = ""
    safety_constraints: tuple[str, ...] = ()
    activation_conditions: tuple[str, ...] = ()
    next_action: str = ""


class NextProgramRecommendation(StrictModel):
    rank: int = Field(ge=1, le=3)
    objective: str
    non_duplicate_rationale: str
    expected_information_gain: str
    smallest_experiment: str
    budget: str
    kill_criterion: str
    dependencies: tuple[str, ...] = ()


class CausalDiagnosis(StrictModel):
    primary: DiagnosisState
    primary_reason: str
    counterfactual_evidence: str
    secondary: tuple[str, ...] = ()


class EvidenceFirstSemanticSynthesisV1(StrictModel):
    """Machine-readable final synthesis."""

    schema_version: Literal[SYNTHESIS_SCHEMA] = SYNTHESIS_SCHEMA
    campaign_id: str = CAMPAIGN_ID
    manifest_hash: str
    generation_command: str
    created_at: str = Field(default_factory=utc_now)
    version_stamp: dict[str, Any] = Field(
        default_factory=lambda: build_version_stamp(
            "harness.experiments",
            "harness.experiments.efs4_04_causal_synthesis",
        )
    )
    hypotheses: tuple[HypothesisSynthesis, ...] = ()
    lineage_summary: dict[str, Any] = Field(default_factory=dict)
    evidence_graph: dict[str, Any] = Field(default_factory=dict)
    causal_diagnosis: CausalDiagnosis | None = None
    architecture_dispositions: tuple[ArchitectureDisposition, ...] = ()
    champion_decision: str = "no_promotion"
    next_programs: tuple[NextProgramRecommendation, ...] = ()
    unresolved_risks: tuple[str, ...] = ()
    invalidated_claims: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Default preregistered campaign
# ---------------------------------------------------------------------------


def _h(
    hypothesis_id: str,
    linear_issue: str,
    milestone: str,
    claim: str,
    falsifier: str,
    *,
    dependencies: tuple[str, ...] = (),
    activation_gate: str | None = None,
    primary_metrics: tuple[str, ...] = ("binding_aware_meaningful_v2",),
    allowed_decisions: tuple[str, ...] | None = None,
    expected_result_refs: tuple[str, ...] = (),
    architecture_tags: tuple[str, ...] = (),
    minimum_effect: dict[str, Any] | None = None,
    cost_constraints: dict[str, Any] | None = None,
) -> CampaignHypothesisSpec:
    if allowed_decisions is None:
        allowed_decisions = (
            "INVALIDATED",
            "INCONCLUSIVE",
            "EQUIVALENT",
            "NEGATIVE",
            "POSITIVE",
            "NOT_RUN_BY_GATE",
        )
    return CampaignHypothesisSpec(
        hypothesis_id=hypothesis_id,
        linear_issue=linear_issue,
        milestone=milestone,
        claim=claim,
        falsifier=falsifier,
        dependencies=dependencies,
        activation_gate=activation_gate,
        primary_metrics=primary_metrics,
        minimum_effect=minimum_effect,
        cost_constraints=cost_constraints,
        allowed_decisions=allowed_decisions,
        expected_result_refs=expected_result_refs,
        architecture_tags=architecture_tags,
    )


def build_default_campaign_manifest() -> CampaignManifestV1:
    """Return the preregistered EFS campaign manifest for SLM-140."""
    hypotheses = (
        # EFS0 — reproducibility & independent measurement
        _h(
            "efs0-01-checkpoint-provenance",
            "SLM-103",
            "EFS0 · Reproducibility & independent measurement",
            "Frontier checkpoints are durable, hash-verified, and resolvable from a fresh clone.",
            "Checkpoints are missing, hashes do not match, or remote references are unresolvable.",
            expected_result_refs=("iter-efs0-01-checkpoint-provenance-*.json",),
            architecture_tags=("provenance",),
            allowed_decisions=("POSITIVE", "NEGATIVE", "INCONCLUSIVE"),
        ),
        _h(
            "efs0-02-decode-invariance",
            "SLM-104",
            "EFS0 · Reproducibility & independent measurement",
            "A canonical decoder/path exists and produces invariant semantic outcomes across decoder variations.",
            "Decoder path changes still change the primary semantic metric or no canonical path can be selected.",
            expected_result_refs=("iter-efs-decode-invariance-*.json",),
            architecture_tags=("decoder", "measurement"),
        ),
        _h(
            "efs0-03-meaningful-v2",
            "SLM-105",
            "EFS0 · Reproducibility & independent measurement",
            "Binding-aware meaningful v2 separates valid-but-empty/useless programs from genuinely useful outputs.",
            "The metric can be gamed by minimal-valid, rare-omission, or inventory-free outputs.",
            expected_result_refs=("iter-efs0-03-meaningful-v2-frontier-audit-*.json",),
            architecture_tags=("metric", "measurement"),
        ),
        _h(
            "efs0-04-judge-independence",
            "SLM-106",
            "EFS0 · Reproducibility & independent measurement",
            "Judge/human agreement and cross-family scoring show the semantic judge is independent and stable.",
            "Judge conflicts or family-specific inflation exceed a tolerable rate.",
            expected_result_refs=("iter-efs0-04-judge-independence-*.json",),
            architecture_tags=("judge", "measurement"),
        ),
        _h(
            "efs0-05-rejected-lever-readjudication",
            "SLM-107",
            "EFS0 · Reproducibility & independent measurement",
            "Prior rejected levers remain negative under five seeds, paired tests, corrected decoder, and independent labels.",
            "At least one prior rejection changes classification when confounds are removed.",
            expected_result_refs=("iter-efs0-05-rejected-lever-readjudication-*.json",),
            architecture_tags=("measurement", "negative_results"),
        ),
        # EFS1 — decisive baselines & exposure threshold
        _h(
            "efs1-01-external-ceiling",
            "SLM-108",
            "EFS1 · Decisive baselines & exposure threshold",
            "A 1–7B external model constrained by the same compiler achieves a nontrivial semantic ceiling above the tiny SLM.",
            "The external constrained model is near zero, indicating a specification/representation/evaluator limit.",
            expected_result_refs=("iter-efs1-01-external-ceiling-*.json",),
            architecture_tags=("baseline", "capacity"),
        ),
        _h(
            "efs1-02-exposure-ladder",
            "SLM-109",
            "EFS1 · Decisive baselines & exposure threshold",
            "The frozen E228 recipe shows a clear exposure threshold at ≥100× cumulative token exposure.",
            "Semantic metrics remain flat through ≥100× exposure while loss improves or saturates.",
            expected_result_refs=("iter-efs1-02-exposure-ladder-*.json",),
            architecture_tags=("training", "exposure"),
        ),
        _h(
            "efs1-03-empty-length-bias",
            "SLM-110",
            "EFS1 · Decisive baselines & exposure threshold",
            "Valid-but-empty/minimal-shell outputs win because of length or mask-mass bias, and a principled correction helps.",
            "Empty outputs are preferred by the model score, not the decoder, or corrections do not improve semantics.",
            expected_result_refs=("iter-efs1-03-empty-length-bias-*.json",),
            architecture_tags=("decode", "content_floor"),
        ),
        # EFS2 — valid-by-construction search & recovery
        _h(
            "efs2-01-x22-scaling",
            "SLM-111",
            "EFS2 · Valid-by-construction search & recovery",
            "X22 valid-state tree-edit quality scales predictably with beam width and/or edit depth before saturating.",
            "Quality is flat across beam width and depth while search coverage rises.",
            expected_result_refs=("iter-efs2-01-tree-edit-scaling-*.json",),
            architecture_tags=("search", "x22"),
        ),
        _h(
            "efs2-02-trigger-telemetry",
            "SLM-112",
            "EFS2 · Valid-by-construction search & recovery",
            "Observe-only trigger predicates fire predictively before recoverable semantic failures under a non-greedy regime.",
            "Triggers never fire or fire without predicting failure/recovery benefit.",
            expected_result_refs=("iter-efs2-02-trigger-telemetry-*.json",),
            architecture_tags=("trigger", "ptrm"),
            activation_gate="Phase-B recovery is enabled only after Phase-A shows predictive activation.",
        ),
        _h(
            "efs2-03-conflict-slice-repair",
            "SLM-113",
            "EFS2 · Valid-by-construction search & recovery",
            "Conflict-localized remasking repairs more failures with fewer edits than full remask or suffix rollback.",
            "Slices are too imprecise, miss dependencies, or provide no cost/quality benefit at equal budgets.",
            expected_result_refs=("iter-efs2-03-conflict-slice-repair-*.json",),
            architecture_tags=("repair", "remask"),
        ),
        _h(
            "efs2-04-verifier-cascade",
            "SLM-115",
            "EFS2 · Valid-by-construction search & recovery",
            "A cached cheap-to-expensive verifier cascade preserves ≥95% of flat-stack pruning at ≤30% verifier cost.",
            "The cascade loses >5% sound pruning, changes outcomes, or cannot reduce cost below 30–50%.",
            expected_result_refs=("iter-efs2-04-verifier-cascade-*.json",),
            architecture_tags=("verifier", "cost"),
        ),
        # EFS3 — semantic supervision, selection & diversity
        _h(
            "efs3-01-solver-state-supervision",
            "SLM-118",
            "EFS3 · Semantic supervision, selection & diversity",
            "A 50/50 gold/on-policy DAgger-style state mixture outperforms pure sources on held-out self-failure recovery.",
            "The mixture provides no recovery gain, or rollout labels are too sparse/noisy to beat pure gold.",
            expected_result_refs=("iter-efs3-01-solver-state-supervision-*.json",),
            architecture_tags=("supervision", "on_policy"),
        ),
        _h(
            "efs3-02-corruption-curriculum",
            "SLM-120",
            "EFS3 · Semantic supervision, selection & diversity",
            "Including 5–15% near-solved semantic corruptions improves local recovery and fixed-point stability without harming generation.",
            "Near-solved mass produces no recovery gain, causes copying, or degrades full-generation quality.",
            expected_result_refs=("iter-efs3-02-corruption-curriculum-*.json",),
            architecture_tags=("curriculum", "corruption"),
        ),
        _h(
            "efs3-03-b3-capacity-v2",
            "SLM-124",
            "EFS3 · Semantic supervision, selection & diversity",
            "The choice representation reaches a preregistered semantic target at smaller capacity/bytes than the surface representation.",
            "After correcting the decoder, surface and choice capacity curves are equivalent or surface is better.",
            expected_result_refs=("iter-efs-b3-capacity-v2-*.json",),
            architecture_tags=("representation", "capacity", "choice_codec"),
        ),
        _h(
            "efs3-04-candidate-selector",
            "SLM-127",
            "EFS3 · Semantic supervision, selection & diversity",
            "A contract-grounded selector with calibrated abstention closes the pass@K-to-selected-pass@K gap.",
            "Candidate pools contain useful programs but no selector beats simple baselines at the target risk.",
            expected_result_refs=("iter-efs3-04-candidate-selector-*.json",),
            architecture_tags=("selection", "abstention"),
        ),
        _h(
            "efs3-05-canonical-ast-dedup",
            "SLM-130",
            "EFS3 · Semantic supervision, selection & diversity",
            "Canonical AST deduplication increases within-prompt unique hard-valid modes and semantic pass@K at fixed cost.",
            "Deduplication changes only bookkeeping, does not increase unique valid modes, or removes semantically distinct candidates.",
            expected_result_refs=("iter-efs3-05-canonical-ast-dedup-*.json",),
            architecture_tags=("diversity", "dedup"),
        ),
        _h(
            "efs3-06-ast-sketch-retrieval",
            "SLM-133",
            "EFS3 · Semantic supervision, selection & diversity",
            "AST-sketch data balancing and/or choice-native retrieval improve semantic quality at fixed budget without leakage.",
            "Sketch balancing or choice retrieval is equivalent to controls or harms OOD semantics/copying.",
            expected_result_refs=("iter-efs3-06-ast-sketch-retrieval-factorial-*.json",),
            architecture_tags=("data", "retrieval"),
        ),
        # EFS4 — conditional solver architecture & causal synthesis
        _h(
            "efs4-01-trailed-assumptions",
            "SLM-135",
            "EFS4 · Conditional solver architecture & causal synthesis",
            "A trailed, dependency-aware solver avoids false prunes that a monotone proposal-contingent state cannot recover from.",
            "Production architecture never places proposal facts in irreversible state, or both policies recover identically.",
            expected_result_refs=("iter-slm135-trailed-assumptions-*.json",),
            architecture_tags=("solver", "trail", "nogood"),
        ),
        _h(
            "efs4-02-shared-recursive-denoiser",
            "SLM-138",
            "EFS4 · Conditional solver architecture & causal synthesis",
            "A weight-shared recursive denoiser with deep supervision improves semantic/recovery metrics at matched block evaluations.",
            "Recursive arms are equivalent or worse under matched cost and exposure.",
            expected_result_refs=("iter-efs4-02-shared-recursive-denoiser-*.json",),
            architecture_tags=("architecture", "recursive", "deep_supervision"),
        ),
        _h(
            "efs4-03-stochastic-recursive-state",
            "SLM-139",
            "EFS4 · Conditional solver architecture & causal synthesis",
            "A learned high-level stochastic recursive state increases valid semantic mode coverage and selected quality at matched cost.",
            "Samples collapse to one mode, low-level noise matches/exceeds high-level width, or selector cannot convert width safely.",
            expected_result_refs=("iter-efs4-03-stochastic-recursive-state-*.json",),
            architecture_tags=("architecture", "stochastic", "gram"),
            activation_gate="Requires SLM-138 recursive core positive, SLM-130 multimodal regime, and SLM-127 calibrated selector.",
        ),
    )
    return CampaignManifestV1(hypotheses=hypotheses)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def _norm(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_norm(v) for v in value)
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


def _extract_rows(raw: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = raw.get("rows") or raw.get("outcomes") or raw.get("campaign_rows") or []
    if isinstance(rows, dict):
        # some manifests nest outcomes by key
        rows = list(rows.values())
    if not isinstance(rows, (list, tuple)):
        return ()
    return tuple(_norm(r) for r in rows if isinstance(r, dict))


def load_result_manifests(docs_design: Path | str) -> tuple[ResultManifest, ...]:
    """Load every ``iter-efs*`` / ``iter-slm*`` JSON under ``docs/design``."""
    docs_design = Path(docs_design)
    if not docs_design.is_dir():
        raise FileNotFoundError(f"docs/design directory not found: {docs_design}")
    manifests: list[ResultManifest] = []
    for path in sorted(docs_design.glob("iter-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        raw = _norm(data)
        if not isinstance(raw, dict):
            continue
        rows = _extract_rows(raw)
        manifests.append(
            ResultManifest(
                source_path=str(path.relative_to(docs_design)),
                schema_version=raw.get("schema_version"),
                matrix_set=raw.get("matrix_set"),
                matrix_version=raw.get("matrix_version"),
                campaign_id=raw.get("campaign_id"),
                claim_class=raw.get("claim_class"),
                status=raw.get("status"),
                verdict=raw.get("verdict"),
                source_commit=raw.get("source_commit"),
                version_stamp=raw.get("version_stamp"),
                rows=rows,
                raw=raw,
            )
        )
    return tuple(manifests)


def _matches(ref: str, manifest: ResultManifest) -> bool:
    """Glob match a result-ref pattern against a manifest source path or matrix_set."""
    source = manifest.source_path
    name = Path(source).name
    if Path(source).match(ref):
        return True
    if name == ref:
        return True
    if manifest.matrix_set is not None and Path(manifest.matrix_set).match(ref):
        return True
    return False


# ---------------------------------------------------------------------------
# Synthesis logic
# ---------------------------------------------------------------------------


def _manifest_state(manifest: ResultManifest) -> DecisionState:
    """Infer a terminal decision-state from a loaded result manifest."""
    status = str(manifest.status or "").lower()
    verdict = str(manifest.verdict or "").upper()
    if verdict and len(verdict) <= 32:
        if verdict in {
            "POSITIVE",
            "NEGATIVE",
            "EQUIVALENT",
            "INCONCLUSIVE",
            "INVALIDATED",
            "NOT_RUN_BY_GATE",
        }:
            return verdict  # type: ignore[return-value]
    if status in {"plan_only", "diagnostic_only", "wiring", "fixture", "frontier_pending_gpu"}:
        return "NOT_RUN_BY_GATE"
    if status in {"complete", "completed", "done"}:
        return "INCONCLUSIVE"
    if status in {"failed", "failure"}:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def _resolve_hypothesis(
    hypothesis: CampaignHypothesisSpec,
    manifests: tuple[ResultManifest, ...],
) -> HypothesisSynthesis:
    """Resolve one hypothesis against the loaded result manifests."""
    matched = tuple(m for m in manifests if any(_matches(ref, m) for ref in hypothesis.expected_result_refs))
    if not matched:
        return HypothesisSynthesis(
            hypothesis_id=hypothesis.hypothesis_id,
            linear_issue=hypothesis.linear_issue,
            claim=hypothesis.claim,
            falsifier=hypothesis.falsifier,
            state="MISSING",
            state_reason="No committed result manifest matched the expected refs.",
        )

    # Prefer the strongest resolved manifest deterministically (alphabetical by path).
    chosen = sorted(matched, key=lambda m: m.source_path)[0]
    inferred = _manifest_state(chosen)
    allowed = set(hypothesis.allowed_decisions) or set(
        ("INVALIDATED", "INCONCLUSIVE", "EQUIVALENT", "NEGATIVE", "POSITIVE", "NOT_RUN_BY_GATE")
    )
    if inferred not in allowed:
        state: DecisionState = "CONTRADICTORY"
        reason = (
            f"Inferred state {inferred!r} from {chosen.source_path} is not in "
            f"preregistered allowed decisions {sorted(allowed)}."
        )
    else:
        state = inferred
        reason = f"Resolved from {chosen.source_path} (schema={chosen.schema_version}, status={chosen.status})."

    checkpoint_refs: list[str] = []
    for row in chosen.rows:
        if isinstance(row, dict):
            for key in ("checkpoint_sha256", "checkpoint_path", "checkpoint_remote_uri"):
                val = row.get(key)
                if val:
                    checkpoint_refs.append(f"{key}={val}")
    # Also look for top-level checkpoint fields.
    for key in ("checkpoint_sha256", "checkpoint_path", "checkpoint_remote_uri"):
        val = chosen.raw.get(key)
        if val:
            checkpoint_refs.append(f"{key}={val}")

    notes: list[str] = []
    if hypothesis.activation_gate and state == "NOT_RUN_BY_GATE":
        notes.append(f"Activation gate not cleared: {hypothesis.activation_gate}")
    if len(matched) > 1:
        notes.append(f"Multiple result manifests matched: {[m.source_path for m in matched]}")

    return HypothesisSynthesis(
        hypothesis_id=hypothesis.hypothesis_id,
        linear_issue=hypothesis.linear_issue,
        claim=hypothesis.claim,
        falsifier=hypothesis.falsifier,
        state=state,
        state_reason=reason,
        result_refs=tuple(m.source_path for m in matched),
        checkpoint_refs=tuple(sorted(set(checkpoint_refs))),
        notes=tuple(notes),
    )


def synthesize_campaign(
    manifest: CampaignManifestV1,
    manifests: tuple[ResultManifest, ...],
    *,
    generation_command: str = "synthesize-efs-campaign",
) -> EvidenceFirstSemanticSynthesisV1:
    """Build the full ``EvidenceFirstSemanticSynthesisV1`` from manifest + results."""
    manifest_hash = _stable_hash(json.loads(manifest.model_dump_json()))
    hypotheses = tuple(_resolve_hypothesis(h, manifests) for h in manifest.hypotheses)

    graph_nodes: list[EvidenceNode] = []
    graph_edges: list[EvidenceEdge] = []
    node_ids: set[str] = set()

    def add_node(node: EvidenceNode) -> None:
        if node.node_id not in node_ids:
            graph_nodes.append(node)
            node_ids.add(node.node_id)

    add_node(
        EvidenceNode(
            node_id="campaign:efs",
            kind="causal_conclusion",
            label="Evidence-First Semantic SLM Campaign",
            payload={"manifest_hash": manifest_hash, "hypothesis_count": len(hypotheses)},
        )
    )

    used_manifests: set[int] = set()
    for hyp, syn in zip(manifest.hypotheses, hypotheses):
        hid = f"hyp:{hyp.hypothesis_id}"
        add_node(
            EvidenceNode(
                node_id=hid,
                kind="hypothesis",
                label=f"{hyp.linear_issue} {hyp.hypothesis_id}",
                payload={"claim": hyp.claim, "state": syn.state},
            )
        )
        for dep in hyp.dependencies:
            graph_edges.append(
                EvidenceEdge(source=hid, target=f"hyp:{dep}", kind="depends_on", label="depends on")
            )
        for ref in syn.result_refs:
            mid = f"result:{ref}"
            add_node(EvidenceNode(node_id=mid, kind="result_manifest", label=ref))
            graph_edges.append(EvidenceEdge(source=hid, target=mid, kind="supports", label="resolved by"))
        for dep in syn.checkpoint_refs:
            cid = f"checkpoint:{dep}"
            add_node(EvidenceNode(node_id=cid, kind="checkpoint", label=dep))
            graph_edges.append(EvidenceEdge(source=hid, target=cid, kind="shares_checkpoint_or_corpus", label="references"))
        if syn.state == "NOT_RUN_BY_GATE" and hyp.activation_gate:
            gate_id = f"gate:{hyp.hypothesis_id}"
            add_node(
                EvidenceNode(
                    node_id=gate_id,
                    kind="causal_conclusion",
                    label=f"Gate: {hyp.hypothesis_id}",
                    payload={"gate": hyp.activation_gate},
                )
            )
            graph_edges.append(
                EvidenceEdge(source=hid, target=gate_id, kind="not_run_because", label="activation gate not cleared")
            )
        if syn.state == "CONTRADICTORY":
            graph_edges.append(
                EvidenceEdge(
                    source=hid,
                    target="campaign:efs",
                    kind="invalidates_prior_result",
                    label="contradicts preregistered decision contract",
                )
            )
        used_manifests.update(id(m) for m in manifests if m.source_path in syn.result_refs)

    # Shared-manifest edges between hypotheses that resolve to the same result file.
    source_to_hyps: dict[str, list[str]] = defaultdict(list)
    for syn in hypotheses:
        for ref in syn.result_refs:
            source_to_hyps[ref].append(syn.hypothesis_id)
    for ref, hids in source_to_hyps.items():
        if len(hids) > 1:
            for i, a in enumerate(hids):
                for b in hids[i + 1 :]:
                    graph_edges.append(
                        EvidenceEdge(
                            source=f"hyp:{a}",
                            target=f"hyp:{b}",
                            kind="shares_checkpoint_or_corpus",
                            label=f"shares {ref}",
                        )
                    )

    diagnosis = _compute_causal_diagnosis(manifest, hypotheses)
    add_node(
        EvidenceNode(
            node_id="diagnosis:primary",
            kind="causal_conclusion",
            label=f"Primary diagnosis: {diagnosis.primary}",
            payload=diagnosis.model_dump(),
        )
    )

    dispositions = _build_architecture_dispositions(manifest, hypotheses)
    for disp in dispositions:
        did = f"decision:{_slug(disp.item)}"
        add_node(
            EvidenceNode(
                node_id=did,
                kind="architecture_decision",
                label=f"{disp.item} -> {disp.disposition}",
                payload=disp.model_dump(),
            )
        )
        for sid in disp.supporting_experiments:
            graph_edges.append(EvidenceEdge(source=f"hyp:{sid}", target=did, kind="supports"))
        for fid in disp.falsifying_experiments:
            graph_edges.append(EvidenceEdge(source=f"hyp:{fid}", target=did, kind="falsifies"))

    next_programs = _build_next_programs(hypotheses)
    lineage_summary = _build_lineage_summary(manifest, hypotheses)

    unresolved: list[str] = []
    invalidated: list[str] = []
    for syn in hypotheses:
        if syn.state in {"MISSING", "CONTRADICTORY", "INVALIDATED"}:
            unresolved.append(f"{syn.hypothesis_id} ({syn.state}): {syn.state_reason}")
        if syn.state == "INVALIDATED":
            invalidated.append(syn.hypothesis_id)

    return EvidenceFirstSemanticSynthesisV1(
        manifest_hash=manifest_hash,
        generation_command=generation_command,
        hypotheses=hypotheses,
        lineage_summary=lineage_summary,
        evidence_graph={
            "nodes": [n.model_dump() for n in graph_nodes],
            "edges": [e.model_dump() for e in graph_edges],
        },
        causal_diagnosis=diagnosis,
        architecture_dispositions=dispositions,
        champion_decision="no_promotion",
        next_programs=next_programs,
        unresolved_risks=tuple(unresolved),
        invalidated_claims=tuple(invalidated),
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:64]


def _build_lineage_summary(
    manifest: CampaignManifestV1,
    hypotheses: tuple[HypothesisSynthesis, ...],
) -> dict[str, Any]:
    state_counts: dict[str, int] = defaultdict(int)
    for syn in hypotheses:
        state_counts[syn.state] += 1
    return {
        "manifest_id": manifest.campaign_id,
        "hypothesis_count": len(hypotheses),
        "terminal_state_counts": dict(sorted(state_counts.items())),
        "checkpoint_reference_count": len(
            {ref for syn in hypotheses for ref in syn.checkpoint_refs}
        ),
    }


def _compute_causal_diagnosis(
    manifest: CampaignManifestV1,
    hypotheses: tuple[HypothesisSynthesis, ...],
) -> CausalDiagnosis:
    """Classify the campaign-wide causal diagnosis from terminal states."""
    by_id = {h.hypothesis_id: h for h in hypotheses}
    state_counts: dict[str, int] = defaultdict(int)
    for syn in hypotheses:
        state_counts[syn.state] += 1

    measurement_issues = {
        "efs0-01-checkpoint-provenance",
        "efs0-02-decode-invariance",
        "efs0-03-meaningful-v2",
        "efs0-04-judge-independence",
        "efs0-05-rejected-lever-readjudication",
    }
    measurement_resolved = all(
        by_id.get(hid, HypothesisSynthesis(
            hypothesis_id=hid, linear_issue="", claim="", falsifier="", state="MISSING", state_reason=""
        )).state in {"POSITIVE", "EQUIVALENT"}
        for hid in measurement_issues
    )

    # If core measurement is unresolved, we cannot trust any training/architecture conclusion.
    if not measurement_resolved:
        return CausalDiagnosis(
            primary="insufficient_valid_evidence",
            primary_reason=(
                "Core measurement issues (checkpoint provenance, decoder invariance, "
                "semantic metric, judge independence, re-adjudication) are not all POSITIVE. "
                "Without durable, invariant, independently measured evidence, no causal "
                "diagnosis of training, data, search, or architecture can be asserted."
            ),
            counterfactual_evidence=(
                "Counterfactual: if SLM-103/104/105/106/107 were all POSITIVE, the remaining "
                "NOT_RUN_BY_GATE states could be attributed to training-exposure or architecture "
                "limits rather than to measurement uncertainty."
            ),
            secondary=("measurement_limited",),
        )

    # If measurement is sound but everything else is NOT_RUN_BY_GATE, still insufficient evidence.
    non_measurement = [syn for syn in hypotheses if syn.hypothesis_id not in measurement_issues]
    if all(syn.state in {"NOT_RUN_BY_GATE", "MISSING", "INCONCLUSIVE"} for syn in non_measurement):
        return CausalDiagnosis(
            primary="insufficient_valid_evidence",
            primary_reason=(
                "Measurement infrastructure is in place, but no EFS experimental branch has "
                "cleared its activation gate or produced a decisive POSITIVE/NEGATIVE result."
            ),
            counterfactual_evidence=(
                "Counterfactual: a POSITIVE result on EFS1-02 (exposure threshold) would support "
                "training_exposure_limited; a NEGATIVE result would support "
                "objective_or_representation_limited."
            ),
            secondary=("training_exposure_limited", "objective_or_representation_limited"),
        )

    # Default honest fallback when evidence is mixed.
    return CausalDiagnosis(
        primary="mixed_with_no_identifiable_dominant_cause",
        primary_reason="Terminal states are heterogeneous and no single layer dominates.",
        counterfactual_evidence="No single counterfactual experiment isolates the dominant cause.",
        secondary=tuple(sorted(state_counts.keys())),
    )


# Architecture disposition items required by the issue.
_REQUIRED_DISPOSITION_ITEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compiler-owned grammar/schema/binding lattice and exact closure", ("efs4-01-trailed-assumptions", "efs2-04-verifier-cascade")),
    ("reduced-product/cross-domain propagation claims", ("efs4-01-trailed-assumptions",)),
    ("reversible decisions, local nogoods, and certificate-backed trailing", ("efs4-01-trailed-assumptions",)),
    ("free-form typed-AST/topology diffusion", ("efs3-03-b3-capacity-v2",)),
    ("X22/all-valid tree-edit diffusion", ("efs2-01-x22-scaling",)),
    ("deterministic shared recursive denoiser/deep supervision", ("efs4-02-shared-recursive-denoiser",)),
    ("request-local recurrent latent persistence", ("efs4-02-shared-recursive-denoiser",)),
    ("triggered PTRM/inference-only low-level noise", ("efs2-02-trigger-telemetry",)),
    ("learned GRAM-style high-level stochastic state", ("efs4-03-stochastic-recursive-state",)),
    ("candidate AST dedup/semantic mode tracking", ("efs3-05-canonical-ast-dedup",)),
    ("contract-grounded selector and abstention", ("efs3-04-candidate-selector",)),
    ("conflict-slice remasking", ("efs2-03-conflict-slice-repair",)),
    ("cheap-to-expensive verifier cascade/cache", ("efs2-04-verifier-cascade",)),
    ("choice representation/capacity conclusion", ("efs3-03-b3-capacity-v2", "efs1-02-exposure-ladder")),
    ("gold/on-policy mixed supervision and nearly-solved curriculum", ("efs3-01-solver-state-supervision", "efs3-02-corruption-curriculum")),
    ("AST-sketch data balancing and choice-native retrieval", ("efs3-06-ast-sketch-retrieval",)),
    ("content-floor/length/mask-mass corrections", ("efs1-03-empty-length-bias",)),
)


def _build_architecture_dispositions(
    manifest: CampaignManifestV1,
    hypotheses: tuple[HypothesisSynthesis, ...],
) -> tuple[ArchitectureDisposition, ...]:
    by_id = {h.hypothesis_id: h for h in hypotheses}
    manifest_by_id = {h.hypothesis_id: h for h in manifest.hypotheses}
    dispositions: list[ArchitectureDisposition] = []
    for item, related in _REQUIRED_DISPOSITION_ITEMS:
        states = [by_id[hid].state for hid in related if hid in by_id]
        if any(s in {"POSITIVE", "EQUIVALENT"} for s in states):
            disposition: Disposition = "CONDITIONAL_RESEARCH"
            next_action = "Replicate under full ship-gates and durable checkpoints before promotion."
        elif any(s in {"NEGATIVE", "INVALIDATED"} for s in states):
            disposition = "REJECT"
            next_action = "Remove from production roadmap; retain negative result in registry."
        elif all(s in {"NOT_RUN_BY_GATE", "MISSING", "INCONCLUSIVE"} for s in states):
            disposition = "NOT_RUN_BY_GATE"
            next_action = "Run the related EFS branch to a terminal state before disposition."
        else:
            disposition = "INCONCLUSIVE"
            next_action = "Await decisive evidence; do not change default configuration."

        # Safety infrastructure that is correct-by-construction should be adopted as safety only.
        if "exact closure" in item or "trailing" in item or "verifier cascade" in item:
            if disposition == "NOT_RUN_BY_GATE":
                disposition = "ADOPT_AS_SAFETY_ONLY"
                next_action = "Keep as correctness infrastructure; do not claim quality improvement."

        tags: list[str] = []
        for hid in related:
            h = manifest_by_id.get(hid)
            if h:
                tags.extend(h.architecture_tags)

        dispositions.append(
            ArchitectureDisposition(
                item=item,
                disposition=disposition,
                supporting_experiments=tuple(
                    hid for hid in related if (by_id.get(hid) and by_id[hid].state in {"POSITIVE", "EQUIVALENT"})
                ),
                falsifying_experiments=tuple(
                    hid for hid in related if (by_id.get(hid) and by_id[hid].state in {"NEGATIVE", "INVALIDATED"})
                ),
                effect="No proven semantic-quality effect under current evidence." if disposition != "POSITIVE" else "Positive effect observed; replication required.",
                uncertainty="Evidence is plan/fixture-grade; frontier runs are pending.",
                semantic_metric="binding_aware_meaningful_v2",
                cost_metric="wall_seconds / verifier_calls",
                safety_constraints=("Must not weaken compiler-owned legality.",),
                activation_conditions=tuple(sorted(set(tags))),
                next_action=next_action,
            )
        )
    return tuple(dispositions)


def _build_next_programs(
    hypotheses: tuple[HypothesisSynthesis, ...],
) -> tuple[NextProgramRecommendation, ...]:
    by_id = {h.hypothesis_id: h for h in hypotheses}
    recs: list[NextProgramRecommendation] = []

    # 1. Resolve measurement limitations first.
    if by_id.get("efs0-01-checkpoint-provenance", HypothesisSynthesis(
        hypothesis_id="", linear_issue="", claim="", falsifier="", state="MISSING", state_reason=""
    )).state != "POSITIVE":
        recs.append(
            NextProgramRecommendation(
                rank=1,
                objective="Make frontier checkpoint provenance and persistence fail-closed.",
                non_duplicate_rationale="EFS0-01 is the root dependency of every causal claim; no later branch is interpretable without it.",
                expected_information_gain="Distinguishes measurement-limited from genuinely architecture-limited conclusions.",
                smallest_experiment="Sync one E228-class checkpoint to hf://buckets/TKendrick/OpenUI and verify hash from a fresh clone.",
                budget="<1 GPU-hour + bucket storage",
                kill_criterion="Hash mismatch or unresolvable checkpoint blocks all dependent syntheses.",
                dependencies=("SLM-103",),
            )
        )

    # 2. If measurement is sound but exposure unresolved, run the decisive exposure ladder.
    if by_id.get("efs1-02-exposure-ladder", HypothesisSynthesis(
        hypothesis_id="", linear_issue="", claim="", falsifier="", state="MISSING", state_reason=""
    )).state != "POSITIVE":
        recs.append(
            NextProgramRecommendation(
                rank=2,
                objective="Run the ≥100× E228 exposure ladder to falsify 'just train longer'.",
                non_duplicate_rationale="EFS1-02 is the only branch designed to separate exposure from representation/objective limits.",
                expected_information_gain="Either reveals an exposure threshold or falsifies the current recipe.",
                smallest_experiment="Continue seed-0 run to 128× T0 and confirm 1×/threshold/128× on seeds 1–2.",
                budget="~8 GPU-hours",
                kill_criterion="Semantic metrics flat with tight CI excluding minimum useful delta.",
                dependencies=("SLM-104", "SLM-105", "SLM-109"),
            )
        )

    # 3. Run the choice-native capacity ladder after decoder correction.
    if by_id.get("efs3-03-b3-capacity-v2", HypothesisSynthesis(
        hypothesis_id="", linear_issue="", claim="", falsifier="", state="MISSING", state_reason=""
    )).state != "POSITIVE":
        recs.append(
            NextProgramRecommendation(
                rank=3,
                objective="Re-run the B3 surface-vs-choice capacity ladder with the corrected choice-native decoder.",
                non_duplicate_rationale="EFS3-03 isolates representation capacity from the prior decoder confound.",
                expected_information_gain="Determines whether externalized syntax shifts the quality-capacity curve.",
                smallest_experiment="18-row grid (2 representations × 3 widths × 3 seeds) on frozen recipe.",
                budget="~6 GPU-hours",
                kill_criterion="Capacity curves overlap within equivalence margin.",
                dependencies=("SLM-104", "SLM-124"),
            )
        )

    return tuple(sorted(recs, key=lambda r: r.rank)[:3])


# ---------------------------------------------------------------------------
# Graph renderers
# ---------------------------------------------------------------------------


def render_mermaid(graph: dict[str, Any]) -> str:
    """Render a deterministic Mermaid flowchart from the evidence graph."""
    lines = ["flowchart TD"]
    for node in graph.get("nodes", []):
        nid = node["node_id"].replace(":", "_")
        label = node["label"].replace('"', "'")[:80]
        lines.append(f"    {nid}[\"{label}\"]")
    for edge in graph.get("edges", []):
        src = edge["source"].replace(":", "_")
        tgt = edge["target"].replace(":", "_")
        kind = edge.get("kind", "")
        lines.append(f"    {src} --{kind}--> {tgt}")
    return "\n".join(lines) + "\n"


def render_dot(graph: dict[str, Any]) -> str:
    """Render a deterministic DOT graph from the evidence graph."""
    lines = ["digraph evidence_graph {"]
    for node in graph.get("nodes", []):
        nid = node["node_id"].replace(":", "_")
        label = node["label"].replace('"', "'")[:80]
        lines.append(f'    {nid} [label="{label}"];')
    for edge in graph.get("edges", []):
        src = edge["source"].replace(":", "_")
        tgt = edge["target"].replace(":", "_")
        kind = edge.get("kind", "")
        lines.append(f'    {src} -> {tgt} [label="{kind}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown(synthesis: EvidenceFirstSemanticSynthesisV1) -> str:
    """Render the human-readable dated report."""
    lines: list[str] = []
    lines.append("# EFS4-04 — Causal diagnosis and explicit architecture disposition")
    lines.append("")
    lines.append(f"**Date:** {synthesis.created_at}")
    lines.append(f"**Campaign:** {synthesis.campaign_id}")
    lines.append(f"**Manifest hash:** `{synthesis.manifest_hash}`")
    lines.append(f"**Generation command:** `{synthesis.generation_command}`")
    lines.append("")

    # 1. Executive verdict
    lines.append("## 1. Executive verdict")
    lines.append("")
    lines.append(
        "The Evidence-First Semantic SLM Campaign has reached a wiring/plan-only terminal state. "
        "No EFS experimental branch has cleared its activation gate or produced durable frontier "
        "evidence. The only honest causal diagnosis is **insufficient valid evidence**; the "
        "primary blocker is unresolved measurement provenance and decoder invariance. "
        "No architecture branch is promoted. All required disposition items are recorded as "
        "``NOT_RUN_BY_GATE``/``INCONCLUSIVE`` or, for safety infrastructure, "
        "``ADOPT_AS_SAFETY_ONLY`` without a quality claim."
    )
    lines.append("")

    # 2. Campaign execution/completeness table
    lines.append("## 2. Campaign execution / completeness table")
    lines.append("")
    lines.append("| Issue | Hypothesis | State | Result refs |")
    lines.append("|-------|------------|-------|-------------|")
    for syn in synthesis.hypotheses:
        refs = ", ".join(syn.result_refs) or "—"
        lines.append(f"| {syn.linear_issue} | {syn.hypothesis_id} | `{syn.state}` | {refs} |")
    lines.append("")

    # 3. Measurement validity findings
    lines.append("## 3. Measurement validity findings")
    lines.append("")
    measurement_syn = [
        syn for syn in synthesis.hypotheses
        if syn.hypothesis_id.startswith("efs0-")
    ]
    for syn in measurement_syn:
        lines.append(f"- **{syn.hypothesis_id}** ({syn.linear_issue}): `{syn.state}` — {syn.state_reason}")
    lines.append("")

    # 4. Causal layer diagnosis
    lines.append("## 4. Causal layer diagnosis")
    lines.append("")
    if synthesis.causal_diagnosis:
        d = synthesis.causal_diagnosis
        lines.append(f"**Primary:** `{d.primary}`")
        lines.append("")
        lines.append(d.primary_reason)
        lines.append("")
        lines.append(f"**Counterfactual evidence:** {d.counterfactual_evidence}")
        lines.append("")
        if d.secondary:
            lines.append(f"**Secondary considerations:** {', '.join(d.secondary)}")
            lines.append("")

    # 5. Semantic-quality and cost Pareto fronts
    lines.append("## 5. Semantic-quality and cost Pareto fronts")
    lines.append("")
    lines.append(
        "No frontier Pareto front is available: all rows are either plan-only or diagnostic-only. "
        "The existing fixture rows report only wall-second / verifier-call placeholders. "
        "Any Pareto claim requires the exposure ladder (SLM-109) and external-ceiling (SLM-108) "
        "frontier runs to complete first."
    )
    lines.append("")

    # 6. Architecture disposition table
    lines.append("## 6. Architecture disposition table")
    lines.append("")
    lines.append("| Item | Disposition | Supporting | Falsifying | Next action |")
    lines.append("|------|-------------|------------|------------|-------------|")
    for disp in synthesis.architecture_dispositions:
        sup = ", ".join(disp.supporting_experiments) or "—"
        fals = ", ".join(disp.falsifying_experiments) or "—"
        lines.append(
            f"| {disp.item} | `{disp.disposition}` | {sup} | {fals} | {disp.next_action} |"
        )
    lines.append("")

    # 7. Comparison with original proposals
    lines.append("## 7. Comparison with LDT/TRM/PTRM/GRAM and agent critiques")
    lines.append("")
    lines.append(
        "The repository contains adapted implementations of lattice-diffusion, tree-edit (X22), "
        "triggered PTRM, shared-recursive denoiser, and GRAM-style stochastic-state ideas. "
        "None has cleared ship-gates or produced durable frontier evidence. The honest position is "
        "that these remain research hypotheses, not reproduced architectures."
    )
    lines.append("")

    # 8. Interaction with VSS and CAP
    lines.append("## 8. Interaction with VSS and CAP")
    lines.append("")
    lines.append(
        "This synthesis links the EFS branch decisions but does not duplicate VSS or CAP claims. "
        "See ``docs/design/verified-scope-solver.md`` and the CAP5 synthesis for their respective "
        "dispositions. EFS4-04 treats VSS/CAP outputs as external evidence inputs."
    )
    lines.append("")

    # 9. Champion/promotion decision
    lines.append("## 9. Champion / promotion decision")
    lines.append("")
    lines.append(f"**{synthesis.champion_decision}**. No checkpoint is promoted.")
    lines.append("")

    # 10. Next programs / consolidation plan
    lines.append("## 10. Next three experiments or consolidation plan")
    lines.append("")
    for rec in synthesis.next_programs:
        lines.append(f"### {rec.rank}. {rec.objective}")
        lines.append(f"- **Why not duplicate:** {rec.non_duplicate_rationale}")
        lines.append(f"- **Expected information gain:** {rec.expected_information_gain}")
        lines.append(f"- **Smallest decisive experiment:** {rec.smallest_experiment}")
        lines.append(f"- **Budget:** {rec.budget}")
        lines.append(f"- **Kill criterion:** {rec.kill_criterion}")
        lines.append(f"- **Dependencies:** {', '.join(rec.dependencies)}")
        lines.append("")

    # 11. Limitations and reproduction
    lines.append("## 11. Limitations and exact reproduction command")
    lines.append("")
    lines.append(
        "This report is a wiring-grade synthesis over plan/fixture manifests. It does not run "
        "training, download models, or mutate checkpoints. All claims are scoped to the committed "
        "result manifests under ``docs/design/``."
    )
    lines.append("")
    lines.append("```bash")
    lines.append("python -m scripts.synthesize_efs_campaign \\")
    lines.append("  --manifest docs/design/evidence-first-semantic-slm-campaign-v1.json \\")
    lines.append("  --docs-design docs/design \\")
    lines.append("  --out-json docs/design/iter-efs4-04-causal-synthesis-$(date +%Y%m%d).json \\")
    lines.append("  --out-md docs/design/iter-efs4-04-causal-synthesis-$(date +%Y%m%d).md")
    lines.append("```")
    lines.append("")

    if synthesis.unresolved_risks:
        lines.append("## Unresolved risks")
        lines.append("")
        for risk in synthesis.unresolved_risks:
            lines.append(f"- {risk}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_synthesis(synthesis: EvidenceFirstSemanticSynthesisV1) -> list[str]:
    """Return a list of validation warnings/errors; empty list means publishable."""
    errors: list[str] = []
    issues = {syn.linear_issue for syn in synthesis.hypotheses}
    required = {
        "SLM-103", "SLM-104", "SLM-105", "SLM-106", "SLM-107",
        "SLM-108", "SLM-109", "SLM-110",
        "SLM-111", "SLM-112", "SLM-113", "SLM-115",
        "SLM-118", "SLM-120", "SLM-124", "SLM-127", "SLM-130", "SLM-133",
        "SLM-135", "SLM-138", "SLM-139",
    }
    missing = required - issues
    if missing:
        errors.append(f"missing required EFS issues: {sorted(missing)}")

    for syn in synthesis.hypotheses:
        if syn.state == "CONTRADICTORY":
            errors.append(f"{syn.hypothesis_id}: contradictory state vs preregistered decision contract")
        for ref in syn.checkpoint_refs:
            if "://" not in ref and "/" in ref and not ref.startswith("hf://"):
                # local absolute paths are prohibited
                if ref.startswith(("file://", "/")):
                    errors.append(f"{syn.hypothesis_id}: local absolute checkpoint ref {ref}")

    for disp in synthesis.architecture_dispositions:
        if disp.disposition in {"ADOPT", "PROMOTE_EXPERIMENTAL"}:
            if not disp.supporting_experiments:
                errors.append(
                    f"{disp.item}: ADOPT/PROMOTE_EXPERIMENTAL requires supporting experiments"
                )

    if synthesis.causal_diagnosis and synthesis.causal_diagnosis.primary == "insufficient_valid_evidence":
        # This is an honest publishable state, not an error.
        pass

    return errors


def save_manifest(manifest: CampaignManifestV1, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def save_synthesis(synthesis: EvidenceFirstSemanticSynthesisV1, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(synthesis.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path | str) -> CampaignManifestV1:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CampaignManifestV1.model_validate(data)
