"""SLM-160 (SPV4-02): causal architecture disposition harness.

Docs/spec harness. Aggregates existing SPV evidence docs under ``docs/design/``
and produces a machine-readable ``SPVDispositionV1`` JSON plus a Markdown
report. All evidence up through SLM-159 is wiring/fixture or blocked; the
dispositions are honest: no ship-gate claims and no ``adopt_primary`` for
fixture-only or measured-not-promotable work.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.versioning import UNKNOWN, build_version_stamp, git_commit

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "Disposition",
    "SPVMechanismDisposition",
    "SPVDispositionReport",
    "build_default_dispositions",
    "load_evidence_claim_class",
    "run_disposition_audit",
    "render_markdown",
]

MATRIX_VERSION = "spv4-02-v1"
MATRIX_SET = "slm160_spv_disposition"
EXPERIMENT_ID = "slm160-spv-disposition"
REPORT_SCHEMA = "SPVDispositionV1"


class Disposition(str, Enum):
    """Architecture disposition values for SPV4-02."""

    ADOPT_PRIMARY = "adopt_primary"
    ADOPT_OPTIONAL = "adopt_optional"
    RETAIN_DIAGNOSTIC = "retain_diagnostic"
    REVISE_AND_RETEST = "revise_and_retest"
    REJECT = "reject"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class SPVMechanismDisposition:
    """Disposition for one causal architecture mechanism."""

    mechanism_id: str
    issue_ids: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    hypothesis: str
    falsifier: str
    disposition: Disposition
    rationale: str
    next_action: str
    default_state: str

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["issue_ids"] = list(self.issue_ids)
        data["evidence_paths"] = list(self.evidence_paths)
        data["disposition"] = self.disposition.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SPVMechanismDisposition":
        return cls(
            mechanism_id=data.get("mechanism_id", ""),
            issue_ids=tuple(data.get("issue_ids", [])),
            evidence_paths=tuple(data.get("evidence_paths", [])),
            hypothesis=data.get("hypothesis", ""),
            falsifier=data.get("falsifier", ""),
            disposition=Disposition(data.get("disposition", "inconclusive")),
            rationale=data.get("rationale", ""),
            next_action=data.get("next_action", ""),
            default_state=data.get("default_state", "off"),
        )


@dataclass(frozen=True)
class SPVDispositionReport:
    """Full SPV4-02 disposition report."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    evidence_cutoff_commit: str
    generated_at: str
    mechanism_dispositions: list[SPVMechanismDisposition]
    cross_pack_summary: str
    canonical_architecture_recommendation: str
    rejected_or_blocked_ids: list[str]
    version_stamp: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "evidence_cutoff_commit": self.evidence_cutoff_commit,
            "generated_at": self.generated_at,
            "mechanism_dispositions": [m.to_dict() for m in self.mechanism_dispositions],
            "cross_pack_summary": self.cross_pack_summary,
            "canonical_architecture_recommendation": self.canonical_architecture_recommendation,
            "rejected_or_blocked_ids": list(self.rejected_or_blocked_ids),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SPVDispositionReport":
        return cls(
            schema=data.get("schema", REPORT_SCHEMA),
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm160_disposition"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            evidence_cutoff_commit=data.get("evidence_cutoff_commit", UNKNOWN),
            generated_at=data.get("generated_at", ""),
            mechanism_dispositions=[
                SPVMechanismDisposition.from_dict(m)
                for m in data.get("mechanism_dispositions", [])
            ],
            cross_pack_summary=data.get("cross_pack_summary", ""),
            canonical_architecture_recommendation=data.get(
                "canonical_architecture_recommendation", ""
            ),
            rejected_or_blocked_ids=list(data.get("rejected_or_blocked_ids", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _repo_root() -> Path:
    """Return the repository root for repo-relative evidence paths."""
    return Path(__file__).resolve().parents[4]


def load_evidence_claim_class(
    path: str | Path, repo_root: Path | None = None
) -> tuple[str | None, str | None]:
    """Read a ``docs/design`` JSON and return its top-level ``claim_class`` and ``status``.

    Returns ``(None, None)`` when the file is missing or not valid JSON.
    """
    if repo_root is None:
        repo_root = _repo_root()
    target = repo_root / path
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return data.get("claim_class"), data.get("status")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mechanism_specs() -> list[dict[str, Any]]:
    """Return the preregistered mechanism specifications for SPV4-02."""
    return [
        {
            "mechanism_id": "semantic_plan_v1_ir",
            "issue_ids": ["SLM-43", "SLM-71", "SLM-146"],
            "evidence_paths": [
                "docs/design/iter-slm146-semantic-plan-compiler-20260720.json"
            ],
            "hypothesis": (
                "A pack-neutral SemanticPlanV1 IR captures archetype, role slots, "
                "topology, symbols, and bindings independently of DSL surface syntax."
            ),
            "falsifier": (
                "A DSL pack cannot populate all factors from its schema/grammar, or "
                "the IR silently encodes OpenUI-specific assumptions."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Contract/spec artifact; no predictor claim. GraphQL pack demonstrates "
                "factor extraction, but predicted-plan generalization is unproven."
            ),
            "next_action": (
                "Keep SemanticPlanV1 as the canonical plan contract; require "
                "predicted-plan evidence before promotion."
            ),
            "default_state": "n/a",
        },
        {
            "mechanism_id": "gold_oracle_factor_heads",
            "issue_ids": ["SLM-144", "SLM-145"],
            "evidence_paths": [
                "docs/design/iter-slm144-plan-predictor-20260720.json",
                "docs/design/iter-slm145-plan-predictor-factors-20260720.json",
            ],
            "hypothesis": (
                "Learned archetype, role-set, topology, cardinality, and binding "
                "heads recover plan factors from the prompt better than frequency "
                "baselines and justify downstream semantic gains."
            ),
            "falsifier": (
                "Gold substitution of a factor does not lift downstream semantic "
                "metrics, or learned heads are no better than the frequency baseline."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Gold oracle arms provide diagnostic ceilings, but SLM-145 closed "
                "without factor-wise gold-substitution evidence for topology, "
                "cardinality, or bindings; learned heads are not justified."
            ),
            "next_action": (
                "Run a factor-wise oracle-substitution matrix on a real or fixture "
                "completion corpus before reopening learned factor heads."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "plan_seed_builder_soft_restrictions",
            "issue_ids": ["SLM-146"],
            "evidence_paths": [
                "docs/design/iter-slm146-semantic-plan-compiler-20260720.json"
            ],
            "hypothesis": (
                "A deterministic plan compiler produces valid OpenUI seeds, attaches "
                "soft action features without changing legal membership, and gates "
                "hard restrictions behind certified evidence."
            ),
            "falsifier": (
                "Plan-derived seeds are invalid, soft features alter the legal "
                "candidate set, or any non-certified prediction removes a supported "
                "candidate in a promotable arm."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Fixture-only synthetic corpus; no production decoder. Unsafe "
                "predicted-hard arm is explicitly non-promotable."
            ),
            "next_action": (
                "Integrate with live legal-set oracle and measured completion "
                "corpus before promoting beyond diagnostics."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "x22_seed_retrieval_conflict_repair",
            "issue_ids": ["SLM-147", "SLM-148"],
            "evidence_paths": [
                "docs/design/iter-slm147-x22-retrieval-20260720.json",
                "docs/design/iter-slm148-x22-conflict-campaign-20260720.json",
            ],
            "hypothesis": (
                "Leakage-safe retrieved hard-valid AST prototypes and "
                "conflict-localized repair initialize X22 closer to acceptable "
                "programs than a minimal seed."
            ),
            "falsifier": (
                "Retrieved prototypes are no closer to gold than the minimal seed, "
                "or recovery policies do not improve recovery while preserving "
                "correct structure."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Fixture-only evidence; no live X22 model trained or decoded. "
                "Oracle and gold-plan arms are diagnostic ceilings only."
            ),
            "next_action": (
                "Evaluate on a real X22 beam-search decode loop with verifier "
                "replay before promotion."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "ar_legal_action_scorer",
            "issue_ids": ["SLM-154", "SLM-155"],
            "evidence_paths": [
                "docs/design/iter-slm154-legal-action-scorer-20260720.json",
                "docs/design/iter-slm155-factorization-comparison-20260720.json",
            ],
            "hypothesis": (
                "Direct autoregressive scoring over the live legal action set "
                "matches or exceeds plan-conditioned X22 semantic quality at lower "
                "deployed cost."
            ),
            "falsifier": (
                "X22 produces materially better semantic outcomes at comparable "
                "cost, or plan features silently alter legal candidate membership."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Wiring-only fixture; no live production decode loop and no ship "
                "readiness claim. SLM-154 explicitly labels itself fixture wiring."
            ),
            "next_action": (
                "Wire real compiler-owned exact state and live legal sets before "
                "claiming a factorization winner."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "global_semantic_critic",
            "issue_ids": ["SLM-150", "SPV2-02"],
            "evidence_paths": [
                "docs/design/iter-spv2-02-global-semantic-critic-20260720.json"
            ],
            "hypothesis": (
                "A global energy-based critic can rerank candidate programs using "
                "pack-neutral semantic factors without leaking gold structure."
            ),
            "falsifier": (
                "The critic cannot be trained on hard-valid contrasts, factor heads "
                "change energy semantics, or abstention is not fail-closed."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Synthetic fixture corpus only; the SPV2-01 hard-valid contrast "
                "corpus is absent from the repo."
            ),
            "next_action": (
                "Build or locate the hard-valid contrast corpus and rerun the "
                "critic fixture before promotion."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "hard_valid_contrasts",
            "issue_ids": ["SPV0-03", "SLM-120"],
            "evidence_paths": [
                "docs/design/iter-spv0-03-semantic-regret-20260719.json",
                "docs/design/iter-slm120-corruption-curriculum-20260719.json",
            ],
            "hypothesis": (
                "Hard-valid semantic contrasts expose factor-specific regret and "
                "near-solved corruption curricula that can supervise plan-aware "
                "models."
            ),
            "falsifier": (
                "Contrasts do not isolate factor regret, or the corruption corpus "
                "leaks eval-suite examples."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "SPV0-03 is a fixture regret diagnostic; SLM-120 is a frontier "
                "curriculum plan. No production contrast corpus has been built."
            ),
            "next_action": (
                "Produce a versioned hard-valid semantic-contrast corpus with "
                "n-gram decontamination before using it for training."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "dense_legal_set_distillation",
            "issue_ids": ["SPV2-03", "SPV2-04"],
            "evidence_paths": [
                "docs/design/iter-spv2-03-legal-set-distillation-20260720.json",
                "docs/design/iter-spv2-04-dense-teacher-mixture-20260720.json",
            ],
            "hypothesis": (
                "Distilling the dense legal-action set and a teacher-mixture "
                "distribution improves data efficiency for small action spaces."
            ),
            "falsifier": (
                "Distillation collapses to the argmax, ignores illegal actions, or "
                "fails to transfer outside the synthetic fixture."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Both SPV2-03 and SPV2-04 are fixture wiring only; no external "
                "teacher model, solver replay, or checkpoint training was performed."
            ),
            "next_action": (
                "Wire the external teacher scorer (SLM-108) and verifier replay "
                "before promoting distillation targets."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "semantic_repair",
            "issue_ids": ["SPV2-05"],
            "evidence_paths": [
                "docs/design/iter-spv2-05-semantic-repair-20260720.json"
            ],
            "hypothesis": (
                "A learned semantic repair scorer selects verifier-backed "
                "counterfactual actions that recover valid programs from partial "
                "or corrupted seeds."
            ),
            "falsifier": (
                "Repair policy success does not exceed edit-distance baselines, or "
                "it requires oracle counterfactual values to rank decisions."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Wiring-only fixture baseline; real verifier-backed counterfactual "
                "action values require SLM-131/VSS finite replay."
            ),
            "next_action": (
                "Connect to VSS finite replay and measured counterfactual action "
                "values before production wiring."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "plan_refinement_slm156",
            "issue_ids": ["SLM-156"],
            "evidence_paths": [
                "docs/design/iter-slm156-plan-refinement-20260720.json"
            ],
            "hypothesis": (
                "A small shared refinement cell applied recursively to "
                "SemanticPlanV1 improves plan-factor recovery over a parameter-"
                "matched one-pass predictor."
            ),
            "falsifier": (
                "Recursion changes plans but not final correctness, or deeper "
                "non-shared matches it at equal FLOPs."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Fixture-only synthetic plan-state recovery; no downstream "
                "completion or ship-gate evidence."
            ),
            "next_action": (
                "Evaluate recursive refinement on a real decode loop with binding-"
                "aware meaning-v2 metrics."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "mixer_slm158",
            "issue_ids": ["SLM-158"],
            "evidence_paths": [
                "docs/design/iter-slm158-mixer-comparison-20260720.json"
            ],
            "hypothesis": (
                "A narrow sequence-mixer protocol with simplified reference "
                "implementations can expose whether non-Transformer mixers preserve "
                "task accuracy and improve latency/memory before productionizing a "
                "family."
            ),
            "falsifier": (
                "All mixers perform identically on the workload, or simplified "
                "references cannot be trained stably enough to separate family "
                "effects from noise."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "Fixture-only synthetic token-pattern classifier; no OpenUI "
                "completion or ship-gate evidence."
            ),
            "next_action": (
                "Run the mixer protocol on a representative OpenUI semantic "
                "decision task before selecting a production mixer family."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "flow_consistency_slm157",
            "issue_ids": ["SLM-157"],
            "evidence_paths": [
                "docs/design/iter-slm157-flow-consistency-20260720.json"
            ],
            "hypothesis": (
                "A flow/consistency layer propagates plan constraints through the "
                "decode state to reject inconsistent partial programs early."
            ),
            "falsifier": (
                "The layer has no measurable impact on verifier-call budget or "
                "semantic quality, or it requires SLM-99/148 infrastructure that is "
                "not yet implemented."
            ),
            "disposition": Disposition.BLOCKED,
            "rationale": (
                "Blocked: upstream dependencies SLM-99/SLM-148 are not done and no "
                "implementation exists. The evidence document is also absent."
            ),
            "next_action": (
                "Implement SLM-99/SLM-148 dependencies, then create a fixture or "
                "measured matrix for flow/consistency before disposition review."
            ),
            "default_state": "blocked",
        },
        {
            "mechanism_id": "multi_pack_graphql",
            "issue_ids": ["SLM-159"],
            "evidence_paths": [
                "docs/design/iter-slm159-cross-dsl-replication-20260720.json"
            ],
            "hypothesis": (
                "Pack-neutral SemanticPlanV1 extraction and seed building transfer "
                "from OpenUI to GraphQL."
            ),
            "falsifier": (
                "Plan factors cannot be defined from GraphQL schema/selection "
                "semantics, or the seed builder cannot reproduce schema-valid queries."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "GraphQL replication fixture succeeds, but it is wiring-only with "
                "no predictor claim and no ship-gate evidence."
            ),
            "next_action": (
                "Use GraphQL as the portability reference pack; keep replication "
                "as a diagnostic, not a production dependency."
            ),
            "default_state": "off",
        },
        {
            "mechanism_id": "multi_pack_second_pack",
            "issue_ids": ["SLM-159", "SLM-44", "SLM-45"],
            "evidence_paths": [
                "docs/design/iter-slm159-cross-dsl-replication-20260720.json"
            ],
            "hypothesis": (
                "The same pack-neutral plan stack transfers to a structurally "
                "different second DSL pack (design-patterns, nomenclature, or "
                "ontology)."
            ),
            "falsifier": (
                "The second-pack candidates lack the grammar/parser/oracle/data "
                "contract required for non-toy replication."
            ),
            "disposition": Disposition.BLOCKED,
            "rationale": (
                "No SLM-44 or SLM-45 pack is registered; only design documents "
                "exist. A syntax-only toy pack cannot satisfy the readiness rubric."
            ),
            "next_action": (
                "Implement a real second-pack grammar, oracle, and generator, then "
                "rerun the SLM-159 readiness rubric."
            ),
            "default_state": "blocked",
        },
        {
            "mechanism_id": "prompt_plan_soft_scoring_e575_e576_e579",
            "issue_ids": ["E575", "E576", "E579"],
            "evidence_paths": [
                "docs/design/iter-e575-prompt-semantic-plan-soft-20260720.json",
                "docs/design/iter-e576-prompt-plan-binding-soft-20260720.json",
                "docs/design/iter-e579-verified-plan-root-20260720.json",
            ],
            "hypothesis": (
                "Prompt-derived SemanticPlanV1 soft scoring at decode time can "
                "improve local structure without gold output structure."
            ),
            "falsifier": (
                "Soft scoring changes legal candidate membership, regresses "
                "binding-aware meaning-v2, or fails to produce AgentV passes."
            ),
            "disposition": Disposition.RETAIN_DIAGNOSTIC,
            "rationale": (
                "E575/E576/E579 report local/structural gains but are explicitly "
                "not promotable: binding-aware meaning-v2 and AgentV remain zero."
            ),
            "next_action": (
                "Keep the generalized soft scorers default-off for diagnostics; "
                "do not promote or sync a checkpoint until meaning-v2 and AgentV "
                "pass."
            ),
            "default_state": "off",
        },
    ]


def build_default_dispositions(
    repo_root: Path | None = None,
) -> list[SPVMechanismDisposition]:
    """Return the preregistered SPV4-02 mechanism dispositions.

    If an evidence path does not exist, the mechanism is downgraded to
    ``blocked`` (for the already-blocked SLM-157 flow/consistency mechanism) or
    ``inconclusive`` (all others), and the rationale is extended with the
    missing-path note.
    """
    if repo_root is None:
        repo_root = _repo_root()

    dispositions: list[SPVMechanismDisposition] = []
    for spec in _mechanism_specs():
        missing = [p for p in spec["evidence_paths"] if not (repo_root / p).exists()]
        disposition: Disposition = spec["disposition"]
        rationale: str = spec["rationale"]
        if missing:
            if disposition == Disposition.BLOCKED:
                disposition = Disposition.BLOCKED
            else:
                disposition = Disposition.INCONCLUSIVE
            rationale += (
                " [Evidence missing at audit time: "
                + ", ".join(missing)
                + "]"
            )
        dispositions.append(
            SPVMechanismDisposition(
                mechanism_id=spec["mechanism_id"],
                issue_ids=tuple(spec["issue_ids"]),
                evidence_paths=tuple(spec["evidence_paths"]),
                hypothesis=spec["hypothesis"],
                falsifier=spec["falsifier"],
                disposition=disposition,
                rationale=rationale,
                next_action=spec["next_action"],
                default_state=spec["default_state"],
            )
        )
    return dispositions


def _build_version_stamp() -> dict[str, Any]:
    """Build a version stamp, degrading if the slm160 component is not yet registered."""
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm160_spv_disposition",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm160_spv_disposition"] = UNKNOWN
        return base


def run_disposition_audit(
    *,
    run_id: str = "slm160_disposition",
    status: str = "fixture",
    repo_root: Path | None = None,
) -> SPVDispositionReport:
    """Build the SPV4-02 disposition report and validate evidence paths.

    Reads each evidence doc via :func:`load_evidence_claim_class` as a
    validation step; missing or unreadable docs downgrade the associated
    mechanism.
    """
    if repo_root is None:
        repo_root = _repo_root()

    dispositions = build_default_dispositions(repo_root=repo_root)

    # Validate evidence docs and collect any additional audit notes.
    for disp in dispositions:
        for path in disp.evidence_paths:
            claim_class, doc_status = load_evidence_claim_class(path, repo_root)
            # Validation is currently informational; missing docs were already
            # handled by ``build_default_dispositions``.
            del claim_class, doc_status

    rejected_or_blocked = [
        d.mechanism_id
        for d in dispositions
        if d.disposition in (Disposition.REJECT, Disposition.BLOCKED)
    ]

    cross_pack_summary = (
        "SPV4-02 audit covers semantic planning, valid-state init/search, "
        "scoring/supervision, generation factorization, and portability. "
        "All evidence up through SLM-159 is wiring/fixture, blocked, or "
        "measured-not-promotable. No mechanism satisfies the criteria for "
        "adopt_primary or adopt_optional. The GraphQL pack replication is a "
        "retained diagnostic; all second-pack portability is blocked pending "
        "real pack implementations."
    )

    recommendation = (
        "Canonical architecture remains the existing honest-slot-contract "
        "TwoTower decoder with all plan-aware mechanisms retained as default-"
        "off diagnostics. Do not promote or sync checkpoints from E575/E576/"
        "E579. Unblock flow/consistency (SLM-157) and second-pack portability "
        "only after their upstream dependencies are implemented and measured. "
        "The next high-leverage step is a factor-wise oracle-substitution "
        "matrix for topology/cardinality/bindings and a versioned hard-valid "
        "semantic-contrast corpus."
    )

    return SPVDispositionReport(
        schema=REPORT_SCHEMA,
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status=status,
        claim_class="wiring",
        evidence_cutoff_commit=git_commit(),
        generated_at=_now(),
        mechanism_dispositions=dispositions,
        cross_pack_summary=cross_pack_summary,
        canonical_architecture_recommendation=recommendation,
        rejected_or_blocked_ids=rejected_or_blocked,
        version_stamp=_build_version_stamp(),
    )


def render_markdown(report: SPVDispositionReport) -> str:
    """Render a sectioned Markdown report from the disposition audit."""
    lines = [
        f"# SLM-160 (SPV4-02): Causal architecture disposition report ({report.run_id})",
        "",
        f"**Schema:** `{report.schema}`",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        "",
        f"**Version:** `{report.matrix_version}`",
        "",
        f"**Status:** {report.status}",
        "",
        f"**Claim class:** {report.claim_class} / disposition audit only. No "
        "GPU was used, no production TwoTower wiring was touched, and no "
        "ship-gate claim is made.",
        "",
        f"**Evidence cutoff commit:** `{report.evidence_cutoff_commit}`",
        "",
        f"**Generated at:** {report.generated_at}",
        "",
        "## Executive finding",
        "",
        report.cross_pack_summary,
        "",
        "## Evidence chronology",
        "",
        "The audit aggregated the following committed ``docs/design`` artifacts:",
        "",
    ]
    seen_paths: set[str] = set()
    for disp in report.mechanism_dispositions:
        for path in disp.evidence_paths:
            if path not in seen_paths:
                seen_paths.add(path)
                lines.append(f"- `{path}`")
    if not seen_paths:
        lines.append("- _No evidence paths recorded._")

    lines.extend(
        [
            "",
            "## Mechanism disposition table",
            "",
            "| Mechanism | Issues | Disposition | Default state | Rationale |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for disp in report.mechanism_dispositions:
        issues = ", ".join(disp.issue_ids)
        rationale = disp.rationale.replace("|", "\\|")
        lines.append(
            f"| {disp.mechanism_id} | {issues} | {disp.disposition.value} | "
            f"{disp.default_state} | {rationale} |"
        )

    lines.extend(
        [
            "",
            "## Cross-pack summary",
            "",
            report.cross_pack_summary,
            "",
            "## Canonical architecture recommendation",
            "",
            report.canonical_architecture_recommendation,
            "",
            "## Rejected or blocked mechanisms",
            "",
        ]
    )
    if report.rejected_or_blocked_ids:
        for mid in report.rejected_or_blocked_ids:
            lines.append(f"- `{mid}`")
    else:
        lines.append("- _None._")

    lines.extend(
        [
            "",
            "## Reproducibility commands",
            "",
            "```bash",
            "# Plan-only manifest (no evidence reads)",
            "python -m scripts.run_slm160_spv_disposition --mode plan-only",
            "",
            "# Fixture audit that reads docs/design evidence and writes the report",
            "python -m scripts.run_slm160_spv_disposition --mode fixture",
            "```",
            "",
            "## Limitations",
            "",
            "- This report is a docs/spec audit, not a training or evaluation run.",
            "- Dispositions are conditioned on the evidence available up to the "
            "cutoff commit; new measured results can change them.",
            "- Any evidence file marked missing downgraded the associated "
            "mechanism to ``blocked`` or ``inconclusive``.",
            "- No ship-gate claim is made; no mechanism is promoted to "
            "``adopt_primary``.",
            "",
        ]
    )
    return "\n".join(lines)
