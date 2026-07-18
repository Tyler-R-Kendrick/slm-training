"""VSS4-02 (SLM-75): verified-scope-solver research campaign as typed specs.

The matched R0-R6 evaluation matrix (see
``slm_training.harnesses.model_build.verified_solver_matrix``) is driven as a
strict-schema :class:`HypothesisMatrix` through the existing autoresearch
engine — no parallel ad-hoc loop.  The campaign encodes the four candidate
hypotheses from the VSS4-02 spec plus a matched control, each grounded in the
committed verified-scope-solver design, the VSS4-02 fixture memo, and the
fixture matrix results, so the engine's validation, bounded command
compilation, and feedback contract apply end-to-end.

Nothing here executes on import: builders return specs; execution stays behind
``scripts/autoresearch.py`` / ``execute_commands`` with the honest ship gates
always appended.  The campaign runs on the ``grammar_diffusion`` track so the
scope/topology solver knobs compile to bounded flags.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from slm_training.autoresearch.schemas import (
    CampaignSpec,
    CategoricalNoveltyAudit,
    EvidenceItem,
    EvidenceSnapshot,
    EvidenceUse,
    ExperimentKnobs,
    ExperimentSpec,
    HypothesisCandidate,
    HypothesisMatrix,
    ResearchSource,
)

# src/slm_training/autoresearch/verified_scope_matrix.py -> repo root parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]

CAMPAIGN_ID = "verified-scope-solver"
MATRIX_ID = "verified-scope-solver-m1"

# Committed evidence trail: one file per validation role.
RESEARCH_DOC = "docs/design/verified-scope-solver.md"
TRACE_DOC = "docs/design/vss4-02-matched-matrix-metrics-20260718.md"
RESULT_DOC = "docs/design/verified-scope-solver-matrix-results.json"
BENCHMARK_DOC = "docs/design/verified-scope-solver-benchmark.md"

_CITATIONS = (RESEARCH_DOC, TRACE_DOC, RESULT_DOC)


def build_vss_campaign() -> CampaignSpec:
    return CampaignSpec(
        campaign_id=CAMPAIGN_ID,
        objective=(
            "Prove-then-measure verified scope solving under matched controls: "
            "exact closure, dependency capsules, cost-to-go energy ranking, and "
            "late surface realization, with correctness gates evaluated before "
            "any quality or search-work gain."
        ),
        primary_metric="held_out.meaningful_program_rate",
        track="grammar_diffusion",
        researcher_mode="agent",
        notes=(
            "VSS4-02 matched evaluation matrix (R0-R6). Correctness authority is "
            "gated separately from search efficiency and output quality: a row "
            "that produces one false certified prune, removes one unknown "
            "candidate, or returns an unverified solved output fails regardless "
            "of quality. Fixture wiring is CPU/torch-free (VSS4-01 benchmark); "
            "frontier rows are specified but not run until VSS4-03."
        ),
    )


def build_vss_evidence(repo_root: Path | str | None = None) -> EvidenceSnapshot:
    """Snapshot the committed VSS4-02 evidence files (real hashes, no stubs)."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    kinds = {
        RESEARCH_DOC: "repo_lineage",  # -> research role
        TRACE_DOC: "run_insight",  # -> prior_trace role
        RESULT_DOC: "evaluation",  # -> prior_result role
    }
    items = []
    for rel, kind in kinds.items():
        data = (root / rel).read_bytes()
        items.append(
            EvidenceItem(
                path=rel,
                kind=kind,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    snapshot_id = "vss402-" + hashlib.sha256(
        "".join(item.sha256 for item in items).encode("utf-8")
    ).hexdigest()[:16]
    return EvidenceSnapshot(
        snapshot_id=snapshot_id,
        roots=("docs/design",),
        items=tuple(items),
    )


def build_vss_sources() -> list[ResearchSource]:
    return [
        ResearchSource(
            source_id="vss-contract",
            kind="repo_lineage",
            title="Bounded verified-scope-solver contract (VSS0-01)",
            uri=RESEARCH_DOC,
        ),
        ResearchSource(
            source_id="vss-benchmark",
            kind="repo_lineage",
            title="Verified-scope-solver matched matrix + benchmark (VSS4-02)",
            uri=BENCHMARK_DOC,
        ),
    ]


def _evidence_uses() -> tuple[EvidenceUse, ...]:
    return (
        EvidenceUse(
            role="research",
            citation=RESEARCH_DOC,
            contribution=(
                "The verified-scope-solver contract fixes SUPPORTED/"
                "UNSUPPORTED/UNKNOWN semantics and the removal-permission rule "
                "this candidate must not violate."
            ),
        ),
        EvidenceUse(
            role="prior_trace",
            citation=TRACE_DOC,
            contribution=(
                "The VSS4-02 fixture memo records the CPU wiring run and the "
                "hard-gate pass this candidate re-measures at frontier scale."
            ),
        ),
        EvidenceUse(
            role="prior_result",
            citation=RESULT_DOC,
            contribution=(
                "The fixture matrix results supply the matched R0 control and "
                "the zero-default metric schema this candidate populates."
            ),
        ),
    )


def _candidate(
    experiment_id: str,
    hypothesis: str,
    rationale: str,
    expected: str,
    knobs: ExperimentKnobs,
    *,
    proposed_element: str,
    regime_transition: bool = False,
) -> HypothesisCandidate:
    experiment = ExperimentSpec(
        experiment_id=experiment_id,
        campaign_id=CAMPAIGN_ID,
        hypothesis=hypothesis,
        rationale=rationale,
        expected_effect=expected,
        falsification_criteria=(
            "Any hard gate fails (false certified prune, removed unknown "
            "candidate, certificate replay failure, unverified solved output, "
            "candidate-set parity failure, or semantic-IR mutation), or the "
            "declared work/quality delta does not improve on the matched "
            "control with every existing ship gate in force.",
        ),
        stop_conditions=(
            "Stop after the configured steps; no correctness or ship gate may "
            "be weakened to obtain a quality or search-work gain.",
        ),
        citations=_CITATIONS,
        knobs=knobs,
    )
    return HypothesisCandidate(
        experiment=experiment,
        evidence_uses=_evidence_uses(),
        novelty=CategoricalNoveltyAudit(
            transition_kind=(
                "regime_transition_candidate"
                if regime_transition
                else "fixed_regime_search"
            ),
            old_schema_elements=("deterministic exact-closure ranking",),
            proposed_schema_elements=(proposed_element,),
            transported_elements=("VSS4-01 family-A fixture scoreboard",),
            transport_analysis=(
                "Fixture support-verdicts do not transfer; the frontier run "
                "re-measures correctness and work from the trained producers.",
            ),
            residual_elements=(f"frontier {experiment_id} coupling regime",),
            preservation_checks=("rerun the matched R0/R1 control rows",),
            stress_tests=(
                "alpha-renamed and unseen-identifier held-outs plus all honest "
                "suites (adversarial, ood)",
            ),
            worthiness_criteria=(
                "search work drops with zero correctness-gate regressions",
            ),
        ),
    )


def build_vss_matrix() -> HypothesisMatrix:
    """Five grounded candidates: a matched control plus the four VSS4-02
    hypotheses (exact closure, capsules, energy ranking, late realization)."""
    candidates = (
        _candidate(
            "vss402-control",
            "The matched control re-measures the baseline scope-solve work and "
            "quality so every lever delta in this matrix is attributable.",
            "Program policy forbids a lever claim without a same-recipe control "
            "row; R0 anchors the matched comparison.",
            "Reproduces the baseline work/quality envelope within noise.",
            ExperimentKnobs(data_source="programspec"),
            proposed_element="matched control baseline",
        ),
        _candidate(
            "vss402-exact-closure",
            "Proof-checked exact closure removes only certified-unsupported "
            "candidates, cutting search work without a single false prune.",
            "The exact closure emits replayable certificates; every removal is "
            "witnessed against the finite domain and pack verifier.",
            "Lower expanded nodes and support queries at zero false-unsupported "
            "count and zero unknown-preservation violations.",
            ExperimentKnobs(
                data_source="programspec",
                scope_contracts=True,
                scope_local_oracle=True,
            ),
            proposed_element="proof-checked exact closure",
        ),
        _candidate(
            "vss402-capsules",
            "Dependency capsules outperform lexical decomposition as coupling "
            "and interface width rise by solving strongly connected scopes "
            "jointly.",
            "Capsule SCC grouping bounds interface width and avoids the "
            "conservative over-approximation lexical scopes impose on coupled "
            "assignments.",
            "Lower joint-versus-singleton work at high coupling with no local/"
            "global verifier disagreement.",
            ExperimentKnobs(
                data_source="programspec",
                scope_contracts=True,
                topology_actions=True,
                topology_bounded_buffer=True,
            ),
            proposed_element="dependency-capsule joint solving",
        ),
        _candidate(
            "vss402-energy",
            "Cost-to-go energy ranking improves search work primarily at high "
            "assignment coupling while remaining strictly order-only.",
            "The energy scorer permutes live candidates only; a candidate-set "
            "parity guard fails closed on any membership change.",
            "Search-work reduction concentrated in high-coupling strata with "
            "zero candidate-set parity failures.",
            ExperimentKnobs(
                data_source="programspec",
                scope_contracts=True,
                scope_contract_negatives=True,
            ),
            proposed_element="learned cost-to-go energy ranking",
            regime_transition=True,
        ),
        _candidate(
            "vss402-late-realization",
            "Late surface realization improves alpha-renaming invariance "
            "without changing the solved semantic program.",
            "Realization touches surface-only slots after the solver certifies "
            "the semantic IR; the semantic fingerprint must be identical "
            "before and after.",
            "Higher alpha-equivalence pass rate at zero semantic-IR mutation "
            "violations and zero structured-slot AR routing.",
            ExperimentKnobs(
                data_source="programspec",
                scope_contracts=True,
                scope_local_oracle=True,
                scope_independent_noise=True,
            ),
            proposed_element="verified late surface realization",
        ),
    )
    return HypothesisMatrix(
        matrix_id=MATRIX_ID,
        campaign_id=CAMPAIGN_ID,
        evidence_snapshot_id=build_vss_evidence().snapshot_id,
        hypotheses=candidates,
        recommended_experiment_id="vss402-exact-closure",
        selection_rationale=(
            "Exact closure is the correctness foundation every other lever "
            "builds on, so it is measured first; the control keeps the "
            "comparison honest and the remaining candidates isolate ranking, "
            "decomposition, and realization one variable at a time."
        ),
    )


__all__ = [
    "CAMPAIGN_ID",
    "MATRIX_ID",
    "build_vss_campaign",
    "build_vss_evidence",
    "build_vss_matrix",
    "build_vss_sources",
]
