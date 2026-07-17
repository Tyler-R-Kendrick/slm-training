"""G1 (SLM-46): DSL diffusion program experiments as typed autoresearch specs.

The research program (Linear project "DSL Diffusion SLM Research Program",
tracks A-G) runs through the existing hypothesizer machinery — no parallel
ad-hoc loop. This module encodes the program's Track A (emptiness wall)
experiments as a strict-schema :class:`HypothesisMatrix` grounded in the
committed evidence trail (A1 diagnosis, E277 A2 fixture row, the program's
literature manifest), so the engine's validation, bounded command
compilation, and feedback acknowledgement apply end-to-end.

Nothing here executes on import: builders return specs; execution stays
behind ``scripts/autoresearch.py`` / ``execute_commands`` with the usual
gates. Hypothesizer-eval benchmarks are untouched (frozen).
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

# src/slm_training/autoresearch/program_matrix.py -> repo root is parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]

CAMPAIGN_ID = "dsl-program-track-a"
MATRIX_ID = "dsl-program-track-a-m1"

# Committed evidence trail: one file per validation role.
RESEARCH_DOC = "docs/design/research-lineage.md"
TRACE_DOC = "docs/design/iter-e248-emptiness-probe-20260716.json"
RESULT_DOC = "docs/design/quality-matrix-results-iter-v14-a2-20260717.json"
LITERATURE_MANIFEST = (
    "src/slm_training/resources/autoresearch/dsl-program-sources.json"
)

_CITATIONS = (RESEARCH_DOC, TRACE_DOC, RESULT_DOC)


def build_track_a_campaign() -> CampaignSpec:
    return CampaignSpec(
        campaign_id=CAMPAIGN_ID,
        objective=(
            "Lift meaningful parse past the valid-but-empty wall by fixing "
            "constraint-mask distortion at decode time (Track A)."
        ),
        primary_metric="held_out.meaningful_program_rate",
        track="twotower",
        researcher_mode="agent",
        notes=(
            "DSL Diffusion SLM Research Program, Track A. Evidence-grounded "
            "per E248 (A1 diagnosis) and E277 (A2 fixture row); literature "
            f"intake committed at {LITERATURE_MANIFEST} (E3/SLM-33)."
        ),
    )


def build_track_a_evidence(repo_root: Path | str | None = None) -> EvidenceSnapshot:
    """Snapshot the committed Track A evidence files (real hashes, no stubs)."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    kinds = {
        RESEARCH_DOC: "repo_lineage",
        TRACE_DOC: "run_insight",
        RESULT_DOC: "evaluation",
        LITERATURE_MANIFEST: "repo_lineage",
    }
    items = []
    for rel, kind in kinds.items():
        path = root / rel
        data = path.read_bytes()
        items.append(
            EvidenceItem(
                path=rel,
                kind=kind,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
        )
    snapshot_id = "trackA-" + hashlib.sha256(
        "".join(item.sha256 for item in items).encode("utf-8")
    ).hexdigest()[:16]
    return EvidenceSnapshot(
        snapshot_id=snapshot_id,
        roots=("docs/design", "src/slm_training/resources/autoresearch"),
        items=tuple(items),
    )


def build_track_a_sources() -> list[ResearchSource]:
    return [
        ResearchSource(
            source_id="lineage",
            kind="repo_lineage",
            title="Research lineage (GAD/ASAp Adapted as A2)",
            uri=RESEARCH_DOC,
        ),
        ResearchSource(
            source_id="program-manifest",
            kind="repo_lineage",
            title="DSL program prior-art manifest (E3/SLM-33)",
            uri=LITERATURE_MANIFEST,
        ),
    ]


def _candidate(
    number: int,
    experiment_id: str,
    hypothesis: str,
    rationale: str,
    expected: str,
    knobs: ExperimentKnobs,
    *,
    regime_transition: bool = False,
) -> HypothesisCandidate:
    experiment = ExperimentSpec(
        experiment_id=experiment_id,
        campaign_id=CAMPAIGN_ID,
        hypothesis=hypothesis,
        rationale=rationale,
        expected_effect=expected,
        falsification_criteria=(
            "Meaningful parse does not improve over the matched control on "
            "held_out with all honest gates in force.",
        ),
        stop_conditions=(
            "Stop after the configured steps; no gate may be weakened.",
        ),
        citations=_CITATIONS,
        knobs=knobs,
    )
    return HypothesisCandidate(
        experiment=experiment,
        evidence_uses=(
            EvidenceUse(
                role="research",
                citation=RESEARCH_DOC,
                contribution=(
                    "GAD/ASAp constraint-distortion lineage motivates the "
                    "decode-time levers."
                ),
            ),
            EvidenceUse(
                role="prior_trace",
                citation=TRACE_DOC,
                contribution=(
                    "A1 probe traces the empty-program preference this "
                    "candidate attacks."
                ),
            ),
            EvidenceUse(
                role="prior_result",
                citation=RESULT_DOC,
                contribution=(
                    "E277 fixture row supplies the matched A2 wiring baseline."
                ),
            ),
        ),
        novelty=CategoricalNoveltyAudit(
            transition_kind=(
                "regime_transition_candidate"
                if regime_transition
                else "fixed_regime_search"
            ),
            old_schema_elements=("constraint-mask renormalized decode",),
            proposed_schema_elements=(
                "distribution-aware decode" if regime_transition else "decode recipe",
            ),
            transported_elements=("E255/E277 fixture scoreboards",),
            transport_analysis=(
                "Fixture deltas do not transfer; the frontier run re-measures "
                "from scratch.",
            ),
            residual_elements=(f"track-A lever combination {number}",),
            preservation_checks=("rerun the matched E255 control",),
            stress_tests=("all honest suites incl. adversarial and ood",),
            worthiness_criteria=(
                "meaningful parse improves without structural regression",
            ),
        ),
    )


def build_track_a_matrix() -> HypothesisMatrix:
    """Five grounded Track A candidates over the program's typed levers."""
    base = dict(output_tokenizer="lexer", mask_pattern="diffusion", steps=200)
    candidates = (
        _candidate(
            0,
            "trackA-a2-asap",
            "Removing observed constraint-violating mass (ASAp) lifts "
            "meaningful parse where renormalized decode prefers empty programs.",
            "A1 (E248) diagnosed decode-time empty-preference; E277 proves the "
            "ledger fires under real decode.",
            "Higher meaningful parse at equal syntax validity.",
            ExperimentKnobs(asap_decode=True, **base),
            regime_transition=True,
        ),
        _candidate(
            1,
            "trackA-a4-floor",
            "An inventory-derived minimum-content floor makes the empty "
            "completion illegal and forces content placement.",
            "A4 wiring (E250) proved the EOS gate; the floor is the hard "
            "sibling of A2's soft reweighting.",
            "Empty-root failures convert to populated layouts or honest dead ends.",
            ExperimentKnobs(decode_min_content=-1, **base),
        ),
        _candidate(
            2,
            "trackA-a2a4-stack",
            "Stacking ASAp mass removal with the content floor closes both "
            "the preference and the legality loopholes simultaneously.",
            "The two levers act at different chokepoints (proposal vs EOS "
            "admission) and compose without interaction by construction.",
            "Largest meaningful-parse lift of the matrix if A1's diagnosis holds.",
            ExperimentKnobs(asap_decode=True, decode_min_content=-1, **base),
        ),
        _candidate(
            3,
            "trackA-a2-dose",
            "ASAp needs revisits to converge: doubling steps gives the ledger "
            "more remask rounds to redistribute removed mass.",
            "ASAp's guarantee is asymptotic in visits; fixture decode showed "
            "204-334 penalties in only a handful of rounds.",
            "Monotone improvement from trackA-a2-asap with diminishing returns.",
            ExperimentKnobs(asap_decode=True, steps=400, output_tokenizer="lexer", mask_pattern="diffusion"),
        ),
        _candidate(
            4,
            "trackA-control",
            "The matched control re-measures the wall so every lever delta in "
            "this matrix is attributable and honest.",
            "Program policy: no lever claim without a same-recipe control row "
            "(E255 lineage).",
            "Reproduces the valid-but-empty wall within noise.",
            ExperimentKnobs(**base),
        ),
    )
    return HypothesisMatrix(
        matrix_id=MATRIX_ID,
        campaign_id=CAMPAIGN_ID,
        evidence_snapshot_id=build_track_a_evidence().snapshot_id,
        hypotheses=candidates,
        recommended_experiment_id="trackA-a2a4-stack",
        selection_rationale=(
            "The stacked candidate tests A1's diagnosis hardest while the "
            "control keeps the comparison honest; singles isolate each lever."
        ),
    )


__all__ = [
    "CAMPAIGN_ID",
    "MATRIX_ID",
    "build_track_a_campaign",
    "build_track_a_evidence",
    "build_track_a_matrix",
    "build_track_a_sources",
]
