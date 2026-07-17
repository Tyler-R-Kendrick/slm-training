"""G1 (SLM-46): program tracks encoded as typed autoresearch matrices.

The recursive self-improvement story runs through the existing hypothesizer
machinery, not a parallel ad-hoc loop. This module encodes a program track's
experiments as a `HypothesisMatrix` of typed `ExperimentSpec`s, grounded in a
real `EvidenceSnapshot` over committed evidence, so it can be submitted
through `engine.validate_hypothesis_matrix` and compiled to bounded commands
by `engine.compile_commands` — no new runner, no touched benchmark.

Track A (the valid-but-empty wall) is encoded here. Its decode-time levers —
A3 coverage-energy remasking (`remask_policy="coverage"`) and A4
minimum-content decode contracts (`decode_min_content`) — became typed
`ExperimentKnobs` in this change so the matrix routes authentically rather
than through incidental knobs. A5 (lattice search) rides the existing
`compiler_search_mode` knob. A1 (the emptiness probe, E248) is the diagnostic
evidence these attacks are grounded in; A2 (ASAp reweighting) is not yet a
model-side lever and is deliberately not fabricated as a routable knob — it
stays a future matrix row (noted in the selection rationale).
"""

from __future__ import annotations

import hashlib

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
from slm_training.bridge_utils import repo_root

TRACK_A_CAMPAIGN_ID = "track-a-emptiness-wall"

# Committed evidence the Track-A attacks are grounded in.
_LINEAGE_DOC = "docs/design/research-lineage.md"  # research role
_PROBE_DOC = "docs/design/iter-e248-emptiness-probe-20260716.md"
_COVERAGE_DOC = "docs/design/iter-e251-coverage-remask-20260716.md"
_MINCONTENT_DOC = "docs/design/iter-e250-min-content-decode-20260716.md"
_LATTICE_DOC = "docs/design/iter-e240-e247-lattice-campaign-20260716.md"


def _evidence_item(rel_path: str, kind: str) -> EvidenceItem:
    path = repo_root() / rel_path
    data = path.read_bytes()
    return EvidenceItem(
        path=rel_path,
        kind=kind,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def track_a_evidence() -> EvidenceSnapshot:
    return EvidenceSnapshot(
        snapshot_id="evidence-track-a",
        roots=("docs/design",),
        items=(
            _evidence_item(_LINEAGE_DOC, "repo_lineage"),
            _evidence_item(_PROBE_DOC, "run_insight"),
            _evidence_item(_COVERAGE_DOC, "run_insight"),
            _evidence_item(_MINCONTENT_DOC, "run_insight"),
            _evidence_item(_LATTICE_DOC, "evaluation"),
        ),
    )


def track_a_sources() -> tuple[ResearchSource, ...]:
    return (
        ResearchSource(
            source_id="e248-emptiness-probe",
            kind="prior_run",
            title="A1 emptiness probe (E248)",
            uri=_PROBE_DOC,
        ),
    )


def track_a_campaign() -> CampaignSpec:
    return CampaignSpec(
        campaign_id=TRACK_A_CAMPAIGN_ID,
        objective="Lift meaningful parse off the valid-but-empty wall without "
        "weakening any honest gate.",
        primary_metric="held_out.meaningful_program_rate",
        track="twotower",
        researcher_mode="fixture",
    )


def _novelty(*, regime: bool, mechanism: str) -> CategoricalNoveltyAudit:
    return CategoricalNoveltyAudit(
        transition_kind=(
            "regime_transition_candidate" if regime else "fixed_regime_search"
        ),
        old_schema_elements=("grammar-constrained MaskGIT decode",),
        proposed_schema_elements=(
            (mechanism,) if regime else ("grammar-constrained MaskGIT decode",)
        ),
        transported_elements=("emptiness-probe NLL diagnostic",),
        transport_analysis=(
            "The empty-vs-populated NLL gap (E248) is not explained by the "
            "unmodified decoder; the proposed lever targets that residual.",
        ),
        residual_elements=(mechanism,),
        preservation_checks=("rerun the matched E255 control unchanged",),
        stress_tests=("every honest suite, meaningful parse primary",),
        worthiness_criteria=(
            "positive meaningful-parse delta with no gate regression",
        ),
    )


def _candidate(
    *,
    experiment_id: str,
    hypothesis: str,
    expected_effect: str,
    knobs: ExperimentKnobs,
    mechanism: str,
    regime: bool,
) -> HypothesisCandidate:
    citations = (
        _LINEAGE_DOC,
        _PROBE_DOC,
        _COVERAGE_DOC,
        _MINCONTENT_DOC,
        _LATTICE_DOC,
    )
    experiment = ExperimentSpec(
        experiment_id=experiment_id,
        campaign_id=TRACK_A_CAMPAIGN_ID,
        hypothesis=hypothesis,
        rationale="The E248 probe shows the empty program is the argmax valid "
        "completion; this lever re-weights decode toward content.",
        expected_effect=expected_effect,
        falsification_criteria=(
            "Meaningful parse does not exceed the matched control on held_out.",
        ),
        stop_conditions=("Stop after the fixture eval; no gate weakening.",),
        citations=citations,
        knobs=knobs,
    )
    return HypothesisCandidate(
        experiment=experiment,
        evidence_uses=(
            EvidenceUse(
                role="research",
                citation=_LINEAGE_DOC,
                contribution="Positions the ASAp/constraint-distortion prior "
                "art the emptiness attack engages.",
            ),
            EvidenceUse(
                role="prior_trace",
                citation=_COVERAGE_DOC,
                contribution="Prior coverage-remask trajectory over the same "
                "fixture corpus.",
            ),
            EvidenceUse(
                role="prior_result",
                citation=_LATTICE_DOC,
                contribution="Matched lattice-campaign baseline result.",
            ),
        ),
        novelty=_novelty(regime=regime, mechanism=mechanism),
    )


def track_a_matrix() -> HypothesisMatrix:
    """Five authentic Track-A emptiness-attack candidates, distinct knob
    signatures, all grounded in the E248 probe evidence."""
    candidates = (
        _candidate(
            experiment_id="a3-coverage-remask",
            hypothesis="Coverage-energy remasking biases decode toward "
            "under-covered named components.",
            expected_effect="Higher meaningful parse and component recall.",
            knobs=ExperimentKnobs(remask_policy="coverage"),
            mechanism="coverage-energy remask policy",
            regime=True,
        ),
        _candidate(
            experiment_id="a4-min-content-auto",
            hypothesis="An inventory-derived minimum-content floor makes the "
            "empty layout an illegal completion.",
            expected_effect="Non-empty roots when the prompt names components.",
            knobs=ExperimentKnobs(decode_min_content=-1),
            mechanism="minimum-content decode contract (auto)",
            regime=True,
        ),
        _candidate(
            experiment_id="a4-min-content-floor",
            hypothesis="A fixed two-item content floor prevents trivial roots.",
            expected_effect="Fewer empty_root_stack failures on adversarial.",
            knobs=ExperimentKnobs(decode_min_content=2),
            mechanism="minimum-content decode contract (fixed)",
            regime=False,
        ),
        _candidate(
            experiment_id="a3-a4-combined",
            hypothesis="Coverage remask plus an auto content floor compounds "
            "the anti-emptiness pressure.",
            expected_effect="Best meaningful parse among the decode levers.",
            knobs=ExperimentKnobs(
                remask_policy="coverage", decode_min_content=-1
            ),
            mechanism="coverage remask + minimum-content contract",
            regime=False,
        ),
        _candidate(
            experiment_id="a5-lattice-search",
            hypothesis="Lattice-guided recursive compiler search escapes the "
            "empty basin via backtracking.",
            expected_effect="Higher meaningful parse at a latency cost.",
            knobs=ExperimentKnobs(compiler_search_mode="lattice"),
            mechanism="lattice recursive compiler search",
            regime=False,
        ),
    )
    return HypothesisMatrix(
        matrix_id="track-a-matrix-1",
        campaign_id=TRACK_A_CAMPAIGN_ID,
        evidence_snapshot_id="evidence-track-a",
        hypotheses=candidates,
        recommended_experiment_id="a3-a4-combined",
        selection_rationale="Combined coverage remask + content floor is the "
        "highest-information safe attack; A1 (probe) is the grounding "
        "diagnostic and A2 (ASAp) awaits a model-side lever before it can "
        "route as a typed knob.",
    )
