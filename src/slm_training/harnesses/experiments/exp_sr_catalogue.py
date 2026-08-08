"""SIE-001 (SLM-467): stable EXP-SR-1..EXP-SR-12 identities and guardrails.

Registers the symbolic-regression/synthesis experiment families named in the
repository roadmap (``docs/design/repository-ownership-map.md``,
``src/slm_training/resources/ownership_map.json``) under the existing
:class:`~slm_training.autoresearch.experiment_campaign.ExperimentCampaignV1`
governance contract *before* any of them executes. This module owns the
stable ``experiment_id``/``campaign_id`` pair for each family and the
per-family primary metric, falsifier, matched-control axis, compute budget,
evidence class, required evidence suites and leakage control. It is
deliberately the only place these twelve identities are minted: no parallel
JSON/YAML registry exists, and every entry round-trips through the real
``ExperimentCampaignV1`` pydantic contract, so a malformed identity fails to
construct rather than silently registering.

None of the twelve entries use ``claim_class in {"promotion_candidate",
"ship_gate"}`` — registering an identity here can never itself satisfy the
contract's promotion-artifact/causal-shape requirements (see
``ExperimentCampaignV1.validate_contract``), so no experiment can mutate
product defaults solely by being cataloged, let alone by running once. Each
sibling SIE/RSP implementation issue extends its family's manifest (adding
seeds, locked eval manifest, causal-shape evidence, etc.) only when it has
real evidence to report; this module never claims that evidence on their
behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignBudget,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ClaimClass,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.lineage.records import canonical_json
import hashlib

_CATALOGUE_VERSION_COMPONENT = "harness.experiments"
# Fixed so every call to _build_campaign for the same family produces a
# byte-identical manifest (a "stable identity" cannot re-stamp its own
# creation time on every lookup) -- mirrors slm287_power_protocol.py's
# fixed created_at for its locked campaign.
_CATALOGUE_REGISTERED_AT = "2026-08-08T00:00:00Z"


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExpSrFamilySpec:
    """Human-authored identity for one EXP-SR family (SLM-467 scope item)."""

    family_id: str
    title: str
    hypothesis: str
    decision: str
    primary_metric: str
    primary_direction: Literal["increase", "decrease"]
    minimum_effect: float
    falsifier: str
    kill_threshold: float
    matched_control_axis: str
    leakage_control: str
    claim_class: ClaimClass
    extra_artifact_kinds: tuple[str, ...] = ()
    max_wall_minutes: float = 10.0
    max_experiments: int = 12


# Ordered 1:1 with the numbered "Scope" list in SLM-467's description, which
# is itself the canonical EXP-SR-1..EXP-SR-12 ordering used by every sibling
# SIE/RSP Linear issue's "(EXP-SR-N)" suffix.
EXP_SR_FAMILY_SPECS: tuple[ExpSrFamilySpec, ...] = (
    ExpSrFamilySpec(
        family_id="exp-sr-1",
        title="Held-out factor-wise semantic oracle localization",
        hypothesis="Semantic-factor scoring errors localize to specific factor families rather than spreading uniformly across the oracle.",
        decision="Prioritize evaluator repair work on the factor families with the largest localized error mass.",
        primary_metric="oracle_factor_localization_precision",
        primary_direction="increase",
        minimum_effect=0.03,
        falsifier="Localized error mass is statistically indistinguishable from a uniform-error null across factor families.",
        kill_threshold=0.0,
        matched_control_axis="Same held-out contrast slice scored by the current oracle and a factor-shuffled null oracle.",
        leakage_control="Held-out factor slice never appears in any evaluator training or calibration split.",
        claim_class="diagnostic",
        extra_artifact_kinds=("agentevals",),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-2",
        title="Prompt-to-partial-requirements prediction",
        hypothesis="A prompt-only predictor recovers a non-trivial share of extracted structured requirement facts before generation runs.",
        decision="Gate the advisory-only requirement-prediction seam on measured precision, not on wiring alone.",
        primary_metric="requirement_predictor_f1",
        primary_direction="increase",
        minimum_effect=0.05,
        falsifier="Predictor F1 does not exceed the majority-class/no-prediction baseline on the held-out prompt slice.",
        kill_threshold=0.0,
        matched_control_axis="Same prompts scored by the predictor and by a requirement-frequency-only baseline.",
        leakage_control="Prompt/requirement pairs are grouped by source template before any train/eval split.",
        claim_class="screening",
        extra_artifact_kinds=("paired_examples",),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-3",
        title="Evaluator construct-validity and human calibration",
        hypothesis="The semantic evaluator's automated verdicts agree with blinded independent human judgment above chance on contrast pairs.",
        decision="Calibrate or replace evaluator components whose human agreement falls below the preregistered floor.",
        primary_metric="evaluator_human_agreement_kappa",
        primary_direction="increase",
        minimum_effect=0.1,
        falsifier="Cohen's kappa between evaluator and blinded human adjudication is at or below chance agreement.",
        kill_threshold=0.0,
        matched_control_axis="Same contrast pairs scored by the evaluator and by blinded independent human raters.",
        leakage_control="Human raters never see evaluator verdicts or source lineage before adjudicating.",
        claim_class="diagnostic",
        extra_artifact_kinds=("agentevals", "agentv"),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-4",
        title="Witness-guided CEGIS repair",
        hypothesis="Counterexample-guided repair using terminal witnesses raises verified-repair yield over unguided resampling at matched compute.",
        decision="Adopt witness-guided repair as an optional lever only if it beats the matched-compute resampling control.",
        primary_metric="verified_repair_yield",
        primary_direction="increase",
        minimum_effect=0.05,
        falsifier="Witness-guided repair yield does not exceed matched-compute unguided resampling.",
        kill_threshold=0.0,
        matched_control_axis="Same broken-candidate corpus repaired by witness-guided CEGIS and by matched-compute resampling.",
        leakage_control="Repair witnesses are drawn only from the candidate's own domain, never from the held-out answer.",
        claim_class="screening",
        extra_artifact_kinds=(),
        max_wall_minutes=15.0,
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-5",
        title="Symbolic value-of-compute controller",
        hypothesis="A learned value-of-compute controller reduces wasted search/repair work without regressing verified-solution yield.",
        decision="Activate the controller only where it is Pareto-better than the fixed-budget control on cost and yield jointly.",
        primary_metric="compute_value_regret",
        primary_direction="decrease",
        minimum_effect=0.05,
        falsifier="Controller regret is not below the fixed-budget control's regret at matched verified-solution yield.",
        kill_threshold=0.0,
        matched_control_axis="Same task stream under the learned controller and under a fixed-budget schedule.",
        leakage_control="Controller state features never include held-out evaluation identity or gold labels.",
        claim_class="diagnostic",
        extra_artifact_kinds=(),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-6",
        title="Neural-guided bounded search and infilling order",
        hypothesis="A neural ranking signal over infilling order improves bounded-search hit rate at matched node budget over left-to-right order.",
        decision="Adopt neural-guided ordering only if it beats the matched-budget left-to-right control.",
        primary_metric="bounded_search_hit_rate",
        primary_direction="increase",
        minimum_effect=0.05,
        falsifier="Neural-guided ordering hit rate does not exceed left-to-right ordering at the same node budget.",
        kill_threshold=0.0,
        matched_control_axis="Same search problems solved under neural-guided order and under left-to-right order at equal node budget.",
        leakage_control="Ranking signal is trained only on problems disjoint from the held-out search benchmark.",
        claim_class="screening",
        extra_artifact_kinds=(),
        max_wall_minutes=20.0,
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-7",
        title="Extended static-artifact and packed semantic-summary experiment",
        hypothesis="A packed semantic summary layered on the certified static artifact preserves exact domain parity while shrinking request-local recomputation.",
        decision="Adopt the packed summary only if domain parity stays exact and recomputation cost measurably drops.",
        primary_metric="static_summary_domain_parity_rate",
        primary_direction="increase",
        minimum_effect=0.0,
        falsifier="Any packed-summary domain diverges from the certified live-construction oracle on the differential corpus.",
        kill_threshold=1.0,
        matched_control_axis="Same prefixes evaluated with the packed summary enabled and disabled against the live oracle.",
        leakage_control="Packed summary keys exclude request-local runtime facts (mirrors PCT-006's static/request-boundary guard).",
        claim_class="diagnostic",
        extra_artifact_kinds=("formal_preflight",),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-8",
        title="Proof-scoped e-graph/equivalence-region experiment",
        hypothesis="A proof-scoped e-graph equivalence-region rewrite preserves semantic equivalence on every certified rewrite.",
        decision="Adopt e-graph regions only where every applied rewrite carries a certified equivalence proof.",
        primary_metric="egraph_certified_equivalence_rate",
        primary_direction="increase",
        minimum_effect=0.0,
        falsifier="Any applied e-graph rewrite lacks a certified equivalence proof or fails runtime re-verification.",
        kill_threshold=1.0,
        matched_control_axis="Same expression corpus rewritten with e-graph regions enabled and with the unrewritten baseline.",
        leakage_control="Equivalence certificates are checked against the live evaluator, never assumed from rule provenance.",
        claim_class="fixture",
        extra_artifact_kinds=("formal_preflight",),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-9",
        title="Prospective certified macro/library-learning experiment",
        hypothesis="Certified macro extraction from repeated structure reduces expression size without changing certified semantics.",
        decision="Adopt learned macros only where size reduction holds with zero certified-semantics regressions.",
        primary_metric="macro_library_size_reduction_rate",
        primary_direction="increase",
        minimum_effect=0.02,
        falsifier="Macro substitution changes certified semantics on any corpus program.",
        kill_threshold=0.0,
        matched_control_axis="Same corpus expressed with and without learned macro substitution.",
        leakage_control="Macro library is learned only from the training-side corpus, never from held-out evaluation programs.",
        claim_class="fixture",
        extra_artifact_kinds=("paired_examples",),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-10",
        title="Quality-diversity corpus generation experiment",
        hypothesis="A quality-diversity generation objective increases structural/topological coverage over rejection sampling at matched compute.",
        decision="Adopt quality-diversity generation only if it beats matched-compute rejection sampling on coverage without a validity drop.",
        primary_metric="corpus_topology_coverage",
        primary_direction="increase",
        minimum_effect=0.05,
        falsifier="Quality-diversity coverage does not exceed matched-compute rejection sampling coverage.",
        kill_threshold=0.0,
        matched_control_axis="Same generation budget spent under quality-diversity search and under rejection sampling.",
        leakage_control="Generated corpus is grouped by source template/lineage before any train/eval split, per VCE-008.",
        claim_class="fixture",
        extra_artifact_kinds=("paired_examples",),
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-11",
        title="Matched PySR and SRBench-compatible symbolic benchmark",
        hypothesis="The in-repo symbolic-regression pack reaches matched-budget parity with PySR/SRBench on shared benchmark problems.",
        decision="Report matched-budget parity honestly; a gap versus PySR/SRBench is not treated as a pack defect without matched compute.",
        primary_metric="srbench_matched_score_gap",
        primary_direction="decrease",
        minimum_effect=0.0,
        falsifier="The matched-budget gap versus PySR/SRBench does not shrink relative to the unmatched baseline comparison.",
        kill_threshold=0.0,
        matched_control_axis="Same SRBench problems solved by the in-repo pack and by the optional isolated PySR/SRBench adapter at equal compute budget.",
        leakage_control="PySR/SRBench adapter runs in an isolated environment manifest (SRP-011) and never shares training data with the pack.",
        claim_class="diagnostic",
        extra_artifact_kinds=(),
        max_wall_minutes=30.0,
    ),
    ExpSrFamilySpec(
        family_id="exp-sr-12",
        title="Real OpenUI plus symbolic-regression second-pack portability certification",
        hypothesis="The generalized pack/completion-artifact seam ports to the symbolic-regression pack without a second bespoke implementation.",
        decision="Certify second-pack portability only when OpenUI and symbolic-regression share the seam with zero pack-specific forks.",
        primary_metric="second_pack_portability_parity_rate",
        primary_direction="increase",
        minimum_effect=0.0,
        falsifier="Any pack-specific fork is required in the shared seam to support the symbolic-regression pack.",
        kill_threshold=1.0,
        matched_control_axis="Same seam exercised end-to-end against the OpenUI pack and against the symbolic-regression pack.",
        leakage_control="Portability fixtures never reuse OpenUI-specific grammar/tokenizer identities for the symbolic-regression pack.",
        claim_class="diagnostic",
        extra_artifact_kinds=("formal_preflight",),
    ),
)


def _build_campaign(spec: ExpSrFamilySpec) -> ExperimentCampaignV1:
    """Assemble a valid, real ``ExperimentCampaignV1`` for one EXP-SR family.

    Every field round-trips through the shipped pydantic contract — a
    malformed spec fails to construct rather than silently registering.
    """

    stamp = build_version_stamp(_CATALOGUE_VERSION_COMPONENT)
    primary_id = f"{spec.family_id}_primary"
    opposite_operator: Literal["ge", "le"] = (
        "ge" if spec.primary_direction == "increase" else "le"
    )
    kill_operator: Literal["le", "ge"] = (
        "le" if spec.primary_direction == "increase" else "ge"
    )
    quality_control_id = f"{spec.family_id}_matched_control"
    leakage_control_id = f"{spec.family_id}_leakage_control"
    seeds = (0, 1, 2)
    artifact_kinds = (
        "version_stamp",
        "seed_result",
        "endpoint_result",
        *spec.extra_artifact_kinds,
    )
    return ExperimentCampaignV1(
        campaign_id=spec.family_id,
        experiment_id=spec.family_id,
        hypothesis=spec.hypothesis,
        decision=spec.decision,
        endpoints=(
            CampaignEndpointV1(
                endpoint_id=primary_id,
                metric=spec.primary_metric,
                role="primary",
                direction=spec.primary_direction,
                minimum_effect=spec.minimum_effect,
            ),
        ),
        arms=(
            CampaignArmV1(
                arm_id=f"{spec.family_id}_control",
                role="control",
                config_sha256=_sha({"family": spec.family_id, "arm": "control"}),
            ),
            CampaignArmV1(
                arm_id=f"{spec.family_id}_candidate",
                role="candidate",
                config_sha256=_sha({"family": spec.family_id, "arm": "candidate"}),
            ),
        ),
        seeds=seeds,
        budget=CampaignBudget(
            max_experiments=spec.max_experiments, max_wall_minutes=spec.max_wall_minutes
        ),
        stopping_rules=(
            f"Falsifier: {spec.falsifier}",
            "Reject any timed-out or incomplete arm; only exact complete coverage counts.",
        ),
        controls=(
            CampaignControlV1(
                control_id=quality_control_id,
                description=spec.matched_control_axis,
                kind="quality",
            ),
            CampaignControlV1(
                control_id=leakage_control_id,
                description=spec.leakage_control,
                kind="negative",
            ),
        ),
        negative_controls=(leakage_control_id,),
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id=f"{spec.family_id}_family",
                hypothesis_ids=(primary_id,),
                alpha=0.05,
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id=f"{spec.family_id}_promotion",
                endpoint_id=primary_id,
                operator=opposite_operator,
                threshold=spec.minimum_effect,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id=f"{spec.family_id}_kill",
                endpoint_id=primary_id,
                operator=kill_operator,
                threshold=spec.kill_threshold,
            ),
        ),
        artifact_requirements=tuple(
            ArtifactRequirementV1(
                kind=kind, minimum_count=len(seeds) if kind == "seed_result" else 1
            )
            for kind in artifact_kinds
        ),
        claim_class=spec.claim_class,
        source_commit=str(stamp["code_commit"]),
        source_dirty=bool(stamp["code_dirty"]),
        author="slm-training",
        created_at=_CATALOGUE_REGISTERED_AT,
    )


def build_exp_sr_catalogue() -> tuple[ExperimentCampaignV1, ...]:
    """The twelve registered EXP-SR campaign identities, in canonical order."""

    return tuple(_build_campaign(spec) for spec in EXP_SR_FAMILY_SPECS)


def exp_sr_campaign(family_id: str) -> ExperimentCampaignV1:
    """Return the one registered campaign for ``family_id`` (e.g. ``"exp-sr-4"``).

    Raises ``KeyError`` for any identity this module did not mint — sibling
    issues must extend a registered family, never invent a new EXP-SR id.
    """

    for spec in EXP_SR_FAMILY_SPECS:
        if spec.family_id == family_id:
            return _build_campaign(spec)
    raise KeyError(f"unregistered EXP-SR family: {family_id!r}")


def plan_only_manifest(family_id: str) -> dict[str, object]:
    """A preview manifest for ``family_id`` that never locks or executes.

    Matches the repository's ``--mode plan-only`` convention: returns the
    manifest payload plus its content digest so a caller can inspect the
    registered identity without calling
    :meth:`~slm_training.autoresearch.storage.CampaignStore.lock_experiment_campaign`.
    """

    manifest = exp_sr_campaign(family_id)
    return {
        "status": "plan_only",
        "claim_class": manifest.claim_class,
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "manifest": manifest.model_dump(mode="json"),
    }
