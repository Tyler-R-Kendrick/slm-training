"""VCE-009 (SLM-468): oracle/contrast fixture campaign in a governed evidence envelope.

Runs the VCE-005 oracle intervention arms
(:mod:`slm_training.data.semantic_plan.oracle`) and the VCE-006/VCE-007
semantic-contrast/metamorphic generators
(:mod:`slm_training.data.semantic_contrast`) through the existing
``ExperimentCampaignV1``/``CampaignStore`` governance
(:mod:`slm_training.autoresearch.experiment_campaign`,
:mod:`slm_training.autoresearch.storage`) instead of hand-built result JSON,
per the ownership map's pre-registered ``experiment_campaign`` extension
point for this issue (``new_owner_justified: false``). Follows the PCT-008
(:mod:`slm_training.harnesses.experiments.pct008_artifact_cold_warm`)
pattern: a frozen dataclass with a ``.manifest()`` method plus a
``run_campaign()`` function that locks the manifest, runs real arms, and
persists a plain-dict result via ``store.write_artifact``/``append_event``.

``claim_class="fixture"`` throughout: no champion/default/checkpoint
promotion path can consume this evidence (see
``docs/design/experiment-campaign-governance.md`` and
``harnesses/experiments/promotion.py``'s ``claim_class_not_promotable``
governance check).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.data.semantic_contrast import SemanticContrastBuilder
from slm_training.data.semantic_contrast.metamorphic import (
    generate_alpha_rename_case,
    generate_ast_rewrite_equivalence_case,
    generate_prompt_paraphrase_case,
    generate_prompt_single_fact_edit_case,
    generate_reorder_case,
)
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.semantic_plan.oracle import (
    InterventionIdentityV1,
    PlanOracleSubstitutor,
    apply_plan_intervention,
    build_baseline_intervention,
    select_shuffled_oracle,
)
from slm_training.dsl.pack import get_pack
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.levers import MAX_RUN_MINUTES

__all__ = [
    "ORACLE_ARMS",
    "CONTRAST_ARMS",
    "Vce009CampaignV1",
    "run_campaign",
]

ORACLE_ARMS: tuple[str, ...] = (
    "oracle_baseline",
    "oracle_one_factor_archetype",
    "oracle_one_factor_roles",
    "oracle_one_factor_topology",
    "oracle_one_factor_bindings",
    "oracle_all_factors",
    "oracle_shuffled",
    "oracle_destructive",
)
CONTRAST_ARMS: tuple[str, ...] = (
    "contrast_corpus_scoreboard",
    "metamorphic_generators",
)


def _frozen_plans(*, count: int = 6, seed: int = 3) -> list[Any]:
    """Deterministic, small fixture source plans -- frozen by fixed seed/config."""
    pack = get_pack("openui")
    extractor = OpenUISemanticPlanExtractor()
    generator = ProgramGenerator(
        GeneratorConfig(
            max_depth=2, max_width=3, components=("TextContent", "Button"), split="train"
        ),
        seed=seed,
    )
    specs = generator.generate(count).programs
    return [extractor.extract(spec, pack) for spec in specs]


def _oracle_arm_result(
    arm: str, plans: list[Any], *, identity: InterventionIdentityV1
) -> dict[str, Any]:
    """Run one oracle arm and return a comparable {outcome, compute, disposition} row."""
    baseline = plans[0]
    started = time.perf_counter()

    if arm == "oracle_baseline":
        record = build_baseline_intervention(baseline, identity=identity)
        disposition = "match" if record.is_no_op() else "mismatch"
    elif arm.startswith("oracle_one_factor_"):
        factor = arm.removeprefix("oracle_one_factor_")
        predicted_oracle = plans[1].model_copy(
            update={
                "identity": plans[1].identity.model_copy(update={"provenance": "predicted"})
            }
        )
        subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=(factor,))
        record = apply_plan_intervention(
            subst, baseline, predicted_oracle, identity=identity
        )
        disposition = (
            "match" if record.touches_only_declared_factors() else "mismatch"
        )
    elif arm == "oracle_all_factors":
        predicted_oracle = plans[1].model_copy(
            update={
                "identity": plans[1].identity.model_copy(update={"provenance": "predicted"})
            }
        )
        subst = PlanOracleSubstitutor(
            plan_source="predicted",
            oracle_factors=("archetype", "roles", "topology", "bindings"),
        )
        record = apply_plan_intervention(
            subst, baseline, predicted_oracle, identity=identity
        )
        disposition = (
            "match" if record.touches_only_declared_factors() else "mismatch"
        )
    elif arm == "oracle_shuffled":
        shuffled = select_shuffled_oracle(baseline, plans[1:], rng_seed=0)
        if shuffled is None:
            # A real, legitimate incomplete outcome -- no compatible candidate
            # in this frozen slice -- never fabricated as a match.
            duration_ms = (time.perf_counter() - started) * 1000.0
            return {
                "arm": arm,
                "role": "candidate",
                "disposition": "inconclusive",
                "reason": "no_compatible_candidate_in_frozen_slice",
                "changed_factors": [],
                "contamination_banner": None,
                "is_contaminated": False,
                "compute": {"forwards": 0, "verifier_calls": 0, "wall_ms": round(duration_ms, 3)},
            }
        subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=("roles",))
        record = apply_plan_intervention(subst, baseline, shuffled, identity=identity)
        disposition = "match" if record.changed_factors else "inconclusive"
    else:  # oracle_destructive
        subst = PlanOracleSubstitutor(plan_source="destructive", oracle_factors=("roles",))
        record = apply_plan_intervention(subst, baseline, None, identity=identity)
        disposition = "match" if record.changed_factors else "inconclusive"

    duration_ms = (time.perf_counter() - started) * 1000.0
    return {
        "arm": arm,
        "role": "control" if arm == "oracle_baseline" else "candidate",
        "disposition": disposition,
        "changed_factors": list(record.changed_factors),
        "declared_factors": list(record.declared_factors),
        "contamination_banner": record.contamination_banner,
        "is_contaminated": record.is_contaminated,
        "compute": {"forwards": 0, "verifier_calls": 1, "wall_ms": round(duration_ms, 3)},
    }


def _contrast_corpus_result() -> dict[str, Any]:
    """Real fixture-scale SemanticContrastBuilder build; reuses its own scoreboard."""
    import tempfile

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        builder = SemanticContrastBuilder(
            output_root=Path(tmp),
            dataset_id="vce009_fixture_slice",
            seed=3,
            source_count=6,
            splits=("train",),
            split_weights=(1.0,),
            wide_sources=True,
            strict_delta=True,
        )
        summary = builder.build()
    duration_ms = (time.perf_counter() - started) * 1000.0
    positive_ok = any(
        row["family"] == "positive" and row["n_admitted"] > 0
        for row in summary["scoreboard"]
    )
    return {
        "arm": "contrast_corpus_scoreboard",
        "role": "candidate",
        "disposition": "match" if positive_ok and summary["pairs"] > 0 else "inconclusive",
        "pairs": summary["pairs"],
        "mutation_classes": summary["mutation_classes"],
        "scoreboard": summary["scoreboard"],
        "contamination_banner": None,
        "is_contaminated": False,
        "compute": {
            "forwards": 0,
            "verifier_calls": summary["pairs"],
            "wall_ms": round(duration_ms, 3),
        },
    }


def _metamorphic_generators_result(plans: list[Any]) -> dict[str, Any]:
    """Runs every VCE-007 generator once on frozen sources; verifies its own contract."""
    started = time.perf_counter()
    pack = get_pack("openui")
    cases: list[dict[str, Any]] = []

    predicted = plans[0].model_copy(
        update={"identity": plans[0].identity.model_copy(update={"provenance": "predicted"})}
    )
    alpha_case = generate_alpha_rename_case(predicted, pack)
    if alpha_case is not None:
        matched = alpha_case.meta["before_seed"] == alpha_case.meta["after_seed"]
        cases.append(
            {
                "family": alpha_case.family,
                "disposition": "match" if matched else "mismatch",
                "expected_invariant": list(alpha_case.expected_invariant),
            }
        )

    reorder_case = generate_reorder_case(plans[0], pack)
    if reorder_case is not None:
        matched = reorder_case.meta["before_seed"] != reorder_case.meta["after_seed"]
        cases.append(
            {
                "family": reorder_case.family,
                "disposition": "match" if matched else "mismatch",
                "expected_changed": list(reorder_case.expected_changed),
            }
        )

    fact_edit = generate_prompt_single_fact_edit_case(
        "Generate an OpenUI program with components: button, text content.",
        "Generate an OpenUI program with components: text content.",
        edited_component="Button",
    )
    cases.append(
        {
            "family": fact_edit.family,
            "disposition": "match"
            if fact_edit.meta["changed_mentions_only_target"]
            else "mismatch",
        }
    )

    paraphrase = generate_prompt_paraphrase_case(
        "Generate an OpenUI program with components: button, text content.",
        "  Generate  an OpenUI program with components: text content, button.  ",
    )
    matched = (
        paraphrase.meta["base_fact_fingerprints"]["fact_set"]
        == paraphrase.meta["paraphrase_fact_fingerprints"]["fact_set"]
    )
    cases.append({"family": paraphrase.family, "disposition": "match" if matched else "mismatch"})

    rewrite_case = generate_ast_rewrite_equivalence_case(
        'root = Stack([t], "column")\nt = TextContent(":a.b")'
    )
    if rewrite_case is not None:
        matched = (
            rewrite_case.meta["semantic_fingerprint_before"]
            == rewrite_case.meta["semantic_fingerprint_after"]
        )
        cases.append(
            {"family": rewrite_case.family, "disposition": "match" if matched else "mismatch"}
        )

    duration_ms = (time.perf_counter() - started) * 1000.0
    all_matched = bool(cases) and all(c["disposition"] == "match" for c in cases)
    return {
        "arm": "metamorphic_generators",
        "role": "candidate",
        "disposition": "match" if all_matched else "inconclusive" if not cases else "mismatch",
        "cases": cases,
        "contamination_banner": None,
        "is_contaminated": False,
        "compute": {
            "forwards": 0,
            "verifier_calls": len(cases),
            "wall_ms": round(duration_ms, 3),
        },
    }


@dataclass(frozen=True)
class Vce009CampaignV1:
    campaign_id: str = "vce009-oracle-contrast-fixture"
    seed: int = 3
    source_count: int = 6
    claim_class: str = "fixture"
    source_commit: str = "0" * 40
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.source_count < 2:
            raise ValueError("source_count must be >= 2 (need a baseline plus an oracle plan)")
        if self.claim_class != "fixture":
            raise ValueError("VCE-009 only exposes the fixture evidence campaign")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        all_arms = (*ORACLE_ARMS, *CONTRAST_ARMS)
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "oracle_baseline" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in all_arms
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=self.campaign_id,
            hypothesis=(
                "Every oracle intervention arm (VCE-005) and semantic-contrast/"
                "metamorphic generator (VCE-006/VCE-007) behaves exactly as its "
                "own declared contract states when run through the repository's "
                "governed evidence writers, on a frozen fixture slice."
            ),
            decision=(
                "Record each arm's declared-vs-observed outcome honestly; "
                "no champion/default/checkpoint promotion claim is made here."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id="arm_contract_match_rate",
                    metric="arm_contract_match_rate",
                    role="primary",
                    direction="increase",
                    minimum_effect=0.0,
                ),
                CampaignEndpointV1(
                    endpoint_id="contaminated_arm_count",
                    metric="contaminated_arm_count",
                    role="secondary",
                    direction="decrease",
                    minimum_effect=0.0,
                ),
            ),
            arms=arms,
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=len(all_arms),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Every arm runs exactly once against the same frozen fixture slice.",
                "An arm with no valid candidate (e.g. no compatible shuffled "
                "record) is recorded inconclusive, never fabricated as a match.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="oracle_baseline_control",
                    description=(
                        "plan_source='none' must equal the baseline within "
                        "deterministic hash equality -- the no-op reference "
                        "every other oracle arm is compared against."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="identity_blind_shuffle_control",
                    description=(
                        "select_shuffled_oracle's compatibility key never reads "
                        "identity.source_program_fingerprint/prompt_context_hash, "
                        "so the shuffled arm cannot leak or infer target identity "
                        "through those hidden fields."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("identity_blind_shuffle_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id="oracle_contrast_fixture_arms",
                    hypothesis_ids=("arm_contract_match_rate",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                # Fixture evidence is never a promotion claim (PCT-008 precedent):
                # deliberately vacuous, not a real promotion bar.
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id="arm_contract_match_rate",
                    operator="ge",
                    threshold=-1.0,
                ),
            ),
            rollback_gates=(
                # No arm here intentionally uses plan_source="gold" (the only
                # source contamination_banner() flags), so any contaminated
                # arm signals a real leak, not an expected outcome -- fires
                # both ways: 0 (normal) doesn't fire, >=1 (leak) does.
                CampaignGateV1(
                    gate_id="unexpected_oracle_contamination",
                    endpoint_id="contaminated_arm_count",
                    operator="gt",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Vce009CampaignV1, *, root: Path) -> dict[str, Any]:
    """Lock the manifest, run every arm once against a frozen fixture slice, persist."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective=(
                "Fixture-scale oracle/contrast evidence through governed writers (VCE-009)"
            ),
            primary_metric="arm_contract_match_rate",
            budget=CampaignBudget(
                max_experiments=len(ORACLE_ARMS) + len(CONTRAST_ARMS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())

    plans = _frozen_plans(count=campaign.source_count, seed=campaign.seed)
    identity = InterventionIdentityV1(
        seed=campaign.seed, verifier_version=campaign.fingerprint[:16]
    )

    arm_results = [_oracle_arm_result(arm, plans, identity=identity) for arm in ORACLE_ARMS]
    arm_results.append(_contrast_corpus_result())
    arm_results.append(_metamorphic_generators_result(plans))

    matches = sum(1 for row in arm_results if row["disposition"] == "match")
    inconclusive = sum(1 for row in arm_results if row["disposition"] == "inconclusive")
    contaminated = [row for row in arm_results if row.get("is_contaminated")]

    result = {
        "schema": "vce009_oracle_contrast_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "arms": arm_results,
        "arm_contract_match_rate": round(matches / len(arm_results), 4),
        "inconclusive_count": inconclusive,
        "contaminated_arm_ids": [row["arm"] for row in contaminated],
        "version_stamp": build_version_stamp("data.semantic_contrast"),
        "scope_disclaimer": (
            "Fixture-scale evidence over a small frozen source slice only. No "
            "champion/default/checkpoint promotion claim is made -- see "
            "claim_class='fixture' and the vacuous fixture_only promotion gate."
        ),
    }
    artifact = store.write_artifact("vce009_oracle_contrast_result", result)
    store.append_event(
        "vce009_oracle_contrast_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={"manifest_sha256": lock.manifest_sha256},
    )
    return result
