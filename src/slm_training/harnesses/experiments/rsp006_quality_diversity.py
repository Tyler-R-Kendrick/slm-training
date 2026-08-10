"""RSP-006 (SLM-481): EXP-SR-10 quality-diversity corpus generation.

Fixture-scale governed campaign comparing three size/exposure-matched corpus
generation arms over the existing OpenUI ProgramSpec owner
(:mod:`slm_training.data.progspec.generate`):

* ``random_matched`` — uniform rejection sampling over the bounded candidate grid
* ``frequency_balanced`` — inverse-frequency balancing over pack-owned descriptor cells
* ``quality_diversity`` — MAP-Elites-style archive maximizing per-cell quality

Descriptor axes (topology / depth / operator family / width / content class) are
frozen at ``QD_DESCRIPTOR_VERSION`` and derived only from pack-owned generator
facts — never from eval corpora or fixture coverage tables.

Primary metric ``corpus_topology_coverage`` (catalogue ``exp-sr-10``). Splits are
assigned from canonical root families *before* any derivative expansion via
:mod:`slm_training.harnesses.train_data.split_policy`. Equivalent programs
(canonically identical roots) never inflate diversity. ``claim_class=fixture``;
never ``promotion_candidate`` / ``ship_gate``; no automatic adoption.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.progspec.generate import (
    GENERATOR_VERSION,
    PROGRAM_FAMILY,
    GeneratorConfig,
    ProgramGenerator,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.experiments.slm267_programspec_coverage_scaling import (
    DEFAULT_CONFIG,
    shard_seed,
)
from slm_training.harnesses.train_data.diversity import fingerprint_record
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-10"
PRIMARY_METRIC = "corpus_topology_coverage"
QD_DESCRIPTOR_VERSION = "rsp006_qd_descriptors/v1"
QD_DESCRIPTOR_AXES: tuple[str, ...] = (
    "depth",
    "reference_topology",
    "operator_family",
    "width",
    "content_class",
)
ARM_IDS: tuple[str, ...] = (
    "random_matched",
    "frequency_balanced",
    "quality_diversity",
)
_ARM_SEED_OFFSET: dict[str, int] = {
    "random_matched": 0,
    "frequency_balanced": 17,
    "quality_diversity": 42,
}

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "DEFAULT_FIXTURE_BUDGET",
    "PRIMARY_METRIC",
    "QD_DESCRIPTOR_AXES",
    "QD_DESCRIPTOR_VERSION",
    "Rsp006CampaignV1",
    "compute_corpus_topology_coverage",
    "descriptor_from_spec",
    "enumerate_descriptor_grid",
    "plan_only_preview",
    "run_arm",
    "run_campaign",
]

DEFAULT_FIXTURE_BUDGET = 48


@dataclass(frozen=True)
class QdDescriptorV1:
    """Frozen pack-owned behavioural descriptor for one accepted program."""

    schema_version: str
    depth: str
    reference_topology: str
    operator_family: str
    width: str
    content_class: str

    def key(self) -> tuple[str, ...]:
        return (
            self.depth,
            self.reference_topology,
            self.operator_family,
            self.width,
            self.content_class,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "depth": self.depth,
            "reference_topology": self.reference_topology,
            "operator_family": self.operator_family,
            "width": self.width,
            "content_class": self.content_class,
        }


def _operator_family(components: tuple[str, ...]) -> str:
    primary = tuple(sorted(name for name in components if name != "Stack"))
    return primary[0] if len(primary) == 1 else "|".join(primary)


def descriptor_from_facts(facts: dict[str, Any]) -> QdDescriptorV1:
    components = tuple(facts.get("components") or ())
    return QdDescriptorV1(
        schema_version=QD_DESCRIPTOR_VERSION,
        depth=str(facts.get("depth", "")),
        reference_topology=str(facts.get("reference_topology", "")),
        operator_family=_operator_family(components),
        width=str(facts.get("width", "")),
        content_class=str(facts.get("content_class", "")),
    )


def descriptor_from_spec(spec: ProgramSpec) -> QdDescriptorV1:
    return descriptor_from_facts(spec.facts)


def enumerate_descriptor_grid(
    config: GeneratorConfig | None = None,
) -> frozenset[tuple[str, ...]]:
    """All descriptor keys reachable from the bounded ProgramSpec candidate grid."""
    generator = ProgramGenerator(config or DEFAULT_CONFIG, seed=0)
    keys: set[tuple[str, ...]] = set()
    for candidate in generator._candidates:  # noqa: SLF001 — fixture owner grid
        facts = {
            "components": list(candidate.components),
            "depth": candidate.depth,
            "width": candidate.width,
            "reference_topology": generator._topology(candidate.depth),  # noqa: SLF001
            "content_class": candidate.content_class,
        }
        keys.add(descriptor_from_facts(facts).key())
    return frozenset(keys)


def _canonical_root_id(spec: ProgramSpec) -> str:
    record = emit_record(
        spec,
        prompt="Generate this typed OpenUI program.",
        task="generation",
        source=PROGRAM_FAMILY,
    )
    return fingerprint_record(record).canonical_root_id


def _assign_root_group(spec: ProgramSpec, root_family_id: str) -> tuple[ProgramSpec, str]:
    """Assign root-family grouping and policy split before derivative expansion.

    ``ProgramSpec.split`` remains the generator's train default; the
    ``RootFamilySplitPolicyV1`` bucket is tracked separately because
    ``ALLOWED_SPLITS`` uses different vocabulary (``held_out`` / ``rico_held``).
    """
    policy = RootFamilySplitPolicyV1()
    policy_split = policy.assign(root_family_id)
    policy.require_inherited(
        root_family_id=root_family_id,
        split_group_id=root_family_id,
        split=policy_split,
    )
    grouped = replace(
        spec,
        program_family_id=root_family_id,
        split_group_id=root_family_id,
    )
    return grouped, policy_split


def _quality_score(spec: ProgramSpec) -> float:
    cells = spec.facts.get("coverage_cells") or []
    return float(len(cells))


@dataclass
class MapElitesArchive:
    """Novelty archive: one best-quality occupant per descriptor cell."""

    cells: dict[tuple[str, ...], tuple[ProgramSpec, float]]

    def consider(self, spec: ProgramSpec) -> bool:
        descriptor = descriptor_from_spec(spec)
        quality = _quality_score(spec)
        key = descriptor.key()
        incumbent = self.cells.get(key)
        if incumbent is None or quality > incumbent[1]:
            self.cells[key] = (spec, quality)
            return True
        return False

    def covered_keys(self) -> set[tuple[str, ...]]:
        return set(self.cells)


def compute_corpus_topology_coverage(
    *,
    covered_keys: set[tuple[str, ...]],
    grid_keys: frozenset[tuple[str, ...]],
) -> float:
    if not grid_keys:
        return 0.0
    return len(covered_keys & grid_keys) / len(grid_keys)


def _available_candidates(generator: ProgramGenerator) -> list[Any]:
    return [
        candidate
        for candidate in generator._candidates  # noqa: SLF001
        if candidate.key() not in generator._used  # noqa: SLF001
    ]


def _descriptor_for_candidate(generator: ProgramGenerator, candidate: Any) -> QdDescriptorV1:
    facts = {
        "components": list(candidate.components),
        "depth": candidate.depth,
        "width": candidate.width,
        "reference_topology": generator._topology(candidate.depth),  # noqa: SLF001
        "content_class": candidate.content_class,
    }
    return descriptor_from_facts(facts)


def _pick_frequency_balanced(
    generator: ProgramGenerator,
    rng: random.Random,
    cell_counts: dict[tuple[str, ...], int],
) -> Any:
    available = _available_candidates(generator)
    if not available:
        raise ValueError("candidate grid exhausted")

    def sort_key(candidate: Any) -> tuple[int, float]:
        descriptor = _descriptor_for_candidate(generator, candidate)
        return (cell_counts.get(descriptor.key(), 0), rng.random())

    return min(available, key=sort_key)


def _pick_map_elites(
    generator: ProgramGenerator,
    rng: random.Random,
    archive: MapElitesArchive,
    uncovered: set[tuple[str, ...]],
) -> Any:
    available = _available_candidates(generator)
    if not available:
        raise ValueError("candidate grid exhausted")
    empty_cells = [
        candidate
        for candidate in available
        if _descriptor_for_candidate(generator, candidate).key() not in archive.cells
    ]
    if empty_cells:
        return rng.choice(empty_cells)
    if uncovered:
        target_keys = uncovered & {
            _descriptor_for_candidate(generator, candidate).key()
            for candidate in available
        }
        if target_keys:
            candidates = [
                candidate
                for candidate in available
                if _descriptor_for_candidate(generator, candidate).key() in target_keys
            ]
            return rng.choice(candidates)
    jitter = {candidate.key(): rng.random() for candidate in available}
    return max(
        available,
        key=lambda candidate: (
            archive.cells.get(
                _descriptor_for_candidate(generator, candidate).key(), (None, -1.0)
            )[1],
            jitter[candidate.key()],
        ),
    )


def _materialize_candidate(generator: ProgramGenerator, candidate: Any) -> ProgramSpec:
    generator._used.add(candidate.key())  # noqa: SLF001
    return generator._materialize(candidate)  # noqa: SLF001


@dataclass(frozen=True)
class ArmResultV1:
    arm_id: str
    seed: int
    generation_budget: int
    proposals: int
    accepted: int
    rejections: int
    duplicate_roots_skipped: int
    test_split_excluded: int
    unique_canonical_roots: int
    corpus_topology_coverage: float
    validity_rate: float
    descriptor_cells_covered: int
    descriptor_grid_size: int
    split_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_arm(
    arm_id: str,
    *,
    generation_budget: int,
    seed: int,
    config: GeneratorConfig | None = None,
) -> ArmResultV1:
    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown arm {arm_id!r}; expected one of {ARM_IDS}")
    if generation_budget < 0:
        raise ValueError("generation_budget must be non-negative")

    cfg = config or DEFAULT_CONFIG
    grid_keys = enumerate_descriptor_grid(cfg)
    generator = ProgramGenerator(cfg, seed=seed)
    rng = random.Random(seed)
    archive = MapElitesArchive(cells={})
    cell_counts: dict[tuple[str, ...], int] = {}
    seen_roots: set[str] = set()
    accepted_specs: list[ProgramSpec] = []
    covered_keys: set[tuple[str, ...]] = set()
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    proposals = 0
    rejections = 0
    duplicate_roots_skipped = 0
    test_split_excluded = 0

    while proposals < generation_budget:
        available = _available_candidates(generator)
        if not available:
            break
        proposals += 1
        try:
            if arm_id == "random_matched":
                candidate = rng.choice(available)
            elif arm_id == "frequency_balanced":
                candidate = _pick_frequency_balanced(generator, rng, cell_counts)
            else:
                candidate = _pick_map_elites(
                    generator,
                    rng,
                    archive,
                    grid_keys - covered_keys,
                )
            spec = _materialize_candidate(generator, candidate)
        except ValueError as exc:
            if str(exc) == "candidate grid exhausted":
                break
            rejections += 1
            continue

        root_id = _canonical_root_id(spec)
        if root_id in seen_roots:
            duplicate_roots_skipped += 1
            continue
        seen_roots.add(root_id)

        spec, policy_split = _assign_root_group(spec, root_id)
        split_counts[policy_split] = split_counts.get(policy_split, 0) + 1
        if policy_split == "test":
            test_split_excluded += 1
            continue

        descriptor = descriptor_from_spec(spec)
        key = descriptor.key()
        cell_counts[key] = cell_counts.get(key, 0) + 1
        covered_keys.add(key)
        if arm_id == "quality_diversity":
            archive.consider(spec)
        accepted_specs.append(spec)

    validity_rate = (len(accepted_specs) / proposals) if proposals else 0.0
    coverage = compute_corpus_topology_coverage(
        covered_keys=covered_keys,
        grid_keys=grid_keys,
    )
    return ArmResultV1(
        arm_id=arm_id,
        seed=seed,
        generation_budget=generation_budget,
        proposals=proposals,
        accepted=len(accepted_specs),
        rejections=rejections,
        duplicate_roots_skipped=duplicate_roots_skipped,
        test_split_excluded=test_split_excluded,
        unique_canonical_roots=len(seen_roots),
        corpus_topology_coverage=round(coverage, 6),
        validity_rate=round(validity_rate, 6),
        descriptor_cells_covered=len(covered_keys),
        descriptor_grid_size=len(grid_keys),
        split_counts=split_counts,
    )


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    grid = enumerate_descriptor_grid(DEFAULT_CONFIG)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "qd_descriptor_version": QD_DESCRIPTOR_VERSION,
        "qd_descriptor_axes": list(QD_DESCRIPTOR_AXES),
        "generator_version": GENERATOR_VERSION,
        "descriptor_grid_size": len(grid),
        "claim_class_execution": "fixture",
        "promotion": False,
        "automatic_adoption": False,
        "eval_target_access": False,
    }


@dataclass(frozen=True)
class Rsp006CampaignV1:
    campaign_id: str = "rsp006-exp-sr-10-quality-diversity"
    generation_budget: int = DEFAULT_FIXTURE_BUDGET
    seed: int = 0
    shards: int = 1
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.generation_budget < 1:
            raise ValueError("generation_budget must be >= 1")
        if self.shards < 1:
            raise ValueError("shards must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError(
                "RSP-006 durable evidence is fixture-only "
                "(catalogue exp-sr-10; never promotion_candidate/ship_gate)"
            )
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def kill_threshold(self) -> float:
        return float(self.catalogue_manifest().rollback_gates[0].threshold)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "random_matched" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt quality-diversity generation only if it beats matched-compute "
                "rejection sampling on corpus_topology_coverage without a validity drop. "
                "Fixture evidence only — never promote or auto-adopt."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=float(catalogue.endpoints[0].minimum_effect),
                ),
            ),
            arms=arms,
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=self.generation_budget * len(ARM_IDS) * self.shards,
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Reject timed-out or incomplete arms.",
                "Equivalent canonical roots must not inflate descriptor coverage.",
                "Split assignment occurs before derivative expansion; no eval corpus reads.",
            ),
            controls=(
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_matched_control",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same generation budget under quality-diversity search and "
                        "rejection sampling."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else (
                        "Generated corpus grouped by root family before train/eval split."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=(f"{CATALOGUE_ID}_leakage_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{CATALOGUE_ID}_kill",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="lt",
                    threshold=self.kill_threshold(),
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
                ArtifactRequirementV1(kind="paired_examples"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def _recommendation(arm_results: dict[str, ArmResultV1]) -> str:
    random_cov = arm_results["random_matched"].corpus_topology_coverage
    qd_cov = arm_results["quality_diversity"].corpus_topology_coverage
    qd_valid = arm_results["quality_diversity"].validity_rate
    random_valid = arm_results["random_matched"].validity_rate
    if qd_cov <= random_cov:
        return "reject_qd_not_better_than_random"
    if qd_valid < random_valid:
        return "reject_qd_validity_regression"
    return "inconclusive_fixture"


def run_campaign(
    campaign: Rsp006CampaignV1,
    *,
    root: Any,
) -> dict[str, Any]:
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-006 / EXP-SR-10 quality-diversity corpus fixture",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=campaign.generation_budget
                * len(ARM_IDS)
                * campaign.shards,
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()
    grid_keys = enumerate_descriptor_grid(DEFAULT_CONFIG)

    arm_results: dict[str, ArmResultV1] = {}
    shard_payloads: list[dict[str, Any]] = []
    for shard_id in range(campaign.shards):
        shard_seed_value = shard_seed(campaign.seed, shard_id, worker_id=0)
        shard_arms: dict[str, dict[str, Any]] = {}
        for arm in ARM_IDS:
            result = run_arm(
                arm,
                generation_budget=campaign.generation_budget,
                seed=shard_seed_value + _ARM_SEED_OFFSET[arm],
            )
            arm_results[arm] = result
            shard_arms[arm] = result.to_dict()
        shard_payloads.append(
            {"shard_id": shard_id, "seed": shard_seed_value, "arms": shard_arms}
        )

    primary_value = arm_results["quality_diversity"].corpus_topology_coverage
    kill_hit = primary_value <= campaign.kill_threshold()
    recommendation = _recommendation(arm_results)

    result = {
        "schema": "rsp006_quality_diversity_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "automatic_adoption": False,
        PRIMARY_METRIC: primary_value,
        "qd_descriptor_version": QD_DESCRIPTOR_VERSION,
        "qd_descriptor_axes": list(QD_DESCRIPTOR_AXES),
        "descriptor_grid_size": len(grid_keys),
        "arm_results": {arm: res.to_dict() for arm, res in arm_results.items()},
        "shards": shard_payloads,
        "comparison": {
            "random_matched_coverage": arm_results["random_matched"].corpus_topology_coverage,
            "frequency_balanced_coverage": arm_results[
                "frequency_balanced"
            ].corpus_topology_coverage,
            "quality_diversity_coverage": primary_value,
            "qd_beats_random": primary_value
            > arm_results["random_matched"].corpus_topology_coverage,
            "qd_validity_rate": arm_results["quality_diversity"].validity_rate,
            "random_validity_rate": arm_results["random_matched"].validity_rate,
        },
        "kill_threshold": campaign.kill_threshold(),
        "kill_gate_triggered": kill_hit,
        "recommendation": recommendation,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-10 evidence for MAP-Elites/novelty-style corpus "
            "generation over frozen pack-owned descriptors on the bounded OpenUI "
            "ProgramSpec candidate grid. Arms are size/exposure-matched by "
            "generation_budget; splits are assigned from canonical root families "
            "before derivatives; test-split roots are excluded from coverage "
            "numerators; eval corpora are never read. Equivalent canonical roots "
            "do not inflate diversity. No automatic adoption from fixture "
            "coverage — claim_class=fixture only."
        ),
    }
    artifact = store.write_artifact("rsp006_quality_diversity_result", result)
    store.append_event(
        "rsp006_quality_diversity_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            PRIMARY_METRIC: primary_value,
            "recommendation": recommendation,
        },
    )
    return result
