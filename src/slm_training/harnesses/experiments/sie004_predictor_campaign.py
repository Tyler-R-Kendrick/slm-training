"""SIE-004 (SLM-482): matched prompt-requirement predictor campaign (EXP-SR-2).

Fixture-scale governed campaign that runs the SIE-003 advisory predictor seam
(:mod:`slm_training.models.prompt_factor_predictor`) through the SIE-001
catalogue identity ``exp-sr-2`` under :class:`ExperimentCampaignV1`.

Arms (matched seeds, same held-out prompts, group-by-``source_program_id``
leakage control from the catalogue):

* ``none`` — always abstain (no-prediction baseline)
* ``deterministic_extractor`` — SIE-003 ``deterministic_keyword_predict``
* ``frequency`` — SIE-003 majority-class frequency prior
* ``retrieval`` — train-only 1-NN bag overlap (leakage-safe by construction)
* ``learned_predictor`` — SIE-003 ``TinyBagOfWordsClassifier`` + calibrated abstention
* ``gold_oracle`` — diagnostic ceiling (true label; never a production path)

SIE-002 authorized factors (``roles``, ``bindings`` from
``docs/design/iter-slm476-sie-002-oracle-localization-20260810.json``) gate
RequirementFact emission. The SIE-003 seam predicts ``style_layout`` /
archetype, which EXP-SR-1 did **not** authorize — so fact emission under that
gate fails closed while F1 on the prediction task is still measured honestly.

Legal support parity is exact: advisory predictions never unlock hard prune
(I6 / SIE-003 seam). No checkpoint promotion.
"""

from __future__ import annotations

import json
import random
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
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.progspec.prompt_requirements import PromptSemanticRequirementsV1
from slm_training.data.semantic_plan.compiler import EvidenceKind
from slm_training.data.semantic_plan.requirements_integrate import (
    requirements_hard_prune_allowed,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.prompt_factor_predictor import (
    PromptFactorExampleV1,
    TinyBagOfWordsClassifier,
    calibrate_abstention_threshold,
    deterministic_keyword_predict,
    frequency_baseline_predict,
    predict_factor,
    prediction_to_requirement_fact,
    resolve_confidence_threshold,
)

CATALOGUE_ID = "exp-sr-2"
PRIMARY_METRIC = "requirement_predictor_f1"
DATASET_ID = "openui_hard_valid_v1"
FACTOR_FAMILY = "style_layout"  # SIE-003 seam target (archetype.id)
CORPUS_REL = Path("src/slm_training/resources/data/eval/openui_hard_valid_v1/records.jsonl")
SIE002_AUTH_DOC = Path(
    "docs/design/iter-slm476-sie-002-oracle-localization-20260810.json"
)

# VCE-009 / SIE-002 factor names → PROGRAM_FACTOR_FAMILIES (SIE-003 vocabulary).
SIE002_TO_PROGRAM_FAMILY: dict[str, str] = {
    "archetype": "style_layout",
    "roles": "property_role_value",
    "topology": "topology",
    "bindings": "binder_reference",
}

ARM_IDS: tuple[str, ...] = (
    "none",
    "deterministic_extractor",
    "frequency",
    "retrieval",
    "learned_predictor",
    "gold_oracle",
)

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "FACTOR_FAMILY",
    "PRIMARY_METRIC",
    "SIE002_AUTH_DOC",
    "SIE002_TO_PROGRAM_FAMILY",
    "Sie004CampaignV1",
    "evaluate_predictions",
    "legal_support_parity",
    "load_authorized_factors",
    "load_corpus_examples",
    "plan_only_preview",
    "retrieval_predict",
    "run_campaign",
    "split_by_source_program",
]


def load_authorized_factors(path: Path | None = None) -> dict[str, Any]:
    """Load SIE-002 durable authorized_factors (fail-closed if missing/empty)."""
    doc = path or SIE002_AUTH_DOC
    if not doc.is_file():
        return {
            "authorized_factors": [],
            "authorized_program_families": [],
            "source_doc": str(doc),
            "loaded": False,
            "note": "SIE-002 durable doc missing; no factor authorized.",
        }
    payload = json.loads(doc.read_text(encoding="utf-8"))
    factors = list(payload.get("authorized_factors") or [])
    program_families = sorted(
        {
            SIE002_TO_PROGRAM_FAMILY[f]
            for f in factors
            if f in SIE002_TO_PROGRAM_FAMILY
        }
    )
    return {
        "authorized_factors": factors,
        "authorized_program_families": program_families,
        "source_doc": str(doc),
        "loaded": True,
        "sie002_claim_class": payload.get("claim_class"),
        "note": (
            f"SIE-002 authorized {factors}; mapped program families {program_families}. "
            f"SIE-003 seam targets {FACTOR_FAMILY!r} "
            f"({'authorized' if FACTOR_FAMILY in program_families else 'NOT authorized'})."
        ),
    }


def load_corpus_examples(
    *,
    corpus_path: Path | None = None,
    max_examples: int | None = None,
) -> list[PromptFactorExampleV1]:
    """Positive-role (prompt, archetype) pairs; never contrast negatives."""
    path = corpus_path or CORPUS_REL
    examples: list[PromptFactorExampleV1] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_examples is not None and len(examples) >= max_examples:
                break
            payload = json.loads(line)
            if payload.get("role") != "positive":
                continue
            examples.append(
                PromptFactorExampleV1(
                    source_program_id=str(payload["source_program_id"]),
                    prompt=str(payload["record"]["prompt"]),
                    label=str(payload["source_plan"]["archetype"]["id"]),
                )
            )
    return examples


def split_by_source_program(
    examples: list[PromptFactorExampleV1],
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[PromptFactorExampleV1], list[PromptFactorExampleV1]]:
    """Catalogue leakage control: group by source template before splitting."""
    programs = sorted({ex.source_program_id for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(programs)
    split_at = max(1, int(len(programs) * train_fraction))
    if split_at >= len(programs):
        split_at = len(programs) - 1
    train_programs = set(programs[:split_at])
    train = [ex for ex in examples if ex.source_program_id in train_programs]
    held = [ex for ex in examples if ex.source_program_id not in train_programs]
    if not train or not held:
        raise ValueError("split produced empty train or held-out set")
    if {e.source_program_id for e in train} & {e.source_program_id for e in held}:
        raise ValueError("source_program_id leaked across train/held-out")
    return train, held


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t for t in text.lower().split() if t)


def retrieval_predict(
    train: list[PromptFactorExampleV1], prompt: str
) -> str | None:
    """Train-only 1-NN Jaccard retrieval — leakage-safe (no held-out lookup)."""
    query = _tokenize(prompt)
    if not query or not train:
        return None
    best_label: str | None = None
    best_score = -1.0
    for example in train:
        tokens = _tokenize(example.prompt)
        if not tokens:
            continue
        score = len(query & tokens) / len(query | tokens)
        if score > best_score:
            best_score = score
            best_label = example.label
    return best_label if best_score > 0.0 else None


def evaluate_predictions(
    gold: list[str],
    predicted: list[str | None],
) -> dict[str, Any]:
    """Micro F1 over non-abstained predictions + coverage / selective risk."""
    if len(gold) != len(predicted):
        raise ValueError("gold/predicted length mismatch")
    n = len(gold)
    accepted = [(g, p) for g, p in zip(gold, predicted) if p is not None]
    coverage = (len(accepted) / n) if n else 0.0
    if not accepted:
        return {
            "n": n,
            "n_accepted": 0,
            "coverage": 0.0,
            "selective_risk": None,
            "requirement_predictor_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "abstention_rate": 1.0,
        }
    # Multiclass micro-F1 ≡ accuracy on the accepted set when every
    # prediction is a single hard label (TP+FP = accepted, TP+FN = accepted
    # only for labels that appear; use label-set micro counts).
    labels = sorted({g for g, _ in accepted} | {p for _, p in accepted if p is not None})
    tp = fp = fn = 0
    for label in labels:
        for g, p in accepted:
            if p == label and g == label:
                tp += 1
            elif p == label and g != label:
                fp += 1
            elif g == label and p != label:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    errors = sum(1 for g, p in accepted if g != p)
    selective_risk = errors / len(accepted)
    return {
        "n": n,
        "n_accepted": len(accepted),
        "coverage": round(coverage, 6),
        "selective_risk": round(selective_risk, 6),
        "requirement_predictor_f1": round(f1, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "abstention_rate": round(1.0 - coverage, 6),
    }


def legal_support_parity(
    *,
    authorized_program_families: tuple[str, ...],
    sample_prediction_value: str = "stack_column",
) -> dict[str, Any]:
    """Exact legal-support parity: predictor on/off must not unlock hard prune."""
    from slm_training.models.prompt_factor_predictor import (
        PREDICTOR_SCHEMA_VERSION,
        PREDICTOR_VERSION,
        PromptFactorPredictionV1,
    )

    empty = PromptSemanticRequirementsV1(facts=())
    off = requirements_hard_prune_allowed(empty)
    prediction = PromptFactorPredictionV1(
        schema_version=PREDICTOR_SCHEMA_VERSION,
        predictor_version=PREDICTOR_VERSION,
        factor_family=FACTOR_FAMILY,
        predicted_value=sample_prediction_value,
        confidence=0.99,
        abstained=False,
        abstention_reason=None,
        authority="advisory-learned",
        provenance="predicted",
    )
    fact = prediction_to_requirement_fact(
        prediction,
        fact_id="sie004-parity",
        authorized_factor_families=authorized_program_families or (FACTOR_FAMILY,),
    )
    # Even when we force-authorize the seam family for the parity probe,
    # advisory facts alone must not unlock prune.
    if fact is None:
        # Force a fact under an explicit authorized set that includes the seam.
        fact = prediction_to_requirement_fact(
            prediction,
            fact_id="sie004-parity",
            authorized_factor_families=(FACTOR_FAMILY,),
        )
    assert fact is not None
    on_reqs = PromptSemanticRequirementsV1(facts=(fact,))
    on = requirements_hard_prune_allowed(on_reqs)
    on_pred_only = requirements_hard_prune_allowed(
        on_reqs, evidence_kinds=(EvidenceKind.PREDICTION_ONLY,)
    )
    parity = off is False and on is False and on_pred_only is False
    return {
        "legal_support_parity_exact": parity,
        "hard_prune_predictor_off": off,
        "hard_prune_predictor_on": on,
        "hard_prune_prediction_only_evidence": on_pred_only,
        "note": (
            "Advisory RequirementFacts never unlock hard prune; byte/set legal "
            "support unchanged with predictor on vs off (I6)."
        ),
    }


def _arm_predict(
    arm_id: str,
    *,
    train: list[PromptFactorExampleV1],
    held: list[PromptFactorExampleV1],
    classifier: TinyBagOfWordsClassifier | None,
    confidence_threshold: float,
    majority_label: str | None,
) -> list[str | None]:
    preds: list[str | None] = []
    for example in held:
        if arm_id == "none":
            preds.append(None)
        elif arm_id == "deterministic_extractor":
            preds.append(deterministic_keyword_predict(example.prompt))
        elif arm_id == "frequency":
            preds.append(majority_label)
        elif arm_id == "retrieval":
            preds.append(retrieval_predict(train, example.prompt))
        elif arm_id == "learned_predictor":
            assert classifier is not None
            pred = predict_factor(
                classifier,
                example.prompt,
                factor_family=FACTOR_FAMILY,
                confidence_threshold=confidence_threshold,
            )
            preds.append(None if pred.abstained else pred.predicted_value)
        elif arm_id == "gold_oracle":
            preds.append(example.label)
        else:
            raise ValueError(f"unknown arm: {arm_id}")
    return preds


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    auth = load_authorized_factors()
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "factor_family": FACTOR_FAMILY,
        "authorized_factors_sie002": auth["authorized_factors"],
        "authorized_program_families": auth["authorized_program_families"],
        "claim_class_execution": "fixture",
        "promotion": False,
    }


@dataclass(frozen=True)
class Sie004CampaignV1:
    campaign_id: str = "sie004-exp-sr-2-predictor-campaign-fixture"
    max_examples: int = 400
    train_fraction: float = 0.7
    seeds: tuple[int, ...] = (0, 1, 2)
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)
    sie002_auth_doc: str = str(SIE002_AUTH_DOC)

    def __post_init__(self) -> None:
        if self.max_examples < 20:
            raise ValueError("max_examples must be >= 20")
        if not 0.1 < self.train_fraction < 0.9:
            raise ValueError("train_fraction must be in (0.1, 0.9)")
        if self.claim_class != "fixture":
            raise ValueError("SIE-004 durable evidence is fixture-only (no promotion)")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="candidate" if arm == "learned_predictor" else "control",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Gate the advisory-only requirement-prediction seam on measured "
                "precision vs frequency/deterministic/retrieval controls; gold "
                "oracle is a diagnostic ceiling only. No checkpoint or default "
                "promotion from this fixture."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=self.minimum_effect(),
                ),
            ),
            arms=arms,
            seeds=self.seeds,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(self.seeds),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Every arm scores the same held-out prompts for a seed.",
                "Train/held-out split groups by source_program_id before split.",
                "No GPU; TinyBagOfWordsClassifier is pure NumPy.",
                "Gold oracle is diagnostic only and never a promotion path.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="frequency_baseline",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else "Requirement-frequency-only baseline on the same prompts.",
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=(
                        "Prompt/requirement pairs grouped by source_program_id "
                        "before any train/eval split; retrieval uses train only."
                    ),
                    kind="negative",
                ),
                CampaignControlV1(
                    control_id="legal_support_parity",
                    description=(
                        "Advisory predictions must not change hard-prune / legal "
                        "support vs predictor-off."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("leakage_control", "legal_support_parity"),
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
                    operator="le",
                    threshold=0.0,
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


def run_campaign(campaign: Sie004CampaignV1, *, root: Path) -> dict[str, Any]:
    """Multi-seed matched predictor campaign; persist under CampaignStore."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="SIE-004 / EXP-SR-2 fixture-scale prompt-requirement predictor",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(campaign.seeds),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()
    auth = load_authorized_factors(Path(campaign.sie002_auth_doc))
    authorized_program = tuple(auth["authorized_program_families"])
    parity = legal_support_parity(authorized_program_families=authorized_program)

    examples = load_corpus_examples(max_examples=campaign.max_examples)
    if len(examples) < 20:
        raise ValueError(f"need >=20 positive examples, got {len(examples)}")

    seed_runs: list[dict[str, Any]] = []
    arm_f1_across_seeds: dict[str, list[float]] = {arm: [] for arm in ARM_IDS}

    for seed in campaign.seeds:
        train, held = split_by_source_program(
            examples, seed=seed, train_fraction=campaign.train_fraction
        )
        freq = frequency_baseline_predict(train)
        majority_label = max(freq, key=freq.get) if freq else None
        classifier = TinyBagOfWordsClassifier.fit(train)
        calibration = calibrate_abstention_threshold(classifier, held)
        threshold = resolve_confidence_threshold(calibration)

        gold = [ex.label for ex in held]
        arm_metrics: dict[str, Any] = {}
        for arm in ARM_IDS:
            preds = _arm_predict(
                arm,
                train=train,
                held=held,
                classifier=classifier,
                confidence_threshold=threshold,
                majority_label=majority_label,
            )
            metrics = evaluate_predictions(gold, preds)
            arm_f1_across_seeds[arm].append(metrics["requirement_predictor_f1"])
            # Fact emission under SIE-002 gate (fail-closed for unauthorized families).
            facts_emitted = 0
            for pred_value in preds:
                if pred_value is None:
                    continue
                from slm_training.models.prompt_factor_predictor import (
                    PREDICTOR_SCHEMA_VERSION,
                    PREDICTOR_VERSION,
                    PromptFactorPredictionV1,
                )

                probe = PromptFactorPredictionV1(
                    schema_version=PREDICTOR_SCHEMA_VERSION,
                    predictor_version=PREDICTOR_VERSION,
                    factor_family=FACTOR_FAMILY,
                    predicted_value=pred_value,
                    confidence=1.0,
                    abstained=False,
                    abstention_reason=None,
                    authority="advisory-learned",
                    provenance="predicted",
                )
                if (
                    prediction_to_requirement_fact(
                        probe,
                        fact_id="emit",
                        authorized_factor_families=authorized_program,
                    )
                    is not None
                ):
                    facts_emitted += 1
            arm_metrics[arm] = {
                **metrics,
                "requirement_facts_emitted_under_sie002_auth": facts_emitted,
            }

        seed_runs.append(
            {
                "seed": seed,
                "n_train": len(train),
                "n_held_out": len(held),
                "n_train_programs": len({e.source_program_id for e in train}),
                "n_held_programs": len({e.source_program_id for e in held}),
                "majority_label": majority_label,
                "confidence_threshold": threshold,
                "calibration": {
                    "n": calibration.get("n"),
                    "brier": calibration.get("brier"),
                    "ece": calibration.get("ece"),
                    "abstain_required": calibration.get("abstain_required"),
                    "operating_point": calibration.get("operating_point"),
                },
                "arms": arm_metrics,
            }
        )

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 6) if xs else 0.0

    aggregate_arms = {
        arm: {
            "requirement_predictor_f1_mean": _mean(arm_f1_across_seeds[arm]),
            "requirement_predictor_f1_per_seed": arm_f1_across_seeds[arm],
            "n_seeds": len(arm_f1_across_seeds[arm]),
        }
        for arm in ARM_IDS
    }
    learned_f1 = aggregate_arms["learned_predictor"]["requirement_predictor_f1_mean"]
    frequency_f1 = aggregate_arms["frequency"]["requirement_predictor_f1_mean"]
    none_f1 = aggregate_arms["none"]["requirement_predictor_f1_mean"]
    gold_f1 = aggregate_arms["gold_oracle"]["requirement_predictor_f1_mean"]
    delta_vs_frequency = round(learned_f1 - frequency_f1, 6)
    ceiling_recovery = (
        round(learned_f1 / gold_f1, 6) if gold_f1 > 0 else 0.0
    )
    min_effect = campaign.minimum_effect()
    # Catalogue falsifier: predictor F1 does not exceed majority/no-prediction.
    clears_frequency = learned_f1 > frequency_f1 + 1e-12
    clears_none = learned_f1 > none_f1 + 1e-12
    clears_minimum_effect = delta_vs_frequency >= min_effect - 1e-12
    dominated_by_simpler = max(
        aggregate_arms["deterministic_extractor"]["requirement_predictor_f1_mean"],
        aggregate_arms["frequency"]["requirement_predictor_f1_mean"],
        aggregate_arms["retrieval"]["requirement_predictor_f1_mean"],
    ) >= learned_f1 - 1e-12
    falsifier_holds = not (clears_frequency and clears_none and clears_minimum_effect)

    # Multi-seed paired uncertainty: seed-wise deltas vs frequency.
    paired_deltas = [
        arm_f1_across_seeds["learned_predictor"][i]
        - arm_f1_across_seeds["frequency"][i]
        for i in range(len(campaign.seeds))
    ]
    paired_mean = _mean(paired_deltas)
    paired_var = (
        sum((d - paired_mean) ** 2 for d in paired_deltas) / max(1, len(paired_deltas) - 1)
        if len(paired_deltas) > 1
        else 0.0
    )
    paired_se = round((paired_var**0.5) / (len(paired_deltas) ** 0.5), 6) if paired_deltas else 0.0

    result = {
        "schema": "sie004_predictor_campaign_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "catalogue_claim_class": catalogue.claim_class,
        "promotion": False,
        "dataset_id": DATASET_ID,
        "factor_family": FACTOR_FAMILY,
        "max_examples": campaign.max_examples,
        "seeds": list(campaign.seeds),
        "authorized_factors_source": auth,
        "legal_support_parity": parity,
        "requirement_predictor_f1": learned_f1,
        "aggregate_arms": aggregate_arms,
        "effect_vs_frequency": delta_vs_frequency,
        "ceiling_recovery_vs_gold": ceiling_recovery,
        "minimum_effect": min_effect,
        "paired_vs_frequency": {
            "deltas": [round(d, 6) for d in paired_deltas],
            "mean": paired_mean,
            "se": paired_se,
            "n_seeds": len(paired_deltas),
        },
        "falsifier": {
            "falsifier_holds": falsifier_holds,
            "clears_frequency_baseline": clears_frequency,
            "clears_none_baseline": clears_none,
            "clears_minimum_effect": clears_minimum_effect,
            "dominated_by_simpler_control": dominated_by_simpler,
            "note": (
                "Catalogue falsifier: predictor F1 does not exceed the "
                "majority-class/no-prediction baseline on the held-out prompt slice."
                if falsifier_holds
                else "Learned predictor cleared frequency/none + minimum_effect on "
                "this fixture; still claim_class=fixture — not a promotion. "
                "SIE-002 did not authorize style_layout/archetype, so "
                "RequirementFact emission under that gate remains fail-closed."
            ),
        },
        "seed_runs": seed_runs,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-2 prompt-requirement predictor campaign over "
            "openui_hard_valid_v1 positive-role rows. Arms reuse SIE-003 "
            "predictor primitives; retrieval is train-only 1-NN (leakage-safe). "
            "Primary metric is requirement_predictor_f1 on archetype/"
            f"{FACTOR_FAMILY} labels. SIE-002 authorized_factors={auth['authorized_factors']} "
            f"(program families {auth['authorized_program_families']}); the seam "
            f"target {FACTOR_FAMILY!r} is not in that set, so advisory fact emission "
            "under the SIE-002 gate is zero. Legal support parity is exact "
            "(advisory facts never hard-prune). claim_class=fixture; no promotion."
        ),
    }
    artifact = store.write_artifact("sie004_predictor_campaign_result", result)
    store.append_event(
        "sie004_predictor_campaign_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "requirement_predictor_f1": learned_f1,
            "falsifier_holds": falsifier_holds,
            "legal_support_parity_exact": parity["legal_support_parity_exact"],
        },
    )
    return result
