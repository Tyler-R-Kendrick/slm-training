"""VCE-010 (SLM-469): evaluator calibration, blinded adjudication, and
risk/coverage protocol.

Composes existing owners instead of reimplementing any of them:

* frozen stratified sample -- :class:`~slm_training.data.semantic_contrast.
  builder.SemanticContrastBuilder` (VCE-006/007); each admitted contrast
  pair's own ``verifier_ok`` ground truth (already computed by the builder
  through the real compiler-legality gate stack) is reused directly as the
  "deterministic evaluator" verdict -- never recomputed here.
* independent external-family judge, mocked adapter, no real credentials --
  :class:`~slm_training.evals.judge_independence.ExternalJudgeAdapter` /
  :class:`~slm_training.evals.judge_independence.JudgeEvidenceV1`.
* blinded human labels, >=2 raters + adjudication policy --
  :mod:`slm_training.harnesses.annotations.judge_audit`
  (``freeze_blinded_pairs`` / ``import_blinded_labels``).
* agreement, pass-set divergence, calibration-error statistics --
  :func:`slm_training.evals.judge_independence.analyze_triple_judges`.
* risk/coverage curve + Brier/ECE --
  :func:`slm_training.dsl.grammar.fastpath.trust_train.gate_calibration_report`.
* preregistered promotion/rollback thresholds --
  :class:`slm_training.autoresearch.experiment_campaign.ExperimentCampaignV1`,
  locked via :class:`slm_training.autoresearch.storage.CampaignStore` *before*
  any external-judge or human-label evidence is gathered.

This module never imports ``slm_training.data.verify`` (the G0-G12
compiler-legality gate stack). Human and independent-judge evidence can
inform calibration diagnostics here, but can never become compiler legality
(AC) -- enforced structurally by this import boundary, not by a runtime
check, and pinned by
``test_module_never_imports_the_compiler_gate_stack``.

The only genuinely new statistic here is :func:`reason_family_confusion_matrix`
-- no crosstab owner exists elsewhere in the repo (``analyze_triple_judges``
only tallies a flat disagreement-reason Counter, not an actual-vs-predicted
family crosstab).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.semantic_contrast.builder import SemanticContrastBuilder
from slm_training.dsl.grammar.fastpath.trust_train import gate_calibration_report
from slm_training.evals.judge_independence import (
    ExternalJudgeAdapter,
    ExternalJudgeConfig,
    analyze_triple_judges,
    text_sha256,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.annotations.judge_audit import (
    freeze_blinded_pairs,
    import_blinded_labels,
)
from slm_training.levers import MAX_RUN_MINUTES

ARM_IDS: tuple[str, ...] = (
    "deterministic_only",
    "external_only_missing_human",
    "human_only_missing_external",
    "full_triple_judge",
)
REASON_FAMILIES: tuple[str, ...] = (
    "positive",
    "content",
    "topology",
    "binding",
    "contract",
)

__all__ = [
    "ARM_IDS",
    "REASON_FAMILIES",
    "EvaluatorCalibrationProtocolV1",
    "reason_family_confusion_matrix",
    "run_protocol",
]


@dataclass(frozen=True)
class EvaluatorCalibrationProtocolV1:
    protocol_id: str = "vce010-evaluator-calibration-protocol"
    source_count: int = 4
    seed: int = 0
    min_raters: int = 2
    calibration_error_max: float = 0.35
    claim_class: str = "diagnostic"
    source_commit: str = "0" * 40
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.source_count < 2:
            raise ValueError("source_count must be >= 2")
        if self.min_raters < 2:
            raise ValueError("min_raters must be >= 2 (>=2 human labels per pair)")
        if self.claim_class not in {"fixture", "diagnostic"}:
            raise ValueError("VCE-010 only exposes fixture/diagnostic evidence classes")
        if not 0.0 < self.calibration_error_max <= 1.0:
            raise ValueError("calibration_error_max must be in (0, 1]")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "deterministic_only" else "candidate",
                config_sha256=content_sha({"protocol": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.protocol_id,
            experiment_id=self.protocol_id,
            hypothesis=(
                "A deterministic evaluator's admission decisions are only as "
                "trustworthy as the independent evidence calibrating them: "
                "external-judge and blinded-human evidence must each be "
                "gathered before a pair can be scored, and the absence of "
                "either must block the corresponding arm rather than default "
                "to the deterministic verdict."
            ),
            decision=(
                "Report calibration diagnostics (agreement, calibration "
                "error, pass-set divergence, risk/coverage) honestly; no "
                "evaluator-promotion or model-quality claim is made here."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id="external_human_calibration_error",
                    metric="full_triple_judge.external_vs_human.calibration_error",
                    role="primary",
                    direction="decrease",
                    minimum_effect=0.0,
                ),
                CampaignEndpointV1(
                    endpoint_id="external_human_cohen_kappa",
                    metric="full_triple_judge.pairwise.external_vs_human.cohen_kappa",
                    role="secondary",
                    direction="increase",
                    minimum_effect=0.0,
                ),
                CampaignEndpointV1(
                    endpoint_id="admission_divergence_rate",
                    metric="full_triple_judge.pairwise.deterministic_vs_human.admission_divergence_rate",
                    role="secondary",
                    direction="decrease",
                    minimum_effect=0.0,
                ),
            ),
            arms=arms,
            selection_rule=SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=self.source_count * len(ARM_IDS),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "An arm missing external or human evidence is reported "
                "blocked, never scored as if the evidence were present.",
                "A pair with an incomplete or disagreeing-without-adjudicator "
                "rater set is retained as ambiguous, never defaulted to a "
                "winner.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="deterministic_only_control",
                    description=(
                        "The deterministic-only arm never consults external "
                        "or human evidence; it establishes the ground-truth "
                        "admission label the other arms are calibrated "
                        "against."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="missing_evidence_blocked_control",
                    description=(
                        "external_only_missing_human and "
                        "human_only_missing_external must both report "
                        "disposition=blocked, never a fabricated "
                        "triple-judge result."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("missing_evidence_blocked_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id="evaluator_calibration",
                    hypothesis_ids=(
                        "external_human_calibration_error",
                        "external_human_cohen_kappa",
                        "admission_divergence_rate",
                    ),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                # Fixture/diagnostic evidence is never a promotion claim;
                # deliberately permissive (mirrors PCT-008/VCE-009's
                # "fixture_only" pattern).
                CampaignGateV1(
                    gate_id="diagnostic_only",
                    endpoint_id="external_human_calibration_error",
                    operator="le",
                    threshold=1.0,
                ),
            ),
            rollback_gates=(
                # Real, non-vacuous: preregistered before any external-judge
                # or human-label evidence exists (AC).
                CampaignGateV1(
                    gate_id="calibration_error_exceeds_preregistered_ceiling",
                    endpoint_id="external_human_calibration_error",
                    operator="gt",
                    threshold=self.calibration_error_max,
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


def _external_config() -> ExternalJudgeConfig:
    return ExternalJudgeConfig(
        provider="mock_external_judge_provider",
        provider_version="fixture-1",
        model_family="mock_external_judge_family",
        model_version="fixture-1",
        rubric_id="vce010_pairwise_acceptability",
        rubric_version="1",
        rubric=(
            "Given two OpenUI programs generated for the same prompt, "
            "choose the one that is semantically valid."
        ),
        temperature=0.0,
        seed=0,
        max_attempts=2,
        max_tokens=256,
        max_cost_usd=1.0,
    )


def _mock_transport(
    rng: random.Random, *, correct_rate: float, family_by_left_openui: dict[str, str]
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Stand in for a real external-judge API call (AC: executable without
    external credentials). Reads only ``prompt``/``left_openui``/
    ``right_openui`` -- never a "positive" label -- and returns the correct
    (canonical-left) side at ``correct_rate``, a labeled fixture-scale
    simulation of an imperfectly calibrated external judge."""

    def _transport(request: dict[str, Any]) -> dict[str, Any]:
        pair = request["pair"]
        family = family_by_left_openui.get(pair["left_openui"], "unknown")
        if rng.random() < correct_rate:
            verdict = "left"
            reason_codes = [family]
            confidence = round(0.75 + 0.2 * rng.random(), 3)
        else:
            verdict = "right"
            reason_codes = [rng.choice(REASON_FAMILIES)]
            confidence = round(0.4 + 0.2 * rng.random(), 3)
        return {
            "verdict": verdict,
            "left_acceptable": verdict == "left",
            "right_acceptable": verdict == "right",
            "score": confidence,
            "confidence": confidence,
            "reason_codes": reason_codes,
            "cost_usd": 0.0001,
        }

    return _transport


def _frozen_stratified_sample(
    protocol: EvaluatorCalibrationProtocolV1, *, root: Path
) -> list[dict[str, Any]]:
    """A real, small, freshly generated semantic-contrast pair sample
    (VCE-006/007) -- admitted pairs only, so positive/negative ground truth
    is already established by the builder's own compiler-legality check.
    This module never calls ``verify_record`` itself."""
    builder = SemanticContrastBuilder(
        output_root=root,
        dataset_id=f"{protocol.protocol_id}-corpus",
        seed=protocol.seed,
        source_count=protocol.source_count,
        splits=("train",),
        split_weights=(1.0,),
        wide_sources=True,
        strict_delta=True,
    )
    builder.build()
    pairs_path = builder.output_dir / "pairs.jsonl"
    return [
        json.loads(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _deterministic_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical order left=positive, right=negative. The "deterministic
    evaluator" verdict reuses the corpus's own ``verifier_ok`` ground truth
    -- never recomputed here."""
    rows = []
    for pair in pairs:
        positive = pair["positive"]["record"]
        negative = pair["negative"]["record"]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "stratum": pair["family"],
                "prompt": positive["prompt"],
                "positive_openui": positive["openui"],
                "negative_openui": negative["openui"],
                "positive_sha256": text_sha256(positive["openui"]),
                "negative_sha256": text_sha256(negative["openui"]),
                "deterministic_verdict": "left",
                "deterministic_score": 1.0,
                "deterministic_pass": True,
            }
        )
    return rows


def _external_arm_rows(
    protocol: EvaluatorCalibrationProtocolV1, deterministic_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    start = time.perf_counter()
    config = _external_config()
    rng = random.Random(f"{protocol.protocol_id}:external:{protocol.seed}")
    family_by_left = {
        row["positive_openui"]: row["stratum"] for row in deterministic_rows
    }
    adapter = ExternalJudgeAdapter(
        config,
        _mock_transport(rng, correct_rate=0.8, family_by_left_openui=family_by_left),
    )
    rows = []
    for det_row in deterministic_rows:
        evidence = adapter.judge(
            audit_id=protocol.protocol_id,
            record_id=det_row["pair_id"],
            generation_id=f"{det_row['pair_id']}:gen",
            pair_id=det_row["pair_id"],
            prompt=det_row["prompt"],
            left_openui=det_row["positive_openui"],
            right_openui=det_row["negative_openui"],
            left_checkpoint_sha256=det_row["positive_sha256"],
            right_checkpoint_sha256=det_row["negative_sha256"],
            candidate_model_families=(
                "semantic_contrast_positive",
                "semantic_contrast_negative",
            ),
            prior_judge_providers=("slm_training_deterministic_evaluator",),
            prior_judge_model_families=("slm_training_deterministic_evaluator",),
            order_seed=protocol.seed,
        )
        rows.append(
            {
                "pair_id": det_row["pair_id"],
                "external_verdict": evidence.verdict,
                "external_score": evidence.score,
                "external_pass": evidence.verdict == det_row["deterministic_verdict"],
                "external_confidence": evidence.confidence,
                "external_reason_codes": list(evidence.reason_codes),
                "external_cost_usd": evidence.cost_usd,
                "external_latency_ms": evidence.latency_ms,
                "_external_independent": evidence.independent,
            }
        )
    return {"rows": rows, "wall_ms": (time.perf_counter() - start) * 1000.0}


def _human_arm_rows(
    protocol: EvaluatorCalibrationProtocolV1,
    deterministic_rows: list[dict[str, Any]],
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze a blinded pair study (AC: >=2 human labels per pair, plus an
    adjudication policy for disagreement), then aggregate it -- both reused
    from :mod:`judge_audit`, never reimplemented."""
    start = time.perf_counter()
    audit_id = f"{protocol.protocol_id}-human"
    output_dir = root / "public"
    private_key_path = root / "private" / "unblinding_key.jsonl"
    rows_in = [
        {
            "pair_id": det_row["pair_id"],
            "prompt": det_row["prompt"],
            "stratum": det_row["stratum"],
            "source_record_id": det_row["pair_id"],
            "candidate_a": {
                "openui": det_row["positive_openui"],
                "candidate_id": f"{det_row['pair_id']}:positive",
                "model_family": "semantic_contrast_positive",
                "run_id": protocol.protocol_id,
                "checkpoint_sha256": det_row["positive_sha256"],
            },
            "candidate_b": {
                "openui": det_row["negative_openui"],
                "candidate_id": f"{det_row['pair_id']}:negative",
                "model_family": "semantic_contrast_negative",
                "run_id": protocol.protocol_id,
                "checkpoint_sha256": det_row["negative_sha256"],
            },
        }
        for det_row in deterministic_rows
    ]
    freeze_blinded_pairs(
        rows_in,
        audit_id=audit_id,
        seed=protocol.seed,
        output_dir=output_dir,
        private_key_path=private_key_path,
        enforce_campaign_coverage=False,
    )
    blind_pairs = [
        json.loads(line)
        for line in (output_dir / "blind_pairs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    positive_sha_by_pair = {
        row["pair_id"]: row["positive_sha256"] for row in deterministic_rows
    }
    correct_side_by_pair = {
        row["pair_id"]: (
            "left"
            if row["left_openui_sha256"] == positive_sha_by_pair[row["pair_id"]]
            else "right"
        )
        for row in blind_pairs
    }

    rng = random.Random(f"{protocol.protocol_id}:human:{protocol.seed}")
    label_rows: list[dict[str, Any]] = []
    disagreement_forced = False
    for index, row in enumerate(blind_pairs):
        pair_id = row["pair_id"]
        correct = correct_side_by_pair[pair_id]
        wrong = "right" if correct == "left" else "left"
        n_raters = max(2, protocol.min_raters)
        force_disagreement = not disagreement_forced and index == 0
        winners = [correct if rng.random() < 0.85 else wrong for _ in range(n_raters)]
        if force_disagreement:
            winners[0] = wrong
            disagreement_forced = True
        for rater_index, winner in enumerate(winners):
            label_rows.append(
                {
                    "schema": "BlindJudgeLabelV1",
                    "audit_id": audit_id,
                    "pair_id": pair_id,
                    "annotator_id": f"ann_rater{rater_index:02d}seed{protocol.seed:04d}fixture",
                    "role": "rater",
                    "winner": winner,
                    "acceptable_left": winner == "left",
                    "acceptable_right": winner == "right",
                    "reasons": [row["stratum"]]
                    if winner == correct
                    else [rng.choice(REASON_FAMILIES)],
                    "confidence": round(0.6 + 0.3 * rng.random(), 3),
                    "duration_ms": int(1000 + 5000 * rng.random()),
                    "submitted_at": "2026-01-01T00:00:00Z",
                }
            )
        if len(set(winners)) > 1:
            label_rows.append(
                {
                    "schema": "BlindJudgeLabelV1",
                    "audit_id": audit_id,
                    "pair_id": pair_id,
                    "annotator_id": f"ann_adjudicatorseed{protocol.seed:04d}fixture",
                    "role": "adjudicator",
                    "winner": correct,
                    "acceptable_left": correct == "left",
                    "acceptable_right": correct == "right",
                    "reasons": [row["stratum"]],
                    "confidence": 0.95,
                    "duration_ms": 4000,
                    "submitted_at": "2026-01-01T00:05:00Z",
                }
            )

    label_path = root / "labels" / "raters.jsonl"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows), encoding="utf-8"
    )

    aggregate = import_blinded_labels(
        [label_path],
        audit_id=audit_id,
        pair_ids={row["pair_id"] for row in blind_pairs},
        output_path=root / "human_aggregate.json",
    )
    rows = _human_rows_from_aggregate(aggregate, correct_side_by_pair)
    return aggregate, {"rows": rows, "wall_ms": (time.perf_counter() - start) * 1000.0}


def _human_rows_from_aggregate(
    aggregate: dict[str, Any], correct_side_by_pair: dict[str, str]
) -> list[dict[str, Any]]:
    """Translate a :func:`judge_audit.import_blinded_labels` aggregate
    (reused, not reimplemented) into canonical-order human evidence rows.

    AC: ambiguous/UNKNOWN cases are retained separately, never defaulted to
    a winner -- an incomplete pair (``complete=False`` or no consensus
    winner, e.g. disagreement without an adjudicator) gets
    ``human_ambiguous=True``, ``human_verdict="error"``, and
    ``human_pass=False`` rather than silently picking a side.
    """
    aggregate_by_pair = {row["pair_id"]: row for row in aggregate["pairs"]}
    rows = []
    for pair_id, correct in correct_side_by_pair.items():
        agg = aggregate_by_pair[pair_id]
        consensus = agg["consensus_winner"]
        ambiguous = not agg["complete"] or consensus is None
        human_verdict = (
            "error" if ambiguous else ("left" if consensus == correct else "right")
        )
        top_reasons = [
            reason
            for reason, _ in sorted(
                agg["reason_counts"].items(), key=lambda kv: -kv[1]
            )[:1]
        ]
        rows.append(
            {
                "pair_id": pair_id,
                "human_verdict": human_verdict,
                "human_score": agg["mean_confidence"]
                if agg["mean_confidence"] is not None
                else 0.0,
                "human_pass": (not ambiguous) and human_verdict == "left",
                "human_ambiguous": ambiguous,
                "human_reason_codes": top_reasons,
            }
        )
    return rows


def _merge_rows(
    deterministic_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    external_by_pair = {row["pair_id"]: row for row in external_rows}
    human_by_pair = {row["pair_id"]: row for row in human_rows}
    merged = []
    for det in deterministic_rows:
        pair_id = det["pair_id"]
        ext = external_by_pair[pair_id]
        hum = human_by_pair[pair_id]
        merged.append(
            {
                "pair_id": pair_id,
                "stratum": det["stratum"],
                "checkpoint_family": det["stratum"],
                "suite": "vce010_fixture",
                "deterministic_verdict": det["deterministic_verdict"],
                "deterministic_score": det["deterministic_score"],
                "deterministic_pass": det["deterministic_pass"],
                "external_verdict": ext["external_verdict"],
                "external_score": ext["external_score"],
                "external_pass": ext["external_pass"],
                "external_confidence": ext["external_confidence"],
                "external_cost_usd": ext["external_cost_usd"],
                "external_latency_ms": ext["external_latency_ms"],
                "external_reason_codes": ext["external_reason_codes"],
                "human_verdict": hum["human_verdict"],
                "human_score": hum["human_score"],
                "human_pass": hum["human_pass"],
                "human_ambiguous": hum["human_ambiguous"],
                "human_reason_codes": hum["human_reason_codes"],
                "reason_codes": [
                    *ext["external_reason_codes"],
                    *hum["human_reason_codes"],
                ],
                "_external_independent": ext["_external_independent"],
            }
        )
    return merged


def reason_family_confusion_matrix(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Crosstab of each pair's true stratum family against the reason-code
    family attributed by any judge that disagreed with the deterministic
    verdict. Genuinely new: no crosstab owner exists elsewhere in the repo
    (``analyze_triple_judges`` only tallies a flat disagreement Counter)."""
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        actual = str(row.get("stratum") or "unknown")
        for judge in ("external", "human"):
            if row.get(f"{judge}_verdict") == row.get("deterministic_verdict"):
                continue
            reasons = row.get(f"{judge}_reason_codes") or ["unknown"]
            predicted = str(reasons[0]) if reasons else "unknown"
            bucket = matrix.setdefault(actual, {})
            bucket[predicted] = bucket.get(predicted, 0) + 1
    return matrix


def run_protocol(
    protocol: EvaluatorCalibrationProtocolV1, *, root: Path
) -> dict[str, Any]:
    """Run all four arms in-process and persist a governed evidence
    artifact. Never spawns a subprocess; never touches the compiler-legality
    gate stack (see module docstring)."""
    store = CampaignStore(protocol.protocol_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=protocol.protocol_id,
            objective="Fixture-scale evaluator calibration / blinded adjudication protocol (VCE-010)",
            primary_metric="external_human_calibration_error",
            budget=CampaignBudget(
                max_experiments=protocol.source_count * len(ARM_IDS),
                max_wall_minutes=protocol.max_wall_minutes,
            ),
        )
    )
    # AC: promotion thresholds are preregistered before any external run --
    # the manifest is locked here, before a single external-judge or
    # human-label evidence row is gathered below.
    lock = store.lock_experiment_campaign(protocol.manifest())

    pairs = _frozen_stratified_sample(protocol, root=root / "corpus")
    if len(pairs) < 2:
        result = {
            "schema": "vce010_evaluator_calibration_result/v1",
            "campaign_id": protocol.protocol_id,
            "manifest_sha256": lock.manifest_sha256,
            "claim_class": protocol.claim_class,
            "disposition": "inconclusive",
            "reason": "frozen_stratified_sample_too_small",
            "pair_n": len(pairs),
        }
        artifact = store.write_artifact("vce010_evaluator_calibration_result", result)
        store.append_event(
            "vce010_evaluator_calibration_completed",
            experiment_id=protocol.protocol_id,
            status="inconclusive",
            artifact_sha256=artifact.stem,
            detail={"manifest_sha256": lock.manifest_sha256},
        )
        return result

    deterministic_rows = _deterministic_rows(pairs)
    external_arm = _external_arm_rows(protocol, deterministic_rows)
    _aggregate, human_arm = _human_arm_rows(
        protocol, deterministic_rows, root=root / "human"
    )
    full_rows = _merge_rows(deterministic_rows, external_arm["rows"], human_arm["rows"])

    analysis = analyze_triple_judges(full_rows)
    calibration_pairs = [row for row in full_rows if not row["human_ambiguous"]]
    risk_coverage = (
        gate_calibration_report(
            [row["external_confidence"] for row in calibration_pairs],
            [
                1.0 if row["external_verdict"] == row["human_verdict"] else 0.0
                for row in calibration_pairs
            ],
            max_selective_risk=0.2,
        )
        if calibration_pairs
        else {"disposition": "blocked", "reason": "no_unambiguous_pairs"}
    )
    confusion = reason_family_confusion_matrix(full_rows)
    independence_ok = all(row["_external_independent"] for row in full_rows)
    ambiguous_pair_ids = [row["pair_id"] for row in full_rows if row["human_ambiguous"]]

    common_compute = lambda wall_ms: {  # noqa: E731
        "forwards": 0,
        "verifier_calls": len(pairs),
        "wall_ms": round(wall_ms, 3),
    }
    arms = [
        {
            "arm": "deterministic_only",
            "disposition": "match",
            "pair_n": len(pairs),
            "compute": common_compute(0.0),
        },
        {
            "arm": "external_only_missing_human",
            "disposition": "blocked",
            "reason": "missing_human_evidence",
            "pair_n": len(pairs),
            "compute": common_compute(external_arm["wall_ms"]),
        },
        {
            "arm": "human_only_missing_external",
            "disposition": "blocked",
            "reason": "missing_external_evidence",
            "pair_n": len(pairs),
            "compute": common_compute(human_arm["wall_ms"]),
        },
        {
            "arm": "full_triple_judge",
            "disposition": "match" if independence_ok else "mismatch",
            "pair_n": len(full_rows),
            "ambiguous_pair_ids": ambiguous_pair_ids,
            "independence_ok": independence_ok,
            "triple_judge_analysis": analysis,
            "risk_coverage": risk_coverage,
            "reason_family_confusion": confusion,
            "compute": common_compute(external_arm["wall_ms"] + human_arm["wall_ms"]),
        },
    ]

    calibration_error = analysis["full"]["external_vs_human"]["calibration_error"]
    cohen_kappa_value = analysis["full"]["pairwise"]["external_vs_human"]["cohen_kappa"]
    divergence_rate = analysis["full"]["pairwise"]["deterministic_vs_human"][
        "admission_divergence_rate"
    ]
    gate = next(
        g
        for g in lock.manifest.rollback_gates
        if g.gate_id == "calibration_error_exceeds_preregistered_ceiling"
    )
    gate_fired = calibration_error is not None and calibration_error > gate.threshold

    result = {
        "schema": "vce010_evaluator_calibration_result/v1",
        "campaign_id": protocol.protocol_id,
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": protocol.claim_class,
        "disposition": "rollback_gate_fired" if gate_fired else "complete",
        "pair_n": len(pairs),
        "arms": arms,
        "external_human_calibration_error": calibration_error,
        "external_human_cohen_kappa": cohen_kappa_value,
        "admission_divergence_rate": divergence_rate,
        "preregistered_calibration_error_max": protocol.calibration_error_max,
        "rollback_gate_fired": gate_fired,
        "independence_ok": independence_ok,
        "ambiguous_pair_ids": ambiguous_pair_ids,
        "scope_disclaimer": (
            "Fixture-scale calibration evidence over a small, freshly "
            "generated semantic-contrast pair sample. The external judge "
            "and its transport are a labeled mock (no real external-model "
            "credentials or network call); human raters are synthetic, "
            "seeded labels standing in for a real blinded pair study. No "
            "evaluator-promotion or production-quality claim is made."
        ),
    }
    artifact = store.write_artifact("vce010_evaluator_calibration_result", result)
    store.append_event(
        "vce010_evaluator_calibration_completed",
        experiment_id=protocol.protocol_id,
        status=result["disposition"],
        artifact_sha256=artifact.stem,
        detail={"manifest_sha256": lock.manifest_sha256},
    )
    return result
