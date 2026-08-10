"""SIE-006 (SLM-483): evaluator construct-validity + human calibration (EXP-SR-3).

Executes the preregistered ``exp-sr-3`` catalogue identity under
:class:`ExperimentCampaignV1`, consuming the **frozen** SIE-005 blinded packet
(``docs/design/iter-slm478-sie-005-blinded-packet-20260810.json``) without
re-claiming packet preparation (SIE-005) or the VCE-010 calibration protocol
owner.

Modes (selected by ``scripts.run_sie006_construct_validity``):

* ``plan-only`` — preview the locked campaign manifest + SIE-005 pin.
* ``fixture`` — exercise agreement/calibration plumbing with **labeled mock**
  external-judge transport and **synthetic** human labels; never presented as
  real human evidence.
* ``external-blocked`` — honestly report ``external_blocked`` / incomplete when
  real external-judge credentials or human labels are unavailable; no fabricated
  kappa or calibration claims.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

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
from slm_training.dsl.grammar.fastpath.trust_train import gate_calibration_report
from slm_training.evals.judge_independence import (
    ExternalJudgeAdapter,
    analyze_triple_judges,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.annotations.judge_audit import import_blinded_labels
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.experiments.sie005_blinded_packet import (
    AUDIT_ID as SIE005_AUDIT_ID,
    REASON_CODES,
    build_external_judge_config,
    build_fixture_audit_rows,
    mock_external_transport,
    prepare_blinded_packet,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

CATALOGUE_ID = "exp-sr-3"
PRIMARY_METRIC = "evaluator_human_agreement_kappa"
SIE005_FROZEN_DOC = Path(
    "docs/design/iter-slm478-sie-005-blinded-packet-20260810.json"
)
SIE005_FROZEN_SEED = 0

ARM_IDS: tuple[str, ...] = (
    "evaluator_only",
    "external_mock_plumbing",
    "human_fixture_plumbing",
    "full_construct_validity",
)

BLOCKED_CLAIMS: tuple[str, ...] = (
    "evaluator_human_agreement_kappa",
    "external_judge_calibration",
    "blinded_human_adjudication",
    "construct_validity_promotion",
)

VERSION_STAMP_COMPONENTS: tuple[str, ...] = (
    "harness.experiments",
    "harness.experiments.slm483_sie006_construct_validity",
)

RunMode = Literal["plan-only", "fixture", "external-blocked"]

__all__ = [
    "ARM_IDS",
    "BLOCKED_CLAIMS",
    "CATALOGUE_ID",
    "PRIMARY_METRIC",
    "SIE005_FROZEN_DOC",
    "SIE005_FROZEN_SEED",
    "Sie006CampaignV1",
    "load_frozen_sie005_pin",
    "materialize_sie005_packet",
    "plan_only_preview",
    "render_markdown",
    "run_external_blocked_campaign",
    "run_fixture_campaign",
]


def load_frozen_sie005_pin(path: Path | None = None) -> dict[str, Any]:
    doc = path or SIE005_FROZEN_DOC
    if not doc.is_file():
        raise FileNotFoundError(f"missing frozen SIE-005 packet doc: {doc}")
    payload = json.loads(doc.read_text(encoding="utf-8"))
    required = (
        "annotation_order_sha256",
        "audit_id",
        "pair_n",
        "rubric_sha256",
        "training_excluded",
        "blinding_ok",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"frozen SIE-005 doc missing keys: {missing}")
    if payload.get("training_excluded") is not True:
        raise ValueError("frozen SIE-005 doc must pin training_excluded=True")
    if payload.get("blinding_ok") is not True:
        raise ValueError("frozen SIE-005 doc must pin blinding_ok=True")
    return payload


def materialize_sie005_packet(
    *,
    root: Path,
    seed: int = SIE005_FROZEN_SEED,
    pin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Regenerate the SIE-005 packet deterministically and verify the frozen pin."""
    frozen = pin or load_frozen_sie005_pin()
    packet_root = root / "sie005_packet"
    report = prepare_blinded_packet(output_dir=packet_root, seed=seed)
    if report.annotation_order_sha256 != frozen["annotation_order_sha256"]:
        raise ValueError(
            "annotation_order_sha256 mismatch vs frozen SIE-005 doc "
            f"(got {report.annotation_order_sha256}, "
            f"expected {frozen['annotation_order_sha256']})"
        )
    if report.rubric_sha256 != frozen["rubric_sha256"]:
        raise ValueError(
            "rubric_sha256 mismatch vs frozen SIE-005 doc "
            f"(got {report.rubric_sha256}, expected {frozen['rubric_sha256']})"
        )
    if report.pair_n != int(frozen["pair_n"]):
        raise ValueError(
            f"pair_n mismatch vs frozen SIE-005 doc ({report.pair_n} != {frozen['pair_n']})"
        )
    public_dir = Path(report.public_dir)
    public_pairs = [
        json.loads(line)
        for line in (public_dir / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    private_key = [
        json.loads(line)
        for line in Path(report.private_key_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    manifest = json.loads((public_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "report": report.to_dict(),
        "public_dir": public_dir,
        "private_key_path": Path(report.private_key_path),
        "public_pairs": public_pairs,
        "private_key": private_key,
        "manifest": manifest,
        "frozen_pin": frozen,
        "seed": seed,
    }


def _deterministic_rows(
    public_pairs: list[dict[str, Any]],
    private_key: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    fixture_rows = build_fixture_audit_rows(seed=seed)
    fixture_by_pair = {str(row["pair_id"]): row for row in fixture_rows}
    private_by_pair = {str(row["pair_id"]): row for row in private_key}
    rows: list[dict[str, Any]] = []
    for blind in public_pairs:
        pair_id = str(blind["pair_id"])
        fixture = fixture_by_pair[pair_id]
        private = private_by_pair[pair_id]
        target = str(fixture["_private_deterministic_verdict"])
        a_ckpt = str(fixture["candidate_a"]["checkpoint_sha256"])
        b_ckpt = str(fixture["candidate_b"]["checkpoint_sha256"])
        if private["left_checkpoint_sha256"] == a_ckpt:
            blind_correct = "left" if target == "left" else "right"
        elif private["left_checkpoint_sha256"] == b_ckpt:
            blind_correct = "right" if target == "left" else "left"
        else:
            raise ValueError(f"{pair_id}: private key does not match fixture candidates")
        rows.append(
            {
                "pair_id": pair_id,
                "stratum": str(blind.get("stratum") or fixture.get("stratum") or "unknown"),
                "prompt": str(blind["prompt"]),
                "deterministic_verdict": blind_correct,
                "deterministic_score": 1.0,
                "deterministic_pass": True,
            }
        )
    return rows


def _external_mock_rows(
    deterministic_rows: list[dict[str, Any]],
    public_pairs: list[dict[str, Any]],
    private_key: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    blind_by_pair = {str(row["pair_id"]): row for row in public_pairs}
    private_by_pair = {str(row["pair_id"]): row for row in private_key}
    config = build_external_judge_config()
    rng = random.Random(f"sie006:external:{seed}")
    transport = mock_external_transport(mode="ok")

    def _noisy_transport(request: dict[str, Any]) -> dict[str, Any]:
        base = transport(request)
        pair = request["pair"]
        pair_id = next(
            pid
            for pid, row in blind_by_pair.items()
            if row["left_openui"] == pair["left_openui"]
            and row["right_openui"] == pair["right_openui"]
        )
        det = next(row for row in deterministic_rows if row["pair_id"] == pair_id)
        if rng.random() < 0.75:
            verdict = det["deterministic_verdict"]
        else:
            verdict = "right" if det["deterministic_verdict"] == "left" else "left"
        return {
            **base,
            "verdict": verdict,
            "left_acceptable": verdict == "left",
            "right_acceptable": verdict == "right",
            "confidence": round(0.55 + 0.35 * rng.random(), 3),
        }

    adapter = ExternalJudgeAdapter(config, _noisy_transport)
    rows: list[dict[str, Any]] = []
    for det in deterministic_rows:
        pair_id = det["pair_id"]
        blind = blind_by_pair[pair_id]
        private = private_by_pair[pair_id]
        evidence = adapter.judge(
            audit_id=SIE005_AUDIT_ID,
            record_id=pair_id,
            generation_id=f"{pair_id}:gen",
            pair_id=pair_id,
            prompt=str(blind["prompt"]),
            left_openui=str(blind["left_openui"]),
            right_openui=str(blind["right_openui"]),
            left_checkpoint_sha256=str(private["left_checkpoint_sha256"]),
            right_checkpoint_sha256=str(private["right_checkpoint_sha256"]),
            candidate_model_families=(
                str(private["left_family"]),
                str(private["right_family"]),
            ),
            prior_judge_providers=("slm_training_deterministic_evaluator",),
            prior_judge_model_families=("slm_training_deterministic_evaluator",),
            order_seed=seed,
        )
        rows.append(
            {
                "pair_id": pair_id,
                "external_verdict": evidence.verdict,
                "external_score": evidence.score,
                "external_pass": evidence.verdict == det["deterministic_verdict"],
                "external_confidence": evidence.confidence,
                "external_reason_codes": list(evidence.reason_codes),
                "_external_independent": evidence.independent,
            }
        )
    return rows


def _fixture_human_labels(
    public_pairs: list[dict[str, Any]],
    deterministic_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Synthetic labels for plumbing only — never real human evidence."""
    det_by_pair = {row["pair_id"]: row for row in deterministic_rows}
    rng = random.Random(f"sie006:human_fixture:{seed}")
    labels: list[dict[str, Any]] = []
    for index, blind in enumerate(public_pairs):
        pair_id = str(blind["pair_id"])
        correct = det_by_pair[pair_id]["deterministic_verdict"]
        wrong = "right" if correct == "left" else "left"
        stratum = str(blind.get("stratum") or "unknown")
        force_ambiguous = stratum == "ambiguity_case"
        if force_ambiguous:
            winners = [correct, wrong]
        else:
            winners = [
                correct if rng.random() < 0.85 else wrong,
                correct if rng.random() < 0.85 else wrong,
            ]
        for rater_i, winner in enumerate(winners[:2]):
            labels.append(
                {
                    "schema": "BlindJudgeLabelV1",
                    "audit_id": SIE005_AUDIT_ID,
                    "pair_id": pair_id,
                    "annotator_id": f"ann_fxrt{rater_i:02d}s{seed:04d}{index:02d}",
                    "role": "rater",
                    "winner": winner,
                    "acceptable_left": winner == "left",
                    "acceptable_right": winner == "right",
                    "reasons": [rng.choice(REASON_CODES)],
                    "confidence": round(0.6 + 0.3 * rng.random(), 3),
                    "duration_ms": 1200 + rater_i,
                    "submitted_at": "2026-08-10T00:00:00Z",
                }
            )
        if len(set(winners[:2])) > 1 and not force_ambiguous:
            labels.append(
                {
                    "schema": "BlindJudgeLabelV1",
                    "audit_id": SIE005_AUDIT_ID,
                    "pair_id": pair_id,
                    "annotator_id": f"ann_fxadjs{seed:04d}{index:02d}",
                    "role": "adjudicator",
                    "winner": correct,
                    "acceptable_left": correct == "left",
                    "acceptable_right": correct == "right",
                    "reasons": [stratum],
                    "confidence": 0.95,
                    "duration_ms": 1800,
                    "submitted_at": "2026-08-10T00:05:00Z",
                }
            )
    return labels


def _human_rows_from_aggregate(
    aggregate: dict[str, Any],
    deterministic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    det_by_pair = {row["pair_id"]: row for row in deterministic_rows}
    rows: list[dict[str, Any]] = []
    for agg in aggregate["pairs"]:
        pair_id = str(agg["pair_id"])
        consensus = agg["consensus_winner"]
        ambiguous = not agg["complete"] or consensus is None
        det = det_by_pair[pair_id]
        if ambiguous:
            human_verdict = "error"
            human_pass = False
        else:
            human_verdict = "left" if consensus == det["deterministic_verdict"] else "right"
            human_pass = human_verdict == "left"
        rows.append(
            {
                "pair_id": pair_id,
                "human_verdict": human_verdict,
                "human_score": agg["mean_confidence"] if agg["mean_confidence"] is not None else 0.0,
                "human_pass": human_pass,
                "human_ambiguous": ambiguous,
                "human_reason_codes": list(agg.get("reason_counts") or {})[:1],
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
    merged: list[dict[str, Any]] = []
    for det in deterministic_rows:
        pair_id = det["pair_id"]
        ext = external_by_pair[pair_id]
        hum = human_by_pair[pair_id]
        merged.append(
            {
                **det,
                **ext,
                **hum,
                "suite": "sie006_fixture",
                "reason_codes": list(ext.get("external_reason_codes") or []),
            }
        )
    return merged


def _real_external_credentials_available() -> bool:
    return bool(os.environ.get("EXTERNAL_JUDGE_API_KEY"))


def plan_only_preview() -> dict[str, Any]:
    preview = plan_only_manifest(CATALOGUE_ID)
    pin = load_frozen_sie005_pin()
    return {
        **preview,
        "catalogue_id": CATALOGUE_ID,
        "primary_metric": PRIMARY_METRIC,
        "fixture_arms": list(ARM_IDS),
        "sie005_frozen_doc": str(SIE005_FROZEN_DOC),
        "sie005_frozen_seed": SIE005_FROZEN_SEED,
        "sie005_pin": {
            "annotation_order_sha256": pin["annotation_order_sha256"],
            "rubric_sha256": pin["rubric_sha256"],
            "pair_n": pin["pair_n"],
            "audit_id": pin["audit_id"],
        },
        "claim_class_execution": "fixture",
        "promotion": False,
        "blocked_without_real_evidence": list(BLOCKED_CLAIMS),
        "scope_disclaimer": (
            "SIE-006 executes EXP-SR-3 through ExperimentCampaignV1 using the "
            "frozen SIE-005 packet pin. Fixture mode exercises plumbing with "
            "labeled mock external transport and synthetic human labels. "
            "External-blocked mode reports incomplete evidence — never fabricates "
            "human ratings as real."
        ),
    }


@dataclass(frozen=True)
class Sie006CampaignV1:
    campaign_id: str = "sie006-exp-sr-3-construct-validity"
    seed: int = SIE005_FROZEN_SEED
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)
    sie005_frozen_doc: str = str(SIE005_FROZEN_DOC)

    def __post_init__(self) -> None:
        if self.claim_class not in {"fixture", "diagnostic"}:
            raise ValueError("SIE-006 only exposes fixture/diagnostic evidence classes")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

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
                role="candidate" if arm == "full_construct_validity" else "control",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Report construct-validity and calibration diagnostics honestly; "
                "calibrate or replace evaluator components only when real blinded "
                "human evidence clears the preregistered floor. Fixture plumbing "
                "never substitutes for external-blocked incomplete evidence."
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
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Audit sample from SIE-005 never enters training.",
                "Ambiguous pairs are retained, never defaulted to a winner.",
                "Missing real external-judge or human evidence reports blocked.",
                "Fixture synthetic labels are labeled fixture_plumbing only.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="sie005_blinded_packet",
                    description=(
                        "Same stratified blinded contrast pairs from the frozen "
                        "SIE-005 packet; human raters never see evaluator verdicts "
                        "or source lineage before adjudicating."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="training_holdout",
                    description=(
                        "SIE-005 manifest pins training_use=False and "
                        "audit_holdout=True for every pair."
                    ),
                    kind="negative",
                ),
                CampaignControlV1(
                    control_id="external_blocked_honesty",
                    description=(
                        "When EXTERNAL_JUDGE_API_KEY and real human labels are "
                        "both absent, primary construct-validity claims remain "
                        "external_blocked — never scored as if evidence existed."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("training_holdout", "external_blocked_honesty"),
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
                ArtifactRequirementV1(kind="agentevals"),
                ArtifactRequirementV1(kind="agentv"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_fixture_campaign(campaign: Sie006CampaignV1, *, root: Path) -> dict[str, Any]:
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="SIE-006 / EXP-SR-3 construct-validity fixture plumbing",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()
    pin = load_frozen_sie005_pin(Path(campaign.sie005_frozen_doc))
    packet = materialize_sie005_packet(root=root, seed=campaign.seed, pin=pin)

    deterministic_rows = _deterministic_rows(
        packet["public_pairs"], packet["private_key"], seed=campaign.seed
    )
    external_rows = _external_mock_rows(
        deterministic_rows,
        packet["public_pairs"],
        packet["private_key"],
        seed=campaign.seed,
    )
    labels = _fixture_human_labels(
        packet["public_pairs"], deterministic_rows, seed=campaign.seed
    )
    label_path = root / "fixture_labels" / "raters.jsonl"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")
    aggregate = import_blinded_labels(
        [label_path],
        audit_id=SIE005_AUDIT_ID,
        pair_ids={row["pair_id"] for row in packet["public_pairs"]},
        output_path=root / "fixture_labels" / "human_aggregate.json",
    )
    human_rows = _human_rows_from_aggregate(aggregate, deterministic_rows)
    full_rows = _merge_rows(deterministic_rows, external_rows, human_rows)

    analysis = analyze_triple_judges(full_rows)
    unambiguous = [row for row in full_rows if not row["human_ambiguous"]]
    risk_coverage = (
        gate_calibration_report(
            [float(row["external_confidence"]) for row in unambiguous],
            [
                1.0 if row["external_verdict"] == row["human_verdict"] else 0.0
                for row in unambiguous
            ],
            max_selective_risk=0.2,
        )
        if unambiguous
        else {"disposition": "blocked", "reason": "no_unambiguous_pairs"}
    )
    kappa = analysis["full"]["pairwise"]["external_vs_human"]["cohen_kappa"]
    calibration_error = analysis["full"]["external_vs_human"]["calibration_error"]
    ambiguous_pair_ids = [row["pair_id"] for row in full_rows if row["human_ambiguous"]]
    min_effect = campaign.minimum_effect()
    clears_minimum = kappa is not None and kappa >= min_effect - 1e-12
    falsifier_holds = kappa is None or kappa <= 0.0 + 1e-12

    arms = [
        {
            "arm": "evaluator_only",
            "disposition": "match",
            "pair_n": len(deterministic_rows),
            "evidence_class": "deterministic_fixture",
        },
        {
            "arm": "external_mock_plumbing",
            "disposition": "match",
            "pair_n": len(external_rows),
            "evidence_class": "mock_external_judge",
            "independence_ok": all(row["_external_independent"] for row in external_rows),
        },
        {
            "arm": "human_fixture_plumbing",
            "disposition": "match",
            "pair_n": len(human_rows),
            "evidence_class": "fixture_plumbing",
            "complete_pair_n": int(aggregate["complete_pair_n"]),
            "ambiguous_pair_ids": ambiguous_pair_ids,
        },
        {
            "arm": "full_construct_validity",
            "disposition": "match",
            "pair_n": len(full_rows),
            "evidence_class": "fixture_plumbing",
            "evaluator_human_agreement_kappa": kappa,
            "external_human_calibration_error": calibration_error,
            "triple_judge_analysis": analysis,
            "risk_coverage": risk_coverage,
            "ambiguous_pair_ids": ambiguous_pair_ids,
        },
    ]

    result = {
        "schema": "sie006_construct_validity_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "status": "fixture",
        "disposition": "complete",
        "claim_class": campaign.claim_class,
        "catalogue_claim_class": catalogue.claim_class,
        "promotion": False,
        "external_blocked": False,
        "evidence_class": "fixture_plumbing",
        "sie005_frozen_doc": campaign.sie005_frozen_doc,
        "sie005_pin": {
            "annotation_order_sha256": pin["annotation_order_sha256"],
            "rubric_sha256": pin["rubric_sha256"],
            "pair_n": pin["pair_n"],
        },
        "training_excluded": packet["manifest"].get("training_use") is False,
        "audit_holdout": packet["manifest"].get("audit_holdout") is True,
        "pair_n": len(full_rows),
        "evaluator_human_agreement_kappa": kappa,
        "external_human_calibration_error": calibration_error,
        "minimum_effect": min_effect,
        "clears_minimum_effect": clears_minimum,
        "falsifier": {
            "falsifier_holds": falsifier_holds,
            "note": (
                "Catalogue falsifier: Cohen's kappa at or below chance."
                if falsifier_holds
                else "Fixture plumbing cleared chance on this tiny sample; "
                "still claim_class=fixture — not real human evidence."
            ),
        },
        "blocked_claims": [],
        "qualified_claims": {
            "evaluator_human_agreement_kappa": "fixture_plumbing_only",
            "external_judge_calibration": "mock_transport_only",
            "blinded_human_adjudication": "synthetic_labels_only",
            "construct_validity_promotion": "blocked_fixture",
        },
        "ambiguous_pair_ids": ambiguous_pair_ids,
        "arms": arms,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-3 construct-validity campaign over the frozen "
            "SIE-005 blinded packet. External judge uses labeled mock transport; "
            "human labels are synthetic fixture_plumbing only. Audit sample is "
            "training-excluded. Ambiguous pairs are retained. No promotion."
        ),
        "version_stamp": build_version_stamp(*VERSION_STAMP_COMPONENTS),
    }
    artifact = store.write_artifact("sie006_construct_validity_result", result)
    store.append_event(
        "sie006_construct_validity_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "evaluator_human_agreement_kappa": kappa,
            "external_blocked": False,
        },
    )
    return result


def run_external_blocked_campaign(campaign: Sie006CampaignV1, *, root: Path) -> dict[str, Any]:
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="SIE-006 / EXP-SR-3 construct-validity external-blocked",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()
    pin = load_frozen_sie005_pin(Path(campaign.sie005_frozen_doc))
    packet = materialize_sie005_packet(root=root, seed=campaign.seed, pin=pin)
    real_external = _real_external_credentials_available()
    real_human = False  # no in-repo human label corpus for this audit

    result = {
        "schema": "sie006_construct_validity_external_blocked_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "status": "external_blocked",
        "disposition": "external_blocked",
        "claim_class": campaign.claim_class,
        "catalogue_claim_class": catalogue.claim_class,
        "promotion": False,
        "external_blocked": True,
        "incomplete": True,
        "evidence_available": {
            "sie005_packet_pin": True,
            "real_external_judge": real_external,
            "real_human_labels": real_human,
        },
        "blocked_claims": list(BLOCKED_CLAIMS),
        "qualified_claims": {
            claim: "external_blocked"
            for claim in BLOCKED_CLAIMS
        },
        "evaluator_human_agreement_kappa": None,
        "external_human_calibration_error": None,
        "pair_n": int(pin["pair_n"]),
        "training_excluded": packet["manifest"].get("training_use") is False,
        "audit_holdout": packet["manifest"].get("audit_holdout") is True,
        "sie005_frozen_doc": campaign.sie005_frozen_doc,
        "sie005_pin": {
            "annotation_order_sha256": pin["annotation_order_sha256"],
            "rubric_sha256": pin["rubric_sha256"],
            "pair_n": pin["pair_n"],
        },
        "reason": (
            "Real blinded human labels are unavailable in-repo and "
            f"EXTERNAL_JUDGE_API_KEY is {'set' if real_external else 'unset'}; "
            "construct-validity endpoints remain external_blocked."
        ),
        "scope_disclaimer": (
            "External-blocked EXP-SR-3 run: verifies the frozen SIE-005 packet "
            "pin and locks the campaign manifest, but does not fabricate human "
            "ratings or score primary construct-validity claims. Audit sample "
            "remains training-excluded. Fixture plumbing (mock transport + "
            "synthetic labels) is validated separately via "
            "`python -m scripts.run_sie006_construct_validity --mode fixture`."
        ),
        "version_stamp": build_version_stamp(*VERSION_STAMP_COMPONENTS),
    }
    artifact = store.write_artifact("sie006_construct_validity_result", result)
    store.append_event(
        "sie006_construct_validity_completed",
        experiment_id=campaign.campaign_id,
        status="external_blocked",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "external_blocked": True,
        },
    )
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    lines = [
        "# SLM-483 (SIE-006): Construct-validity + human calibration (EXP-SR-3)",
        "",
        f"**Claim class:** `{payload.get('claim_class')}` execution on catalogue "
        f"`{payload.get('catalogue_claim_class')}`",
        "",
        f"**Status:** `{status}` / `{payload.get('disposition')}`",
        "",
        f"**Catalogue:** `{payload.get('catalogue_id', CATALOGUE_ID)}`",
        "",
        f"**Frozen SIE-005 pin:** `{payload.get('sie005_frozen_doc', SIE005_FROZEN_DOC)}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| pair_n | {payload.get('pair_n')} |",
        f"| training_excluded | {payload.get('training_excluded')} |",
        f"| audit_holdout | {payload.get('audit_holdout')} |",
        f"| external_blocked | {payload.get('external_blocked')} |",
        f"| evaluator_human_agreement_kappa | {payload.get('evaluator_human_agreement_kappa')} |",
        f"| promotion | {payload.get('promotion')} |",
        "",
    ]
    blocked = payload.get("blocked_claims") or []
    if blocked:
        lines.extend(["## Blocked claims", ""])
        for claim in blocked:
            lines.append(f"- `{claim}`")
        lines.append("")
    qualified = payload.get("qualified_claims") or {}
    if qualified:
        lines.extend(["## Qualified claims", ""])
        for claim, qual in qualified.items():
            lines.append(f"- `{claim}`: {qual}")
        lines.append("")
    if payload.get("ambiguous_pair_ids"):
        lines.extend(
            [
                "## Ambiguous pairs retained",
                "",
                ", ".join(f"`{pid}`" for pid in payload["ambiguous_pair_ids"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Scope",
            "",
            str(payload.get("scope_disclaimer", "")),
            "",
        ]
    )
    return "\n".join(lines)
