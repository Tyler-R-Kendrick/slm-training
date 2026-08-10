"""RSP-008 (SLM-487): EXP-SR-12 second-pack portability certification.

Governed diagnostic campaign that scores ``second_pack_portability_parity_rate``
by exercising the same SRP-010 seams against OpenUI (control) and
symbolic-regression (candidate). Reuses
:mod:`srp010_pack_portability` for the SR arm; does not re-derive that
inventory.

Locks ``ExperimentCampaignV1`` via ``CampaignStore`` at ``claim_class=diagnostic``
only (catalogue identity). Never ``promotion_candidate`` / ``ship_gate``.
No learned legality; cross-pack mismatch fails closed; OpenUI defaults stay.
"""

from __future__ import annotations

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
from slm_training.dsl.pack import PackSlotUnavailable, get_pack, list_packs
from slm_training.dsl.symbolic_regression_pack import SYMBOLIC_REGRESSION_PACK_ID
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.experiments.slm159_cross_dsl_replication import (
    assess_pack_readiness,
)
from slm_training.harnesses.experiments.srp010_pack_portability import (
    RELATED_CATALOGUE_ID,
    assess_symbolic_pack_portability,
    build_tiny_sr_problem,
    require_pack_id,
    require_pack_slot,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-12"
PRIMARY_METRIC = "second_pack_portability_parity_rate"
OPENUI_PACK_ID = "openui"
SHARED_SLOTS: tuple[str, ...] = (
    "backend",
    "placeholder_policy",
    "canonicalize",
    "oracle",
)
ARM_IDS: tuple[str, ...] = ("openui_control", "symbolic_regression_candidate")

# Ordered seam checks that compose the primary metric. Every check is binary;
# ``zero_required_shared_seam_forks`` encodes the catalogue falsifier so any
# inventored shared-seam fork drops the rate below the kill threshold (1.0).
SEAM_CHECK_IDS: tuple[str, ...] = (
    "default_isolation_openui",
    "both_packs_registered",
    "openui_shared_slots",
    "sr_shared_slots",
    "openui_readiness_core",
    "sr_e2e_parse_canonicalize_oracle_evidence",
    "sr_vsp_and_sygus",
    "sr_import_audit_clean",
    "cross_pack_mismatch_fails_closed",
    "no_learned_legality",
    "zero_required_shared_seam_forks",
)

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "OPENUI_PACK_ID",
    "PRIMARY_METRIC",
    "SEAM_CHECK_IDS",
    "Rsp008CampaignV1",
    "exercise_openui_control_arm",
    "plan_only_preview",
    "score_second_pack_portability",
    "run_campaign",
]


def _slot_filled(pack_id: str, slot: str) -> bool:
    return getattr(get_pack(pack_id), slot, None) is not None


def exercise_openui_control_arm() -> dict[str, Any]:
    """Matched-control OpenUI seam: registration, shared slots, readiness.

    Does not invoke the OpenUI Node bridge (env-fragile); slot presence +
    :func:`assess_pack_readiness` are the durable control signals.
    """
    available = OPENUI_PACK_ID in set(list_packs())
    if not available:
        return {
            "arm_id": "openui_control",
            "pack_id": OPENUI_PACK_ID,
            "pack_available": False,
            "shared_slots_filled": False,
            "filled_shared_slots": [],
            "readiness_pass": False,
            "parser_available": False,
            "oracle_available": False,
            "canonicalizer_available": False,
        }
    pack = get_pack(OPENUI_PACK_ID)
    filled = [s for s in SHARED_SLOTS if getattr(pack, s, None) is not None]
    readiness = assess_pack_readiness(OPENUI_PACK_ID)
    return {
        "arm_id": "openui_control",
        "pack_id": pack.pack_id,
        "pack_available": True,
        "shared_slots_filled": filled == list(SHARED_SLOTS),
        "filled_shared_slots": filled,
        "readiness_pass": bool(readiness.readiness_pass),
        "parser_available": bool(readiness.parser_available),
        "oracle_available": bool(readiness.oracle_available),
        "canonicalizer_available": bool(readiness.canonicalizer_available),
        "unsupported_features": list(readiness.unsupported_features),
    }


def _mismatch_fails_closed() -> bool:
    problem = build_tiny_sr_problem()
    try:
        require_pack_id(problem, OPENUI_PACK_ID)
    except ValueError:
        return True
    return False


def _no_learned_legality(sr_report: Any) -> bool:
    """Legality is pack-oracle / grammar only — no model/learned fields."""
    evidence = dict(sr_report.evidence or {})
    banned = ("learned_legality", "neural_mask", "unconstrained_fallback", "logit_mask")
    if any(k in evidence for k in banned):
        return False
    # Fail closed if a required legality slot is missing (never invent legality).
    pack = get_pack(SYMBOLIC_REGRESSION_PACK_ID)
    try:
        require_pack_slot(pack, "oracle")
        require_pack_slot(pack, "canonicalize")
    except PackSlotUnavailable:
        return False
    return sr_report.oracle_ok is True


def score_second_pack_portability(
    *,
    openui_arm: dict[str, Any],
    sr_report: Any,
) -> dict[str, Any]:
    """Binary seam checks → ``second_pack_portability_parity_rate``."""
    forks = list(sr_report.forks or ())
    fork_ids = [f.fork_id for f in forks]
    import_clean = all(not f.offenders for f in (sr_report.import_audit or ()))
    sygus_ok = (
        isinstance(sr_report.sygus_capability, dict)
        and bool(sr_report.sygus_capability.get("conformance"))
        and bool(sr_report.vsp_digest)
    )
    e2e_ok = (
        bool(sr_report.pack_available)
        and bool(sr_report.oracle_ok)
        and bool(sr_report.canonical_expression)
        and bool(sr_report.problem_fingerprint)
        and bool(sr_report.evidence)
        and sr_report.status in {"fixture", "diagnostic", "certified", "fixture_failed"}
    )
    # fixture_failed from SRP-010 (default isolation / import leak) still
    # contributes evidence; those failures surface in their own checks.
    if sr_report.status == "fixture_failed":
        e2e_ok = (
            bool(sr_report.pack_available)
            and bool(sr_report.oracle_ok)
            and bool(sr_report.canonical_expression)
            and bool(sr_report.evidence)
        )

    checks: dict[str, bool] = {
        "default_isolation_openui": bool(
            sr_report.default_isolation.openui_is_default
            and sr_report.default_isolation.env_unset
        ),
        "both_packs_registered": bool(
            openui_arm.get("pack_available") and sr_report.pack_available
        ),
        "openui_shared_slots": bool(openui_arm.get("shared_slots_filled")),
        "sr_shared_slots": all(
            _slot_filled(SYMBOLIC_REGRESSION_PACK_ID, s) for s in SHARED_SLOTS
        ),
        "openui_readiness_core": bool(
            openui_arm.get("parser_available")
            and openui_arm.get("oracle_available")
            and openui_arm.get("canonicalizer_available")
        ),
        "sr_e2e_parse_canonicalize_oracle_evidence": e2e_ok,
        "sr_vsp_and_sygus": sygus_ok,
        "sr_import_audit_clean": import_clean,
        "cross_pack_mismatch_fails_closed": _mismatch_fails_closed(),
        "no_learned_legality": _no_learned_legality(sr_report),
        "zero_required_shared_seam_forks": len(forks) == 0,
    }
    if set(checks) != set(SEAM_CHECK_IDS):
        raise RuntimeError(
            f"seam check id drift: {sorted(checks)} vs {list(SEAM_CHECK_IDS)}"
        )

    passed = sum(1 for ok in checks.values() if ok)
    total = len(SEAM_CHECK_IDS)
    rate = (passed / total) if total else 0.0
    falsifier_holds = not checks["zero_required_shared_seam_forks"]
    return {
        "second_pack_portability_parity_rate": rate,
        "checks_passed": passed,
        "checks_total": total,
        "checks": checks,
        "fork_ids": fork_ids,
        "unsupported_hooks": [h.to_dict() for h in (sr_report.unsupported_hooks or ())],
        "falsifier_holds": falsifier_holds,
        "falsifier_note": (
            "Pack-specific forks are required in the shared seam "
            f"(fork_ids={fork_ids}); catalogue falsifier holds — not certified."
            if falsifier_holds
            else "No inventored shared-seam forks; parity subject to remaining checks."
        ),
        "certified": rate >= 1.0 and not falsifier_holds,
    }


def _repo_relative(path: str | Path, *, repo_root: Path | None = None) -> str:
    """Strip worktree absolute prefixes so durable docs stay portable."""
    text = str(path)
    root = repo_root or Path(__file__).resolve().parents[4]
    try:
        return str(Path(text).resolve().relative_to(root.resolve()))
    except ValueError:
        marker = "/src/slm_training/"
        if marker in text:
            return "src/slm_training/" + text.split(marker, 1)[1]
        return text


def plan_only_preview() -> dict[str, object]:
    """Catalogue plan-only preview plus the matched arm surface RSP-008 runs."""
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "seam_checks": list(SEAM_CHECK_IDS),
        "primary_metric": PRIMARY_METRIC,
        "claim_class_execution": "diagnostic",
        "promotion": False,
        "srp010_precursor": RELATED_CATALOGUE_ID,
    }


@dataclass(frozen=True)
class Rsp008CampaignV1:
    campaign_id: str = "rsp008-exp-sr-12-second-pack-portability"
    seeds: tuple[int, ...] = (0, 1, 2)
    source_commit: str = "0" * 40
    claim_class: str = "diagnostic"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.claim_class != "diagnostic":
            raise ValueError(
                "RSP-008 durable evidence is diagnostic-only "
                "(catalogue exp-sr-12; never promotion_candidate/ship_gate)"
            )
        if self.claim_class in {"promotion_candidate", "ship_gate"}:
            raise ValueError("claim_class must not be promotion_candidate/ship_gate")
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

    def kill_threshold(self) -> float:
        gate = self.catalogue_manifest().rollback_gates[0]
        return float(gate.threshold)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "openui_control" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Certify second-pack portability only when OpenUI and "
                "symbolic-regression share the SRP-010 seam with zero "
                "pack-specific forks. Diagnostic claim only — never promote."
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
                "Deterministic seam checks; seeds confirm stability only.",
                "Cross-pack pack_id mismatch must raise (fail closed).",
                "No GPU / learned legality; SRP-010 inventory is authoritative for forks.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="openui_default_isolation",
                    description=(
                        "Unresolved / auto / default pack aliases remain OpenUI "
                        "with SLM_GRAMMAR_DSL unset."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="matched_seam_control",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same seam exercised end-to-end against OpenUI and "
                        "symbolic-regression."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else (
                        "Portability fixtures never reuse OpenUI-specific "
                        "grammar/tokenizer identities for the SR pack."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("leakage_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                # Diagnostic evidence is never a promotion claim.
                CampaignGateV1(
                    gate_id="diagnostic_only",
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
                ArtifactRequirementV1(kind="formal_preflight"),
            ),
            claim_class=self.claim_class,  # type: ignore[arg-type]
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Rsp008CampaignV1, *, root: Path) -> dict[str, Any]:
    """Run multi-seed portability certification and persist under CampaignStore."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-008 / EXP-SR-12 second-pack portability certification",
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

    seed_payloads: list[dict[str, Any]] = []
    rates: list[float] = []
    last_score: dict[str, Any] | None = None
    last_openui: dict[str, Any] | None = None
    last_sr: dict[str, Any] | None = None

    for seed in campaign.seeds:
        openui_arm = exercise_openui_control_arm()
        sr_report = assess_symbolic_pack_portability(
            run_id=f"rsp008_seed_{seed}"
        )
        score = score_second_pack_portability(
            openui_arm=openui_arm, sr_report=sr_report
        )
        rates.append(float(score["second_pack_portability_parity_rate"]))
        seed_payloads.append(
            {
                "seed": seed,
                "openui_control": openui_arm,
                "sr_status": sr_report.status,
                "sr_claim_class": sr_report.claim_class,
                "fork_ids": score["fork_ids"],
                "checks": score["checks"],
                "second_pack_portability_parity_rate": score[
                    "second_pack_portability_parity_rate"
                ],
                "falsifier_holds": score["falsifier_holds"],
                "certified": score["certified"],
            }
        )
        last_score = score
        last_openui = openui_arm
        last_sr = {
            "status": sr_report.status,
            "claim_class": sr_report.claim_class,
            "pack_id": sr_report.pack_id,
            "default_isolation": sr_report.default_isolation.to_dict(),
            "forks": [f.to_dict() for f in sr_report.forks],
            "unsupported_hooks": [h.to_dict() for h in sr_report.unsupported_hooks],
            "sygus_conformance": (sr_report.sygus_capability or {}).get("conformance"),
            "import_audit": [
                {
                    **f.to_dict(),
                    "path": _repo_relative(f.path),
                }
                for f in sr_report.import_audit
            ],
            "missing_slots": list(sr_report.missing_slots),
            "filled_slots": list(sr_report.filled_slots),
            "notes": list(sr_report.notes),
        }

    assert last_score is not None and last_openui is not None and last_sr is not None
    # Deterministic seam: every seed must agree on the primary metric.
    rate_stable = all(r == rates[0] for r in rates)
    aggregate_rate = rates[0] if rates else 0.0
    kill_hit = aggregate_rate < campaign.kill_threshold()

    result = {
        "schema": "rsp008_second_pack_portability_diagnostic_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "seeds": list(campaign.seeds),
        "second_pack_portability_parity_rate": aggregate_rate,
        "rate_stable_across_seeds": rate_stable,
        "kill_threshold": campaign.kill_threshold(),
        "kill_operator": "lt",
        "kill_gate_triggered": kill_hit,
        "analysis": last_score,
        "openui_control": last_openui,
        "symbolic_regression_candidate": last_sr,
        "seed_runs": seed_payloads,
        "certified": bool(last_score["certified"] and rate_stable and not kill_hit),
        "scope_disclaimer": (
            "Diagnostic EXP-SR-12 certification over SRP-010 seams on OpenUI "
            "(control) and symbolic_regression (candidate). Reuses "
            "assess_symbolic_pack_portability; inventored forks are reported "
            "honestly and count against zero_required_shared_seam_forks. "
            "claim_class=diagnostic; never promotion_candidate/ship_gate. "
            "No learned legality; OpenUI defaults unchanged."
        ),
    }
    artifact = store.write_artifact("rsp008_second_pack_portability_result", result)
    store.append_event(
        "rsp008_second_pack_portability_completed",
        experiment_id=campaign.campaign_id,
        status="diagnostic",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "second_pack_portability_parity_rate": aggregate_rate,
            "falsifier_holds": last_score["falsifier_holds"],
            "certified": result["certified"],
            "kill_gate_triggered": kill_hit,
        },
    )
    return result
