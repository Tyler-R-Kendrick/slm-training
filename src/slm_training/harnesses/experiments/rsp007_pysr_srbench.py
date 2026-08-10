"""RSP-007 (SLM-486): EXP-SR-11 matched PySR/SRBench benchmark.

Fixture-scale diagnostic campaign comparing the in-repo pack enumerative
baseline (SRP-010 / :mod:`symbolic_expr_enumerate`) against the optional
isolated PySR/SRBench adapter (SRP-011) at a matched compute budget on shared
tiny SRBench-compatible problems.

When the PySR/Julia environment is unavailable, the adapter path runs a
honest dry-run: ``build_adapter_run_input`` plus ``run_adapter`` with an
injected provider that fails closed as ``external_blocked`` — never as a
benchmark loss. Durable manifests record env/version fingerprints either way.

``claim_class=diagnostic`` (catalogue ``exp-sr-11``). Never
``promotion_candidate`` / ``ship_gate``. No SOTA claims.
"""

from __future__ import annotations

import importlib.util
import sys
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
from slm_training.dsl.symbolic_expr_enumerate import run_enumerative_baseline
from slm_training.dsl.symbolic_expr_pysr_adapter import (
    DEFAULT_PYSR_MANIFEST,
    AdapterEnvironmentManifestV1,
    AdapterRunInputV1,
    AdapterRunResultV1,
    build_adapter_run_input,
    run_adapter,
)
from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.experiments.srp010_pack_portability import (
    build_tiny_sr_problem,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-11"
PRIMARY_METRIC = "srbench_matched_score_gap"
EVIDENCE_EXTERNAL_BLOCKED = "external-blocked"
ARM_IDS: tuple[str, ...] = ("pack_enumerate_matched", "pysr_adapter_matched")

# Matched compute: enumerate state cap == adapter iteration budget.
MATCHED_ITERATION_BUDGET = DEFAULT_PYSR_MANIFEST.iteration_budget
ENUM_MAX_DEPTH = 3
ENUM_MAX_NODES = 5
ENUM_MAX_STATES = MATCHED_ITERATION_BUDGET

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "DEFAULT_MANIFEST",
    "ENUM_MAX_DEPTH",
    "ENUM_MAX_NODES",
    "ENUM_MAX_STATES",
    "EVIDENCE_EXTERNAL_BLOCKED",
    "MATCHED_ITERATION_BUDGET",
    "PRIMARY_METRIC",
    "Rsp007CampaignV1",
    "best_validation_loss",
    "compute_matched_score_gap",
    "dry_run_pysr_adapter",
    "plan_only_preview",
    "probe_pysr_environment",
    "run_campaign",
    "run_pack_enumerate_arm",
    "run_pysr_adapter_arm",
]

DEFAULT_MANIFEST = DEFAULT_PYSR_MANIFEST


def _validation_loss_from_evidence(evidence: dict[str, Any]) -> float | None:
    for split in evidence.get("splits", ()):
        if split.get("split") == "validation":
            loss = split.get("loss")
            if isinstance(loss, (int, float)) and loss != float("inf"):
                return float(loss)
            return None
    return None


def best_validation_loss(records: tuple[Any, ...]) -> float | None:
    """Minimum finite validation loss across evidence records, if any."""
    best: float | None = None
    for record in records:
        payload = record.to_dict() if hasattr(record, "to_dict") else record
        loss = _validation_loss_from_evidence(payload)
        if loss is None:
            continue
        best = loss if best is None else min(best, loss)
    return best


def probe_pysr_environment(
    manifest: AdapterEnvironmentManifestV1 = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Record whether a live PySR import is available — never installs anything."""
    spec = importlib.util.find_spec("pysr")
    available = spec is not None
    return {
        "tool_name": manifest.tool_name,
        "tool_version": manifest.tool_version,
        "python_version": sys.version.split()[0],
        "manifest_fingerprint": manifest.fingerprint(),
        "pysr_import_available": available,
        "live_execution_ready": available,
    }


def run_pack_enumerate_arm(
    problem: SymbolicRegressionProblemV1,
    *,
    max_depth: int = ENUM_MAX_DEPTH,
    max_nodes: int = ENUM_MAX_NODES,
    max_states: int = ENUM_MAX_STATES,
) -> dict[str, Any]:
    """Matched-budget in-repo enumerative baseline (SRP-010 enumerator)."""
    result = run_enumerative_baseline(
        problem,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_states=max_states,
    )
    best = best_validation_loss(result.records)
    counters = result.counters.to_dict()
    return {
        "arm_id": "pack_enumerate_matched",
        "status": "complete",
        "evidence": "pack_enumerate",
        "validation_loss": best,
        "optimality_claim": result.optimality_claim,
        "bound_exhausted": bool(counters.get("bound_exhausted")),
        "states_generated": int(counters.get("states_generated", 0)),
        "unique_canonical_states": int(counters.get("unique_canonical_states", 0)),
        "matched_iteration_budget": max_states,
        "max_depth": max_depth,
        "max_nodes": max_nodes,
    }


def dry_run_pysr_adapter(
    problem: SymbolicRegressionProblemV1,
    manifest: AdapterEnvironmentManifestV1 = DEFAULT_MANIFEST,
) -> tuple[AdapterRunInputV1, AdapterRunResultV1]:
    """Translate + invoke adapter with a fail-closed missing-env provider."""
    adapter_input = build_adapter_run_input(problem, manifest)

    def _missing_env_provider(_: AdapterRunInputV1) -> str:
        raise ImportError(
            "PySR/SRBench live environment unavailable; "
            "fixture records external-blocked evidence only"
        )

    result = run_adapter(
        problem,
        manifest,
        raw_output_provider=_missing_env_provider,
    )
    return adapter_input, result


def run_pysr_adapter_arm(
    problem: SymbolicRegressionProblemV1,
    *,
    manifest: AdapterEnvironmentManifestV1 = DEFAULT_MANIFEST,
    env_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """SRP-011 adapter arm: live when PySR importable, else dry-run incomplete."""
    probe = env_probe if env_probe is not None else probe_pysr_environment(manifest)
    adapter_input = build_adapter_run_input(problem, manifest)

    if not probe.get("live_execution_ready"):
        _, blocked = dry_run_pysr_adapter(problem, manifest)
        return {
            "arm_id": "pysr_adapter_matched",
            "status": "incomplete",
            "evidence": EVIDENCE_EXTERNAL_BLOCKED,
            "adapter_status": blocked.status,
            "validation_loss": None,
            "error_detail": blocked.error_detail,
            "manifest_fingerprint": manifest.fingerprint(),
            "adapter_input_fingerprint": content_sha(adapter_input.to_dict()),
            "matched_iteration_budget": manifest.iteration_budget,
            "env_probe": probe,
            "dry_run": True,
        }

    def _live_provider(_: AdapterRunInputV1) -> str:
        raise RuntimeError(
            "PySR live execution is out of fixture scope; "
            "use an isolated runner with raw_output_provider injection"
        )

    result = run_adapter(problem, manifest, raw_output_provider=_live_provider)
    validation_loss = (
        _validation_loss_from_evidence(result.evidence)
        if result.evidence is not None
        else None
    )
    complete = result.status == "ok" and validation_loss is not None
    return {
        "arm_id": "pysr_adapter_matched",
        "status": "complete" if complete else "incomplete",
        "evidence": "pysr_adapter" if complete else EVIDENCE_EXTERNAL_BLOCKED,
        "adapter_status": result.status,
        "validation_loss": validation_loss,
        "error_detail": result.error_detail,
        "manifest_fingerprint": manifest.fingerprint(),
        "adapter_input_fingerprint": content_sha(adapter_input.to_dict()),
        "matched_iteration_budget": manifest.iteration_budget,
        "env_probe": probe,
        "dry_run": False,
    }


def compute_matched_score_gap(
    pack_arm: dict[str, Any],
    pysr_arm: dict[str, Any],
) -> dict[str, Any]:
    """Primary metric with honest incomplete handling — never fabricates a loss."""
    pack_loss = pack_arm.get("validation_loss")
    pysr_loss = pysr_arm.get("validation_loss")
    pysr_evidence = pysr_arm.get("evidence")
    pysr_status = pysr_arm.get("status")

    if pysr_evidence == EVIDENCE_EXTERNAL_BLOCKED or pysr_status != "complete":
        return {
            PRIMARY_METRIC: None,
            "evidence": EVIDENCE_EXTERNAL_BLOCKED,
            "complete": False,
            "pack_validation_loss": pack_loss,
            "pysr_validation_loss": None,
            "note": (
                "External PySR/SRBench arm incomplete — gap not scored as a "
                "benchmark loss."
            ),
        }
    if pack_loss is None or pysr_loss is None:
        return {
            PRIMARY_METRIC: None,
            "evidence": "incomplete",
            "complete": False,
            "pack_validation_loss": pack_loss,
            "pysr_validation_loss": pysr_loss,
            "note": "Missing validation loss on one or both arms.",
        }
    gap = round(float(pack_loss) - float(pysr_loss), 6)
    return {
        PRIMARY_METRIC: gap,
        "evidence": "matched_complete",
        "complete": True,
        "pack_validation_loss": pack_loss,
        "pysr_validation_loss": pysr_loss,
        "note": "Positive gap means pack validation loss exceeds PySR.",
    }


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "matched_iteration_budget": MATCHED_ITERATION_BUDGET,
        "enumerate_budget": {
            "max_depth": ENUM_MAX_DEPTH,
            "max_nodes": ENUM_MAX_NODES,
            "max_states": ENUM_MAX_STATES,
        },
        "pysr_manifest_fingerprint": DEFAULT_MANIFEST.fingerprint(),
        "claim_class_execution": "diagnostic",
        "promotion": False,
        "evidence_when_blocked": EVIDENCE_EXTERNAL_BLOCKED,
        "no_sota_claims": True,
    }


@dataclass(frozen=True)
class Rsp007CampaignV1:
    campaign_id: str = "rsp007-exp-sr-11-pysr-srbench"
    seed: int = 0
    source_commit: str = "0" * 40
    claim_class: str = "diagnostic"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.claim_class != "diagnostic":
            raise ValueError(
                "RSP-007 durable evidence is diagnostic-only "
                "(catalogue exp-sr-11; never promotion_candidate/ship_gate)"
            )
        if self.claim_class in {"promotion_candidate", "ship_gate"}:
            raise ValueError("claim_class must not be promotion_candidate/ship_gate")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def kill_threshold(self) -> float:
        return float(self.catalogue_manifest().rollback_gates[0].threshold)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "pack_enumerate_matched" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Report matched-budget parity honestly; external PySR failures "
                "are incomplete evidence, never benchmark losses. Diagnostic "
                "only — never promote or claim SOTA."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="decrease",
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
                "Reject timed-out or incomplete external arms without scoring a loss.",
                "Matched iteration budget shared between enumerate and adapter manifest.",
                "PySR adapter runs in isolated manifest (SRP-011); no training-data leakage.",
            ),
            controls=(
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_matched_control",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same SRBench-compatible problems at equal compute budget."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id=f"{CATALOGUE_ID}_leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else (
                        "PySR/SRBench adapter in isolated environment manifest."
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
                    operator="gt",
                    threshold=self.kill_threshold(),
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
            ),
            claim_class=self.claim_class,  # type: ignore[arg-type]
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(
    campaign: Rsp007CampaignV1,
    *,
    root: Path,
    problem: SymbolicRegressionProblemV1 | None = None,
) -> dict[str, Any]:
    """Lock ``exp-sr-11``, run matched arms, persist under CampaignStore."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-007 / EXP-SR-11 matched PySR/SRBench benchmark",
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

    fixture_problem = problem if problem is not None else build_tiny_sr_problem()
    env_probe = probe_pysr_environment()
    pack_arm = run_pack_enumerate_arm(fixture_problem)
    pysr_arm = run_pysr_adapter_arm(fixture_problem, env_probe=env_probe)
    gap_payload = compute_matched_score_gap(pack_arm, pysr_arm)

    gap_value = gap_payload[PRIMARY_METRIC]
    kill_hit = (
        gap_payload["complete"]
        and gap_value is not None
        and float(gap_value) > campaign.kill_threshold()
    )

    result = {
        "schema": "rsp007_pysr_srbench_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "no_sota_claims": True,
        "seed": campaign.seed,
        "problem_fingerprint": fixture_problem.fingerprint,
        "matched_iteration_budget": MATCHED_ITERATION_BUDGET,
        "env_probe": env_probe,
        "pack_enumerate_matched": pack_arm,
        "pysr_adapter_matched": pysr_arm,
        "gap_analysis": gap_payload,
        PRIMARY_METRIC: gap_value,
        "evidence": gap_payload["evidence"],
        "complete": gap_payload["complete"],
        "kill_threshold": campaign.kill_threshold(),
        "kill_operator": "gt",
        "kill_gate_triggered": kill_hit,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-11 matched-budget wiring over SRP-010 "
            "enumerator and SRP-011 PySR adapter on a tiny SRBench-compatible "
            "problem. External PySR/Julia absence yields "
            f"evidence={EVIDENCE_EXTERNAL_BLOCKED!r} incomplete results — "
            "never benchmark losses or SOTA claims. claim_class=diagnostic; "
            "never promotion_candidate/ship_gate."
        ),
    }
    artifact = store.write_artifact("rsp007_pysr_srbench_result", result)
    store.append_event(
        "rsp007_pysr_srbench_completed",
        experiment_id=campaign.campaign_id,
        status="incomplete" if not gap_payload["complete"] else "diagnostic",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            PRIMARY_METRIC: gap_value,
            "evidence": gap_payload["evidence"],
            "complete": gap_payload["complete"],
            "kill_gate_triggered": kill_hit,
        },
    )
    return result
