"""Anti-E237 advisory semantic-factor frontier campaign with claim verdicts.

Preregistered causal arms across factor representations. Never changes legal
support. Complete singletons perform zero factor/propagation/ranker/neural work.
Emits real ranking metrics (accuracy, MRR, paired deltas, CIs) and validates
or invalidates each registered claim from measurements.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignBudget,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    CampaignLockV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.data.progspec.semantic_evidence import (
    FactorRepresentation,
    build_factor_representation,
    project_program_factors,
    reconstruct_membership,
)
from slm_training.harnesses.experiments.semantic_factor_claims import evaluate_claims
from slm_training.harnesses.experiments.semantic_factor_config import (
    build_suite_from_config,
    load_sff_campaign,
)
from slm_training.harnesses.experiments.semantic_factor_metric_engine import (
    apply_control_relative_metrics,
    compute_arm_metrics,
)
from slm_training.harnesses.experiments.semantic_factor_metrics import (
    campaign_declares_runtime_endpoints,
    validate_scoreboard_metrics,
)
from slm_training.models.semantic_factor_propagation import (
    column_stochastic_S,
    propagate_restart,
)
from slm_training.models.semantic_residual_scorer import (
    SemanticResidualScorerConfig,
    SemanticResidualScorerV1,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "build_campaign",
    "build_ranked_suite",
    "lock_campaign",
    "run_semantic_factor_frontier",
    "validate_scoreboard_metrics",
]


def build_ranked_suite() -> tuple[dict[str, Any], ...]:
    """Materialize suite from external suite.v1.json (generator + static examples)."""

    return build_suite_from_config(load_sff_campaign().suite)



def _arm_digest(arm_id: str, arm_payload: Mapping[str, Any] | None = None) -> str:
    blob = {"arm_id": arm_id, "config": arm_payload or {}}
    return hashlib.sha256(
        json.dumps(blob, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_campaign(*, source_commit: str, source_dirty: bool) -> ExperimentCampaignV1:
    """Build ExperimentCampaignV1 from external campaign + metrics + scorer files."""

    sff = load_sff_campaign()
    camp = sff.payload
    metrics = sff.metrics
    scorer = sff.scorer
    control_id = scorer.control_arm_id
    arms = tuple(
        CampaignArmV1(
            arm_id=arm_id,
            role="control" if arm_id == control_id else "candidate",
            config_sha256=_arm_digest(arm_id, scorer.arm(arm_id)),
        )
        for arm_id in scorer.arm_ids
    )
    endpoints = tuple(
        CampaignEndpointV1(
            endpoint_id=str(ep["endpoint_id"]),
            metric=str(ep["metric"]),
            role=str(ep["role"]),  # type: ignore[arg-type]
            direction=str(ep["direction"]),  # type: ignore[arg-type]
            minimum_effect=float(ep["minimum_effect"]),
        )
        for ep in metrics.endpoints
    )
    budget_raw = dict(camp.get("budget") or {})
    return ExperimentCampaignV1(
        campaign_id=str(camp["campaign_id"]),
        experiment_id=str(camp["experiment_id"]),
        hypothesis=str(camp["hypothesis"]),
        decision=str(camp["decision"]),
        endpoints=endpoints,
        arms=arms,
        seeds=tuple(int(s) for s in camp["seeds"]),
        budget=CampaignBudget(
            max_experiments=int(budget_raw.get("max_experiments", 12)),
            max_wall_minutes=float(budget_raw.get("max_wall_minutes", 3.0)),
        ),
        stopping_rules=tuple(str(x) for x in camp["stopping_rules"]),
        controls=tuple(
            CampaignControlV1(
                control_id=str(c["control_id"]),
                description=str(c["description"]),
                kind=str(c["kind"]),  # type: ignore[arg-type]
            )
            for c in camp["controls"]
        ),
        negative_controls=tuple(str(x) for x in camp["negative_controls"]),
        multiplicity_families=tuple(
            MultiplicityFamilyV1(
                family_id=str(f["family_id"]),
                hypothesis_ids=tuple(str(h) for h in f["hypothesis_ids"]),
                alpha=float(f["alpha"]),
                method=str(f.get("method") or "holm"),  # type: ignore[arg-type]
            )
            for f in camp["multiplicity_families"]
        ),
        promotion_gates=tuple(
            CampaignGateV1(
                gate_id=str(g["gate_id"]),
                endpoint_id=str(g["endpoint_id"]),
                operator=str(g["operator"]),  # type: ignore[arg-type]
                threshold=float(g["threshold"]),
            )
            for g in camp["promotion_gates"]
        ),
        rollback_gates=tuple(
            CampaignGateV1(
                gate_id=str(g["gate_id"]),
                endpoint_id=str(g["endpoint_id"]),
                operator=str(g["operator"]),  # type: ignore[arg-type]
                threshold=float(g["threshold"]),
            )
            for g in camp["rollback_gates"]
        ),
        artifact_requirements=(
            ArtifactRequirementV1(kind="version_stamp"),
            ArtifactRequirementV1(kind="seed_result"),
            ArtifactRequirementV1(kind="endpoint_result"),
        ),
        formal_obligations=(),
        claim_class=str(camp.get("claim_class") or "fixture"),  # type: ignore[arg-type]
        source_commit=source_commit,
        source_dirty=source_dirty,
        author="semantic-factor-frontier-harness",
        requires_rl=False,
    )


def lock_campaign(campaign: ExperimentCampaignV1) -> CampaignLockV1:
    endpoint_ids = tuple(e.endpoint_id for e in campaign.endpoints)
    if not campaign_declares_runtime_endpoints(endpoint_ids):
        from slm_training.harnesses.experiments.semantic_factor_metrics import (
            REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS,
        )

        missing = sorted(REQUIRED_CAMPAIGN_RUNTIME_ENDPOINTS - set(endpoint_ids))
        raise ValueError(
            "SFF campaign missing required runtime endpoints for future runs: "
            f"{missing}"
        )
    digest = campaign_manifest_sha256(campaign)
    return CampaignLockV1(manifest_sha256=digest, manifest=campaign)


def _arm_config(arm_id: str, seed: int) -> SemanticResidualScorerConfig:
    """Resolve scorer config from external scorer_params.v1.json only."""

    scorer = load_sff_campaign().scorer
    return SemanticResidualScorerConfig.from_external(
        arm=scorer.arm(arm_id),
        defaults=scorer.defaults,
        seed=seed,
    )


def _reciprocal_rank(ordered: Sequence[str], gold: str | None) -> float:
    if gold is None:
        return 0.0
    for i, cid in enumerate(ordered):
        if cid == gold:
            return 1.0 / float(i + 1)
    return 0.0


def _math_properties() -> dict[str, Any]:
    """Executable math checks from external math_probes.v1.json (C13/C14)."""

    probes = load_sff_campaign().math_probes.payload
    cs = dict(probes.get("column_stochastic") or {})
    B = np.array(cs.get("incidence_B") or [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    S, col, notes = column_stochastic_S(B)
    atol = float(cs.get("column_sum_atol", 1e-9))
    mass_atol = float(cs.get("mass_atol", 1e-8))
    col_ok = bool(np.allclose(col, 1.0, atol=atol))
    r = np.array(cs.get("restart") or [1.0, 0.0, 0.0], dtype=np.float64)
    prop = propagate_restart(B, r)
    mass_ok = abs(float(prop.confidence.sum()) - 1.0) < mass_atol
    st = dict(probes.get("soft_token_collision") or {})
    a = list(st.get("a") or [1, 0, 0])
    b = list(st.get("b") or [0, 1, 0])
    coeffs = list(st.get("coeffs") or [7, 7, 3])
    soft_a = sum(int(x) * int(c) for x, c in zip(a, coeffs, strict=False))
    soft_b = sum(int(x) * int(c) for x, c in zip(b, coeffs, strict=False))
    soft_ok = soft_a == soft_b and a != b
    return {
        "column_stochastic_ok": col_ok and mass_ok,
        "column_sums": [float(x) for x in col.tolist()],
        "propagation_converged": bool(prop.converged),
        "propagation_steps": int(prop.steps),
        "soft_token_collision_ok": soft_ok,
        "soft_token_values": [soft_a, soft_b],
        "notes": list(notes),
    }



@dataclass
class _ExampleOutcome:
    example_id: str
    family: str
    arm_id: str
    seed: int
    selected: str | None
    gold: str | None
    correct: bool | None
    reciprocal_rank: float
    base_selected: str | None
    quality_improving: bool
    quality_harming: bool
    result: dict[str, Any]
    project_ms: float
    score_ms: float
    decision_ms: float  # project (amortized 0 when shared) + score + select


def run_semantic_factor_frontier(
    *,
    seeds: Sequence[int] | None = None,
    out_dir: Path | None = None,
    source_commit: str | None = None,
    source_dirty: bool = False,
) -> dict[str, Any]:
    """Execute the locked campaign and return metrics + claim verdicts."""

    commit = source_commit or ("0" * 40)
    sff_early = load_sff_campaign()
    if seeds is None:
        seeds = sff_early.seeds
    campaign = build_campaign(source_commit=commit, source_dirty=source_dirty)
    lock = lock_campaign(campaign)
    suite = build_ranked_suite()
    started = time.perf_counter()
    started_process = time.process_time()
    outcomes: list[_ExampleOutcome] = []
    membership_roundtrip_ok = True
    support_preservation_ok = True

    for seed in seeds:
        for example in suite:
            # Per-arm full decision path is timed independently so runtime is a
            # fair first-class metric (no free shared projection across arms).
            legal = tuple(example["legal"])
            baseline = {cid: float(example["baseline"].get(cid, 0.0)) for cid in legal}
            base_selected = (
                None
                if example["coverage"] != "complete"
                else max(legal, key=lambda c: (baseline[c], c))
            )
            for arm in campaign.arms:
                decision_t0 = time.perf_counter()
                project_t0 = time.perf_counter()
                snap = project_program_factors(
                    program_id=str(example["example_id"]),
                    components=tuple(example["components"]),
                    binder_edges=tuple(example.get("binder_edges") or ()),
                    legal_candidate_ids=legal,
                    coverage=str(example["coverage"]),
                )
                view = build_factor_representation(
                    snap, FactorRepresentation.FACTOR_NODE, seed=int(seed)
                )
                project_ms = (time.perf_counter() - project_t0) * 1000.0
                if view is not None:
                    reconstructed = reconstruct_membership(view)
                    expected = {f.factor_id: f.membership() for f in snap.factors}
                    if reconstructed != expected:
                        membership_roundtrip_ok = False
                cfg = _arm_config(arm.arm_id, int(seed))
                scorer = SemanticResidualScorerV1(cfg)
                score_t0 = time.perf_counter()
                result = scorer.score(
                    legal_candidate_ids=legal,
                    coverage=str(example["coverage"]),
                    snapshot=snap,
                    expected_source_digest=snap.source_digest,
                    baseline_scores=baseline,
                )
                score_ms = (time.perf_counter() - score_t0) * 1000.0
                # Support preservation: applied keys ⊆ legal.
                if any(k not in set(legal) for k in result.deltas):
                    support_preservation_ok = False
                select_t0 = time.perf_counter()
                combined = {
                    cid: baseline[cid] + float(result.deltas.get(cid, 0.0))
                    for cid in legal
                }
                if example["coverage"] != "complete":
                    selected = None
                    correct = None
                    ordered: list[str] = []
                elif len(legal) == 1:
                    selected = legal[0]
                    correct = selected == example["gold"]
                    ordered = [selected]
                else:
                    ordered = sorted(
                        legal, key=lambda c: (combined[c], c), reverse=True
                    )
                    selected = ordered[0]
                    correct = selected == example["gold"]
                select_ms = (time.perf_counter() - select_t0) * 1000.0
                decision_ms = (time.perf_counter() - decision_t0) * 1000.0
                # Keep decomposition consistent with wall decision_ms.
                _ = select_ms
                rr = _reciprocal_rank(ordered, example["gold"])
                improving = (
                    correct is True
                    and base_selected is not None
                    and base_selected != example["gold"]
                    and selected == example["gold"]
                )
                harming = (
                    correct is False
                    and base_selected == example["gold"]
                    and selected != example["gold"]
                )
                outcomes.append(
                    _ExampleOutcome(
                        example_id=str(example["example_id"]),
                        family=str(example.get("family") or "ranked"),
                        arm_id=arm.arm_id,
                        seed=int(seed),
                        selected=selected,
                        gold=example["gold"],
                        correct=correct,
                        reciprocal_rank=rr,
                        base_selected=base_selected,
                        quality_improving=bool(improving),
                        quality_harming=bool(harming),
                        result=result.to_dict(),
                        project_ms=project_ms,
                        score_ms=score_ms,
                        decision_ms=decision_ms,
                    )
                )

    sff = load_sff_campaign()
    metrics_cfg = sff.metrics
    scorer_cfg = sff.scorer
    control_arm_id = scorer_cfg.control_arm_id

    arm_rows: list[dict[str, Any]] = []
    n_ranked = 0
    for arm in campaign.arms:
        rows = [o for o in outcomes if o.arm_id == arm.arm_id]
        # Flatten outcome objects into engine rows (metrics file field names).
        engine_rows: list[dict[str, Any]] = []
        for o in rows:
            engine_rows.append(
                {
                    "family": o.family,
                    "correct": o.correct,
                    "reciprocal_rank": o.reciprocal_rank,
                    "applications": int(o.result["applications"]),
                    "choice_changed": bool(o.result["choice_changed"]),
                    "quality_improving": o.quality_improving,
                    "quality_harming": o.quality_harming,
                    "singleton_skip": bool(o.result["singleton_skip"]),
                    "incomplete_skip": bool(o.result["incomplete_skip"]),
                    "unknown_preserved": bool(o.result["unknown_preserved"]),
                    "decision_ms": float(o.decision_ms),
                    "project_ms": float(o.project_ms),
                    "score_ms": float(o.score_ms),
                }
            )
        applications = sum(int(r["applications"]) for r in engine_rows)
        choice_changes = sum(1 for r in engine_rows if r["choice_changed"])
        kill: list[str] = []
        for rule in sff.kill_rules:
            rtype = str(rule.get("type") or "")
            rid = str(rule.get("id") or rtype)
            scope = str(rule.get("scope") or "")
            if rtype == "apps_without_choice":
                if applications > 0 and choice_changes == 0:
                    if scope != "decode_on" or scorer_cfg.is_decode_on(arm.arm_id):
                        kill.append(rid)
            elif rtype == "singleton_work":
                ex_id = str(rule.get("example_id") or "singleton_root")
                for o in rows:
                    if o.example_id != ex_id:
                        continue
                    if not o.result["singleton_skip"] or o.result["applications"] != 0:
                        kill.append(rid)
            elif rtype == "singleton_neural":
                ex_id = str(rule.get("example_id") or "singleton_root")
                for o in rows:
                    if o.example_id != ex_id:
                        continue
                    if o.result["telemetry"].get("neural_forwards", 0) != 0:
                        kill.append(rid)
        row = compute_arm_metrics(
            metrics_cfg,
            arm_id=arm.arm_id,
            role=arm.role,
            rows=engine_rows,
            kills=sorted(set(kill)),
            trainable_parameters=0,
            claim_class=str(sff.payload.get("claim_class") or "fixture"),
        )
        n_ranked = max(n_ranked, int(row.get("n_ranked") or 0))
        arm_rows.append(row)

    apply_control_relative_metrics(
        metrics_cfg, arm_rows, control_arm_id=control_arm_id
    )

    control = next(r for r in arm_rows if r["arm_id"] == control_arm_id)
    control_acc = control.get("ranking_accuracy") or 0.0
    control_ms = float(control.get("wall_ms_mean") or 0.0)
    zero_eps = float(metrics_cfg.numerics.get("ratio_zero_eps", 1e-12))
    for row in arm_rows:
        acc = row.get("ranking_accuracy")
        for rule in sff.kill_rules:
            if str(rule.get("type")) != "quality_decreased":
                continue
            rid = str(rule.get("id") or "quality_decreased_with_choice_changes")
            eps = float(rule.get("eps", zero_eps))
            scope = str(rule.get("scope") or "")
            if scope == "decode_on" and not scorer_cfg.is_decode_on(str(row["arm_id"])):
                continue
            if (
                acc is not None
                and int(row.get("choice_changes") or 0) > 0
                and float(acc) + eps < float(control_acc)
            ):
                row["kill_criteria_hit"] = sorted(
                    set(row["kill_criteria_hit"]) | {rid}
                )

    # Paired comparisons from campaign resource.
    paired_blocks: dict[str, Any] = {}
    ranked_fams = set(
        str(x)
        for x in (sff.suite.payload.get("ranked_families") or ("ranked", "adversarial"))
    )
    for pair in sff.paired_comparisons:
        pair_id = str(pair.get("id") or "paired")
        treatment = str(pair["treatment"])
        ctrl_id = str(pair.get("control") or control_arm_id)
        families = set(str(x) for x in pair.get("families") or ranked_fams)
        control_by_key = {
            (o.example_id, o.seed): o
            for o in outcomes
            if o.arm_id == ctrl_id and o.family in families
        }
        deltas: list[float] = []
        for o in outcomes:
            if o.arm_id != treatment:
                continue
            if o.family not in families or o.correct is None:
                continue
            base = control_by_key.get((o.example_id, o.seed))
            if base is None or base.correct is None:
                continue
            deltas.append(float(o.correct) - float(base.correct))
        paired_blocks[pair_id] = {
            "treatment": treatment,
            "control": ctrl_id,
            "n": len(deltas),
            "mean_delta_accuracy": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "n_improved": sum(1 for d in deltas if d > 0),
            "n_regressed": sum(1 for d in deltas if d < 0),
        }
    # Back-compat alias used by older docs/tests.
    primary_pair = next(iter(paired_blocks.values()), None)
    paired_role_ported = primary_pair or {
        "n": 0,
        "mean_delta_accuracy": None,
        "n_improved": 0,
        "n_regressed": 0,
    }

    decode_off_arm = str(
        sff.payload.get("decode_off_arm_id") or "train_on_decode_off_role_ported"
    )
    decode_off_identical = True
    control_sel = {
        (o.example_id, o.seed): o.selected
        for o in outcomes
        if o.arm_id == control_arm_id
    }
    for o in outcomes:
        if o.arm_id == decode_off_arm:
            if control_sel.get((o.example_id, o.seed)) != o.selected:
                decode_off_identical = False

    math_props = _math_properties()
    payload: dict[str, Any] = {
        "kind": metrics_cfg.scoreboard_kind,
        "metrics_contract": str(
            metrics_cfg.payload.get("metrics_contract_id") or "semantic_factor_metrics/v1"
        ),
        "config_resources": sff.resource_identity(),
        "claim_class": str(sff.payload.get("claim_class") or "fixture"),
        "campaign_lock": {
            "manifest_sha256": lock.manifest_sha256,
            "campaign_id": campaign.campaign_id,
            "experiment_id": campaign.experiment_id,
        },
        "suite": {
            "n_examples": len(suite),
            "n_ranked_design": sum(1 for e in suite if e.get("family") in ranked_fams),
            "seeds": list(seeds),
            "resource": sff.suite.resource_identity(),
        },
        "n_ranked_per_arm": n_ranked,
        "membership_roundtrip_ok": membership_roundtrip_ok,
        "support_preservation_ok": support_preservation_ok,
        "decode_off_identical_to_control": decode_off_identical,
        "math_properties": math_props,
        "paired_comparisons": paired_blocks,
        "paired_role_ported_vs_control": {
            "n": paired_role_ported.get("n"),
            "mean_delta_accuracy": paired_role_ported.get("mean_delta_accuracy"),
            "n_improved": paired_role_ported.get("n_improved"),
            "n_regressed": paired_role_ported.get("n_regressed"),
        },
        "arms": arm_rows,
        "outcomes": [
            {
                "example_id": o.example_id,
                "family": o.family,
                "arm_id": o.arm_id,
                "seed": o.seed,
                "selected": o.selected,
                "base_selected": o.base_selected,
                "gold": o.gold,
                "correct": o.correct,
                "reciprocal_rank": o.reciprocal_rank,
                "quality_improving": o.quality_improving,
                "quality_harming": o.quality_harming,
                "project_ms": o.project_ms,
                "score_ms": o.score_ms,
                "decision_ms": o.decision_ms,
                "result": o.result,
            }
            for o in outcomes
        ],
        "unimplemented": {
            "rl": True,
            "spectral_training": True,
            "production_topology_authority": True,
            "graph_pruning": True,
            "recurrent_semantic_inference": True,
            "faithful_shift_soft_tokens": True,
            "search_r1_policy_training": True,
        },
        "runtime": {
            "campaign_wall_s": time.perf_counter() - started,
            "campaign_process_s": time.process_time() - started_process,
            "timer": metrics_cfg.timer,
            "decision_path": metrics_cfg.decision_path,
            "control_wall_ms_mean": control_ms,
            "note": (
                "decision_ms is the first-class runtime metric for efficiency. "
                "Slight quality loss can be preferred if wall_ms_mean is ~100× lower."
            ),
        },
        "wall_time_s": time.perf_counter() - started,
        "version_stamp": build_version_stamp(),
        "honesty": (
            "Fixture/synthetic ranked suite (n_ranked design ≥24×seeds). "
            "Not held_out promotion. Anti-E237 bar requires held_out n≥20. "
            "Runtime is a first-class endpoint (wall_ms_mean, quality_per_ms). "
            "Metrics, scorer params, claims, suite, projection, and math probes "
            "loaded from versioned resources (config_resources)."
        ),
    }
    claims = evaluate_claims(
        payload, claims_cfg=sff.claims, metrics_cfg=metrics_cfg
    )
    payload["claims"] = [c.to_dict() for c in claims]
    payload["claims_summary"] = {
        "validated": sum(1 for c in claims if c.verdict == "validated"),
        "invalidated": sum(1 for c in claims if c.verdict == "invalidated"),
        "inconclusive": sum(1 for c in claims if c.verdict == "inconclusive"),
    }
    mech = dict(sff.payload.get("mechanism_useful") or {})
    prefix = str(mech.get("arm_id_prefix") or "train_on_decode_on")
    any_useful = any(
        str(r["arm_id"]).startswith(prefix)
        and (not mech.get("require_choice_changes", True) or r["choice_changes"] > 0)
        and (not mech.get("require_no_kills", True) or not r["kill_criteria_hit"])
        and (
            not mech.get("require_acc_gt_control", True)
            or (r["ranking_accuracy"] or 0) > control_acc
        )
        for r in arm_rows
    )
    payload["mechanism_causally_useful"] = bool(
        any_useful and membership_roundtrip_ok and decode_off_identical
    )

    # Fail closed: future runs cannot drop runtime efficiency fields.
    validate_scoreboard_metrics(payload)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "campaign_lock.json").write_text(
            lock.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "scoreboard.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_dir / "claims.json").write_text(
            json.dumps(payload["claims"], indent=2) + "\n", encoding="utf-8"
        )
    return payload
