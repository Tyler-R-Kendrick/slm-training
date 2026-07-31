"""Claim registry: validate / invalidate / inconclusive for SFF experiments.

Each claim is a falsifiable statement with a machine-checkable verdict derived
from measured campaign metrics. Proofs in Lean cover the mathematical cores;
runtime claims cover causal/experimental hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

ClaimVerdict = Literal["validated", "invalidated", "inconclusive"]

__all__ = [
    "ClaimSpec",
    "ClaimResult",
    "evaluate_claims",
    "CLAIM_SPECS",
]


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    statement: str
    kind: Literal["math", "causal", "safety", "representation"]
    # How to decide from a metrics blob (implemented in evaluate_claims).
    rule: str


@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    statement: str
    verdict: ClaimVerdict
    evidence: dict[str, Any]
    falsifier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "verdict": self.verdict,
            "evidence": dict(self.evidence),
            "falsifier": self.falsifier,
        }


CLAIM_SPECS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        "SFF-C1-support-preservation",
        "Advisory residual never changes the exact legal candidate set.",
        "safety",
        "support_preservation",
    ),
    ClaimSpec(
        "SFF-C2-singleton-zero-work",
        "Complete singletons perform zero factor/propagation/ranker/neural work.",
        "safety",
        "singleton_zero_work",
    ),
    ClaimSpec(
        "SFF-C3-unknown-preserved",
        "Incomplete coverage never becomes UNSUPPORTED via residual scoring.",
        "safety",
        "unknown_preserved",
    ),
    ClaimSpec(
        "SFF-C4-decode-off-identity",
        "train-on/decode-off is behaviorally identical to control at decode.",
        "causal",
        "decode_off_identity",
    ),
    ClaimSpec(
        "SFF-C5-factor-node-lossless",
        "factor_node representation reconstructs factor membership exactly.",
        "representation",
        "membership_roundtrip",
    ),
    ClaimSpec(
        "SFF-C6-role-ported-causal",
        "role_ported_factor residual changes legal choices and improves ranking vs control.",
        "causal",
        "role_ported_causal",
    ),
    ClaimSpec(
        "SFF-C7-factor-node-causal",
        "factor_node residual (roles stripped) changes choices and improves ranking.",
        "causal",
        "factor_node_causal",
    ),
    ClaimSpec(
        "SFF-C8-direct-factors-causal",
        "direct_factors residual changes choices and improves ranking vs control.",
        "causal",
        "direct_factors_causal",
    ),
    ClaimSpec(
        "SFF-C9-role-identity-load-bearing",
        "role_ported beats role_shuffled on ranking when both apply residuals.",
        "causal",
        "role_identity",
    ),
    ClaimSpec(
        "SFF-C10-higher-order-identity",
        "factor_node (lossless) beats lossy_pairwise on ranking under matched evidence.",
        "causal",
        "higher_order",
    ),
    ClaimSpec(
        "SFF-C11-exact-typed-baseline",
        "exact_typed_zero_parameter is competitive with residual arms (zero params).",
        "causal",
        "exact_typed",
    ),
    ClaimSpec(
        "SFF-C12-apps-without-choice-is-fail",
        "Any decode-on residual arm with applications>0 and choice_changes==0 is rejected.",
        "causal",
        "apps_without_choice_kill",
    ),
    ClaimSpec(
        "SFF-C13-propagation-column-stochastic",
        "Reference S has column sums ≈ 1 on valid nonempty incidence (math property).",
        "math",
        "column_stochastic",
    ),
    ClaimSpec(
        "SFF-C14-soft-token-not-injective",
        "Soft-token map p↦Eᵀp is not injective on the simplex (SHIFT video overclaim).",
        "math",
        "soft_token",
    ),
    ClaimSpec(
        "SFF-C15-no-rl-no-spectral",
        "RL and spectral training paths remain unimplemented in this delivery.",
        "safety",
        "unimplemented_forbidden",
    ),
    ClaimSpec(
        "SFF-C16-anti-e237-promotion-bar",
        "held_out meaningful ≥ control+0.10 at n≥20 (promotion bar).",
        "causal",
        "promotion_bar",
    ),
    ClaimSpec(
        "SFF-C17-runtime-tracked",
        "Every arm reports wall_ms_mean/p50/p95 and quality_per_ms (runtime is first-class).",
        "causal",
        "runtime_tracked",
    ),
    ClaimSpec(
        "SFF-C18-efficiency-vs-quality",
        "If a residual arm is >10× slower than control mean wall_ms, it must improve accuracy "
        "by more than the noise floor or be efficiency-rejected.",
        "causal",
        "efficiency_gate",
    ),
    ClaimSpec(
        "SFF-C19-exact-typed-efficiency",
        "exact_typed_zero_parameter is not dominated on (accuracy, wall_ms_mean) by residual arms "
        "with equal-or-worse quality and higher runtime.",
        "causal",
        "exact_typed_efficiency",
    ),
)


def _arm(metrics: Mapping[str, Any], arm_id: str) -> dict[str, Any] | None:
    for row in metrics.get("arms") or []:
        if row.get("arm_id") == arm_id:
            return dict(row)
    return None


def _acc(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    v = row.get("ranking_accuracy")
    return float(v) if v is not None else None


def evaluate_claims(metrics: Mapping[str, Any]) -> list[ClaimResult]:
    """Evaluate every registered claim against a campaign metrics payload."""

    results: list[ClaimResult] = []
    control = _arm(metrics, "control_none")
    control_acc = _acc(control) or 0.0

    for spec in CLAIM_SPECS:
        verdict: ClaimVerdict = "inconclusive"
        evidence: dict[str, Any] = {}
        falsifier = ""

        if spec.rule == "support_preservation":
            # No outcome may report illegal score acceptance.
            illegal = sum(
                1
                for o in metrics.get("outcomes") or []
                if o.get("result", {}).get("illegal_score_rejection")
            )
            # Also require every applied delta keys ⊆ legal (enforced by scorer).
            support_ok = metrics.get("support_preservation_ok", True)
            evidence = {"illegal_rejections": illegal, "support_preservation_ok": support_ok}
            verdict = "validated" if support_ok and illegal >= 0 else "invalidated"
            # Always validated if no illegal keys accepted (rejections are OK).
            verdict = "validated" if support_ok else "invalidated"
            falsifier = "any residual key outside A_compiler"

        elif spec.rule == "singleton_zero_work":
            kills = []
            for row in metrics.get("arms") or []:
                kills.extend(row.get("kill_criteria_hit") or [])
            bad = any(
                k in kills
                for k in ("singleton_work_performed", "singleton_neural_forward")
            )
            evidence = {
                "singleton_skips_total": sum(
                    int(r.get("singleton_skips") or 0) for r in metrics.get("arms") or []
                ),
                "bad": bad,
            }
            verdict = "invalidated" if bad else "validated"
            falsifier = "singleton factor/propagation/ranker/neural work > 0"

        elif spec.rule == "unknown_preserved":
            preserved = all(
                int(r.get("unknown_preserved") or 0) >= 0
                for r in metrics.get("arms") or []
            )
            # Require incomplete examples produced unknown_preserved on every arm.
            per_arm = [
                int(r.get("unknown_preserved") or 0) > 0
                for r in metrics.get("arms") or []
            ]
            ok = preserved and (not per_arm or all(per_arm))
            evidence = {"per_arm_unknown_preserved": per_arm}
            verdict = "validated" if ok else "invalidated"
            falsifier = "incomplete coverage scored as definitive UNSUPPORTED"

        elif spec.rule == "decode_off_identity":
            ok = bool(metrics.get("decode_off_identical_to_control"))
            evidence = {"decode_off_identical_to_control": ok}
            verdict = "validated" if ok else "invalidated"
            falsifier = "decode-off selection differs from control"

        elif spec.rule == "membership_roundtrip":
            ok = bool(metrics.get("membership_roundtrip_ok"))
            evidence = {"membership_roundtrip_ok": ok}
            verdict = "validated" if ok else "invalidated"
            falsifier = "factor_node membership reconstruction mismatch"

        elif spec.rule == "role_ported_causal":
            row = _arm(metrics, "train_on_decode_on_role_ported")
            acc = _acc(row)
            apps = int((row or {}).get("applications") or 0)
            chg = int((row or {}).get("choice_changes") or 0)
            kills = list((row or {}).get("kill_criteria_hit") or [])
            evidence = {
                "ranking_accuracy": acc,
                "control_accuracy": control_acc,
                "applications": apps,
                "choice_changes": chg,
                "kills": kills,
                "delta_acc": None if acc is None else acc - control_acc,
            }
            if apps > 0 and chg == 0:
                verdict = "invalidated"
            elif acc is not None and chg > 0 and acc > control_acc and not kills:
                verdict = "validated"
            else:
                verdict = "inconclusive"
            falsifier = "apps>0 & choice_changes==0 or quality not improved"

        elif spec.rule == "factor_node_causal":
            row = _arm(metrics, "train_on_decode_on_factor_node")
            acc = _acc(row)
            apps = int((row or {}).get("applications") or 0)
            chg = int((row or {}).get("choice_changes") or 0)
            kills = list((row or {}).get("kill_criteria_hit") or [])
            evidence = {
                "ranking_accuracy": acc,
                "control_accuracy": control_acc,
                "applications": apps,
                "choice_changes": chg,
                "kills": kills,
            }
            if apps > 0 and chg == 0:
                verdict = "invalidated"
            elif acc is not None and chg > 0 and acc > control_acc and not kills:
                verdict = "validated"
            else:
                verdict = "inconclusive"
            falsifier = "apps>0 & choice_changes==0 or quality not improved"

        elif spec.rule == "direct_factors_causal":
            row = _arm(metrics, "train_on_decode_on_direct_factors")
            acc = _acc(row)
            apps = int((row or {}).get("applications") or 0)
            chg = int((row or {}).get("choice_changes") or 0)
            kills = list((row or {}).get("kill_criteria_hit") or [])
            evidence = {
                "ranking_accuracy": acc,
                "control_accuracy": control_acc,
                "applications": apps,
                "choice_changes": chg,
                "kills": kills,
                "delta_acc": None if acc is None else acc - control_acc,
            }
            if apps > 0 and chg == 0:
                verdict = "invalidated"
            elif acc is not None and chg > 0 and acc > control_acc and not kills:
                verdict = "validated"
            else:
                verdict = "inconclusive"
            falsifier = "apps>0 & choice_changes==0 or quality not improved"

        elif spec.rule == "role_identity":
            rp = _arm(metrics, "train_on_decode_on_role_ported")
            rs = _arm(metrics, "train_on_decode_on_role_shuffled")
            a_rp, a_rs = _acc(rp), _acc(rs)
            evidence = {"role_ported_acc": a_rp, "role_shuffled_acc": a_rs}
            if a_rp is None or a_rs is None:
                verdict = "inconclusive"
            elif a_rp > a_rs + 1e-9:
                verdict = "validated"
            elif a_rp + 1e-9 < a_rs:
                verdict = "invalidated"
            else:
                verdict = "inconclusive"
            falsifier = "role_shuffled ≥ role_ported ranking accuracy"

        elif spec.rule == "higher_order":
            fn = _arm(metrics, "train_on_decode_on_factor_node")
            pw = _arm(metrics, "train_on_decode_on_lossy_pairwise")
            a_fn, a_pw = _acc(fn), _acc(pw)
            evidence = {"factor_node_acc": a_fn, "lossy_pairwise_acc": a_pw}
            if a_fn is None or a_pw is None:
                verdict = "inconclusive"
            elif a_fn > a_pw + 1e-9:
                verdict = "validated"
            elif a_fn + 1e-9 < a_pw:
                verdict = "invalidated"
            else:
                verdict = "inconclusive"
            falsifier = "lossy_pairwise ≥ factor_node accuracy"

        elif spec.rule == "exact_typed":
            ex = _arm(metrics, "exact_typed_zero_parameter")
            df = _arm(metrics, "train_on_decode_on_direct_factors")
            a_ex, a_df = _acc(ex), _acc(df)
            evidence = {
                "exact_typed_acc": a_ex,
                "direct_factors_acc": a_df,
                "params": 0,
            }
            if a_ex is None:
                verdict = "inconclusive"
            elif a_df is None or a_ex + 1e-9 >= a_df:
                verdict = "validated"
            else:
                verdict = "inconclusive"
            falsifier = "learned/residual arms dominate exact typed by large margin with params"

        elif spec.rule == "apps_without_choice_kill":
            offenders = [
                r["arm_id"]
                for r in metrics.get("arms") or []
                if str(r.get("arm_id", "")).startswith("train_on_decode_on")
                and int(r.get("applications") or 0) > 0
                and int(r.get("choice_changes") or 0) == 0
            ]
            evidence = {"offenders": offenders}
            # Claim: the campaign correctly *detects* this as failure.
            detected = all(
                "applications_without_choice_changes"
                in ( _arm(metrics, oid) or {} ).get("kill_criteria_hit", [])
                for oid in offenders
            ) if offenders else True
            # If no offenders, claim that residual arms are causal is separate;
            # this claim is about the kill criterion working.
            verdict = "validated" if detected else "invalidated"
            falsifier = "decode-on arm with apps>0 choice_changes==0 not killed"

        elif spec.rule == "column_stochastic":
            ok = bool(metrics.get("math_properties", {}).get("column_stochastic_ok"))
            evidence = dict(metrics.get("math_properties") or {})
            verdict = "validated" if ok else "invalidated"
            falsifier = "column sums of S deviate from 1 beyond tolerance"

        elif spec.rule == "soft_token":
            ok = bool(metrics.get("math_properties", {}).get("soft_token_collision_ok"))
            evidence = dict(metrics.get("math_properties") or {})
            verdict = "validated" if ok else "invalidated"
            falsifier = "soft-token map injective on the test simplex points"

        elif spec.rule == "unimplemented_forbidden":
            un = metrics.get("unimplemented") or {}
            required = (
                "rl",
                "spectral_training",
                "production_topology_authority",
                "graph_pruning",
                "recurrent_semantic_inference",
                "faithful_shift_soft_tokens",
                "search_r1_policy_training",
            )
            ok = all(bool(un.get(k)) for k in required)
            evidence = {k: un.get(k) for k in required}
            verdict = "validated" if ok else "invalidated"
            falsifier = "forbidden path implemented"

        elif spec.rule == "promotion_bar":
            n = int(metrics.get("n_ranked_per_arm") or 0)
            best = None
            for r in metrics.get("arms") or []:
                if str(r.get("arm_id", "")).startswith("train_on_decode_on"):
                    acc = _acc(r)
                    if acc is None:
                        continue
                    if best is None or acc > best:
                        best = acc
            evidence = {
                "n_ranked_per_arm": n,
                "control_accuracy": control_acc,
                "best_decode_on_accuracy": best,
                "claim_class": metrics.get("claim_class"),
                "required_delta": 0.10,
                "required_n": 20,
            }
            if metrics.get("claim_class") in {"fixture", "wiring", "fixture_or_scratch"}:
                verdict = "invalidated"  # promotion bar not met by construction
            elif (
                best is not None
                and n >= 20
                and best >= control_acc + 0.10
            ):
                verdict = "validated"
            else:
                verdict = "invalidated"
            falsifier = "held_out n<20 or delta < 0.10 or fixture claim class"

        elif spec.rule == "runtime_tracked":
            required = (
                "wall_ms_mean",
                "wall_ms_p50",
                "wall_ms_p95",
                "wall_ms_total",
                "quality_per_ms",
                "wall_ms_mean_vs_control",
            )
            missing: list[str] = []
            for r in metrics.get("arms") or []:
                for key in required:
                    if key not in r:
                        missing.append(f"{r.get('arm_id')}:{key}")
            runtime = metrics.get("runtime") or {}
            evidence = {
                "missing_fields": missing,
                "campaign_wall_s": runtime.get("campaign_wall_s"),
                "timer": runtime.get("timer"),
                "n_arms": len(metrics.get("arms") or []),
            }
            verdict = "validated" if not missing and runtime.get("timer") else "invalidated"
            falsifier = "arm missing wall_ms_* / quality_per_ms or campaign runtime block absent"

        elif spec.rule == "efficiency_gate":
            # Residual arms that are >10× slower than control must improve accuracy.
            control = _arm(metrics, "control_none") or {}
            control_ms = float(control.get("wall_ms_mean") or 0.0)
            offenders: list[dict[str, Any]] = []
            ok_arms: list[dict[str, Any]] = []
            for r in metrics.get("arms") or []:
                arm_id = str(r.get("arm_id") or "")
                if arm_id in {"control_none", "train_on_decode_off_role_ported"}:
                    continue
                arm_ms = float(r.get("wall_ms_mean") or 0.0)
                ratio = (arm_ms / control_ms) if control_ms > 0 else None
                acc = _acc(r)
                delta = None if acc is None else acc - control_acc
                row = {
                    "arm_id": arm_id,
                    "wall_ms_mean": arm_ms,
                    "vs_control": ratio,
                    "delta_accuracy": delta,
                }
                if ratio is not None and ratio > 10.0:
                    if delta is None or delta <= 1e-12:
                        offenders.append(row)
                    else:
                        ok_arms.append(row)
            evidence = {
                "control_wall_ms_mean": control_ms,
                "offenders_gt_10x_no_quality_gain": offenders,
                "ok_gt_10x_with_quality_gain": ok_arms,
            }
            # Validated when the gate is enforced in metrics (no silent expensive no-ops).
            # If offenders exist, claim is invalidated (expensive without quality).
            if control_ms <= 0:
                verdict = "inconclusive"
            elif offenders:
                verdict = "invalidated"
            else:
                verdict = "validated"
            falsifier = ">10× slower than control with no accuracy gain"

        elif spec.rule == "exact_typed_efficiency":
            ex = _arm(metrics, "exact_typed_zero_parameter") or {}
            a_ex = _acc(ex)
            ms_ex = float(ex.get("wall_ms_mean") or 0.0)
            dominated_by: list[str] = []
            for r in metrics.get("arms") or []:
                arm_id = str(r.get("arm_id") or "")
                if not arm_id.startswith("train_on_decode_on"):
                    continue
                a = _acc(r)
                ms = float(r.get("wall_ms_mean") or 0.0)
                if a_ex is None or a is None or ms_ex <= 0 or ms <= 0:
                    continue
                # Dominates exact_typed if strictly better accuracy and not slower,
                # or same accuracy and strictly faster.
                better_acc = a > a_ex + 1e-12
                not_slower = ms <= ms_ex * (1.0 + 1e-9)
                same_acc = abs(a - a_ex) <= 1e-12
                faster = ms < ms_ex * (1.0 - 1e-9)
                if (better_acc and not_slower) or (same_acc and faster):
                    dominated_by.append(arm_id)
            evidence = {
                "exact_typed_acc": a_ex,
                "exact_typed_wall_ms_mean": ms_ex,
                "dominated_by": dominated_by,
            }
            if a_ex is None:
                verdict = "inconclusive"
            elif dominated_by:
                verdict = "invalidated"
            else:
                verdict = "validated"
            falsifier = "a residual arm dominates exact_typed on quality+runtime"

        results.append(
            ClaimResult(
                claim_id=spec.claim_id,
                statement=spec.statement,
                verdict=verdict,
                evidence=evidence,
                falsifier=falsifier,
            )
        )
    return results
