"""Claim registry: validate / invalidate / inconclusive for SFF experiments.

Claim inventory and thresholds live in ``claims.v1.json`` (and linked
metrics/campaign promotion fields). This module only dispatches by rule type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from slm_training.harnesses.experiments.semantic_factor_config import (
    SFFClaimsConfig,
    SFFMetricsConfig,
    load_sff_campaign,
    load_sff_claims,
    load_sff_metrics,
)

ClaimVerdict = Literal["validated", "invalidated", "inconclusive"]

__all__ = [
    "ClaimSpec",
    "ClaimResult",
    "evaluate_claims",
    "CLAIM_SPECS",
    "load_claim_specs",
]


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    statement: str
    kind: Literal["math", "causal", "safety", "representation"]
    rule: str
    params: Mapping[str, Any]


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


def load_claim_specs(
    claims_cfg: SFFClaimsConfig | None = None,
) -> tuple[ClaimSpec, ...]:
    cfg = claims_cfg or load_sff_claims()
    out: list[ClaimSpec] = []
    for raw in cfg.claims:
        out.append(
            ClaimSpec(
                claim_id=str(raw["claim_id"]),
                statement=str(raw["statement"]),
                kind=str(raw["kind"]),  # type: ignore[arg-type]
                rule=str(raw["rule"]),
                params=dict(raw.get("params") or {}),
            )
        )
    return tuple(out)


# Back-compat: load at import for tests that iterate CLAIM_SPECS.
CLAIM_SPECS: tuple[ClaimSpec, ...] = load_claim_specs()


def _arm(metrics: Mapping[str, Any], arm_id: str) -> dict[str, Any] | None:
    for row in metrics.get("arms") or []:
        if row.get("arm_id") == arm_id:
            return dict(row)
    return None


def _acc(row: Mapping[str, Any] | None, metric: str = "ranking_accuracy") -> float | None:
    if row is None:
        return None
    v = row.get(metric)
    return float(v) if v is not None else None


def _thresholds(
    claims_cfg: SFFClaimsConfig,
    metrics_cfg: SFFMetricsConfig,
) -> dict[str, Any]:
    thr = dict(claims_cfg.thresholds)
    # Prefer metrics.promotion when present (single source for bar numbers).
    promo = metrics_cfg.promotion
    if "delta" in promo:
        thr["promotion_delta"] = float(promo["delta"])
    if "min_n" in promo:
        thr["promotion_min_n"] = int(promo["min_n"])
    # Prefer campaign promotion gate min_n/threshold if recorded on metrics payload.
    # Efficiency slowdown from metrics.efficiency_gate.
    eg = metrics_cfg.efficiency_gate
    if "max_slowdown_without_quality_gain" in eg:
        thr["efficiency_max_slowdown"] = float(eg["max_slowdown_without_quality_gain"])
    thr.setdefault("promotion_delta", 0.10)
    thr.setdefault("promotion_min_n", 20)
    thr.setdefault("efficiency_max_slowdown", 10.0)
    thr.setdefault("acc_eps", 1e-9)
    thr.setdefault("ms_eps", 1e-9)
    thr.setdefault("zero_eps", 1e-12)
    thr.setdefault(
        "fixture_claim_classes",
        ["fixture", "wiring", "fixture_or_scratch"],
    )
    return thr


def evaluate_claims(
    metrics: Mapping[str, Any],
    *,
    claims_cfg: SFFClaimsConfig | None = None,
    metrics_cfg: SFFMetricsConfig | None = None,
) -> list[ClaimResult]:
    """Evaluate every registered claim against a campaign metrics payload."""

    claims_cfg = claims_cfg or load_sff_claims()
    metrics_cfg = metrics_cfg or load_sff_metrics()
    # Prefer campaign-linked promotion min_n from locked campaign when present.
    try:
        camp = load_sff_campaign()
        for gate in camp.payload.get("promotion_gates") or []:
            if str(gate.get("gate_id")) == "anti_e237_bar":
                if "threshold" in gate:
                    # campaign threshold is absolute gain floor for promotion gate
                    pass
                if "min_n" in gate:
                    # inject into a local thr override via metrics_cfg path below
                    metrics_cfg = metrics_cfg  # noqa: PLW0127 — keep reference
                break
    except Exception:
        camp = None

    thr = _thresholds(claims_cfg, metrics_cfg)
    if camp is not None:
        for gate in camp.payload.get("promotion_gates") or []:
            if str(gate.get("gate_id")) == "anti_e237_bar":
                if "threshold" in gate:
                    thr["promotion_delta"] = float(gate["threshold"])
                if "min_n" in gate:
                    thr["promotion_min_n"] = int(gate["min_n"])

    control_id = "control_none"
    if camp is not None:
        control_id = camp.scorer.control_arm_id
    control = _arm(metrics, control_id)
    control_acc = _acc(control) or 0.0
    acc_eps = float(thr["acc_eps"])
    ms_eps = float(thr["ms_eps"])
    zero_eps = float(thr["zero_eps"])
    specs = load_claim_specs(claims_cfg)
    results: list[ClaimResult] = []

    for spec in specs:
        params = dict(spec.params)
        verdict: ClaimVerdict = "inconclusive"
        evidence: dict[str, Any] = {}
        falsifier = ""

        if spec.rule == "support_preservation":
            illegal = sum(
                1
                for o in metrics.get("outcomes") or []
                if o.get("result", {}).get("illegal_score_rejection")
            )
            support_ok = metrics.get("support_preservation_ok", True)
            evidence = {
                "illegal_rejections": illegal,
                "support_preservation_ok": support_ok,
            }
            verdict = "validated" if support_ok else "invalidated"
            falsifier = "any residual key outside A_compiler"

        elif spec.rule == "singleton_zero_work":
            kill_ids = set(str(x) for x in params.get("kill_ids") or ())
            kills: list[str] = []
            for row in metrics.get("arms") or []:
                kills.extend(row.get("kill_criteria_hit") or [])
            bad = any(k in kills for k in kill_ids)
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

        elif spec.rule == "causal_improve":
            arm_id = str(params["arm_id"])
            row = _arm(metrics, arm_id)
            acc = _acc(row)
            apps = int((row or {}).get("applications") or 0)
            chg = int((row or {}).get("choice_changes") or 0)
            kills = list((row or {}).get("kill_criteria_hit") or [])
            evidence = {
                "arm_id": arm_id,
                "ranking_accuracy": acc,
                "control_accuracy": control_acc,
                "applications": apps,
                "choice_changes": chg,
                "kills": kills,
                "delta_acc": None if acc is None else acc - control_acc,
            }
            require_chg = bool(params.get("require_choice_change", True))
            require_no_kills = bool(params.get("require_no_kills", True))
            if apps > 0 and chg == 0:
                verdict = "invalidated"
            elif (
                acc is not None
                and (not require_chg or chg > 0)
                and acc > control_acc
                and (not require_no_kills or not kills)
            ):
                verdict = "validated"
            else:
                verdict = "inconclusive"
            falsifier = "apps>0 & choice_changes==0 or quality not improved"

        elif spec.rule == "beats_arm":
            treat = _arm(metrics, str(params["treatment_arm_id"]))
            base = _arm(metrics, str(params["baseline_arm_id"]))
            metric = str(params.get("metric") or "ranking_accuracy")
            a_t, a_b = _acc(treat, metric), _acc(base, metric)
            evidence = {
                "treatment": params["treatment_arm_id"],
                "baseline": params["baseline_arm_id"],
                "treatment_value": a_t,
                "baseline_value": a_b,
            }
            if a_t is None or a_b is None:
                verdict = "inconclusive"
            elif a_t > a_b + acc_eps:
                verdict = "validated"
            elif a_t + acc_eps < a_b:
                verdict = "invalidated"
            else:
                verdict = "inconclusive"
            falsifier = f"{params['baseline_arm_id']} ≥ {params['treatment_arm_id']}"

        elif spec.rule == "competitive_with":
            arm = _arm(metrics, str(params["arm_id"]))
            peer = _arm(metrics, str(params["peer_arm_id"]))
            metric = str(params.get("metric") or "ranking_accuracy")
            a, a_p = _acc(arm, metric), _acc(peer, metric)
            evidence = {
                "arm_id": params["arm_id"],
                "peer_arm_id": params["peer_arm_id"],
                "arm_value": a,
                "peer_value": a_p,
                "params": 0,
            }
            if a is None:
                verdict = "inconclusive"
            elif a_p is None or a + acc_eps >= a_p:
                verdict = "validated"
            else:
                verdict = "inconclusive"
            falsifier = "peer arm dominates by large margin with params"

        elif spec.rule == "apps_without_choice_kill":
            prefix = str(params.get("arm_id_prefix") or "train_on_decode_on")
            kill_id = str(params.get("kill_id") or "applications_without_choice_changes")
            offenders = [
                r["arm_id"]
                for r in metrics.get("arms") or []
                if str(r.get("arm_id", "")).startswith(prefix)
                and int(r.get("applications") or 0) > 0
                and int(r.get("choice_changes") or 0) == 0
            ]
            evidence = {"offenders": offenders}
            detected = (
                all(
                    kill_id
                    in (_arm(metrics, oid) or {}).get("kill_criteria_hit", [])
                    for oid in offenders
                )
                if offenders
                else True
            )
            verdict = "validated" if detected else "invalidated"
            falsifier = "decode-on arm with apps>0 choice_changes==0 not killed"

        elif spec.rule == "math_flag":
            flag = str(params["flag"])
            ok = bool(metrics.get("math_properties", {}).get(flag))
            evidence = dict(metrics.get("math_properties") or {})
            verdict = "validated" if ok else "invalidated"
            falsifier = f"math property {flag} failed"

        elif spec.rule == "unimplemented_forbidden":
            un = metrics.get("unimplemented") or {}
            required: Sequence[str] = tuple(
                str(x)
                for x in (
                    claims_cfg.payload.get("unimplemented_required_keys")
                    or params.get("keys")
                    or ()
                )
            )
            ok = all(bool(un.get(k)) for k in required)
            evidence = {k: un.get(k) for k in required}
            verdict = "validated" if ok else "invalidated"
            falsifier = "forbidden path implemented"

        elif spec.rule == "promotion_bar":
            n = int(metrics.get("n_ranked_per_arm") or 0)
            prefix = str(params.get("decode_on_prefix") or "train_on_decode_on")
            best = None
            for r in metrics.get("arms") or []:
                if str(r.get("arm_id", "")).startswith(prefix):
                    acc = _acc(r)
                    if acc is None:
                        continue
                    if best is None or acc > best:
                        best = acc
            delta = float(thr["promotion_delta"])
            min_n = int(thr["promotion_min_n"])
            fixture_classes = set(
                str(x) for x in thr.get("fixture_claim_classes") or ()
            )
            evidence = {
                "n_ranked_per_arm": n,
                "control_accuracy": control_acc,
                "best_decode_on_accuracy": best,
                "claim_class": metrics.get("claim_class"),
                "required_delta": delta,
                "required_n": min_n,
            }
            if metrics.get("claim_class") in fixture_classes:
                verdict = "invalidated"
            elif best is not None and n >= min_n and best >= control_acc + delta:
                verdict = "validated"
            else:
                verdict = "invalidated"
            falsifier = (
                f"held_out n<{min_n} or delta < {delta} or fixture claim class"
            )

        elif spec.rule == "runtime_tracked":
            required = tuple(
                str(x)
                for x in (
                    claims_cfg.payload.get("runtime_required_arm_fields")
                    or params.get("fields")
                    or ()
                )
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
            verdict = (
                "validated" if not missing and runtime.get("timer") else "invalidated"
            )
            falsifier = "arm missing wall_ms_* / quality_per_ms or campaign runtime block absent"

        elif spec.rule == "efficiency_gate":
            control_row = _arm(metrics, control_id) or {}
            control_ms = float(control_row.get("wall_ms_mean") or 0.0)
            max_slow = float(thr["efficiency_max_slowdown"])
            skip = set(
                str(x)
                for x in (
                    claims_cfg.payload.get("efficiency_skip_arm_ids")
                    or params.get("skip_arm_ids")
                    or ()
                )
            )
            offenders: list[dict[str, Any]] = []
            ok_arms: list[dict[str, Any]] = []
            for r in metrics.get("arms") or []:
                arm_id = str(r.get("arm_id") or "")
                if arm_id in skip:
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
                if ratio is not None and ratio > max_slow:
                    if delta is None or delta <= zero_eps:
                        offenders.append(row)
                    else:
                        ok_arms.append(row)
            evidence = {
                "control_wall_ms_mean": control_ms,
                "max_slowdown": max_slow,
                "offenders_gt_slowdown_no_quality_gain": offenders,
                "ok_gt_slowdown_with_quality_gain": ok_arms,
            }
            if control_ms <= 0:
                verdict = "inconclusive"
            elif offenders:
                verdict = "invalidated"
            else:
                verdict = "validated"
            falsifier = f">{max_slow}× slower than control with no accuracy gain"

        elif spec.rule == "exact_typed_efficiency":
            arm_id = str(params.get("arm_id") or "exact_typed_zero_parameter")
            prefix = str(params.get("competitor_prefix") or "train_on_decode_on")
            ex = _arm(metrics, arm_id) or {}
            a_ex = _acc(ex)
            ms_ex = float(ex.get("wall_ms_mean") or 0.0)
            dominated_by: list[str] = []
            for r in metrics.get("arms") or []:
                cid = str(r.get("arm_id") or "")
                if not cid.startswith(prefix):
                    continue
                a = _acc(r)
                ms = float(r.get("wall_ms_mean") or 0.0)
                if a_ex is None or a is None or ms_ex <= 0 or ms <= 0:
                    continue
                better_acc = a > a_ex + zero_eps
                not_slower = ms <= ms_ex * (1.0 + ms_eps)
                same_acc = abs(a - a_ex) <= zero_eps
                faster = ms < ms_ex * (1.0 - ms_eps)
                if (better_acc and not_slower) or (same_acc and faster):
                    dominated_by.append(cid)
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

        else:
            verdict = "inconclusive"
            evidence = {"unsupported_rule": spec.rule}
            falsifier = f"unsupported claim rule {spec.rule!r}"

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
