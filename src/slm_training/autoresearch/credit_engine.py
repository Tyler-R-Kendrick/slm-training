"""Authoritative statistical credit: observations + analysis plan → credit report.

Decision-bearing endpoints, paired effects, Holm multiplicity, and gate
outcomes for promotion/ship must be recomputed here from immutable observation
rows. Caller-supplied summaries are never trusted as sole authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.lineage.records import canonical_json

__all__ = [
    "CREDIT_DEFAULTS_SCHEMA",
    "CreditEngineError",
    "ObservationRow",
    "ObservationTable",
    "AnalysisPlan",
    "CreditReport",
    "SealedScoreLedger",
    "load_credit_defaults",
    "credit_defaults_digest",
    "observation_table_from_mapping",
    "analysis_plan_from_mapping",
    "credit_report_from_mapping",
    "compute_credit_report",
    "verify_credit_against_caller",
    "assert_score_once",
    "digest_mapping",
]

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULTS_PATH = (
    _PACKAGE_ROOT
    / "resources"
    / "experiments"
    / "authoritative_credit"
    / "defaults.v1.json"
)
CREDIT_DEFAULTS_SCHEMA = "authoritative_credit_defaults/v1"


class CreditEngineError(ValueError):
    """Observation/plan/report is invalid or credit recomputation failed."""


def digest_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


@lru_cache(maxsize=4)
def load_credit_defaults(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else _DEFAULTS_PATH
    if not p.is_file():
        raise CreditEngineError(f"missing credit defaults: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CreditEngineError("credit defaults root must be object")
    if str(raw.get("schema") or "") != CREDIT_DEFAULTS_SCHEMA:
        raise CreditEngineError(
            f"schema {raw.get('schema')!r} != {CREDIT_DEFAULTS_SCHEMA!r}"
        )
    return raw


def credit_defaults_digest(defaults: Mapping[str, Any] | None = None) -> str:
    payload = defaults if defaults is not None else load_credit_defaults()
    return digest_mapping(payload)


@dataclass(frozen=True)
class ObservationRow:
    arm_id: str
    seed: int
    example_id: str
    metric_id: str
    value: float
    direction: str  # increase | decrease
    split: str  # search | seal | ship | screening


@dataclass(frozen=True)
class ObservationTable:
    schema: str
    version: str
    campaign_id: str
    experiment_id: str
    manifest_sha256: str
    locked_eval_manifest_sha256: str | None
    rows: tuple[ObservationRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "campaign_id": self.campaign_id,
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "locked_eval_manifest_sha256": self.locked_eval_manifest_sha256,
            "rows": [
                {
                    "arm_id": r.arm_id,
                    "seed": r.seed,
                    "example_id": r.example_id,
                    "metric_id": r.metric_id,
                    "value": r.value,
                    "direction": r.direction,
                    "split": r.split,
                }
                for r in self.rows
            ],
        }

    def content_sha256(self) -> str:
        return digest_mapping(self.to_dict())


@dataclass(frozen=True)
class AnalysisPlan:
    schema: str
    version: str
    primary_metric: str
    primary_endpoint_id: str
    direction: str
    minimum_effect: float
    control_arm_id: str
    candidate_arm_id: str
    holm_hypothesis_ids: tuple[str, ...]
    holm_alpha: float
    seal_split: str
    score_once: bool
    underpowered_n_below: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "primary_metric": self.primary_metric,
            "primary_endpoint_id": self.primary_endpoint_id,
            "direction": self.direction,
            "minimum_effect": self.minimum_effect,
            "control_arm_id": self.control_arm_id,
            "candidate_arm_id": self.candidate_arm_id,
            "holm_hypothesis_ids": list(self.holm_hypothesis_ids),
            "holm_alpha": self.holm_alpha,
            "seal_split": self.seal_split,
            "score_once": self.score_once,
            "underpowered_n_below": self.underpowered_n_below,
        }

    def content_sha256(self) -> str:
        return digest_mapping(self.to_dict())


@dataclass(frozen=True)
class HolmCreditRow:
    hypothesis_id: str
    raw_p_value: float
    rank: int
    threshold: float
    adjusted_p_value: float
    rejected: bool


@dataclass(frozen=True)
class CreditReport:
    schema: str
    version: str
    observation_table_sha256: str
    analysis_plan_sha256: str
    endpoint_values: dict[str, float]
    paired_effect: float
    paired_n: int
    underpowered: bool
    disposition: str  # promotable | inconclusive | reject
    promotable_empirical: bool
    holm_results: tuple[HolmCreditRow, ...]
    promotion_gate_pass: bool
    rollback_gate_hit: bool
    sealed_score_once_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "observation_table_sha256": self.observation_table_sha256,
            "analysis_plan_sha256": self.analysis_plan_sha256,
            "endpoint_values": dict(self.endpoint_values),
            "paired_effect": self.paired_effect,
            "paired_n": self.paired_n,
            "underpowered": self.underpowered,
            "disposition": self.disposition,
            "promotable_empirical": self.promotable_empirical,
            "holm_results": [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "raw_p_value": h.raw_p_value,
                    "rank": h.rank,
                    "threshold": h.threshold,
                    "adjusted_p_value": h.adjusted_p_value,
                    "rejected": h.rejected,
                }
                for h in self.holm_results
            ],
            "promotion_gate_pass": self.promotion_gate_pass,
            "rollback_gate_hit": self.rollback_gate_hit,
            "sealed_score_once_ok": self.sealed_score_once_ok,
        }

    def content_sha256(self) -> str:
        return digest_mapping(self.to_dict())


def observation_table_from_mapping(raw: Mapping[str, Any]) -> ObservationTable:
    if str(raw.get("schema") or "") != "observation_table/v1":
        raise CreditEngineError(
            f"observation table schema {raw.get('schema')!r} != observation_table/v1"
        )
    rows_raw = raw.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise CreditEngineError("observation_table requires non-empty rows")
    rows: list[ObservationRow] = []
    for item in rows_raw:
        if not isinstance(item, Mapping):
            raise CreditEngineError("observation row must be object")
        try:
            value = float(item["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CreditEngineError(f"invalid observation value: {exc}") from exc
        if not math.isfinite(value):
            raise CreditEngineError("observation value must be finite")
        direction = str(item.get("direction") or "increase")
        if direction not in {"increase", "decrease"}:
            raise CreditEngineError(f"invalid direction {direction!r}")
        rows.append(
            ObservationRow(
                arm_id=str(item["arm_id"]),
                seed=int(item["seed"]),
                example_id=str(item["example_id"]),
                metric_id=str(item["metric_id"]),
                value=value,
                direction=direction,
                split=str(item.get("split") or "search"),
            )
        )
    return ObservationTable(
        schema="observation_table/v1",
        version=str(raw.get("version") or "v1"),
        campaign_id=str(raw["campaign_id"]),
        experiment_id=str(raw["experiment_id"]),
        manifest_sha256=str(raw["manifest_sha256"]),
        locked_eval_manifest_sha256=(
            str(raw["locked_eval_manifest_sha256"])
            if raw.get("locked_eval_manifest_sha256")
            else None
        ),
        rows=tuple(rows),
    )


def analysis_plan_from_mapping(raw: Mapping[str, Any]) -> AnalysisPlan:
    if str(raw.get("schema") or "") != "analysis_plan/v1":
        raise CreditEngineError(
            f"analysis plan schema {raw.get('schema')!r} != analysis_plan/v1"
        )
    hyps = raw.get("holm_hypothesis_ids") or ()
    if not isinstance(hyps, (list, tuple)) or not hyps:
        raise CreditEngineError("analysis_plan requires holm_hypothesis_ids")
    direction = str(raw.get("direction") or "increase")
    if direction not in {"increase", "decrease"}:
        raise CreditEngineError(f"invalid plan direction {direction!r}")
    primary_endpoint_id = str(
        raw.get("primary_endpoint_id") or raw.get("primary_metric") or ""
    ).strip()
    if not primary_endpoint_id:
        raise CreditEngineError("analysis_plan requires primary_endpoint_id")
    return AnalysisPlan(
        schema="analysis_plan/v1",
        version=str(raw.get("version") or "v1"),
        primary_metric=str(raw["primary_metric"]),
        primary_endpoint_id=primary_endpoint_id,
        direction=direction,
        minimum_effect=float(raw.get("minimum_effect") or 0.0),
        control_arm_id=str(raw["control_arm_id"]),
        candidate_arm_id=str(raw["candidate_arm_id"]),
        holm_hypothesis_ids=tuple(str(h) for h in hyps),
        holm_alpha=float(raw.get("holm_alpha") or 0.05),
        seal_split=str(raw.get("seal_split") or "seal"),
        score_once=bool(raw.get("score_once", True)),
        underpowered_n_below=int(raw.get("underpowered_n_below") or 2),
    )


def credit_report_from_mapping(raw: Mapping[str, Any]) -> CreditReport:
    if str(raw.get("schema") or "") != "credit_report/v1":
        raise CreditEngineError(
            f"credit report schema {raw.get('schema')!r} != credit_report/v1"
        )
    holm_raw = raw.get("holm_results") or ()
    holm: list[HolmCreditRow] = []
    for item in holm_raw:
        if not isinstance(item, Mapping):
            raise CreditEngineError("holm row must be object")
        holm.append(
            HolmCreditRow(
                hypothesis_id=str(item["hypothesis_id"]),
                raw_p_value=float(item["raw_p_value"]),
                rank=int(item["rank"]),
                threshold=float(item["threshold"]),
                adjusted_p_value=float(item["adjusted_p_value"]),
                rejected=bool(item["rejected"]),
            )
        )
    endpoints = raw.get("endpoint_values") or {}
    if not isinstance(endpoints, Mapping):
        raise CreditEngineError("endpoint_values must be object")
    return CreditReport(
        schema="credit_report/v1",
        version=str(raw.get("version") or "v1"),
        observation_table_sha256=str(raw["observation_table_sha256"]),
        analysis_plan_sha256=str(raw["analysis_plan_sha256"]),
        endpoint_values={str(k): float(v) for k, v in endpoints.items()},
        paired_effect=float(raw["paired_effect"]),
        paired_n=int(raw["paired_n"]),
        underpowered=bool(raw["underpowered"]),
        disposition=str(raw["disposition"]),
        promotable_empirical=bool(raw["promotable_empirical"]),
        holm_results=tuple(holm),
        promotion_gate_pass=bool(raw["promotion_gate_pass"]),
        rollback_gate_hit=bool(raw["rollback_gate_hit"]),
        sealed_score_once_ok=bool(raw.get("sealed_score_once_ok", True)),
    )


def _signed_delta(candidate: float, control: float, direction: str) -> float:
    raw = candidate - control
    return raw if direction == "increase" else -raw


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _two_sided_p_from_signed_effects(effects: Sequence[float]) -> float:
    """Simple deterministic sign test approximation for paired effects.

    Uses a normal approximation to the binomial sign test (excluding zeros).
    Sufficient for hermetic unit authority; not a claim of optimal power.
    """

    nonzero = [e for e in effects if e != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = sum(1 for e in nonzero if e > 0)
    # Normal approx to Binomial(n, 0.5): z = (k - n/2) / sqrt(n/4)
    mean = n / 2.0
    sd = math.sqrt(n / 4.0)
    if sd == 0:
        return 1.0
    z = abs(positives - mean) / sd
    # Two-sided p ≈ 2 * (1 - Φ(z)); Φ approx via erfc
    p = math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, p))


def _holm_adjust(
    raw_p: Mapping[str, float], alpha: float
) -> tuple[HolmCreditRow, ...]:
    ordered = sorted(raw_p.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(ordered)
    rows: list[HolmCreditRow] = []
    running_max = 0.0
    for rank, (hyp, p) in enumerate(ordered, start=1):
        threshold = alpha / (m - rank + 1)
        adjusted = min(1.0, p * (m - rank + 1))
        running_max = max(running_max, adjusted)
        monotone = min(1.0, running_max)
        rows.append(
            HolmCreditRow(
                hypothesis_id=hyp,
                raw_p_value=p,
                rank=rank,
                threshold=threshold,
                adjusted_p_value=monotone,
                rejected=monotone <= alpha and p <= threshold,
            )
        )
    # Re-rank by original rank order for stable output? Keep Holm order (by p).
    return tuple(rows)


def compute_credit_report(
    table: ObservationTable,
    plan: AnalysisPlan,
    *,
    promotion_gate_threshold: float | None = None,
    promotion_gate_operator: str = "ge",
    rollback_gate_threshold: float | None = None,
    rollback_gate_operator: str = "lt",
    sealed_already_scored: bool = False,
) -> CreditReport:
    """Recompute decision-bearing stats from observation rows + locked plan."""

    if not table.rows:
        raise CreditEngineError("empty observation table")
    primary_rows = [
        r
        for r in table.rows
        if r.metric_id == plan.primary_metric
        and r.split not in {plan.seal_split}  # primary credit uses non-seal by default
    ]
    # If all rows are seal-only (ship path), allow seal rows for credit.
    if not primary_rows:
        primary_rows = [r for r in table.rows if r.metric_id == plan.primary_metric]
    if not primary_rows:
        raise CreditEngineError(
            f"no observations for primary_metric {plan.primary_metric!r}"
        )

    control = {
        (r.seed, r.example_id): r.value
        for r in primary_rows
        if r.arm_id == plan.control_arm_id
    }
    candidate = {
        (r.seed, r.example_id): r.value
        for r in primary_rows
        if r.arm_id == plan.candidate_arm_id
    }
    keys = sorted(set(control) & set(candidate))
    if not keys:
        raise CreditEngineError("no paired control/candidate observations")
    effects = [
        _signed_delta(candidate[k], control[k], plan.direction) for k in keys
    ]
    paired_effect = _mean(effects)
    paired_n = len(effects)
    underpowered = paired_n < plan.underpowered_n_below
    raw_p = _two_sided_p_from_signed_effects(effects)

    # Authoritative endpoint values are keyed by *manifest endpoint_id*, not
    # metric name. Gates and caller summaries bind to endpoint_id; the signed
    # paired effect is the decision-bearing quantity for the primary endpoint.
    endpoint_values: dict[str, float] = {
        plan.primary_endpoint_id: paired_effect,
    }

    # Holm over preregistered hypotheses: primary gets raw_p; extras get 1.0
    # unless they have dedicated rows (extension point).
    hyp_p = {hid: 1.0 for hid in plan.holm_hypothesis_ids}
    if plan.holm_hypothesis_ids:
        # Map first hypothesis to primary effect p-value
        hyp_p[plan.holm_hypothesis_ids[0]] = raw_p
    holm = _holm_adjust(hyp_p, plan.holm_alpha)

    def _gate(op: str, value: float, thr: float) -> bool:
        return {
            "ge": value >= thr,
            "gt": value > thr,
            "le": value <= thr,
            "lt": value < thr,
            "eq": value == thr,
        }.get(op, False)

    # Promotion uses paired_effect vs minimum_effect by default
    thr = (
        float(promotion_gate_threshold)
        if promotion_gate_threshold is not None
        else float(plan.minimum_effect)
    )
    promotion_pass = _gate(promotion_gate_operator, paired_effect, thr)
    rollback_hit = False
    if rollback_gate_threshold is not None:
        rollback_hit = _gate(
            rollback_gate_operator, paired_effect, float(rollback_gate_threshold)
        )

    primary_holm_sig = False
    if plan.holm_hypothesis_ids:
        primary_holm_sig = any(
            h.rejected and h.hypothesis_id == plan.holm_hypothesis_ids[0] for h in holm
        )
    else:
        primary_holm_sig = paired_effect > plan.minimum_effect

    sealed_ok = True
    if plan.score_once and sealed_already_scored:
        sealed_ok = False

    effect_clears = paired_effect > plan.minimum_effect
    # Empirical promotable: clear minimum effect, significant under Holm (or
    # effect-only when no family), not underpowered, not rollback, seal ok.
    promotable = bool(
        sealed_ok
        and not underpowered
        and not rollback_hit
        and effect_clears
        and primary_holm_sig
        and promotion_pass
    )
    if not sealed_ok or rollback_hit:
        disposition = "reject"
    elif underpowered and not primary_holm_sig:
        disposition = "inconclusive"
    elif underpowered and primary_holm_sig:
        # Significant-looking but underpowered → still inconclusive
        disposition = "inconclusive"
        promotable = False
    elif promotable:
        disposition = "promotable"
    else:
        disposition = "reject"

    return CreditReport(
        schema="credit_report/v1",
        version="v1",
        observation_table_sha256=table.content_sha256(),
        analysis_plan_sha256=plan.content_sha256(),
        endpoint_values=endpoint_values,
        paired_effect=paired_effect,
        paired_n=paired_n,
        underpowered=underpowered,
        disposition=disposition,
        promotable_empirical=promotable,
        holm_results=holm,
        promotion_gate_pass=promotion_pass and promotable,
        rollback_gate_hit=rollback_hit,
        sealed_score_once_ok=sealed_ok,
    )


def verify_credit_against_caller(
    report: CreditReport,
    *,
    caller_endpoints: Mapping[str, float] | None,
    caller_holm: Sequence[Mapping[str, Any]] | None,
    required_endpoint_ids: Sequence[str] | None = None,
    tolerance: float | None = None,
) -> tuple[str, ...]:
    """Return failures when caller summaries disagree with recomputed credit.

    Endpoint authority is the recomputed report (keyed by endpoint_id). Every
    required endpoint and every caller-supplied endpoint must match; invented
    values that only appear on the caller side fail closed.
    """

    tol = (
        float(tolerance)
        if tolerance is not None
        else float(load_credit_defaults().get("caller_stats_tolerance") or 1e-9)
    )
    failures: list[str] = []
    auth = dict(report.endpoint_values)
    required = set(required_endpoint_ids) if required_endpoint_ids is not None else set(auth)
    if caller_endpoints is not None:
        for eid in sorted(required):
            if eid not in auth:
                failures.append(f"credit_endpoint_missing:{eid}")
                continue
            if eid not in caller_endpoints:
                failures.append(f"caller_endpoint_missing:{eid}")
                continue
            try:
                claimed = float(caller_endpoints[eid])
            except (TypeError, ValueError):
                failures.append(f"caller_endpoint_non_numeric:{eid}")
                continue
            if abs(claimed - float(auth[eid])) > tol:
                failures.append(f"caller_endpoint_mismatch:{eid}")
        for key, raw_val in caller_endpoints.items():
            key_s = str(key)
            if key_s not in auth:
                # Caller invented a key the credit engine did not recompute.
                failures.append(f"caller_endpoint_unknown:{key_s}")
                continue
            if key_s in required:
                continue  # already compared above
            try:
                claimed = float(raw_val)
            except (TypeError, ValueError):
                failures.append(f"caller_endpoint_non_numeric:{key_s}")
                continue
            if abs(claimed - float(auth[key_s])) > tol:
                failures.append(f"caller_endpoint_mismatch:{key_s}")
        # Paired effect field (if caller still sends it) must match report.
        if "paired_effect" in caller_endpoints and "paired_effect" not in auth:
            try:
                claimed_pe = float(caller_endpoints["paired_effect"])
            except (TypeError, ValueError):
                failures.append("caller_endpoint_non_numeric:paired_effect")
            else:
                if abs(claimed_pe - report.paired_effect) > tol:
                    failures.append("caller_endpoint_mismatch:paired_effect")
    if caller_holm is not None:
        by_id = {h.hypothesis_id: h for h in report.holm_results}
        for row in caller_holm:
            if not isinstance(row, Mapping):
                failures.append("caller_holm_invalid_row")
                continue
            hid = str(row.get("hypothesis_id") or "")
            if hid not in by_id:
                failures.append(f"caller_holm_unknown:{hid}")
                continue
            true = by_id[hid]
            try:
                raw_p = float(row["raw_p_value"])
                adj = float(row["adjusted_p_value"])
            except (KeyError, TypeError, ValueError):
                failures.append(f"caller_holm_non_numeric:{hid}")
                continue
            if abs(raw_p - true.raw_p_value) > tol or abs(adj - true.adjusted_p_value) > tol:
                failures.append(f"caller_holm_mismatch:{hid}")
            if bool(row.get("rejected")) != true.rejected:
                failures.append(f"caller_holm_reject_mismatch:{hid}")
    return tuple(failures)


@dataclass
class SealedScoreLedger:
    """Tracks score-once sealed evaluation identities."""

    scored: set[str]

    def __init__(self) -> None:
        self.scored = set()

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "sealed_score_ledger/v1", "scored": sorted(self.scored)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SealedScoreLedger:
        led = cls()
        for item in raw.get("scored") or []:
            led.scored.add(str(item))
        return led

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SealedScoreLedger:
        if not path.is_file():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def assert_score_once(
    ledger: SealedScoreLedger,
    *,
    seal_identity: str,
    record: bool = True,
) -> None:
    """Deny a second score of the same sealed evaluation identity."""

    if seal_identity in ledger.scored:
        raise CreditEngineError(
            f"sealed score-once violation: identity already scored: {seal_identity}"
        )
    if record:
        ledger.scored.add(seal_identity)


def seal_identity_from_table(table: ObservationTable, plan: AnalysisPlan) -> str:
    """Stable seal identity from table digest + plan seal split + campaign."""

    payload = {
        "table_sha256": table.content_sha256(),
        "plan_sha256": plan.content_sha256(),
        "seal_split": plan.seal_split,
        "campaign_id": table.campaign_id,
        "experiment_id": table.experiment_id,
    }
    return digest_mapping(payload)


def load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CreditEngineError(f"missing artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CreditEngineError(f"corrupt artifact JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CreditEngineError(f"artifact root must be object: {path}")
    # Reject kind-only placeholders
    if set(raw.keys()) <= {"kind"} or (
        "kind" in raw and "schema" not in raw and "rows" not in raw and "endpoint_values" not in raw
    ):
        if "schema" not in raw:
            raise CreditEngineError(
                f"semantic validation failed for {path.name}: kind-only placeholder"
            )
    return raw
