"""Screening sample-size range: a certified answer to "what n should the
climb loop screen at?".

The continuous loop screened at a hard-coded ``screening_smoke_n: 3``, which
is below the paired sign-test decidability floor (minimum two-sided p = 0.25
at n=3 > alpha = 1/20; see ``autoresearch.power``). Every cycle then landed on
``fixture_insufficient_n`` — undecidable noise that can never rehold. This
module computes the range per cycle instead of hard-coding it:

* **Floor (theorem-backed, exact).** The least per-arm n whose sign-test
  p-floor reaches the policy alpha, searched against the
  ``bound.screening_n.decidability_lower.v1`` predicate; the search's
  soundness and minimality are proved in
  ``LeverProofLean.ScreeningSampleSize`` (``signTestFloorFrom_sound`` /
  ``signTestFloorFrom_minimal``).
* **Power floor (assumption-backed).** ``power.required_n_for_effect`` under
  the normal approximation when the policy declares a minimum effect and a
  paired SD *measured on the primary metric* is supplied
  (``lookup_paired_sd_for_metric``: expectations slot -> metric-keyed ledger
  -> tagged constant). Declared approximation, never theorem-labeled. An
  unmeasured SD yields ``power_floor_status="unmeasured"`` — never a floor
  borrowed from another metric's variance (RC1: ``smoke.structural_similarity``
  SD demanded n=96 for a ``smoke.eval_nll`` primary and parked the loop).
* **Ceilings.** Arm-wall budget (``bound.screening_n.budget_upper.v1``,
  proved: ``screeningBudgetUpperBound_fits``) and screening-suite volume —
  today the committed smoke suites carry 3 records, so suite volume binds as
  hard as the wall budget.

Verdicts:

* ``feasible`` — ``n_min <= min(budget, suite)``; ``chosen_n = n_min`` (the
  smallest sufficient n, mirroring the parameter-efficiency law: size is a
  budget, never a free knob).
* ``infeasible_range_empty`` — the floor clears neither ceiling;
  ``binding_constraints`` names ``wall_budget`` and/or ``suite_volume``.
  ``suite_volume`` is a generate-and-publish signal (``must_generate``), not a
  license to screen at an undecidable fallback n.
* ``insufficient_evidence`` — no decode-cost observation yet (``chosen_n``
  None; the caller keeps its configured fallback n), or the power SD is
  unmeasured while the exact floor is affordable (``chosen_n`` =
  ``min(suite, budget, max_candidate_n)``: the screen spends the affordable
  n as an advisory design). ``must_generate`` only when the suite is below
  the exact decidability floor.

Design rules (docs/design/screening-sample-size-bounds.md):

* **Never a gate.** Reports carry ``promotion_authority: false``; findings
  carry ``authority: climb_signal_not_gate``. Ship, admission, and promotion
  gates are untouched.
* **Exact vs approximate is labeled.** The sign-test floor and budget ceiling
  are theorem-backed exact arithmetic; the power floor is an explicitly
  declared normal approximation (``improve-lean-optimums`` discipline: Lean
  proves arithmetic, not calibration).
* **Fail closed.** Degenerate inputs produce infeasible/insufficient
  verdicts with reasons, never an exception. An empty range never authorizes
  screening at a known-undecidable n.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from slm_training.autoresearch.power import required_n_for_effect
from slm_training.autoresearch.schemas import StrictModel
from slm_training.formal.bound_ast import (
    BOUND_SCREENING_N_BUDGET_UPPER,
    BOUND_SCREENING_N_DECIDABILITY_LOWER,
    evaluate_bound,
    evaluate_predicate,
    resolve_bound_ast,
)
from slm_training.harness_core.lineage.records import content_sha

SCREENING_SAMPLE_SIZE_SCHEMA = "screening_sample_size/v1"

# Search cap for the exact sign-test floor. 64 pairs reaches min two-sided
# p ~ 1e-19, far past any usable alpha; the cap only bounds the search.
DEFAULT_MAX_CANDIDATE_N = 64
DEFAULT_ALPHA = Fraction(1, 20)

ScreeningSampleSizeVerdict = Literal[
    "insufficient_evidence",
    "feasible",
    "infeasible_range_empty",
]

FINDING_RANGE_EMPTY = "screening_n_range_empty"
FINDING_FLOOR_BEYOND_SEARCH_CAP = "screening_n_floor_beyond_search_cap"
FINDING_POWER_SD_UNMEASURED = "screening_power_sd_unmeasured"

# Provenance of the assumption-backed power floor. ``unmeasured`` means the
# policy declared a minimum effect but no paired SD exists *for that metric*;
# the exact decidability floor then stands alone (never a borrowed SD).
PowerFloorStatus = Literal["not_requested", "measured", "unmeasured"]
PairedSdSource = Literal[
    "policy",
    "metric_expectations",
    "evidence_ledger",
    "measured_constant",
    "unmeasured",
]


class ScreeningSampleSizeObservation(StrictModel):
    """Per-cycle observed inputs; every field is re-measured, never cached."""

    # Declared sign-test alpha (policy power_gate.alpha). int or "num/den".
    alpha: int | str | Fraction = DEFAULT_ALPHA
    max_candidate_n: int = Field(default=DEFAULT_MAX_CANDIDATE_N, ge=1)
    # Records in the resolved screening (smoke) suite.
    suite_records: int = Field(ge=0)
    # Integral second budgets (caller ceils fractional observations).
    arm_wall_seconds: int = Field(ge=0)
    min_train_floor_seconds: int = Field(default=0, ge=0)
    suite_overhead_seconds: int = Field(default=0, ge=0)
    # Conservative per-record decode floor in whole seconds. None = not yet
    # observed this cycle (budget ceiling undecidable -> insufficient_evidence).
    per_record_decode_floor_seconds: int | None = Field(default=None, ge=1)
    # Assumption-backed power floor inputs. ``minimum_effect`` alone means the
    # policy declares a target effect whose paired SD is *unmeasured* for the
    # primary metric (power floor reported as unmeasured, never substituted
    # from another metric). ``observed_sd`` without an effect is meaningless.
    minimum_effect: int | str | Fraction | None = None
    observed_sd: int | str | Fraction | None = None
    # Provenance of ``observed_sd`` (see ``lookup_paired_sd_for_metric``).
    observed_sd_source: str | None = None
    observed_sd_metric: str | None = None

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, value: Any) -> Fraction:
        alpha = Fraction(value)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        return alpha

    @model_validator(mode="after")
    def _paired_power_fields(self) -> ScreeningSampleSizeObservation:
        if self.observed_sd is not None and self.minimum_effect is None:
            raise ValueError("observed_sd requires a minimum_effect")
        if self.minimum_effect is not None and Fraction(self.minimum_effect) <= 0:
            raise ValueError("minimum_effect must be positive")
        if self.observed_sd is not None and Fraction(self.observed_sd) < 0:
            raise ValueError("observed_sd must be non-negative")
        return self


class BoundEvaluation(StrictModel):
    """One registry-backed bound evaluation, self-describing and re-checkable."""

    bound_ast_id: str
    env: dict[str, Any]
    value: str
    authority: Literal["theorem_backed_exact", "assumption_backed"]
    assumption: str


class ScreeningSampleSizeReport(StrictModel):
    schema_version: Literal["screening_sample_size/v1"] = SCREENING_SAMPLE_SIZE_SCHEMA
    alpha: str
    max_candidate_n: int
    # Exact sign-test attainability floor (None = beyond the search cap).
    decidability_floor_n: int | None
    # Normal-approximation power floor (None = effect/sd not supplied).
    power_floor_n: int | None
    n_min: int | None
    budget_ceiling_n: int | None
    suite_ceiling_n: int
    n_max: int | None
    chosen_n: int | None
    binding_constraints: tuple[Literal["wall_budget", "suite_volume"], ...]
    verdict: ScreeningSampleSizeVerdict
    bounds: tuple[BoundEvaluation, ...]
    findings: tuple[dict[str, Any], ...]
    # True when suite_volume binds: grow and persist smoke records, do not screen.
    must_generate: bool = False
    # Power-floor provenance: not_requested (no minimum_effect), measured
    # (same-metric paired SD supplied), unmeasured (effect declared, no
    # same-metric SD -> the exact decidability floor stands alone and the
    # verdict is advisory ``insufficient_evidence`` with the affordable n).
    power_floor_status: PowerFloorStatus = "not_requested"
    observed_sd_source: str | None = None
    observed_sd_metric: str | None = None
    # Climb verdicts recommend; they never gate admission or promotion.
    promotion_authority: Literal[False] = False

    def certificate_sha256(self) -> str:
        return content_sha(self.model_dump(mode="json"))


def _finding(code: str, evidence: dict[str, Any], suggestion: str) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "experiment_recommendation",
        "authority": "climb_signal_not_gate",
        "evidence": dict(evidence),
        "suggestion": suggestion,
    }


def _decidability_floor(alpha: Fraction, max_candidate_n: int) -> tuple[int | None, list[BoundEvaluation]]:
    """Least n in [1, max_candidate_n] whose sign-test p-floor reaches alpha.

    Searches the registry predicate ``2*alpha_den <= alpha_num*2^n`` upward;
    the soundness and minimality of exactly this search are proved in
    ``LeverProofLean.ScreeningSampleSize.signTestFloorFrom_*``. Records the
    first attaining candidate as the auditable bound evaluation.
    """

    doc = resolve_bound_ast(BOUND_SCREENING_N_DECIDABILITY_LOWER)
    if doc.predicate is None:
        raise ValueError(
            f"{BOUND_SCREENING_N_DECIDABILITY_LOWER} must carry a predicate"
        )
    bounds: list[BoundEvaluation] = []
    for n in range(1, max_candidate_n + 1):
        env = {
            "alpha_num": alpha.numerator,
            "alpha_den": alpha.denominator,
            "n_candidate": n,
            "two_pow_n_candidate": 2**n,
        }
        if evaluate_predicate(doc.predicate, variables=doc.variables, env=env):
            value = evaluate_bound(BOUND_SCREENING_N_DECIDABILITY_LOWER, env)
            bounds.append(
                BoundEvaluation(
                    bound_ast_id=BOUND_SCREENING_N_DECIDABILITY_LOWER,
                    env=env,
                    value=value.value,
                    authority="theorem_backed_exact",
                    assumption=(
                        "sign-test attainability combinatorics only; search "
                        "soundness/minimality proved by LeverProofLean."
                        "ScreeningSampleSize.signTestFloorFrom_sound/_minimal; "
                        "not an empirical power claim"
                    ),
                )
            )
            return n, bounds
    return None, bounds


def compute_screening_sample_size(
    observation: ScreeningSampleSizeObservation,
) -> ScreeningSampleSizeReport:
    """Evaluate the screening-n floor, ceilings, and the cycle verdict."""

    alpha = Fraction(observation.alpha)
    floor_n, bounds = _decidability_floor(alpha, observation.max_candidate_n)
    findings: list[dict[str, Any]] = []

    power_floor: int | None = None
    power_status: PowerFloorStatus = "not_requested"
    if observation.minimum_effect is not None and observation.observed_sd is None:
        # Effect declared, SD unmeasured for this metric: no power floor, and
        # never one borrowed from another metric's variance.
        power_status = "unmeasured"
        metric = observation.observed_sd_metric or "<primary metric>"
        findings.append(
            _finding(
                FINDING_POWER_SD_UNMEASURED,
                {
                    "minimum_effect": str(Fraction(observation.minimum_effect)),
                    "metric": observation.observed_sd_metric,
                    "observed_sd_source": observation.observed_sd_source
                    or "unmeasured",
                    "decidability_floor_n": floor_n,
                },
                f"measure the paired-delta SD of {metric} and record it under "
                "metric_expectations.screening.v1.json "
                "observed_paired_sd_by_metric[<metric leaf>]; until then the "
                "exact decidability floor stands alone and the screen spends "
                "the affordable n (advisory, never a borrowed power floor)",
            )
        )
    elif observation.minimum_effect is not None:
        power_status = "measured"
        power_floor = required_n_for_effect(
            float(Fraction(observation.minimum_effect)),
            float(Fraction(observation.observed_sd)),
            float(alpha),
            paired=True,
        )
        bounds.append(
            BoundEvaluation(
                bound_ast_id="power.required_n_for_effect",
                env={
                    "minimum_effect": str(Fraction(observation.minimum_effect)),
                    "observed_sd": str(Fraction(observation.observed_sd)),
                    "observed_sd_source": observation.observed_sd_source,
                    "observed_sd_metric": observation.observed_sd_metric,
                    "alpha": str(alpha),
                    "paired": True,
                },
                value=str(power_floor),
                authority="assumption_backed",
                assumption=(
                    "normal-approximation z-power over paired differences; "
                    "budgeting approximation declared by autoresearch.power, "
                    "not proved arithmetic"
                ),
            )
        )

    floors = [f for f in (floor_n, power_floor) if f is not None]
    n_min = max(floors) if floors else None

    if floor_n is None:
        findings.append(
            _finding(
                FINDING_FLOOR_BEYOND_SEARCH_CAP,
                {"alpha": str(alpha), "max_candidate_n": observation.max_candidate_n},
                "no n within the search cap attains rejection at alpha; raise "
                "max_candidate_n or revisit the declared alpha before climbing",
            )
        )

    # Budget ceiling (exact arithmetic over caller-declared second budgets).
    budget_ceiling: int | None = None
    if observation.per_record_decode_floor_seconds is not None:
        usable = max(
            0,
            observation.arm_wall_seconds
            - observation.min_train_floor_seconds
            - observation.suite_overhead_seconds,
        )
        env = {
            "usable_wall_seconds": usable,
            "min_decode_seconds": observation.per_record_decode_floor_seconds,
        }
        result = evaluate_bound(BOUND_SCREENING_N_BUDGET_UPPER, env)
        budget_ceiling = int(result.fraction)
        bounds.append(
            BoundEvaluation(
                bound_ast_id=BOUND_SCREENING_N_BUDGET_UPPER,
                env=env,
                value=result.value,
                authority="theorem_backed_exact",
                assumption=(
                    "caller-declared integral second budgets; fitness of every "
                    "n under the ceiling proved by LeverProofLean."
                    "ScreeningSampleSize.screeningBudgetUpperBound_fits"
                ),
            )
        )

    suite_ceiling = observation.suite_records

    verdict: ScreeningSampleSizeVerdict
    n_max: int | None = None
    chosen: int | None = None
    binding: list[Literal["wall_budget", "suite_volume"]] = []
    if budget_ceiling is None:
        verdict = "insufficient_evidence"
    elif n_min is None:
        # Only reachable when the floor search failed (findings already carry it).
        verdict = "infeasible_range_empty"
    else:
        n_max = min(budget_ceiling, suite_ceiling)
        if n_min <= n_max and power_status == "unmeasured":
            # The exact floor is affordable but power is unknown: spend the
            # affordable n (bounded by the search cap) as an advisory screen.
            verdict = "insufficient_evidence"
            chosen = min(n_max, observation.max_candidate_n)
        elif n_min <= n_max:
            verdict = "feasible"
            chosen = n_min
        else:
            verdict = "infeasible_range_empty"
            if budget_ceiling < n_min:
                binding.append("wall_budget")
            if suite_ceiling < n_min:
                binding.append("suite_volume")
            suggestions = []
            if "suite_volume" in binding:
                suggestions.append(
                    f"generate and persist {n_min - suite_ceiling} smoke "
                    f"records (floor is {n_min}); do not screen at {suite_ceiling}"
                )
            if "wall_budget" in binding:
                suggestions.append(
                    f"the arm wall affords {budget_ceiling} records at the "
                    "declared decode floor — cheaper per-record decode or a "
                    "larger stage share, never a silent wall++"
                )
            findings.append(
                _finding(
                    FINDING_RANGE_EMPTY,
                    {
                        "n_min": n_min,
                        "decidability_floor_n": floor_n,
                        "power_floor_n": power_floor,
                        "budget_ceiling_n": budget_ceiling,
                        "suite_ceiling_n": suite_ceiling,
                        "binding_constraints": binding,
                    },
                    "; ".join(suggestions),
                )
            )

    return ScreeningSampleSizeReport(
        alpha=str(alpha),
        max_candidate_n=observation.max_candidate_n,
        decidability_floor_n=floor_n,
        power_floor_n=power_floor,
        n_min=n_min,
        budget_ceiling_n=budget_ceiling,
        suite_ceiling_n=suite_ceiling,
        n_max=n_max,
        chosen_n=chosen,
        binding_constraints=tuple(binding),
        must_generate="suite_volume" in binding,
        verdict=verdict,
        bounds=tuple(bounds),
        findings=tuple(findings),
        power_floor_status=power_status,
        observed_sd_source=(
            observation.observed_sd_source
            if observation.minimum_effect is not None
            else None
        ),
        observed_sd_metric=(
            observation.observed_sd_metric
            if observation.minimum_effect is not None
            else None
        ),
    )


SCREENING_SMOKE6_EVAL_VERSION = "e938_role_safe_all_targets_smoke6_v1"
SCREENING_SMOKE24_EVAL_VERSION = "e938_role_safe_all_targets_smoke24_v1"
FROZEN_EVAL_SNAPSHOTS = frozenset(
    {
        "e938_role_safe_all_targets_v2",
        "e938_role_safe_all_targets_smoke6_v1",
    }
)
# Paired-delta SD of smoke.structural_similarity from continuous-openui-local
# 20260820 control vs candidate eval_smoke.json (n_deltas=405). Budgeting
# prior for power_floor_n; re-measured in docs/design/screening-power-analysis.
MEASURED_PAIRED_SD = 0.1741
#: The metric MEASURED_PAIRED_SD was measured on. It is evidence for that
#: metric only: a screening primary on any other metric (e.g. smoke.eval_nll)
#: is *unmeasured* until its own paired SD is recorded — never borrowed.
MEASURED_PAIRED_SD_METRIC = "smoke.structural_similarity"
MEASURED_PAIRED_SD_BY_METRIC: Mapping[str, float] = {
    MEASURED_PAIRED_SD_METRIC: MEASURED_PAIRED_SD,
}
TARGET_SMOKE_N = 24

_CLIMB_RESOURCE_DIR = (
    Path(__file__).resolve().parents[1] / "resources" / "experiments" / "autotrain_climb"
)
#: Per-metric paired-SD slot written by the screening power measurement:
#: ``observed_paired_sd_by_metric[<metric leaf or full metric>]`` holds a
#: number/string or an object with ``sd`` (plus provenance fields).
DEFAULT_SCREENING_EXPECTATIONS_PATH = (
    _CLIMB_RESOURCE_DIR / "metric_expectations.screening.v1.json"
)
OBSERVED_PAIRED_SD_BY_METRIC_KEY = "observed_paired_sd_by_metric"
DEFAULT_EVIDENCE_LEDGER_PATH = _CLIMB_RESOURCE_DIR / "evidence_ledger.v1.json"
#: The committed evidence ledger does not tag its deltas with a metric; they
#: accumulated under the v1 screening primary. A ledger declaring ``metric``
#: (or per-arm ``by_metric`` stats) overrides this legacy tag.
LEDGER_LEGACY_METRIC = "smoke.structural_similarity"
#: Minimum pooled degrees of freedom before a ledger SD counts as measured.
MIN_LEDGER_DOF = 20


def metric_leaf(metric: str) -> str:
    """``smoke.eval_nll`` -> ``eval_nll``; a bare leaf is returned unchanged."""

    return str(metric or "").rsplit(".", 1)[-1]


def _metric_matches(key: str, metric: str) -> bool:
    """A table key names ``metric`` when it is the full name or its leaf.

    ``held_out.eval_nll`` never matches ``smoke.eval_nll``: suites carry their
    own variance, and a leaf key is the only suite-agnostic spelling.
    """

    key = str(key or "").strip()
    metric = str(metric or "").strip()
    if not key or not metric:
        return False
    if key == metric or key == metric_leaf(metric):
        return True
    return "." not in metric and metric_leaf(key) == metric


def _sd_from_value(value: Any) -> Fraction | None:
    if isinstance(value, Mapping):
        for field in ("sd", "observed_sd", "paired_sd", "value"):
            if field in value:
                return _sd_from_value(value[field])
        return None
    if isinstance(value, bool) or value is None:
        return None
    try:
        sd = Fraction(str(value)) if isinstance(value, str) else Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    if sd < 0:
        return None
    return sd


def _lookup_sd_table(table: Any, metric: str) -> tuple[Fraction, str] | None:
    if not isinstance(table, Mapping):
        return None
    for key in (metric, metric_leaf(metric)):
        if key in table:
            sd = _sd_from_value(table[key])
            if sd is not None:
                return sd, str(key)
    return None


def expectations_paired_sd(
    metric: str, expectations_path: Path | str | None = None
) -> tuple[Fraction, dict[str, Any]] | None:
    """``observed_paired_sd_by_metric`` entry for ``metric`` (full or leaf key)."""

    path = Path(expectations_path or DEFAULT_SCREENING_EXPECTATIONS_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    hit = _lookup_sd_table(payload.get(OBSERVED_PAIRED_SD_BY_METRIC_KEY), metric)
    if hit is None:
        return None
    sd, key = hit
    return sd, {"path": path.name, "key": key}


def pooled_ledger_paired_sd(
    ledger_path: Path | str, metric: str | None = None
) -> tuple[float, int] | None:
    """Pooled SD of per-run deltas across arms: sqrt(sum m2 / sum (n-1)).

    With ``metric`` the ledger contributes only stats keyed to that metric:
    per-arm ``by_metric[<metric>]`` entries, or the arm-level deltas when the
    ledger's declared ``metric`` (legacy default: ``LEDGER_LEGACY_METRIC``)
    names it. Returns ``(sd, dof)`` or None when the ledger is missing,
    unreadable, foreign to the metric, or carries fewer than
    ``MIN_LEDGER_DOF`` degrees of freedom.
    """

    try:
        payload = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    arms = payload.get("arms")
    if not isinstance(arms, Mapping):
        return None
    declared = str(
        payload.get("metric") or payload.get("delta_metric") or LEDGER_LEGACY_METRIC
    )
    arm_level_applies = metric is None or _metric_matches(declared, metric)
    m2_total = 0.0
    dof = 0
    for arm in arms.values():
        if not isinstance(arm, Mapping):
            continue
        stats: Mapping[str, Any] | None = None
        if metric is not None and isinstance(arm.get("by_metric"), Mapping):
            for key, entry in arm["by_metric"].items():
                if _metric_matches(str(key), metric) and isinstance(entry, Mapping):
                    stats = entry
                    break
        if stats is None and arm_level_applies:
            stats = arm
        if stats is None:
            continue
        m2 = stats.get("m2_delta")
        n_delta = stats.get("n_delta")
        if isinstance(m2, bool) or isinstance(n_delta, bool):
            continue
        if not isinstance(m2, (int, float)) or not isinstance(n_delta, int):
            continue
        if not math.isfinite(float(m2)) or float(m2) < 0 or n_delta < 2:
            continue
        m2_total += float(m2)
        dof += n_delta - 1
    if dof < MIN_LEDGER_DOF:
        return None
    sd = math.sqrt(m2_total / dof)
    if not math.isfinite(sd) or sd <= 0:
        return None
    return sd, dof


class PairedSdLookup(StrictModel):
    """Same-metric paired-delta SD with provenance; ``unmeasured`` is honest."""

    metric: str
    metric_leaf: str
    observed_sd: str | None
    source: PairedSdSource
    measured: bool
    detail: dict[str, Any] = Field(default_factory=dict)


def lookup_paired_sd_for_metric(
    metric: str,
    *,
    expectations_path: Path | str | None = None,
    ledger_path: Path | str | None = None,
) -> PairedSdLookup:
    """Resolve the paired-delta SD for ``metric`` without borrowing.

    Order: ``metric_expectations.screening.v1.json``
    ``observed_paired_sd_by_metric`` -> evidence-ledger stats keyed to the
    metric -> the module's tagged measured constant -> ``unmeasured``.
    Never raises; every failure degrades to ``unmeasured``.
    """

    metric = str(metric or "").strip()
    leaf = metric_leaf(metric)
    if not metric:
        return PairedSdLookup(
            metric="", metric_leaf="", observed_sd=None, source="unmeasured",
            measured=False, detail={"reason": "empty metric"},
        )
    try:
        hit = expectations_paired_sd(metric, expectations_path)
        if hit is not None:
            sd, detail = hit
            return PairedSdLookup(
                metric=metric, metric_leaf=leaf, observed_sd=str(sd),
                source="metric_expectations", measured=True, detail=detail,
            )
        pooled = pooled_ledger_paired_sd(
            ledger_path or DEFAULT_EVIDENCE_LEDGER_PATH, metric
        )
        if pooled is not None:
            sd_f, dof = pooled
            return PairedSdLookup(
                metric=metric, metric_leaf=leaf,
                observed_sd=str(Fraction(sd_f).limit_denominator(10**9)),
                source="evidence_ledger", measured=True,
                detail={"dof": dof, "path": Path(ledger_path or DEFAULT_EVIDENCE_LEDGER_PATH).name},
            )
        constant = _lookup_sd_table(MEASURED_PAIRED_SD_BY_METRIC, metric)
        if constant is not None:
            sd, key = constant
            return PairedSdLookup(
                metric=metric, metric_leaf=leaf, observed_sd=str(sd),
                source="measured_constant", measured=True,
                detail={"key": key, "note": "docs/design/screening-power-analysis.md"},
            )
    except Exception as exc:  # noqa: BLE001 — lookup failure is not evidence
        return PairedSdLookup(
            metric=metric, metric_leaf=leaf, observed_sd=None,
            source="unmeasured", measured=False,
            detail={"reason": f"{type(exc).__name__}: {exc}"},
        )
    return PairedSdLookup(
        metric=metric, metric_leaf=leaf, observed_sd=None, source="unmeasured",
        measured=False,
        detail={"reason": f"no paired SD recorded for {metric} (leaf {leaf})"},
    )


class FrozenEvalSnapshotError(ValueError):
    """Publishing must not mutate a frozen eval snapshot directory."""


def assert_eval_publish_target_writable(dataset_id: str) -> None:
    """Refuse writes that would mutate a frozen e938_* snapshot."""

    if dataset_id in FROZEN_EVAL_SNAPSHOTS:
        raise FrozenEvalSnapshotError(
            f"refusing to mutate frozen eval snapshot {dataset_id!r}"
        )
    if dataset_id.startswith("e938_") and dataset_id in FROZEN_EVAL_SNAPSHOTS:
        raise FrozenEvalSnapshotError(
            f"refusing to mutate frozen eval snapshot {dataset_id!r}"
        )


# Deficit smoke records are sampled from the certified corpus, never from a
# hand-written tuple: harnesses.test_data.certified draws root-family
# validation-bucket records (RootFamilySplitPolicyV1 buckets 80-89), n-gram /
# structure decontaminated against the certified train bucket, stratified by
# source and seeded, and never returns an id the target suite already holds.
# The test bucket (90-99) is reserved for the locked held-out suite.
CERTIFIED_SMOKE_SPLITS: tuple[str, ...] = ("validation",)


def _policy_train_manifest() -> Path | None:
    """Manifest of the climb policy's default train dataset, when resolvable.

    The certified train bucket is always decontaminated against; the policy
    train set is added so a deficit record can never be a leak of whatever the
    climb currently trains on, even when that differs from the bucket.
    """

    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy
        from slm_training.data.store import DataStore

        train_version = str(load_climb_policy().defaults.get("train_version") or "")
        if not train_version:
            return None
        manifest = DataStore().resolve("train", train_version).path / "manifest.json"
    except Exception:  # noqa: BLE001 — unresolvable policy set adds nothing
        return None
    return manifest if manifest.is_file() else None


def extra_smoke_fixtures_for_deficit(
    existing_ids: set[str],
    need: int,
    *,
    seed: int = 0,
    extra_train_manifests: Iterable[Path | str] = (),
) -> list[dict[str, Any]]:
    """Return up to ``need`` unused certified smoke records as seed dicts.

    ``existing_ids`` is updated in place with the returned ids so repeated
    calls keep growing the suite without duplicates. Records are plain dicts
    in the ``test_seeds.jsonl`` shape (``split``/``meta.suite`` = smoke) so
    the driver can append them and rebuild with ``--source fixture``.
    Candidates are decontaminated against the certified train bucket, the
    climb policy's default train dataset, and ``extra_train_manifests``.
    """

    if need <= 0:
        return []
    from slm_training.harnesses.test_data.certified import (
        sample_certified_candidates,
    )

    manifests: list[Path | str] = list(extra_train_manifests)
    policy_manifest = _policy_train_manifest()
    if policy_manifest is not None:
        manifests.append(policy_manifest)
    sample = sample_certified_candidates(
        existing_ids=set(existing_ids),
        need=need,
        suite="smoke",
        splits=CERTIFIED_SMOKE_SPLITS,
        seed=seed,
        extra_train_manifests=manifests,
    )
    out: list[dict[str, Any]] = []
    for record in sample.records:
        if record.id in existing_ids:
            continue
        existing_ids.add(record.id)
        out.append(record.to_dict())
    return out


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MAX_CANDIDATE_N",
    "FINDING_FLOOR_BEYOND_SEARCH_CAP",
    "FINDING_RANGE_EMPTY",
    "SCREENING_SAMPLE_SIZE_SCHEMA",
    "BoundEvaluation",
    "ScreeningSampleSizeObservation",
    "ScreeningSampleSizeReport",
    "ScreeningSampleSizeVerdict",
    "compute_screening_sample_size",
    "SCREENING_SMOKE6_EVAL_VERSION",
    "SCREENING_SMOKE24_EVAL_VERSION",
    "FROZEN_EVAL_SNAPSHOTS",
    "FrozenEvalSnapshotError",
    "MEASURED_PAIRED_SD",
    "MEASURED_PAIRED_SD_METRIC",
    "MEASURED_PAIRED_SD_BY_METRIC",
    "DEFAULT_SCREENING_EXPECTATIONS_PATH",
    "DEFAULT_EVIDENCE_LEDGER_PATH",
    "OBSERVED_PAIRED_SD_BY_METRIC_KEY",
    "LEDGER_LEGACY_METRIC",
    "MIN_LEDGER_DOF",
    "FINDING_POWER_SD_UNMEASURED",
    "PowerFloorStatus",
    "PairedSdSource",
    "PairedSdLookup",
    "metric_leaf",
    "expectations_paired_sd",
    "pooled_ledger_paired_sd",
    "lookup_paired_sd_for_metric",
    "TARGET_SMOKE_N",
    "CERTIFIED_SMOKE_SPLITS",
    "assert_eval_publish_target_writable",
    "extra_smoke_fixtures_for_deficit",
]
