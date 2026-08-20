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
  the normal approximation when the policy declares a minimum effect and an
  observed SD is supplied. Declared approximation, never theorem-labeled.
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
* ``insufficient_evidence`` — no decode-cost observation yet; the caller
  keeps its configured fallback n and the report is advisory only.

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

from fractions import Fraction
from typing import Any, Literal

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
    # Assumption-backed power floor inputs — both or neither.
    minimum_effect: int | str | Fraction | None = None
    observed_sd: int | str | Fraction | None = None

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, value: Any) -> Fraction:
        alpha = Fraction(value)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        return alpha

    @model_validator(mode="after")
    def _paired_power_fields(self) -> ScreeningSampleSizeObservation:
        if (self.minimum_effect is None) != (self.observed_sd is None):
            raise ValueError(
                "minimum_effect and observed_sd are supplied together or not at all"
            )
        if self.minimum_effect is not None:
            if Fraction(self.minimum_effect) <= 0:
                raise ValueError("minimum_effect must be positive")
            if Fraction(self.observed_sd) < 0:
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

    power_floor: int | None = None
    if observation.minimum_effect is not None:
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

    findings: list[dict[str, Any]] = []
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
        if n_min <= n_max:
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
    )


SCREENING_SMOKE6_EVAL_VERSION = "e938_role_safe_all_targets_smoke6_v1"

# Extra smoke programs (I9 grammar + placeholders only). Appended when Lean n
# exceeds the committed smoke count; never duplicates of the original 3.
_EXTRA_SMOKE_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "smoke_tabs_01",
        "prompt": "Two-tab panel with overview and details placeholders.",
        "openui": (
            "root = Stack([panel], \"column\")\n"
            "overview = TextContent(\":smoke.tabs.overview\")\n"
            "details = TextContent(\":smoke.tabs.details\")\n"
            "tab1 = TabItem(\"$0\", \":smoke.tabs.tab1\", [overview])\n"
            "tab2 = TabItem(\"$1\", \":smoke.tabs.tab2\", [details])\n"
            "panel = Tabs([tab1, tab2])"
        ),
        "placeholders": [
            ":smoke.tabs.overview",
            ":smoke.tabs.details",
            ":smoke.tabs.tab1",
            ":smoke.tabs.tab2",
        ],
        "split": "smoke",
        "source": "fixture",
        "meta": {"suite": "smoke"},
    },
    {
        "id": "smoke_form_01",
        "prompt": "Email field with label and submit button.",
        "openui": (
            "root = Stack([field, actions], \"column\")\n"
            "email = Input(\"$0\", \":smoke.form.email\", \"email\")\n"
            "field = FormControl(\":smoke.form.email.label\", email)\n"
            "submit = Button(\":smoke.form.submit\")\n"
            "actions = Buttons([submit])"
        ),
        "placeholders": [
            ":smoke.form.email",
            ":smoke.form.email.label",
            ":smoke.form.submit",
        ],
        "split": "smoke",
        "source": "fixture",
        "meta": {"suite": "smoke"},
    },
    {
        "id": "smoke_switch_01",
        "prompt": "Settings switch with a supporting note.",
        "openui": (
            "root = Stack([notify, note], \"column\")\n"
            "notify = SwitchItem(\":smoke.settings.notify\", "
            "\":smoke.settings.notify.desc\", \"$0\")\n"
            "note = Callout(\"info\", \":smoke.settings.hint.title\", "
            "\":smoke.settings.hint.body\")"
        ),
        "placeholders": [
            ":smoke.settings.notify",
            ":smoke.settings.notify.desc",
            ":smoke.settings.hint.title",
            ":smoke.settings.hint.body",
        ],
        "split": "smoke",
        "source": "fixture",
        "meta": {"suite": "smoke"},
    },
)


def extra_smoke_fixtures_for_deficit(
    *, existing_ids: set[str], need: int
) -> list[dict[str, Any]]:
    """Return unused extra smoke fixtures, up to ``need`` records."""

    out: list[dict[str, Any]] = []
    for fixture in _EXTRA_SMOKE_FIXTURES:
        if need <= 0:
            break
        fid = str(fixture["id"])
        if fid in existing_ids:
            continue
        out.append(dict(fixture))
        existing_ids.add(fid)
        need -= 1
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
    "extra_smoke_fixtures_for_deficit",
]
