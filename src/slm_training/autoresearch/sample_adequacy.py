"""Sample-size adequacy: a self-recalibrating data-volume signal.

Answers, per cycle, from the current build's evidence: which components are
under-witnessed (generate more, *targeted*), is the corpus inside its
adequate band, or has measured marginal utility flattened (stop generating,
change trajectory)?

Design rules (docs/design/sample-size-adequacy-bounds.md):

* **Coverage drives generation, per component.** The trigger is the observed
  per-component witness deficit, and the compiled action is a targeted
  ``until_coverage`` rebuild with a raised component coverage minimum —
  never a blind global volume raise. The projected record floor
  (``bound.sample_size.coverage_lower.v1``, proved arithmetic in
  ``LeverProofLean.SampleAdequacy``) is reported for planning; the
  fail-closed build, not the projection, decides reachability.
* **Saturation requires measurement.** ``saturated_change_trajectory`` fires
  only on a measured flat marginal-gain classification from the data
  adequacy ladder (``harnesses.experiments.data_adequacy_ladder``) — never
  from the capacity prior alone. Exceeding the memorization-capacity prior
  is *expected* in the compression regime and is surfaced as a diagnostic
  (``memorization_regime_expected``), not a stop signal.
* **Capacity prior is diagnostic.** The Collins et al. 3-6
  task-bits/parameter interval (``bound.sample_size.capacity_upper.v1``,
  assumption-backed) is computed only when the caller deliberately supplies
  ``trainable_params`` and description bits; it never gates and never fires
  a verdict by itself.
* **Never a gate.** Reports recommend; admission, ship, and promotion gates
  are untouched, and capacity may only grow charged (``EG_params``).

Eval-side sample floors are owned by ``autoresearch.power`` and the
promotion suite floors; the ladder classification consumes them so a flat
verdict cannot be claimed from an undecidable eval.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from slm_training.formal.bound_ast import (
    BOUND_SAMPLE_SIZE_CAPACITY_UPPER,
    BOUND_SAMPLE_SIZE_COVERAGE_LOWER,
    evaluate_bound,
)
from slm_training.harness_core.lineage.records import content_sha

from slm_training.autoresearch.schemas import StrictModel

SAMPLE_ADEQUACY_SCHEMA = "sample_adequacy/v2"

# Collins et al. (ICLR 2017) report 3-6 task-information bits/parameter for
# trained recurrent architectures; Morris et al. independently estimate 3.6
# for GPT-style models. Assumption box, never a measured constant — the same
# prior src/slm_training/web/routes.py applies to grammar-memory sizing.
CAPACITY_PRIOR_BITS_PER_PARAM_LOW = Fraction(3)
CAPACITY_PRIOR_BITS_PER_PARAM_HIGH = Fraction(6)

# Witnesses each tracked component needs before its mapping is trainable at
# all; overridable per policy, floor of 1.
DEFAULT_WITNESSES_PER_COMPONENT = 4

AdequacyVerdict = Literal[
    "insufficient_evidence",
    "generate_more",
    "sufficient",
    "saturated_change_trajectory",
]

# Feedback vocabulary (mirrored in harnesses.train_data.feedback.FINDING_CODES).
FINDING_BELOW_FLOOR = "sample_size_below_coverage_floor"
FINDING_ABOVE_CEILING = "sample_size_above_capacity_ceiling"


class SampleAdequacyObservation(StrictModel):
    """Per-cycle observed inputs; every field is re-measured, never cached."""

    observed_records: int = Field(ge=0)
    # Per-component witness counts from the build's component histogram.
    component_witnesses: Mapping[str, int]
    witnesses_per_component: int = Field(
        default=DEFAULT_WITNESSES_PER_COMPONENT, ge=1
    )
    # Generator's reachable unique-root ceiling when known (e.g. the measured
    # 1781-candidate grid, docs/design/compiler-inverted-program-data.md).
    reachable_unique_roots: int | None = Field(default=None, ge=1)
    # Measured marginal-gain classification from the data adequacy ladder.
    # None = not measured this cycle; True = flat (saturation evidence);
    # False = still rising. Source names the ladder artifact for the audit
    # trail and is required whenever a classification is supplied.
    marginal_gain_flat: bool | None = None
    marginal_gain_source: str | None = None
    # Capacity diagnostic inputs — both or neither. Fraction spelled as int
    # or "num/den" string; exact arithmetic only.
    trainable_params: int | None = Field(default=None, ge=1)
    mean_example_description_bits: int | str | None = None

    @field_validator("component_witnesses")
    @classmethod
    def _non_negative_counts(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        clean: dict[str, int] = {}
        for name, count in value.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(
                    f"witness count for {name!r} must be a non-negative int"
                )
            clean[str(name)] = count
        return clean

    @model_validator(mode="after")
    def _paired_fields(self) -> SampleAdequacyObservation:
        if (self.trainable_params is None) != (
            self.mean_example_description_bits is None
        ):
            raise ValueError(
                "trainable_params and mean_example_description_bits are "
                "supplied together or not at all"
            )
        if self.mean_example_description_bits is not None and (
            Fraction(self.mean_example_description_bits) <= 0
        ):
            raise ValueError("mean_example_description_bits must be positive")
        if self.marginal_gain_flat is not None and not self.marginal_gain_source:
            raise ValueError(
                "a marginal-gain classification requires marginal_gain_source "
                "naming the ladder artifact it came from"
            )
        return self


class BoundEvaluation(StrictModel):
    """One registry-backed bound evaluation, self-describing and re-checkable."""

    bound_ast_id: str
    env: Mapping[str, Any]
    value: str
    authority: Literal["theorem_backed_projection", "assumption_backed"]
    assumption: str


class SampleAdequacyReport(StrictModel):
    schema_version: Literal["sample_adequacy/v2"] = SAMPLE_ADEQUACY_SCHEMA
    observed_records: int
    witnesses_per_component: int
    tracked_components: int
    # Components still needing witnesses: {component: witnesses missing}.
    coverage_deficits: Mapping[str, int]
    zero_witness_components: tuple[str, ...]
    min_witness_count: int | None
    # Projected records to close the scarcest deficit at the observed rate
    # (planning number; the fail-closed targeted build is authoritative).
    coverage_floor_records: int | None
    generator_ceiling_limits_floor: bool
    # Capacity diagnostics (None unless params were deliberately supplied).
    capacity_ceiling_records_low_prior: int | None
    capacity_ceiling_records_high_prior: int | None
    memorization_regime_expected: bool | None
    # Measured marginal-gain evidence consumed this cycle, if any.
    marginal_gain_flat: bool | None
    marginal_gain_source: str | None
    verdict: AdequacyVerdict
    recommended_records: int | None
    bounds: tuple[BoundEvaluation, ...]
    findings: tuple[Mapping[str, Any], ...]
    # Climb verdicts recommend; they never gate admission or promotion.
    promotion_authority: Literal[False] = False

    def certificate_sha256(self) -> str:
        return content_sha(self.model_dump(mode="json"))


def _finding(
    code: str, evidence: Mapping[str, Any], suggestion: str
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "experiment_recommendation",
        "authority": "climb_signal_not_gate",
        "evidence": dict(evidence),
        "suggestion": suggestion,
    }


def compute_sample_adequacy(
    observation: SampleAdequacyObservation,
) -> SampleAdequacyReport:
    """Evaluate coverage deficits, diagnostics, and the cycle verdict."""

    witnesses = dict(observation.component_witnesses)
    n = observation.observed_records
    k = observation.witnesses_per_component
    zero = tuple(sorted(name for name, count in witnesses.items() if count == 0))
    deficits = {
        name: k - count
        for name, count in sorted(witnesses.items())
        if count < k
    }
    positive = [count for count in witnesses.values() if count > 0]
    min_witness = min(positive) if positive else None

    bounds: list[BoundEvaluation] = []
    findings: list[dict[str, Any]] = []

    floor_records: int | None = None
    if n > 0 and min_witness is not None and min_witness < k:
        env = {
            "witnesses_target": k,
            "observed_records": n,
            "min_witness_count": min_witness,
        }
        result = evaluate_bound(BOUND_SAMPLE_SIZE_COVERAGE_LOWER, env)
        floor_records = int(result.fraction)
        bounds.append(
            BoundEvaluation(
                bound_ast_id=BOUND_SAMPLE_SIZE_COVERAGE_LOWER,
                env=env,
                value=result.value,
                authority="theorem_backed_projection",
                assumption=(
                    "synthesis keeps witnessing the scarcest component at its "
                    "observed rate; arithmetic proved by "
                    "LeverProofLean.SampleAdequacy.coverageLowerBound_covers"
                ),
            )
        )

    ceiling_low: int | None = None
    ceiling_high: int | None = None
    memorization_expected: bool | None = None
    if observation.trainable_params is not None:
        for prior in (
            CAPACITY_PRIOR_BITS_PER_PARAM_LOW,
            CAPACITY_PRIOR_BITS_PER_PARAM_HIGH,
        ):
            env = {
                "trainable_params": observation.trainable_params,
                "bits_per_param": str(prior),
                "bits_per_example": str(
                    Fraction(observation.mean_example_description_bits)
                ),
            }
            result = evaluate_bound(BOUND_SAMPLE_SIZE_CAPACITY_UPPER, env)
            value = int(result.fraction)
            if prior == CAPACITY_PRIOR_BITS_PER_PARAM_LOW:
                ceiling_low = value
            else:
                ceiling_high = value
            bounds.append(
                BoundEvaluation(
                    bound_ast_id=BOUND_SAMPLE_SIZE_CAPACITY_UPPER,
                    env=env,
                    value=result.value,
                    authority="assumption_backed",
                    assumption=(
                        f"capacity prior {prior} task-bits/parameter (Collins "
                        "et al. interval endpoint); diagnostic only — "
                        "exceeding memorization capacity is expected in the "
                        "compression regime and is not a stop signal; budget "
                        "arithmetic proved by LeverProofLean.SampleAdequacy."
                        "capacityUpperBound_within_budget"
                    ),
                )
            )
        memorization_expected = bool(ceiling_high is not None and n > ceiling_high)

    generator_limited = bool(
        floor_records is not None
        and observation.reachable_unique_roots is not None
        and floor_records > observation.reachable_unique_roots
    )

    verdict: AdequacyVerdict
    recommended: int | None = None
    if n == 0 or not witnesses:
        verdict = "insufficient_evidence"
    elif deficits:
        verdict = "generate_more"
        if floor_records is not None:
            recommended = floor_records
            if generator_limited:
                recommended = observation.reachable_unique_roots
        findings.append(
            _finding(
                FINDING_BELOW_FLOOR,
                {
                    "observed_records": n,
                    "coverage_deficits": deficits,
                    "zero_witness_components": list(zero),
                    "coverage_floor_records": floor_records,
                    "witnesses_per_component": k,
                    "generator_ceiling_limits_floor": generator_limited,
                    "reachable_unique_roots": observation.reachable_unique_roots,
                },
                "components are under-witnessed; rebuild with "
                "generation_mode=until_coverage and component_coverage_minimum "
                f"= {k} (targeted, fail-closed) — a failed targeted build, not "
                "a projection, is what closes this approach",
            )
        )
    elif observation.marginal_gain_flat is True:
        verdict = "saturated_change_trajectory"
        findings.append(
            _finding(
                FINDING_ABOVE_CEILING,
                {
                    "observed_records": n,
                    "marginal_gain_source": observation.marginal_gain_source,
                    "capacity_ceiling_records_high_prior": ceiling_high,
                },
                "measured marginal gain is flat at current coverage; more "
                "volume is waste — change trajectory (quality/coverage "
                "levers, or charged capacity growth via EG_params)",
            )
        )
    else:
        verdict = "sufficient"

    return SampleAdequacyReport(
        observed_records=n,
        witnesses_per_component=k,
        tracked_components=len(witnesses),
        coverage_deficits=deficits,
        zero_witness_components=zero,
        min_witness_count=min_witness,
        coverage_floor_records=floor_records,
        generator_ceiling_limits_floor=generator_limited,
        capacity_ceiling_records_low_prior=ceiling_low,
        capacity_ceiling_records_high_prior=ceiling_high,
        memorization_regime_expected=memorization_expected,
        marginal_gain_flat=observation.marginal_gain_flat,
        marginal_gain_source=observation.marginal_gain_source,
        verdict=verdict,
        recommended_records=recommended,
        bounds=tuple(bounds),
        findings=tuple(findings),
    )


def observation_from_train_stats(
    stats: Mapping[str, Any],
    *,
    witnesses_per_component: int = DEFAULT_WITNESSES_PER_COMPONENT,
    reachable_unique_roots: int | None = None,
    marginal_gain_flat: bool | None = None,
    marginal_gain_source: str | None = None,
) -> SampleAdequacyObservation:
    """Build an observation from a train build's ``stats.json`` payload.

    Uses ``record_count`` and ``component_histogram`` — the fields every
    build already emits. Capacity diagnostics stay off (no params supplied):
    the live loop consumes coverage evidence only.
    """

    histogram = stats.get("component_histogram")
    if not isinstance(histogram, Mapping):
        histogram = {}
    return SampleAdequacyObservation(
        observed_records=int(stats.get("record_count") or 0),
        component_witnesses={
            str(name): int(count) for name, count in histogram.items()
        },
        witnesses_per_component=witnesses_per_component,
        reachable_unique_roots=reachable_unique_roots,
        marginal_gain_flat=marginal_gain_flat,
        marginal_gain_source=marginal_gain_source,
    )


__all__ = [
    "CAPACITY_PRIOR_BITS_PER_PARAM_HIGH",
    "CAPACITY_PRIOR_BITS_PER_PARAM_LOW",
    "DEFAULT_WITNESSES_PER_COMPONENT",
    "FINDING_ABOVE_CEILING",
    "FINDING_BELOW_FLOOR",
    "SAMPLE_ADEQUACY_SCHEMA",
    "BoundEvaluation",
    "SampleAdequacyObservation",
    "SampleAdequacyReport",
    "compute_sample_adequacy",
    "observation_from_train_stats",
]
