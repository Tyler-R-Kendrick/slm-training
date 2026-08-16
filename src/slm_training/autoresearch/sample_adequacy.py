"""Sample-size adequacy bounds: a self-recalibrating data-volume signal.

Answers, per cycle, from the current build's evidence and the current
architecture's parameter budget: does the training corpus need more records,
is it inside its adequate band, or has volume passed what the parameter
budget can absorb (change trajectory instead of generating more)?

Two bounds frame the interval, both evaluated through the EVID-04 bound AST
registry so the arithmetic is exact, digest-pinned, and Lean-mirrored
(``LeverProofLean.SampleAdequacy``):

* ``bound.sample_size.coverage_lower.v1`` — records needed for every tracked
  component to accumulate the witness target, projecting the scarcest
  component's observed witness rate. The arithmetic is proved
  (``coverageLowerBound_covers``); the rate-persistence projection is a
  declared assumption, restated on every report.
* ``bound.sample_size.capacity_upper.v1`` — records whose total description
  length fits the parameter budget under the Collins et al. 3-6
  task-bits/parameter capacity prior (assumption-backed interval; the same
  prior the dashboard uses for grammar-memory sizing, never a
  generalization claim).

The bounds recalibrate automatically: every input (witness counts, record
count, description bits, trainable parameters) is re-observed each cycle,
so improving the model or the synthesis distribution moves the band. A
verdict is a climb signal, never a gate — it may schedule a data rebuild or
recommend a trajectory change, and it may not weaken any admission gate or
buy capacity without charging ``EG_params`` (decode-invariants VI).

Eval-side sample floors are owned elsewhere and only referenced here:
``autoresearch.power.required_n_for_effect`` (statistical decidability) and
the promotion suite floors. This module bounds the *training* corpus.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Literal

from pydantic import Field, field_validator

from slm_training.formal.bound_ast import (
    BOUND_SAMPLE_SIZE_CAPACITY_UPPER,
    BOUND_SAMPLE_SIZE_COVERAGE_LOWER,
    evaluate_bound,
)
from slm_training.harness_core.lineage.records import content_sha

from slm_training.autoresearch.schemas import StrictModel

SAMPLE_ADEQUACY_SCHEMA = "sample_adequacy/v1"

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
    "coverage_gap_blocked",
    "generate_more",
    "sufficient",
    "saturated_change_trajectory",
    "conflict_change_trajectory",
]

# Feedback vocabulary (mirrored in harnesses.train_data.feedback.FINDING_CODES).
FINDING_BELOW_FLOOR = "sample_size_below_coverage_floor"
FINDING_ABOVE_CEILING = "sample_size_above_capacity_ceiling"
FINDING_COVERAGE_GAP = "coverage_cell_gap"


def _fraction(value: int | str | Fraction) -> Fraction:
    return Fraction(value)


class SampleAdequacyObservation(StrictModel):
    """Per-cycle observed inputs; every field is re-measured, never cached."""

    observed_records: int = Field(ge=0)
    component_witnesses: Mapping[str, int]
    trainable_params: int = Field(ge=1)
    # Fraction spelled as int or "num/den" string; exact arithmetic only.
    mean_example_description_bits: int | str
    witnesses_per_component: int = Field(
        default=DEFAULT_WITNESSES_PER_COMPONENT, ge=1
    )
    # Generator's reachable unique-root ceiling when known (e.g. the measured
    # 1781-candidate grid, docs/design/compiler-inverted-program-data.md).
    reachable_unique_roots: int | None = Field(default=None, ge=1)

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

    @field_validator("mean_example_description_bits")
    @classmethod
    def _positive_bits(cls, value: int | str) -> int | str:
        if _fraction(value) <= 0:
            raise ValueError("mean_example_description_bits must be positive")
        return value


class BoundEvaluation(StrictModel):
    """One registry-backed bound evaluation, self-describing and re-checkable."""

    bound_ast_id: str
    env: Mapping[str, Any]
    value: str
    authority: Literal["theorem_backed_projection", "assumption_backed"]
    assumption: str


class SampleAdequacyReport(StrictModel):
    schema_version: Literal["sample_adequacy/v1"] = SAMPLE_ADEQUACY_SCHEMA
    observed_records: int
    witnesses_per_component: int
    tracked_components: int
    zero_witness_components: tuple[str, ...]
    min_witness_count: int | None
    coverage_floor_records: int | None
    capacity_ceiling_records_low_prior: int | None
    capacity_ceiling_records_high_prior: int | None
    generator_ceiling_limits_floor: bool
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
    """Evaluate the [coverage-floor, capacity-ceiling] band and verdict."""

    witnesses = dict(observation.component_witnesses)
    zero = tuple(sorted(name for name, count in witnesses.items() if count == 0))
    positive = [count for count in witnesses.values() if count > 0]
    n = observation.observed_records
    k = observation.witnesses_per_component

    bounds: list[BoundEvaluation] = []
    findings: list[dict[str, Any]] = []

    min_witness = min(positive) if positive else None
    floor_records: int | None = None
    if n > 0 and min_witness is not None:
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
    if n >= 0:
        for prior, label in (
            (CAPACITY_PRIOR_BITS_PER_PARAM_LOW, "low"),
            (CAPACITY_PRIOR_BITS_PER_PARAM_HIGH, "high"),
        ):
            env = {
                "trainable_params": observation.trainable_params,
                "bits_per_param": str(prior),
                "bits_per_example": str(
                    _fraction(observation.mean_example_description_bits)
                ),
            }
            result = evaluate_bound(BOUND_SAMPLE_SIZE_CAPACITY_UPPER, env)
            value = int(result.fraction)
            if label == "low":
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
                        f"capacity prior {prior} task-bits/parameter "
                        "(Collins et al. interval endpoint); budget arithmetic "
                        "proved by LeverProofLean.SampleAdequacy."
                        "capacityUpperBound_within_budget"
                    ),
                )
            )

    generator_limited = bool(
        floor_records is not None
        and observation.reachable_unique_roots is not None
        and floor_records > observation.reachable_unique_roots
    )

    verdict: AdequacyVerdict
    recommended: int | None = None
    if n == 0 or min_witness is None:
        verdict = "insufficient_evidence"
    elif zero:
        # Volume alone can never witness a zero-rate component; the
        # distribution (generator/coverage), not the count, must change.
        verdict = "coverage_gap_blocked"
        findings.append(
            _finding(
                FINDING_COVERAGE_GAP,
                {"zero_witness_components": list(zero)},
                "widen the generator/coverage distribution; more records at "
                "the observed rate cannot witness these components",
            )
        )
    elif floor_records is not None and ceiling_high is not None and (
        floor_records > ceiling_high
    ):
        verdict = "conflict_change_trajectory"
        findings.append(
            _finding(
                FINDING_ABOVE_CEILING,
                {
                    "coverage_floor_records": floor_records,
                    "capacity_ceiling_records_high_prior": ceiling_high,
                    "trainable_params": observation.trainable_params,
                },
                "coverage demand exceeds what the parameter budget can absorb "
                "even at the generous prior endpoint; change trajectory — "
                "charge capacity growth via EG_params, shrink per-example "
                "description length, or reduce the tracked-coverage target",
            )
        )
    elif floor_records is not None and n < floor_records:
        verdict = "generate_more"
        recommended = floor_records
        if generator_limited:
            recommended = observation.reachable_unique_roots
        findings.append(
            _finding(
                FINDING_BELOW_FLOOR,
                {
                    "observed_records": n,
                    "coverage_floor_records": floor_records,
                    "min_witness_count": min_witness,
                    "witnesses_per_component": k,
                    "generator_ceiling_limits_floor": generator_limited,
                    "reachable_unique_roots": observation.reachable_unique_roots,
                },
                "corpus is below the projected coverage floor; rebuild with a "
                "higher unique-root target"
                + (
                    " (floor exceeds the generator's reachable grid — widening "
                    "the generator is the successor approach, see "
                    "docs/design/compiler-inverted-program-data.md)"
                    if generator_limited
                    else ""
                ),
            )
        )
    elif ceiling_high is not None and n > ceiling_high:
        verdict = "saturated_change_trajectory"
        findings.append(
            _finding(
                FINDING_ABOVE_CEILING,
                {
                    "observed_records": n,
                    "capacity_ceiling_records_high_prior": ceiling_high,
                    "trainable_params": observation.trainable_params,
                },
                "corpus already exceeds what the parameter budget can absorb "
                "at the generous prior endpoint; generating more is waste — "
                "change trajectory (quality/coverage levers, or charged "
                "capacity growth via EG_params)",
            )
        )
    else:
        verdict = "sufficient"

    return SampleAdequacyReport(
        observed_records=n,
        witnesses_per_component=k,
        tracked_components=len(witnesses),
        zero_witness_components=zero,
        min_witness_count=min_witness,
        coverage_floor_records=floor_records,
        capacity_ceiling_records_low_prior=ceiling_low,
        capacity_ceiling_records_high_prior=ceiling_high,
        generator_ceiling_limits_floor=generator_limited,
        verdict=verdict,
        recommended_records=recommended,
        bounds=tuple(bounds),
        findings=tuple(findings),
    )


def mean_description_bits(total_bytes: int, records: int) -> Fraction:
    """Observed mean per-record description length: serialized bytes * 8 / n."""

    if records <= 0:
        raise ValueError("records must be positive")
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    return Fraction(total_bytes * 8, records)


__all__ = [
    "CAPACITY_PRIOR_BITS_PER_PARAM_HIGH",
    "CAPACITY_PRIOR_BITS_PER_PARAM_LOW",
    "DEFAULT_WITNESSES_PER_COMPONENT",
    "FINDING_ABOVE_CEILING",
    "FINDING_BELOW_FLOOR",
    "FINDING_COVERAGE_GAP",
    "SAMPLE_ADEQUACY_SCHEMA",
    "BoundEvaluation",
    "SampleAdequacyObservation",
    "SampleAdequacyReport",
    "compute_sample_adequacy",
    "mean_description_bits",
]
