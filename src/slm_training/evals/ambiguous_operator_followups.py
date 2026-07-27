"""SLM-418 (DSH5-10): ambiguous operator follow-up disposition (sixth slice).

Assembles the honest, fixture-scale disposition for the issue's own
"Measure action/operator/argument/reference/branch accuracy, unintended
mutations, correction/undo rate, calibration, trace replay, and CAP0/CAP1/CAP2
retention" bullet, by combining:

* :mod:`slm_training.dsl.operators.replay_preference_context_views` -- the
  five context-view / five turn-depth structural ablation grid.
* :mod:`slm_training.harnesses.preference.replay_preference_context_view_variants`
  -- the bounded, fixture-scale matched context-view comparison this
  disposition reports on.

This is deliberately a disposition, not a ship-readiness claim: the corpus is
small and synthetic, the trained "variant" per cell is a two-feature linear
scorer (never a DSH3 policy/control head), and ``held_out_benefit`` records
whichever of {benefit_observed_fixture_scale, no_benefit_fixture_scale} the
real run actually produced -- never a fabricated positive result. See
``docs/design/dsh5-10-replay-preference-rows.md`` for the full history and
what remains explicitly out of scope (real policy training, a powered
corpus, CAP0/CAP1/CAP2 retention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_training.dsl.operators.replay_preference_context_views import (
    build_ablation_report,
)
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    ReplayPreferenceContextViewComparisonReportV1,
    ReplayPreferenceSessionV1,
    synthesize_bounded_session_corpus,
    train_replay_preference_context_view_variants,
)

#: Metrics the issue's own bullet names that this slice does NOT measure, and
#: why -- printed verbatim into the report so a reader never has to guess.
#: ``held_out_benefit_statistical_power`` is corpus-size-dependent and is
#: filled in per-call by ``_out_of_scope_metrics`` below, never hardcoded,
#: since ``evaluate_ambiguous_operator_followups`` accepts an arbitrary
#: caller-supplied corpus (a test passes exactly two sessions).
OUT_OF_SCOPE_METRICS: dict[str, str] = {
    "real_action_operator_argument_accuracy": (
        "measured only via a two-feature diagnostic linear scorer's pairwise "
        "accuracy, never a trained DSH3 policy/control head; not a real "
        "action/operator/argument accuracy claim"
    ),
    "unintended_mutations": (
        "structurally inapplicable -- this slice only scores existing legal "
        "candidates and never calls OperatorLibraryV1.apply"
    ),
    "cap0_cap1_cap2_retention": (
        "requires the full CAP-gated eval suite integrated with a trained "
        "policy checkpoint; out of scope for a fixture-scale diagnostic scorer"
    ),
}


def _out_of_scope_metrics(comparison: ReplayPreferenceContextViewComparisonReportV1) -> dict[str, str]:
    """``OUT_OF_SCOPE_METRICS`` plus a corpus-size-accurate statistical-power note."""
    return {
        **OUT_OF_SCOPE_METRICS,
        "held_out_benefit_statistical_power": (
            f"the corpus supplied to this run is {comparison.row_count} row(s) "
            f"across {comparison.session_count} session(s); any verdict below "
            "is fixture-scale wiring evidence, never a powered or certified "
            "held-out-benefit claim"
        ),
    }


@dataclass(frozen=True)
class AmbiguousOperatorFollowupEvalReportV1:
    """The sixth-slice SLM-418 disposition: real numbers, honestly scoped."""

    comparison: ReplayPreferenceContextViewComparisonReportV1
    ablation_cell_counts: dict
    out_of_scope_metrics: dict
    version_stamp: dict
    schema: str = "ambiguous_operator_followup_eval_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "comparison": self.comparison.to_dict(),
            "ablation_cell_counts": dict(self.ablation_cell_counts),
            "out_of_scope_metrics": dict(self.out_of_scope_metrics),
            "version_stamp": self.version_stamp,
        }


def evaluate_ambiguous_operator_followups(
    sessions: tuple[ReplayPreferenceSessionV1, ...] | None = None,
) -> AmbiguousOperatorFollowupEvalReportV1:
    """Run the bounded, fixture-scale comparison and assemble the disposition.

    ``sessions`` defaults to :func:`synthesize_bounded_session_corpus`'s
    deterministic synthetic corpus; a caller may pass a different corpus (a
    test does, to exercise this on a smaller fixture) as long as it obeys the
    same train/held-out-by-group split contract.
    """
    if sessions is None:
        sessions = synthesize_bounded_session_corpus()

    comparison = train_replay_preference_context_view_variants(sessions)

    ablation_cell_counts: dict[str, int] = {}
    for session in sessions:
        ablation = build_ablation_report(
            session.report, state_lookup=session.state_lookup, receipts=session.receipts
        )
        for cell in ablation.cells:
            key = f"{cell.view.value}:{cell.turn_depth}"
            ablation_cell_counts[key] = ablation_cell_counts.get(key, 0) + 1

    return AmbiguousOperatorFollowupEvalReportV1(
        comparison=comparison,
        ablation_cell_counts=ablation_cell_counts,
        out_of_scope_metrics=_out_of_scope_metrics(comparison),
        version_stamp=build_version_stamp(
            "harness.preference.replay_preference_context_view_variants"
        ),
    )
