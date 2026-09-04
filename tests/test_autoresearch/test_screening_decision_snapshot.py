"""Characterization pin for the screening sample-size decision.

This is the surface where the loop decides *whether it can measure anything*,
and it is where the loop's worst failure lived: it returned n=0 for 543 cycles
and parked, so no arm ever ran. The defect was not in any single branch -- each
one was individually defensible -- it was in the payload the branches produced
together. So the whole payload is pinned, for a matrix that spans both empty-
range causes and both feasible ones.

Snapshots are the repo's external test-case resources (the local equivalent of
a Verify-style approval file): the expectation lives in
``src/slm_training/resources/test_cases/...json`` beside this module, and
``python -m scripts.refresh_test_cases <this file>`` re-records it. Blind
re-recording is the failure mode of every approval-test system, so each case
also carries an ``invariant`` -- a subset that must hold in the *actual* value
before the snapshot is even compared, and that a refresh copies through
unchanged. The invariants here are the laws the recovery established:

* ``parks_at_zero`` false wherever a decidable screen is affordable, so the
  n=0 park can never come back through a refreshed snapshot;
* ``asks_for_data_only_when_data_is_short`` -- ``must_generate`` may only be
  true when the suite is what binds, because no volume of records clears a
  wall;
* ``never_screens_below_the_decidability_floor`` -- a chosen n is either at or
  above the exact sign-test floor, or the decision refuses outright.

A refresh can update every number in the payload. It cannot make any of those
three false without the test failing on the invariant, which is the property
an approval file alone does not give you.
"""

from __future__ import annotations

from typing import Any

import pytest

from slm_training.autoresearch.climb_policy import (
    load_climb_policy,
    screening_smoke_n_for_policy,
)
from tests.casefiles import snapshot_cases

pytestmark = pytest.mark.snapshot

CASES = snapshot_cases(__file__, "screening_decisions")


def _decide(inputs: dict[str, Any]) -> dict[str, Any]:
    """The full decision payload plus the laws derived from it."""
    chosen, report = screening_smoke_n_for_policy(
        load_climb_policy(),
        arm_wall_seconds=inputs["arm_wall_seconds"],
        suite_records=inputs["suite_records"],
        per_record_decode_floor_seconds=inputs.get("per_record_decode_floor_seconds"),
    )
    payload: dict[str, Any] = dict(report or {})
    binding = tuple(payload.get("binding_constraints") or ())
    floor = payload.get("decidability_floor_n")
    return {
        "chosen_n": int(chosen),
        "report": payload,
        "laws": {
            "parks_at_zero": int(chosen) <= 0,
            "asks_for_data_only_when_data_is_short": (
                not payload.get("must_generate") or "suite_volume" in binding
            ),
            "never_screens_below_the_decidability_floor": (
                int(chosen) <= 0 or floor is None or int(chosen) >= int(floor)
            ),
            "promotion_authority": bool(payload.get("promotion_authority")),
        },
    }


def test_decisions_match_the_recorded_snapshot() -> None:
    for case in CASES:
        case.expect(_decide(case.input))


def test_every_case_is_reproducible() -> None:
    """A decision that varies run to run cannot be a snapshot at all."""
    for case in CASES:
        assert _decide(case.input) == _decide(case.input), case.id


def test_the_matrix_covers_both_empty_range_causes() -> None:
    """Guard the fixture itself: a matrix that lost its hard cases proves little.

    Both ``suite_volume`` and ``wall_budget`` must appear, or the invariants
    about them are vacuous on every case.
    """
    binding = {
        constraint
        for case in CASES
        for constraint in (case.expected["report"].get("binding_constraints") or ())
    }
    assert {"suite_volume", "wall_budget"} <= binding

    verdicts = {case.expected["report"].get("verdict") for case in CASES}
    assert "infeasible_range_empty" in verdicts
    assert verdicts - {"infeasible_range_empty"}, "no feasible case in the matrix"
