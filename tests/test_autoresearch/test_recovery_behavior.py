"""Behavioral scenarios for the continuous loop's recovery invariants.

Written as given/when/then over the loop's *observable* behaviour rather than
over any one function's return value, because every defect these cover was
invisible at unit level: each individual call did what its own docstring said
while the loop as a whole reported progress it had not made.

Scope is deliberately the seams between components -- what the park writes for
the next owner, what admission does with a record the trainer would refuse,
what the decode lane reports about a substituted program. Each scenario names
the observation that motivated it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.behavior

REPO_ROOT = Path(__file__).resolve().parents[2]


def _driver() -> Any:
    """Import the continuous driver by path (it is a script, not a package)."""
    if "run_autotrain_continuous" in sys.modules:
        return sys.modules["run_autotrain_continuous"]
    path = REPO_ROOT / "scripts" / "run_autotrain_continuous.py"
    spec = importlib.util.spec_from_file_location("run_autotrain_continuous", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_autotrain_continuous"] = module
    spec.loader.exec_module(module)
    return module


def _handoff_root(tmp_path: Path, campaign_id: str) -> Path:
    """A campaign directory carrying the typed predecessor handoff the park needs."""
    from slm_training.autoresearch.schemas import (
        AutotrainActionV1,
        AutotrainCycleHandoffV1,
    )

    campaign_dir = tmp_path / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    handoff = AutotrainCycleHandoffV1(
        loop_id="behavior-loop",
        campaign_id=campaign_id,
        cycle_index=1,
        upstream_commit="a" * 40,
        integration_commit="b" * 40,
        cycle_role="screening",
        cycle_intent="screening",
        evidence_class="fixture",
        climb_state="rejected",
        ship_state="blocked",
        primary_metric="smoke.eval_nll",
        actions=(
            AutotrainActionV1(
                kind="next_experiment",
                owner="autotrain",
                reason="predecessor handoff the park rewrites",
                evidence_ids=(f"campaign:{campaign_id}",),
            ),
        ),
    )
    (campaign_dir / "cycle_handoff.json").write_text(
        handoff.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return campaign_dir


def _park(tmp_path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the n-deficit park and return the actions it queued."""
    campaign_id = "fixture-campaign"
    campaign_dir = _handoff_root(tmp_path, campaign_id)
    _driver()._park_screening_n_deficit(
        root=tmp_path,
        loop_id="behavior-loop",
        campaign_id=campaign_id,
        cycle_index=0,
        report=report,
    )
    written = json.loads(
        (campaign_dir / "cycle_handoff.json").read_text(encoding="utf-8")
    )
    return list(written["actions"])


# ---------------------------------------------------------------------------
# Scenario: a park must ask the owner who can actually clear the block.
# ---------------------------------------------------------------------------


def test_a_suite_volume_deficit_asks_for_data(tmp_path: Path) -> None:
    """Given the published suite is below the decidability floor,
    when screening parks,
    then the handoff asks the synthesis owner to generate records.
    """
    actions = _park(
        tmp_path,
        {
            "n_min": 6,
            "suite_ceiling_n": 3,
            "budget_ceiling_n": 21,
            "binding_constraints": ["suite_volume"],
            "must_generate": True,
        },
    )

    kinds = [a["kind"] for a in actions]
    assert "rebuild_data" in kinds
    assert "repair_harness" not in kinds
    rebuild = next(a for a in actions if a["kind"] == "rebuild_data")
    assert rebuild["owner"] == "synthesis-feedback"


def test_a_wall_budget_deficit_never_asks_for_data(tmp_path: Path) -> None:
    """Given the arm wall affords fewer records than the floor,
    when screening parks,
    then it must not queue a data rebuild -- no volume of records clears a wall.

    The regression this pins: the park hard-coded "suite_volume binds" and a
    `rebuild_data` action for every empty range. Under a wall bind that asks
    the synthesis owner for work that cannot lift the block; they publish
    records, the range stays empty, and the next cycle parks again. Naming the
    wrong cause is how a loop spends cycles without moving.
    """
    actions = _park(
        tmp_path,
        {
            "n_min": 6,
            "suite_ceiling_n": 96,
            "budget_ceiling_n": 1,
            "binding_constraints": ["wall_budget"],
            "must_generate": False,
        },
    )

    kinds = [a["kind"] for a in actions]
    assert "rebuild_data" not in kinds, "a wall bind is not a data deficit"
    assert "repair_harness" in kinds
    repair = next(a for a in actions if a["kind"] == "repair_harness")
    assert "wall_budget" in repair["reason"]
    assert "never a silent wall++" in repair["reason"]


def test_both_constraints_binding_queue_both_remedies(tmp_path: Path) -> None:
    actions = _park(
        tmp_path,
        {
            "n_min": 6,
            "suite_ceiling_n": 3,
            "budget_ceiling_n": 1,
            "binding_constraints": ["wall_budget", "suite_volume"],
            "must_generate": True,
        },
    )

    kinds = [a["kind"] for a in actions]
    assert "rebuild_data" in kinds
    assert "repair_harness" in kinds


def test_an_unclassified_deficit_still_asks_for_something(tmp_path: Path) -> None:
    """A report with no recorded cause must not park silently."""
    actions = _park(tmp_path, {})

    assert [a["kind"] for a in actions][0] == "rebuild_data"
    assert any("cause unrecorded" in a["reason"] for a in actions)


def test_every_park_leaves_a_next_experiment_action(tmp_path: Path) -> None:
    """The loop always records how it may resume, whatever bound it."""
    for index, report in enumerate(
        (
            {"binding_constraints": ["suite_volume"], "must_generate": True},
            {"binding_constraints": ["wall_budget"]},
            {},
        )
    ):
        # A fresh root per report: the park rewrites the handoff in place, so
        # reusing one would score the previous scenario's actions.
        actions = _park(tmp_path / f"park-{index}", report)
        assert actions[-1]["kind"] == "next_experiment"


# ---------------------------------------------------------------------------
# Scenario: admission and the trainer agree on what a usable record is.
# ---------------------------------------------------------------------------


def test_a_record_the_trainer_refuses_never_reaches_a_train_bucket(
    tmp_path: Path,
) -> None:
    """Given a corpus mixing clean records with ones the trainer refuses,
    when the certified partition is built,
    then the bad records are rejected at the role-safety stage and the
    surviving buckets contain only records the trainer accepts.

    The regression: admission applied one contract while the trainer applied
    four, so 29 records the trainer refuses reached the certified train bucket
    and every screening arm exited non-zero on the first batch.
    """
    from slm_training.harnesses.test_data.certified import partition_certified_corpus

    clean = 'root = Stack([c1])\nc1 = TextContent([":slot_0"])'
    rows = [
        {
            "id": f"clean-{index}",
            "prompt": f"Show card {index}",
            "openui": clean,
            "placeholders": [":slot_0"],
            "meta": {},
            "accepted_outputs": [],
            "split": "train",
            "source": "fixture",
        }
        for index in range(6)
    ]
    rows.append(
        {
            "id": "placeholder-in-non-content-property",
            "prompt": "Show a chart",
            "openui": 'root = Stack([c1])\nc1 = RadialChart([":slot_0"], [1])',
            "placeholders": [":slot_0"],
            "meta": {},
            "accepted_outputs": [],
            "split": "train",
            "source": "fixture",
        }
    )
    corpus = tmp_path / "records.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    part = partition_certified_corpus(corpus)

    rejected_ids = {row["id"] for row in part.rejected}
    assert "placeholder-in-non-content-property" in rejected_ids
    stages = {
        row["stage"]
        for row in part.rejected
        if row["id"] == "placeholder-in-non-content-property"
    }
    assert stages == {"role_safety"}

    from slm_training.harnesses.test_data.certified import _assert_certified_role_safe

    for split in ("train", "validation", "test"):
        for record in part.records(split):
            _assert_certified_role_safe(record)


# ---------------------------------------------------------------------------
# Scenario: a legal output is not automatically a successful generation.
# ---------------------------------------------------------------------------


def test_a_certified_substitution_is_never_persisted_as_a_generation() -> None:
    """Given decode falls back to a certified program to satisfy I6,
    when the serving harness inspects the attempt,
    then it refuses to persist it as a genuine generation.

    I6 ("never emit invalid grammar") is satisfied by the substitution, which
    is exactly why it needs its own channel: the output is legal, so nothing
    downstream can tell it apart from a real generation by looking at it.
    """
    from slm_training.web.service import (
        SubstitutedGeneration,
        _raise_on_substituted_generation,
    )

    class _Backend:
        def __init__(self, rows: list[dict]) -> None:
            self._rows = rows

        def consume_generation_evidence(self) -> list[dict]:
            rows, self._rows = self._rows, []
            return rows

    with pytest.raises(SubstitutedGeneration):
        _raise_on_substituted_generation(
            _Backend([{"fallback_used": False}, {"fallback_used": True}])
        )

    _raise_on_substituted_generation(_Backend([{"fallback_used": False}]))


def test_an_unrecognized_binding_constraint_still_asks_for_something(
    tmp_path: Path,
) -> None:
    """A third constraint must not make the park request nothing at all.

    Remedies are queued per recognized constraint, so a constraint neither
    branch knows would queue none -- a hole the old unconditional
    `rebuild_data` did not have. The fallback names the constraint so the owner
    can see it was not understood here.
    """
    actions = _park(
        tmp_path,
        {
            "n_min": 6,
            "suite_ceiling_n": 96,
            "budget_ceiling_n": 21,
            "binding_constraints": ["some_future_constraint"],
            "must_generate": False,
        },
    )

    kinds = [action["kind"] for action in actions]
    assert kinds[-1] == "next_experiment"
    assert len(kinds) > 1, "the park queued no remedy at all"
    assert any("some_future_constraint" in action["reason"] for action in actions)
