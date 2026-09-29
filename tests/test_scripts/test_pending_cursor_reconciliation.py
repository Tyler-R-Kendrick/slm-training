"""Readiness dispatch and cursor continuation regressions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.autotrain_pending import drain_driver_pending
from scripts.autotrain_readiness_probe import _verified_pending_advance
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from tests.test_autoresearch.test_evaluation_continuation import MANIFEST, outcome, pending
from tests.test_scripts.test_readiness_pending_dispatch import (
    _matrix_fixture,
    common_for,
    waiting_driver,
)

pytest_plugins = ("tests.test_scripts.test_readiness_pending_dispatch",)


@pytest.mark.parametrize("binding", ["unknown", "wall_budget"])
def test_driver_wall_or_unknown_dispatches_source_repair_not_data(compiled, monkeypatch, binding):
    from scripts import autotrain_readiness as readiness
    from scripts.autotrain_pending import resolve_screening_matrix

    matrix = _matrix_fixture(compiled)
    def forbidden(*args, **kwargs):
        pytest.fail("unknown/wall constraint generated evaluation data")
    monkeypatch.setattr(readiness, "resolve_matrix_readiness", forbidden)
    _, pending = resolve_screening_matrix(matrix, {"eval_version": "original"},
        {"n_min": 6, "binding_constraints": [binding]}, context={"cwd": compiled["cwd"],
            "root": compiled["cwd"] / "campaigns", "loop_id": "pair", "fitted_candidates": 1,
            "policy": SimpleNamespace(identity_dict=lambda: {"fixture": "controlled-selection"})})
    assert pending["blocker"]["kind"] == "repair_harness"
    assert pending["blocker"]["original_reproducer"]["argv"][2] == "scripts.autotrain_readiness_probe"
@pytest.mark.parametrize("remaining,restored", [(3, True), (4, False), (5, False)])
def test_pending_row_wake_uses_canonical_monotone_progress(tmp_path, remaining, restored):
    from scripts.autoresearch_command_cursor import record_execution_outcome

    previous, current = outcome("stopped", pending(n=4)), outcome("stopped", pending(n=remaining))
    store = CampaignStore(current.campaign_id, tmp_path)
    record_execution_outcome(store, current, MANIFEST, pending=True)
    value = {"order": [current.experiment_id], "arms": {current.experiment_id: {"manifest_digest": MANIFEST}}}
    assert _verified_pending_advance(store, value, set(),
        {"last_yield": previous.model_dump(mode="json")},
        {"index": 0, "last_yield": current.model_dump(mode="json")}) is restored
def test_pending_dispatch_round_robin_is_bounded_and_survives_controller_restart(tmp_path, monkeypatch):
    common = common_for({"cwd": tmp_path}, monkeypatch)
    store = CampaignStore("runtime", Path(common["root"]) / "loops/pair")
    pending = {"schema_version": "driver_pending/v1", "measurement_complete": False,
        "outcome": "capability", "reason": "fixture capability absent",
        "wake": {"predicate": "provider restored", "source": "provider_grant", "identity_digest": "e" * 64},
        "blocker": {"kind": "repair_harness", "blocker_code": "harness_code_failure"}}
    with ActivityRuntime(store) as runtime:
        for name in ("a", "b", "c"):
            waiting_driver(runtime, common, pending, name=name)
    calls = []
    def repair(_runtime, request, **kwargs):
        calls.append(request["affected_activity_id"])
        return None  # Unavailable capability; not a successful repair.
    for cycle in range(6):
        with ActivityRuntime(store) as runtime:
            drain_driver_pending(runtime, common, cycle, lambda _: None, repair)
            assert len(calls) == cycle + 1  # At most one remedy per pass.
            assert all(s.status == "waiting_capability" for s in runtime.snapshot().values())
    assert calls == ["a", "b", "c", "a", "b", "c"]
