"""Independent real-producer and real-process evidence-boundary falsifiers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_autoresearch.test_search_preflight_ingestion import (
    arms as fixture_arms,
    production_inputs as fixture_production_inputs,
    train_dir as fixture_train_dir,
)

arms = fixture_arms
production_inputs = fixture_production_inputs
train_dir = fixture_train_dir


def test_new_success_cannot_borrow_previous_attempt_loss_rows(production_inputs):
    from scripts.autotrain_nll import run_arm_eval_nll
    from scripts.autotrain_search import record_cycle_credit
    from slm_training.autoresearch.preflight.compiled_treatment import begin_attempt
    from tests.test_autoresearch.test_search_preflight_ingestion import _lock

    store, _, _, eval_root, model = production_inputs
    pair, _ = _lock(production_inputs)
    for eid in pair["arm_ids"]:
        attempt = begin_attempt(store, pair, eid)
        store.append_event(
            "experiment_attempt_returned",
            experiment_id=eid,
            detail={**attempt, "exit_code": 0},
        )
        run_arm_eval_nll(
            store.root / "runs" / eid, {"test_dir": eval_root, "model": model}
        )
    # The second successful invocation emitted no measurement. Its predecessor's
    # real, unmodified row file and matching smoke digest remain on disk.
    eid = pair["arm_ids"][1]
    attempt = begin_attempt(store, pair, eid)
    store.append_event(
        "experiment_attempt_returned",
        experiment_id=eid,
        detail={**attempt, "exit_code": 0},
    )
    record_cycle_credit(
        SimpleNamespace(store=store, value={"locked_designs": {eid: pair}})
    )
    events = store.verify_event_chain()
    assert not any(e["event_type"] == "search_effect_recorded" for e in events)
    assert any(e["event_type"] == "search_ingestion_refused" for e in events)


