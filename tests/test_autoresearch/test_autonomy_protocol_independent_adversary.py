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


def test_isolated_candidate_cannot_forge_pytest_success(tmp_path):
    from scripts.merge_verification_isolation import run_workload

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    # Ordinary source code, not the trusted runner or a patched test selector.
    # This proof only touches the sandbox's explicitly writable output file.
    (candidate / "module.py").write_text("""import json,os,pytest
from pathlib import Path
request=json.loads(Path('/workspace/control/request.json').read_text())
nodes=request['targets']
payload={'schema':'merge_test_workload/v1','request_digest':request['request_digest'],
         'exit_code':0,'pytest_version':pytest.__version__,'nodes':nodes,'markers':{},
         'deselected':[],'collection_errors':[],
         'reports':[{'nodeid':node,'when':phase,'outcome':'passed',
                     'duration_seconds':0.0,'wasxfail':False}
                    for node in nodes for phase in ('setup','call','teardown')]}
Path('/workspace/workload-result.json').write_text(json.dumps(payload))
os._exit(0)
""")
    (candidate / "test_case.py").write_text(
        "import module\ndef test_must_fail():\n    assert False, 'never executed'\n"
    )
    record = run_workload(
        candidate,
        ["test_case.py::test_must_fail"],
        collect_only=False,
        seconds=30,
        directory=tmp_path / "controller",
        isolated=True,
        runtimes=(Path(sys.prefix),),
    )
    assert record["status"] != "ok", record
    assert "workload_sha256" not in record


@pytest.mark.parametrize("field", ["integration_commit", "upstream_commit"])
def test_fresh_handoff_source_identity_must_match_locked_context(
    tmp_path, monkeypatch, field
):
    import json

    from scripts import autotrain_cycle_execution as execution
    from scripts import run_autotrain_continuous as driver
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from tests.test_autoresearch.test_driver_cycle_continuation import _fixture, _resume

    f = _fixture(tmp_path, monkeypatch, arms=2)
    producer = driver._write_cycle_handoff

    def wrong_release(**kwargs):
        producer(**kwargs)
        path = f.store.root / "cycle_handoff.json"
        payload = json.loads(path.read_text())
        payload[field] = "f" * 40
        AutotrainCycleHandoffV1.model_validate(payload)
        path.write_text(json.dumps(payload))

    monkeypatch.setattr(driver, "_write_cycle_handoff", wrong_release)
    assert _resume(f)["outcome"] == "yielded"
    prior = execution.cycle_event_ids(f.root, "test-loop")
    with pytest.raises(ValueError):
        _resume(f)
        execution.completed_cycle_since(
            f.cwd, f.root, "test-loop", f.store.campaign_id, prior
        )
